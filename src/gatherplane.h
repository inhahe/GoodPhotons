#pragma once
// Which stored photons a mode-M or mode-S density estimate keeps (MERGE-PROXIMITY): photonmap.h's
// adaptive radius, the gathers in photonmap_render.h and sppm_render.h, and the footprints that
// divide their estimates (photonmap_render.h's probe, gafootprint.h). The scalar rules are
// templates both backends call, so the device's vector glue (render_cuda.cu, dOnGatherSurfaceSided)
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
FT_GP_HD inline float  gpEps(float)  { return 1.1920929e-7f; }            // FLT_EPSILON
FT_GP_HD inline double gpEps(double) { return 2.220446049250313e-16; }    // DBL_EPSILON

// MERGE-PROXIMITY: which stored photons lie on the surface a gather point is on. A gather takes the
// photons within a ball, and a second surface inside it -- the floor under a table top, once the
// radius outgrows the table's height, or the plate a few millimetres behind a sheet -- would lend
// it light it never received. A photon belongs if it lies near the gather point's tangent plane:
// off it by at most sin(theta) times its distance d from the point, theta the angle between their
// normals, plus `slack` for rounding (gatherBandSlack). A smooth surface bends about d*sin(theta)/2
// off its tangent plane over d, and a facet tilted by theta meets its neighbour's plane within
// d*sin(theta), so a curved or faceted surface keeps its photons, where a parallel surface above or
// below does not, however close. `withinGatherBand` takes the offset `off` (>= 0), d^2 and the
// cosine `c` between the normals.
// And it faces the gather point's way, within 60 degrees: the normal test of every photon gather,
// on true normals oriented to the side their light came from.
constexpr double kGatherFacingCos = 0.5;
// The footprint's low leg (gafootprint.h walkSegment, the device's dWalkSegment): its height above
// a gather point's tangent plane, as a share of the radius -- under any lid the parity can count,
// over the surface's own relief.
constexpr double kLowLegFrac = 0.01;

template <class T> FT_GP_HD inline bool withinGatherBand(T off, T d2, T c, T slack) {
    if (off <= slack) return true;
    T q = T(1) - c * c;
    if (q < T(0)) q = T(0);
    const T e = off - slack;
    return e * e <= q * d2;
}

// The rounding the slack allows a position computed in T at the coordinate scale `pscale`, in units
// of T's epsilon: two positions' offset along a normal carries a few of them (the ray origin, the
// hit's t, the deposit).
constexpr double kGatherRoundUlps = 8.0;

// The slack of a gather of radius `radius` at a point on a curve of radius `fiberR` (0 off a curve)
// whose positions round at the coordinate scale `pscale` (Hit::pscale, the hit's primitive's
// distance from the origin): 1e-3 of the radius, plus the rounding of positions computed in T
// there -- what lets the band, which has no flat allowance, keep a flat surface's own photons. In
// double (the host) that rounding is far below the radius's thousandth, and the host passes none;
// in float (the device under FTRACE_GPU_FP32) it is about 1e-7 of the coordinates, which outgrows
// a small radius's thousandth far from the origin (a room 3 km out, gathered without it, lost its
// own surface to the band and read 4x bright). Or none of the band at all on a strand thinner than
// the radius. Such a strand -- `type diffuse` fur, grass, wire -- has no tangent plane at the
// radius's scale: the photons around a fur gather lie on the neighbouring strands, which the band
// would drop while the estimate still divides by the whole disc. Only a `diffuse_transmit` or
// fluorescent strand gathers at all: modes M and S walk on past a `type diffuse` one
// (CURVE-GATHER). A curve as thick as the radius is a surface at that scale, and takes the band as
// one. The facing test stays. One value per gather point, so its count, sum and footprint agree.
template <class T> FT_GP_HD inline T gatherBandSlack(T radius, T fiberR, T pscale = T(0)) {
    if (fiberR > T(0) && fiberR < radius) return T(HUGE_VAL);
    return radius * T(1e-3) + T(kGatherRoundUlps) * gpEps(T(0)) * pscale;
}

// THE GATHER TEST: a photon at `pp`, true normal `pn` (on the side its light came from), for a gather
// at `x`, true normal `n`; `d2` = |pp - x|^2, which every caller has from its radius test. `slack` must
// be the same wherever one gather's photons are counted and summed (mode M's radius, its sum and its
// footprint: PhotonMap::gatherSlack), or they see different populations.
//
// A gather on a TWO-SIDED surface (`twoSided`: a `translucent` visible point, TRANSLUCENT-MS) takes the
// photons on either side of it: those facing the gather point's way (its near side, lit as a
// reflector) and those facing the other (`farSide`: the light the sheet transmits from behind, which
// it deposits on its far face). The facing test on the normals' cosine `c`, the band on |c|: a
// far-side photon must lie on the sheet itself, within its curvature (d sin theta) and the slack, as
// a near one must -- the far face of a thin sheet is the sheet, while the back face of a thick
// translucent solid, parallel to it a thickness away, is another surface whose photons are the
// light arriving at that face, not the light this one transmits. One-sided (`twoSided` false), the
// near side's alone.
template <class T> FT_GP_HD inline bool gatherFacing(T c, bool twoSided, bool& farSide) {
    farSide = false;
    if (c >= T(kGatherFacingCos)) return true;
    if (twoSided && c <= -T(kGatherFacingCos)) { farSide = true; return true; }
    return false;
}
inline bool onGatherSurfaceSided(const Vec3& x, const Vec3& n, const Vec3& pp, const Vec3& pn, double d2,
                                 double slack, bool twoSided, bool& farSide) {
    const double c = dot(pn, n);
    if (!gatherFacing(c, twoSided, farSide)) return false;
    return withinGatherBand(std::fabs(dot(pp - x, n)), d2, std::fabs(c), slack);
}
