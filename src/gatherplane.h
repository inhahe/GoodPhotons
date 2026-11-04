#pragma once
// Which stored photons and light vertices a density estimate's gather keeps (MERGE-PROXIMITY), for
// every gather on the host: modes M and S (photonmap.h, and the footprint that divides their
// estimate, gafootprint.h), U (vcm.h) and J (bdpt.h). The device's twins are in render_cuda.cu.
#include <algorithm>
#include <cmath>
#include "linalg.h"

// MERGE-PROXIMITY: which stored photons lie on the surface a gather point is on. A gather takes the
// photons within a ball, and a second surface inside it -- the floor under a table top, once the
// radius outgrows the table's height -- would lend it light it never received. A photon belongs if
// it lies near the gather point's tangent plane: off it by at most (kGatherPlaneTol + sin(theta))
// times its distance d from the point, theta the angle between their normals, plus `slack` for
// rounding. A smooth surface bends about d*sin(theta)/2 off its tangent plane over d, so a curved
// (or faceted) one keeps its photons, where a parallel surface above or below does not.
// `withinGatherBand` takes the offset `off` (>= 0), d^2 and the cosine `c` between the normals.
constexpr double kGatherPlaneTol = 0.1;
inline bool withinGatherBand(double off, double d2, double c, double slack) {
    if (off <= slack) return true;
    const double s = std::sqrt(std::max(0.0, 1.0 - c * c));
    const double k = kGatherPlaneTol + s;
    return (off - slack) * (off - slack) <= k * k * d2;
}
// The slack of a gather of radius `radius` at a point on a curve of radius `fiberR` (0 off a curve):
// 1e-3 of the radius, or none of the band at all on a curve. A strand -- `type diffuse` fur, grass,
// wire -- is far thinner than the radius, so it has no tangent plane at that scale: the photons
// around a fur gather lie on the neighbouring strands, which the band would drop while the estimate
// still divides by the whole disc (on fur_creature, up to 30 % darker in mode M). The facing test
// stays. Mode J never merges at a curve vertex (surfMergeSite); modes M, S and U do.
inline double gatherBandSlack(double radius, double fiberR) {
    return fiberR > 0.0 ? HUGE_VAL : radius * 1e-3;
}
// Modes M and S: a photon at `pp`, true normal `pn` (on the side its light came from), for a gather
// at `x`, true normal `n`: facing the same way, the 60-degree test every such gather keeps, and in
// the band. `slack` must be the same wherever one gather's photons are counted and summed (mode M's
// radius and its sum: PhotonMap::gatherSlack), or the two see different populations.
inline bool onGatherSurface(const Vec3& x, const Vec3& n, const Vec3& pp, const Vec3& pn, double slack) {
    const double c = dot(pn, n);
    if (c < 0.5) return false;
    const Vec3 d = pp - x;
    return withinGatherBand(std::fabs(dot(d, n)), dot(d, d), c, slack);
}
// Modes U and J, whose stored vertices' normals are not oriented to a side (mergeFromBehind decides
// that from the light's direction): in the band, whichever way the normal points.
inline bool nearGatherPlane(const Vec3& x, const Vec3& n, const Vec3& pp, const Vec3& pn, double slack) {
    const Vec3 d = pp - x;
    return withinGatherBand(std::fabs(dot(d, n)), dot(d, d), std::fabs(dot(pn, n)), slack);
}
