// Realistic (physical) multi-element camera lens — the "mesh-lens" camera.
//
// A real photographic lens is a stack of glass elements separated by air gaps,
// with a diaphragm (aperture stop) somewhere in the middle. Each element face is a
// spherical (or planar) refracting interface. Instead of tessellating those faces
// into triangle meshes (which would only add discretisation error), we represent
// each interface *analytically* by its signed radius of curvature — the exact
// geometry of the real glass surface. Rays are refracted (Snell, per-wavelength, so
// dispersion / chromatic aberration is automatic) through the whole stack; the
// diaphragm and each element's clear semi-diameter clip stray rays, giving true
// vignetting. Focus, depth of field, distortion (including fisheye barrel), coma and
// spherical aberration all emerge from the geometry — there is no analytic
// projection or thin-lens approximation here.
//
// This is the backward "realistic camera" formulation (cf. PBRT's RealisticCamera):
// the backward reference tracer (mode R) samples a point on the film and a point on
// the rear element, traces that ray from the film out through the elements into the
// world, and path-traces from there. Vignetted rays contribute nothing; surviving
// rays carry a radiometric weight (cos^4 / Z^2 * pupil-area) so exposure, DoF and
// natural corner darkening are physically correct.
//
// Coordinate convention (lens-local, millimetres): the optical axis is +z, pointing
// from the sensor toward the scene. The sensor plane is at z = filmZ (0 at design
// focus); each surface vertex sits at a cached zpos. Surfaces are stored front-to-
// rear, i.e. scene-side first (index 0) to sensor-side last. Each surface's `ior` is
// the index of the medium immediately on its *sensor* side (the standard lens-
// prescription convention: the index of the space that follows the surface toward
// the image). The camera maps this lens-local frame to world via (u,v,w) with the
// front vertex plane pinned at the camera eye.
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include "linalg.h"
#include "geometry.h"   // Ray, PI
#include "spectrum.h"           // Spectrum, iorConstant
#include "spectral_library.h"   // defaultGlass (BK7/SF10 dispersion)
#include "parallel.h"           // ft::parallelFor (the pupil table's build)
#include "rng.h"                // Pcg32

// One refracting interface of the lens (a spherical cap, or a plane when radius==0).
struct LensSurface {
    double   radius   = 0.0;    // signed radius of curvature (mm); 0 => planar (stop/flat)
    double   thickness = 0.0;   // axial gap to the next surface toward the sensor (mm);
                                // for the rear surface this is the surface->sensor distance
    Spectrum ior;               // index of the medium on the sensor side of this surface
                                // (fn of wavelength; air => 1). Empty => treated as air.
    double   aperture = 0.0;    // clear semi-diameter / stop radius (mm)
    bool     isStop   = false;  // true for the diaphragm (aperture stop) plane
};

struct LensSystem {
    std::vector<LensSurface> surf;   // front (scene, idx 0) -> rear (sensor, idx M-1)
    double filmW_mm = 36.0;          // sensor width  (mm) — default full-frame
    double filmH_mm = 24.0;          // sensor height (mm)
    std::string name = "lens";       // for logging

    // Derived (filled by finalize):
    std::vector<double> zpos;        // vertex z of each surface (mm); sensor nominally at 0
    double T      = 0.0;             // total track: front-surface vertex z (mm)
    double filmZ  = 0.0;             // sensor plane z (mm); autofocus shifts this (<=0 moves back)

    // Index of the medium on the sensor side of surface j (air if unset).
    double iorSensorSide(int j, double lambda) const {
        return surf[j].ior ? surf[j].ior(lambda) : 1.0;
    }
    // Index of the medium on the scene side of surface j (= sensor side of j-1; air at front).
    double iorSceneSide(int j, double lambda) const {
        return j == 0 ? 1.0 : iorSensorSide(j - 1, lambda);
    }

    // Cache vertex z-positions. Sensor at z=0, front surface at z=T (mm).
    void finalize() {
        int M = (int)surf.size();
        zpos.assign(M, 0.0);
        T = 0.0;
        for (const auto& s : surf) T += s.thickness;
        if (M > 0) {
            zpos[0] = T;
            for (int j = 1; j < M; ++j) zpos[j] = zpos[j - 1] - surf[j - 1].thickness;
        }
        filmZ = 0.0;
    }
    double rearAperture() const { return surf.empty() ? 0.0 : surf.back().aperture; }
    double rearZ()        const { return zpos.empty() ? 0.0 : zpos.back(); }

    // Refract incident unit dir `d` at unit normal `n` (oriented against d) with
    // eta = n_incident / n_transmitted. Returns false on total internal reflection.
    static bool refractDir(const Vec3& d, const Vec3& n, double eta, Vec3& out) {
        double cosi = -dot(d, n);                 // >0 (n faces the incoming ray)
        double k = 1.0 - eta * eta * (1.0 - cosi * cosi);
        if (k < 0.0) return false;                // TIR — treat as a blocked ray
        out = d * eta + n * (eta * cosi - std::sqrt(k));
        out = normalize(out);
        return true;
    }

    // Intersect a lens-local ray (o,d) with surface j. Returns the hit point and the
    // surface normal oriented against d. `false` if it misses or is clipped by the
    // surface's clear aperture (vignetting). Planar surfaces (radius 0) intersect the
    // vertex plane z = zpos[j].
    bool hitSurface(int j, const Vec3& o, const Vec3& d, Vec3& hitP, Vec3& nrm) const {
        double zv = zpos[j];
        double R  = surf[j].radius;
        double ap = surf[j].aperture;
        if (R == 0.0) {                                   // planar (aperture stop / flat)
            if (std::fabs(d.z) < 1e-12) return false;
            double t = (zv - o.z) / d.z;
            if (t < 1e-9) return false;
            hitP = o + d * t;
            nrm = Vec3(0, 0, d.z > 0 ? -1.0 : 1.0);       // face against the ray
        } else {
            Vec3 C(0, 0, zv + R);                         // sphere centre on the axis
            Vec3 op = o - C;
            double b = dot(op, d);                        // a = 1 (d is unit)
            double c = dot(op, op) - R * R;
            double disc = b * b - c;
            if (disc < 0.0) return false;
            double sq = std::sqrt(disc);
            double t0 = -b - sq, t1 = -b + sq;
            // Pick the intersection on the correct side (PBRT sphere-element rule).
            bool closer = (d.z > 0.0) ^ (R < 0.0);
            double t = closer ? std::fmin(t0, t1) : std::fmax(t0, t1);
            if (t < 1e-9) { t = closer ? std::fmax(t0, t1) : std::fmin(t0, t1); }
            if (t < 1e-9) return false;
            hitP = o + d * t;
            nrm = normalize(hitP - C);
            if (dot(nrm, d) > 0.0) nrm = -nrm;            // face against the ray
        }
        if (hitP.x * hitP.x + hitP.y * hitP.y > ap * ap) return false;   // clipped
        return true;
    }

    // Trace a lens-local ray through every interface. `sensorToScene` picks the
    // iteration order (rear->front for camera-ray generation; front->rear for the
    // paraxial focus probe). Returns the outgoing ray in air on success.
    bool trace(const Vec3& o0, const Vec3& d0, double lambda,
               bool sensorToScene, Ray& outLocal) const {
        Vec3 o = o0, d = normalize(d0);
        int M = (int)surf.size();
        if (sensorToScene) {
            for (int j = M - 1; j >= 0; --j) {
                Vec3 hp, n;
                if (!hitSurface(j, o, d, hp, n)) return false;
                o = hp;
                if (surf[j].radius != 0.0) {
                    double eta = iorSensorSide(j, lambda) / iorSceneSide(j, lambda);
                    Vec3 nd;
                    if (!refractDir(d, n, eta, nd)) return false;
                    d = nd;
                }
            }
        } else {
            for (int j = 0; j < M; ++j) {
                Vec3 hp, n;
                if (!hitSurface(j, o, d, hp, n)) return false;
                o = hp;
                if (surf[j].radius != 0.0) {
                    double eta = iorSceneSide(j, lambda) / iorSensorSide(j, lambda);
                    Vec3 nd;
                    if (!refractDir(d, n, eta, nd)) return false;
                    d = nd;
                }
            }
        }
        outLocal = Ray{o, d};
        return true;
    }

    // Paraxial autofocus: shift the sensor plane (filmZ) so an on-axis object point
    // at `focusDist` metres in front of the front vertex images sharply. focusDist<=0
    // => focus at infinity (parallel rays). Uses a near-axis probe ray traced
    // scene->sensor at the design wavelength.
    void focusAt(double focusDist) {
        if (surf.empty()) return;
        const double lambdaD = 587.6;                     // Fraunhofer d-line
        const double h = std::fmax(0.05, 0.02 * rearAperture());   // small probe height (mm)
        Vec3 o, d;
        if (focusDist > 0.0) {
            double pmm = focusDist * 1000.0;              // object distance (mm)
            Vec3 O(0.0, 0.0, T + pmm);                    // on-axis object point (scene side)
            Vec3 aim(h, 0.0, zpos[0]);                    // aim at the front rim height h
            o = O; d = normalize(aim - O);
        } else {
            o = Vec3(h, 0.0, T + 1.0);                    // parallel ray at height h
            d = Vec3(0.0, 0.0, -1.0);
        }
        Ray out;
        if (!trace(o, d, lambdaD, /*sensorToScene=*/false, out)) return;   // keep default
        if (std::fabs(out.d.x) < 1e-12) return;
        double t = -out.o.x / out.d.x;                    // where the ray crosses the axis
        double zCross = out.o.z + out.d.z * t;            // paraxial image plane z
        filmZ = zCross;                                   // move the sensor to it
    }

    // ---- a surface's clear cap, for sampling points on it ------------------------------
    // Light tracing picks where a scene point's light enters the lens, a point on the clear
    // cap of the FRONT element (surface 0), and traces it through to the sensor (Hanika &
    // Dachsbacher 2014, "Efficient Monte Carlo rendering with realistic lenses", sec. 3.2).
    // The camera, when its rear element is curved, picks where its ray leaves toward the rear
    // the same way. Sampling the cap uniformly by area covers every ray that can cross the
    // clear part of a curved surface, from any direction; a disc in a plane does not.
    //
    // Area of surface j's clear part (mm^2): a spherical cap, or a disc when planar.
    double surfaceArea(int j) const {
        const double R = std::fabs(surf[j].radius), ap = surf[j].aperture;
        if (R == 0.0) return PI * ap * ap;
        const double sag = R - std::sqrt(std::fmax(0.0, R * R - ap * ap));
        return 2.0 * PI * R * sag;
    }
    double frontArea() const { return surf.empty() ? 0.0 : surfaceArea(0); }
    // The radius about surface j's vertex that holds its whole clear cap: the rim's distance
    // from the vertex, sqrt(aperture^2 + sag^2).
    double capRadius(int j) const {
        const double R = std::fabs(surf[j].radius), ap = surf[j].aperture;
        const double sag = (R == 0.0) ? 0.0 : R - std::sqrt(std::fmax(0.0, R * R - ap * ap));
        return std::sqrt(ap * ap + sag * sag);
    }
    // A point uniform by area on surface j's clear cap, and the unit surface normal there on
    // the +z side. Archimedes: on a sphere, area is uniform in the axial coordinate, so the
    // cap is sampled by a uniform cos(alpha).
    Vec3 sampleSurface(int j, double u1, double u2, Vec3& n) const {
        const double R = surf[j].radius, ap = surf[j].aperture, zv = zpos[j];
        const double phi = 2.0 * PI * u2;
        if (R == 0.0) {
            const double r = ap * std::sqrt(u1);
            n = Vec3(0, 0, 1);
            return Vec3(r * std::cos(phi), r * std::sin(phi), zv);
        }
        const double aR = std::fabs(R);
        const double cosMax = std::sqrt(std::fmax(0.0, 1.0 - (ap / aR) * (ap / aR)));
        const double ca = 1.0 - u1 * (1.0 - cosMax);           // uniform in [cosMax, 1]
        const double sa = std::sqrt(std::fmax(0.0, 1.0 - ca * ca));
        // The centre sits at zv + R; the vertex lies from it along -sign(R) z.
        const double s = (R < 0.0) ? 1.0 : -1.0;
        Vec3 P(aR * sa * std::cos(phi), aR * sa * std::sin(phi), zv + R + s * aR * ca);
        Vec3 out = normalize(P - Vec3(0, 0, zv + R));
        n = (out.z >= 0.0) ? out : out * -1.0;
        return P;
    }
    // Trace a lens-local ray that comes FROM the scene (d.z < 0) through every interface
    // and onto the sensor plane z = filmZ. `rearExit`, if given, receives the point where it
    // left the last surface. False if any element or the stop clips it, on total internal
    // reflection, or if it never reaches the sensor plane. The caller starts `o` near the
    // lens: from a scene point thousands of mm out, the first sphere solve would cancel most
    // of its digits.
    bool traceToFilm(const Vec3& o, const Vec3& d, double lambda, Vec3& onFilm,
                     Vec3& dFilm, Vec3* rearExit = nullptr) const {
        if (surf.empty()) return false;
        Ray out;
        if (!trace(o, d, lambda, /*sensorToScene=*/false, out)) return false;
        if (!(out.d.z < -1e-12)) return false;               // not heading for the sensor
        const double t = (filmZ - out.o.z) / out.d.z;
        if (!(t > 0.0)) return false;
        onFilm = out.o + out.d * t;
        dFilm = out.d;
        if (rearExit) *rearExit = out.o;
        return true;
    }

    // Effective focal length (mm), measured from a near-axis parallel probe at the
    // design wavelength: f = h / tan(angle of the emergent ray). Used for reporting.
    double focalLengthMM() const {
        if (surf.empty()) return 0.0;
        const double lambdaD = 587.6;
        double h = std::fmax(0.05, 0.02 * rearAperture());
        Vec3 o(h, 0.0, T + 1.0), d(0.0, 0.0, -1.0);
        Ray out;
        if (!trace(o, d, lambdaD, /*sensorToScene=*/false, out)) return 0.0;
        double slope = -out.d.x / out.d.z;                // dx/dz of the emergent ray
        if (std::fabs(slope) < 1e-12) return 0.0;
        return std::fabs(h / slope);                      // report the magnitude (mm)
    }
};

// ---------------------------------------------------------------------------
// The entrance pupil, for aiming light at the stop.
//
// Light tracing picks where light from a scene point enters the lens. Uniform over the front
// element's clear cap (sampleSurface) is exact for any lens, but when the front element is
// much wider than the beam the stop admits -- a fast wide-angle, any lens stopped down -- most
// picks are clipped inside, and for a point outside the field of view every pick is traced
// through the glass only to miss the film. Hanika & Dachsbacher (2014, sec. 5.2) and Schrade
// et al. (2016) aim at the stop instead, solving per sample for the direction that reaches a
// sampled point of it (Newton steps on a fitted polynomial): several traces per sample. This
// table aims for the price of a lookup.
//
// The rays from a scene point that get through to the film cross a plane at the entrance
// pupil (the stop's image seen from the scene) in a region that, a pupil being what it is,
// barely moves with the point's distance and moves slowly with its angle off the axis. The
// table bins scene points by that angle and by inverse distance and keeps, per bin, a region
// on that plane -- an ellipse, or a rectangle where an ellipse fits badly -- holding every
// crossing that reaches the film. lensEntry then draws the crossing uniformly in the region
// and aims at it. The lens is rotationally symmetric, so a bin's region is built for points at
// azimuth 0 and serves every azimuth rotated.
//
// The regions come from tracing, not from paraxial optics, so aberrations, vignetting by any
// element and the film's edge are all in them. Per bin, over scene points from the bin
// widened by 15% on every side: rays drawn the old way (uniform on the front cap) find where
// light gets through; rays aimed inside a growing ellipse find its edge; the region is fitted
// and padded; then fresh batches of uniform rays check that nothing reaching the film falls
// outside, enlarging the region over anything that does, until a whole batch finds nothing
// (the count is kept for -checklens). A bin keeps the front sampler where aiming would not
// pay -- the stop already fills the front element -- or where its region never comes clean,
// and a bin where no light got through, nor in any neighbouring bin, rejects outright, before
// any tracing. The film edge used is a circle round the sensor's corners, which the lens's
// rotational symmetry needs; a camera whose raster outgrows that circle, or whose
// focus moves, stops using the table (Camera::pupil).
struct LensPupilCell {
    double cx = 0.0, a = 0.0, b = 0.0;   // region on z = zRef (mm) for azimuth 0: centred at
                                         // (cx, 0), half-widths a (radial, along x) and b (along y)
    int kind = 0;                        // LensPupil::FRONT / AIM / NONE / AIMBOX
};
struct LensPupil {
    // AIM draws in the ellipse with semi-axes a, b; AIMBOX in the rectangle with half-widths
    // a, b, for a region an ellipse fits badly (a crescent at the edge of the field, where
    // the film's rim cuts the beam).
    enum { FRONT = 0, AIM = 1, NONE = 2, AIMBOX = 3 };
    double zRef = 0.0;     // the plane crossings are drawn on (mm): the entrance pupil's
    double zStart = 0.0;   // above every point of the front cap: where an aimed leg starts
    double dMin = 0.0;     // scene points nearer than this to (0, 0, zRef) keep the front sampler
    double racc2 = 0.0;    // squared film radius (mm^2) the table admits rays within
    double filmZ = 0.0;    // the sensor plane it was built for
    int nTheta = 0, nU = 0;
    double dTheta = 0.0;   // bin width in the angle off the axis, over [0, pi/2)
    std::vector<LensPupilCell> cells;   // [iTheta * nU + iU], iU = floor(nU * dMin / distance)
    int stop = -1;                      // the surface taken as the stop (for the report)
    int nAim = 0, nFront = 0, nNone = 0;
    long long enlarged = 0;             // uniform rays that reached the film outside their bin's
                                        // fitted region, each of which enlarged it

    // Which sampler light from lens-local point p takes: a bin index >= 0 to aim (cphi, sphi
    // then hold p's azimuth), -1 for the front sampler, -2 when no light from p reaches the
    // film. Points beside the glass or near it keep the front sampler.
    int classify(const Vec3& p, double& cphi, double& sphi) const {
        if (!(p.z > zStart)) return -1;
        const double r2 = p.x * p.x + p.y * p.y, qz = p.z - zRef;
        const double D2 = r2 + qz * qz;
        if (!(D2 >= dMin * dMin)) return -1;
        const double D = std::sqrt(D2);
        int it = (int)(std::acos(std::min(1.0, qz / D)) / dTheta);
        int iu = (int)(dMin / D * nU);
        it = std::min(std::max(it, 0), nTheta - 1);
        iu = std::min(std::max(iu, 0), nU - 1);
        const int c = it * nU + iu;
        const int k = cells[c].kind;
        if (k == NONE) return -2;
        if (k != AIM && k != AIMBOX) return -1;
        const double r = std::sqrt(r2);
        if (r > 0.0) { cphi = p.x / r; sphi = p.y / r; } else { cphi = 1.0; sphi = 0.0; }
        return c;
    }
    // A crossing uniform in bin c's region, rotated to azimuth (cphi, sphi).
    Vec3 sampleCrossing(int c, double cphi, double sphi, double u1, double u2) const {
        const LensPupilCell& C = cells[c];
        double ex, ey;
        if (C.kind == AIMBOX) {
            ex = C.cx + C.a * (2.0 * u1 - 1.0);
            ey = C.b * (2.0 * u2 - 1.0);
        } else {
            const double rr = std::sqrt(u1), an = 2.0 * PI * u2;
            ex = C.cx + C.a * rr * std::cos(an);
            ey = C.b * rr * std::sin(an);
        }
        return Vec3(ex * cphi - ey * sphi, ex * sphi + ey * cphi, zRef);
    }
    // How far crossing X lies out in bin c's region at azimuth (cphi, sphi): <= 1 inside.
    double cellNorm(int c, const Vec3& X, double cphi, double sphi) const {
        const LensPupilCell& C = cells[c];
        const double ex = X.x * cphi + X.y * sphi, ey = -X.x * sphi + X.y * cphi;
        return regionNorm(C, ex, ey);
    }
    static double regionNorm(const LensPupilCell& C, double ex, double ey) {
        const double qx = (ex - C.cx) / C.a, qy = ey / C.b;
        return (C.kind == AIMBOX) ? std::max(std::fabs(qx), std::fabs(qy)) : std::sqrt(qx * qx + qy * qy);
    }
    // Is crossing X inside bin c's region? The slack covers the rounding between a crossing
    // drawn and the same crossing recomputed from its ray.
    bool inCell(int c, const Vec3& X, double cphi, double sphi) const {
        return cellNorm(c, X, cphi, sphi) <= 1.0 + 1e-9;
    }
    double cellArea(int c) const {
        const LensPupilCell& C = cells[c];
        return (C.kind == AIMBOX ? 4.0 : PI) * C.a * C.b;
    }
};

// Build the table for lens L focused as it is, admitting rays that land within `racc` mm of
// the sensor's centre. Null if the lens is empty or the build was stopped (-stop).
inline std::shared_ptr<const LensPupil> buildLensPupil(const LensSystem& L, double racc) {
    const int M = (int)L.surf.size();
    if (M == 0 || !(racc > 0.0)) return nullptr;
    auto P = std::make_shared<LensPupil>();
    const double lamD = 587.6;
    // The stop: the marked one, else the surface whose paraxial entrance pupil (its aperture
    // scaled by the height a ray parallel to the axis keeps from the scene to it) is smallest,
    // the textbook definition.
    int stop = -1;
    for (int j = 0; j < M; ++j) if (L.surf[j].isStop) { stop = j; break; }
    if (stop < 0) {
        const double h0 = 1e-3 * std::max(1e-3, L.surf[0].aperture);
        Vec3 o(h0, 0.0, L.T + 1.0), d(0.0, 0.0, -1.0);
        double best = 1e300;
        for (int j = 0; j < M; ++j) {
            Vec3 hp, n;
            if (!L.hitSurface(j, o, d, hp, n)) break;
            const double yj = std::fabs(hp.x);
            if (yj > 1e-15) {
                const double rep = L.surf[j].aperture * h0 / yj;
                if (rep < best) { best = rep; stop = j; }
            }
            o = hp;
            if (L.surf[j].radius != 0.0) {
                Vec3 nd;
                if (!LensSystem::refractDir(d, n, L.iorSceneSide(j, lamD) / L.iorSensorSide(j, lamD), nd)) break;
                d = nd;
            }
        }
        if (stop < 0) stop = 0;
    }
    // The entrance pupil's plane: where the chief ray, leaving the stop's centre for the scene
    // at a small angle, crosses the axis in object space. A plane elsewhere would still be
    // exact, only less tight.
    double zRef = L.zpos[stop];
    if (stop > 0) {
        const double ang = 1e-3;
        Vec3 o(0.0, 0.0, L.zpos[stop]), d(std::sin(ang), 0.0, std::cos(ang));
        bool ok = true;
        for (int j = stop - 1; j >= 0 && ok; --j) {
            Vec3 hp, n;
            if (!L.hitSurface(j, o, d, hp, n)) { ok = false; break; }
            o = hp;
            if (L.surf[j].radius != 0.0) {
                Vec3 nd;
                if (!LensSystem::refractDir(d, n, L.iorSensorSide(j, lamD) / L.iorSceneSide(j, lamD), nd)) ok = false;
                else d = nd;
            }
        }
        if (ok && std::fabs(d.x) > 1e-12) {
            const double z = o.z - o.x * d.z / d.x;
            if (std::isfinite(z)) zRef = z;
        }
    }
    const double capR = L.capRadius(0), ap0 = L.surf[0].aperture;
    const double track = L.zpos[0] - L.zpos[M - 1];
    zRef = std::max(std::min(zRef, L.zpos[0]), L.zpos[M - 1] - track - 10.0);
    P->stop = stop;
    P->zRef = zRef;
    P->zStart = L.zpos[0] + capR + 1.0;
    P->dMin = (P->zStart - zRef) + 4.0 * capR;
    P->racc2 = racc * racc;
    P->filmZ = L.filmZ;
    P->nTheta = 64; P->nU = 8;
    P->dTheta = 0.5 * PI / P->nTheta;
    const int nCells = P->nTheta * P->nU;
    P->cells.assign((size_t)nCells, LensPupilCell{});
    std::vector<char> lit((size_t)nCells, 0);          // did any light get through in this bin?
    std::vector<long long> enl((size_t)nCells, 0);
    const double lamLo = 360.0, lamHi = 830.0;          // every wavelength a scene can carry
    const bool done = ft::parallelFor((size_t)nCells, 1, [&](size_t ci) {
        const int it = (int)ci / P->nU, iu = (int)ci % P->nU;
        Pcg32 rng; rng.seed(0x9e3779b97f4a7c15ULL + ci, 0x5e1ec7ULL);
        // A scene point at azimuth 0 (the +x half of the x-z plane) in this bin widened by 15% of
        // its width on every side: a region's extremes sit at the bin's edges, which points
        // drawn only inside would approach but never reach. False where the point lies beside
        // the glass, which classify never aims from.
        auto point = [&](Vec3& p) {
            constexpr double kWiden = 0.15;
            const double th = std::min(std::max((it - kWiden + (1.0 + 2.0 * kWiden) * rng.uniform()) * P->dTheta,
                                                0.0), 0.5 * PI - 1e-9);
            const double u = std::max((iu - kWiden + (1.0 + 2.0 * kWiden) * (1.0 - rng.uniform())) / P->nU,
                                      1e-7);
            const double D = P->dMin / u;
            p = Vec3(D * std::sin(th), 0.0, zRef + D * std::cos(th));
            return p.z > P->zStart;
        };
        auto reaches = [&](const Vec3& o, const Vec3& d) {
            Vec3 f, df;
            if (!L.traceToFilm(o, d, lamLo + (lamHi - lamLo) * rng.uniform(), f, df)) return false;
            return f.x * f.x + f.y * f.y <= P->racc2;
        };
        // One ray the front sampler's way: -1 no usable point, 0 lost, 1 reached the film
        // (X = its crossing, y folded to >= 0 by the mirror symmetry about the x-z plane).
        auto frontRay = [&](Vec3& X) {
            Vec3 p;
            if (!point(p)) return -1;
            Vec3 nY;
            const Vec3 Y = L.sampleSurface(0, rng.uniform(), rng.uniform(), nY);
            Vec3 d = Y - p;
            const double dm = length(d);
            if (!(dm > 1e-9)) return 0;
            d = d / dm;
            if (!(d.z < 0.0)) return 0;
            const double back = 2.0 * capR + 1.0;
            const Vec3 o = (dm > back) ? Y - d * back : p;
            Vec3 h, n;
            if (!L.hitSurface(0, o, d, h, n)) return 0;
            const Vec3 g = h - Y;
            if (dot(g, g) > 1e-12) return 0;
            if (!reaches(o, d)) return 0;
            X = Y + d * ((zRef - Y.z) / d.z);
            X.y = std::fabs(X.y);
            return 1;
        };
        // One ray aimed at crossing (ex, ey) the way lensEntry aims.
        auto aimRay = [&](double ex, double ey) {
            Vec3 p;
            if (!point(p)) return -1;
            const Vec3 X(ex, ey, zRef);
            Vec3 d = X - p;
            d = d / length(d);
            const Vec3 o = X - d * ((P->zStart - zRef) / -d.z);
            return reaches(o, d) ? 1 : 0;
        };
        std::vector<double> xs, ys;                    // crossings that reached the film
        long long nUni = 0, okUni = 0;                 // uniform rays tried / through
        auto uniformBatch = [&](int n) {
            for (int k = 0; k < n; ++k) {
                Vec3 X;
                const int r = frontRay(X);
                if (r < 0) continue;
                ++nUni;
                if (r) { ++okUni; xs.push_back(X.x); ys.push_back(X.y); }
            }
        };
        uniformBatch(1536);
        if (xs.empty()) uniformBatch(2048);
        LensPupilCell& C = P->cells[ci];
        if (xs.empty()) { C.kind = LensPupil::NONE; return; }   // dilated below
        lit[ci] = 1;
        auto fit = [&](double& cx, double& a, double& b) {
            double x0 = 1e300, x1 = -1e300, y1 = 0.0;
            for (size_t i = 0; i < xs.size(); ++i) {
                x0 = std::min(x0, xs[i]); x1 = std::max(x1, xs[i]); y1 = std::max(y1, ys[i]);
            }
            cx = 0.5 * (x0 + x1);
            a = std::max(0.5 * (x1 - x0), 1e-6 * ap0);
            b = std::max(y1, 1e-6 * ap0);
        };
        // Find the edge: aim inside a padded ellipse round what the uniform rays found, and
        // grow it until nothing that gets through comes near its rim.
        double cx, a, b;
        fit(cx, a, b);
        double ra = 1.3 * a + 0.02 * ap0, rb = 1.3 * b + 0.02 * ap0;
        bool bounded = false;
        for (int round = 0; round < 8 && !bounded; ++round) {
            std::vector<double> bx, by;
            double edge = 0.0;
            for (int k = 0; k < 1536; ++k) {
                const double rr = std::sqrt(rng.uniform()), an = 2.0 * PI * rng.uniform();
                const double ex = cx + ra * rr * std::cos(an), ey = rb * rr * std::sin(an);
                const int r = aimRay(ex, ey);
                if (r == 1) { bx.push_back(ex); by.push_back(std::fabs(ey)); edge = std::max(edge, rr); }
            }
            if (edge < 0.92) {
                bounded = true;
                xs.insert(xs.end(), bx.begin(), bx.end());
                ys.insert(ys.end(), by.begin(), by.end());
            } else {
                ra *= 1.5; rb *= 1.5;
            }
        }
        if (!bounded) { C.kind = LensPupil::FRONT; return; }
        // The region: centred on the crossings' extent. An ellipse when the crossings fit one
        // (scaled to hold every one, it grows by at most 5%), else the extent's rectangle -- an
        // ellipse round a crescent's horns would need its sparse tips sampled to be sure of
        // holding them, a rectangle only the extent. Padded 3%.
        fit(cx, a, b);
        double s = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) {
            const double qx = (xs[i] - cx) / a, qy = ys[i] / b;
            s = std::max(s, std::sqrt(qx * qx + qy * qy));
        }
        LensPupilCell R;
        R.cx = cx;
        if (s <= 1.05) { R.kind = LensPupil::AIM;    R.a = a * s * 1.03 + 1e-4 * ap0; R.b = b * s * 1.03 + 1e-4 * ap0; }
        else           { R.kind = LensPupil::AIMBOX; R.a = a * 1.03 + 1e-4 * ap0;     R.b = b * 1.03 + 1e-4 * ap0; }
        // Check with fresh uniform rays that whatever reaches the film falls inside, enlarging
        // over any that does not, until a whole batch finds nothing outside. A region that
        // never comes clean keeps the front sampler.
        bool clean = false;
        for (int round = 0; round < 6 && !clean; ++round) {
            const size_t from = xs.size();
            uniformBatch(2048);
            double grow = 1.0;
            for (size_t i = from; i < xs.size(); ++i) {
                const double q = LensPupil::regionNorm(R, xs[i], ys[i]);
                if (q > 1.0) { ++enl[ci]; grow = std::max(grow, q); }
            }
            if (grow > 1.0) { R.a *= grow * 1.03; R.b *= grow * 1.03; }
            else clean = true;
        }
        if (!clean) { C.kind = LensPupil::FRONT; return; }
        cx = R.cx; a = R.a; b = R.b;
        // Aim only where it pays: the share of aimed rays that get through against the share
        // of uniform ones, measured on the same points.
        long long nE = 0, okE = 0;
        for (int k = 0; k < 1024; ++k) {
            double ex, ey;
            if (R.kind == LensPupil::AIMBOX) {
                ex = cx + a * (2.0 * rng.uniform() - 1.0);
                ey = b * (2.0 * rng.uniform() - 1.0);
            } else {
                const double rr = std::sqrt(rng.uniform()), an = 2.0 * PI * rng.uniform();
                ex = cx + a * rr * std::cos(an);
                ey = b * rr * std::sin(an);
            }
            const int r = aimRay(ex, ey);
            if (r < 0) continue;
            ++nE; okE += r;
        }
        const double fFront = nUni ? (double)okUni / (double)nUni : 0.0;
        const double fAim = nE ? (double)okE / (double)nE : 0.0;
        C.cx = cx; C.a = a; C.b = b;
        C.kind = (fAim > 1.1 * fFront) ? R.kind : LensPupil::FRONT;
    });
    if (!done) return nullptr;
    // A bin where no light got through keeps the front sampler if a neighbour saw any (its
    // footprint may be too thin for the batch to have hit), and rejects only when none did.
    for (int it = 0; it < P->nTheta; ++it)
        for (int iu = 0; iu < P->nU; ++iu) {
            LensPupilCell& C = P->cells[(size_t)(it * P->nU + iu)];
            if (lit[(size_t)(it * P->nU + iu)]) continue;
            bool byLit = false;
            for (int dt = -1; dt <= 1 && !byLit; ++dt)
                for (int du = -1; du <= 1 && !byLit; ++du) {
                    const int jt = it + dt, ju = iu + du;
                    if (jt >= 0 && jt < P->nTheta && ju >= 0 && ju < P->nU && lit[(size_t)(jt * P->nU + ju)])
                        byLit = true;
                }
            C.kind = byLit ? LensPupil::FRONT : LensPupil::NONE;
        }
    for (const LensPupilCell& C : P->cells) {
        if (C.kind == LensPupil::AIM || C.kind == LensPupil::AIMBOX) ++P->nAim;
        else if (C.kind == LensPupil::NONE) ++P->nNone;
        else ++P->nFront;
    }
    for (long long e : enl) P->enlarged += e;
    return P;
}

// The table for a shared lens and film radius, built once and kept while the lens lives (a
// flyby's hundreds of frames share one lens). Prints a line when it builds one.
inline std::shared_ptr<const LensPupil> lensPupilFor(const std::shared_ptr<LensSystem>& L, double racc) {
    if (!L || L->surf.empty()) return nullptr;
    struct Entry { std::weak_ptr<LensSystem> lens; double racc, filmZ; std::shared_ptr<const LensPupil> tab; };
    static std::mutex mtx;
    static std::vector<Entry> cache;
    std::lock_guard<std::mutex> g(mtx);
    cache.erase(std::remove_if(cache.begin(), cache.end(),
                               [](const Entry& e) { return e.lens.expired(); }), cache.end());
    for (const Entry& e : cache)
        if (e.lens.lock() == L && e.racc == racc && e.filmZ == L->filmZ) return e.tab;
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<const LensPupil> tab = buildLensPupil(*L, racc);
    if (!tab) return nullptr;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[lens] '%s': pupil table aims light at the stop (surface %d, entrance pupil at "
                "z=%.2f mm) in %d of %d bins, keeps the front sampler in %d, rejects %d; %.0f ms\n",
                L->name.c_str(), tab->stop, tab->zRef, tab->nAim, (int)tab->cells.size(),
                tab->nFront, tab->nNone, ms);
    cache.push_back({L, racc, L->filmZ, tab});
    return tab;
}

// ---------------------------------------------------------------------------
// Generators — build a lens prescription from high-level parameters. All lengths
// in millimetres. These are physically derived (lensmaker / achromat equations),
// not fabricated data, so the focal length and (for the achromat) the chromatic
// correction are correct by construction; dispersion during rendering comes from
// the real Sellmeier glass indices.
// ---------------------------------------------------------------------------

// Abbe number V_d = (n_d - 1)/(n_F - n_C) from a Sellmeier index function.
inline double abbeNumber(const Spectrum& ior) {
    double nF = ior(486.1), nd = ior(587.6), nC = ior(656.3);
    double den = nF - nC;
    return (std::fabs(den) < 1e-9) ? 0.0 : (nd - 1.0) / den;
}

// A single biconvex glass singlet of focal length `focalMM` at f/`fstop`. The clear
// semi-diameter (= focal/(2*fstop)) doubles as the aperture stop. Strong spherical
// and chromatic aberration on purpose — the simplest "real lens".
inline LensSystem makeSinglet(double focalMM, double fstop, const Spectrum& glass) {
    double nd = glass(587.6);
    double f  = focalMM;
    double R  = 2.0 * (nd - 1.0) * f;                     // equiconvex: 1/f=(n-1)(2/R)
    double semi = std::fmax(1.0, f / (2.0 * std::fmax(0.5, fstop)));
    double ct = std::fmax(1.5, 0.12 * semi);              // a plausible centre thickness
    LensSystem L; L.name = "singlet";
    // The lensmaker radii above are in the object->image sign convention; our lens
    // frame stores curvature as `centre = vertex + radius` with +z toward the scene
    // (the PBRT convention), which is the opposite sign — so the front convex surface
    // takes -R and the rear +R to make a converging element.
    L.surf.push_back({ -R, ct,  glass,            semi });   // front (convex toward scene)
    L.surf.push_back({ +R, f,   iorConstant(1.0), semi });   // back; rear gap ~ f (autofocus refines)
    L.finalize();
    L.focusAt(0.0);
    return L;
}

// A cemented achromatic doublet (crown + flint) of focal length `focalMM` at
// f/`fstop`. Element powers are split by the Abbe numbers to cancel first-order
// chromatic aberration (phi1/V1 + phi2/V2 = 0, phi1+phi2 = 1/f); the crown is made
// equiconvex and the flint's rear radius follows from its (negative) power. A thin
// stop just behind the doublet sets the f-number.
inline LensSystem makeAchromat(double focalMM, double fstop,
                               const Spectrum& crown, const Spectrum& flint) {
    double n1 = crown(587.6), n2 = flint(587.6);
    double V1 = abbeNumber(crown), V2 = abbeNumber(flint);
    double f  = focalMM;
    double phi = 1.0 / f;
    // The achromat condition phi1/V1 + phi2/V2 = 0 has no solution for two glasses of the same
    // dispersion, or of none (a constant index has an infinite Abbe number): refuse with an empty
    // system, which the callers report, rather than design infinite radii and a NaN camera.
    if (!(std::isfinite(V1) && std::isfinite(V2) && std::fabs(V1 - V2) >= 1e-6)) return LensSystem{};
    double denom = V1 - V2;
    double phi1 =  phi * V1 / denom;                      // crown power (>0)
    double phi2 = -phi * V2 / denom;                      // flint power (<0)
    double R1 =  2.0 * (n1 - 1.0) / phi1;                 // crown equiconvex
    double R2 = -2.0 * (n1 - 1.0) / phi1;                 // shared cemented surface
    // flint power phi2 = (n2-1)(1/R2 - 1/R4)  ->  1/R4 = 1/R2 - phi2/(n2-1)
    double invR4 = 1.0 / R2 - phi2 / (n2 - 1.0);
    double R4 = (std::fabs(invR4) < 1e-9) ? 0.0 : 1.0 / invR4;
    // Convert from the lensmaker (object->image) sign convention to our lens frame,
    // where curvature is stored as `centre = vertex + radius` with +z toward the
    // scene (PBRT convention) — the opposite sign. Negate every radius so the
    // doublet actually converges.
    R1 = -R1; R2 = -R2; R4 = -R4;
    double semi = std::fmax(1.5, f / (2.0 * std::fmax(0.5, fstop)));
    double ct1 = std::fmax(2.0, 0.16 * semi);            // crown centre thickness
    double ct2 = std::fmax(1.2, 0.08 * semi);            // flint centre thickness
    double gap = std::fmax(1.0, 0.5 * semi);             // stop stand-off
    LensSystem L; L.name = "achromat";
    L.surf.push_back({ R1, ct1, crown,            semi * 1.05 });          // crown front
    L.surf.push_back({ R2, ct2, flint,            semi * 1.05 });          // cemented crown/flint
    L.surf.push_back({ R4, gap, iorConstant(1.0), semi * 1.05 });          // flint rear -> air
    LensSurface stop; stop.radius = 0.0; stop.thickness = f; stop.ior = iorConstant(1.0);
    stop.aperture = semi; stop.isStop = true;
    L.surf.push_back(stop);                                                 // diaphragm, rear gap ~ f
    L.finalize();
    L.focusAt(0.0);
    return L;
}

// Resolve a named lens preset into a prescription, scaled to `focalMM` at f/`fstop`.
// Known names: singlet/biconvex, achromat/doublet, telephoto (achromat), wide/wide-
// angle (short-focal achromat). Returns false on an unknown name.
inline bool resolveLensPreset(const std::string& name, double focalMM, double fstop,
                              LensSystem& out) {
    std::string k;
    for (char c : name) { if (c == ' ' || c == '_' || c == '-') continue; k += (char)std::tolower((unsigned char)c); }
    if (focalMM <= 0.0) focalMM = 50.0;
    if (fstop   <= 0.0) fstop   = 2.8;
    const Spectrum bk7 = defaultGlass("BK7"), sf10 = defaultGlass("SF10");
    if (k == "singlet" || k == "biconvex" || k == "simple") {
        out = makeSinglet(focalMM, fstop, bk7); out.name = "singlet"; return true;
    }
    if (k == "achromat" || k == "doublet") {
        out = makeAchromat(focalMM, fstop, bk7, sf10); out.name = "achromat"; return true;
    }
    if (k == "telephoto" || k == "tele") {                 // long achromat
        double f = focalMM > 0.0 ? focalMM : 135.0;
        out = makeAchromat(f, fstop, bk7, sf10); out.name = "telephoto"; return true;
    }
    if (k == "wide" || k == "wideangle") {                 // short achromat
        double f = focalMM > 0.0 ? focalMM : 28.0;
        out = makeAchromat(f, fstop, bk7, sf10); out.name = "wide"; return true;
    }
    return false;
}
