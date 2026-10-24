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
#include <atomic>
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
#include "color.h"              // LAMBDA_MIN / LAMBDA_MAX

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
    // surface normal oriented against d. `false` if it misses or, unless `clip` is false, is
    // clipped by the surface's clear aperture (vignetting). Planar surfaces (radius 0)
    // intersect the vertex plane z = zpos[j]. The paraxial probes pass clip = false: paraxial
    // optics has no apertures, and a probe clipped by a small stop would measure nothing.
    bool hitSurface(int j, const Vec3& o, const Vec3& d, Vec3& hitP, Vec3& nrm, bool clip = true) const {
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
        if (clip && hitP.x * hitP.x + hitP.y * hitP.y > ap * ap) return false;   // clipped
        return true;
    }

    // Trace a lens-local ray through every interface. `sensorToScene` picks the
    // iteration order (rear->front for camera-ray generation; front->rear for the
    // paraxial focus probe). Returns the outgoing ray in air on success.
    bool trace(const Vec3& o0, const Vec3& d0, double lambda,
               bool sensorToScene, Ray& outLocal, bool clip = true) const {
        Vec3 o = o0, d = normalize(d0);
        int M = (int)surf.size();
        if (sensorToScene) {
            for (int j = M - 1; j >= 0; --j) {
                Vec3 hp, n;
                if (!hitSurface(j, o, d, hp, n, clip)) return false;
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
                if (!hitSurface(j, o, d, hp, n, clip)) return false;
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
        if (!trace(o, d, lambdaD, /*sensorToScene=*/false, out, /*clip=*/false)) return;   // keep default
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

    // The unaimed samplers, shared by the camera, the pupil tables' build and -checklens, so
    // what the tables are fitted and checked against is what renders draw.
    //
    // Light: from lens-local point `pl`, the leg to a point Y uniform on the front cap (with
    // the cap's +z-side normal nY), started just in front of the glass at `o` with unit
    // direction `d` (Camera::lensEntry explains the rebased start). False if the leg does not
    // head for the sensor, or Y is not the first point of the cap it meets (a convex element's
    // far side is hidden).
    bool frontLeg(const Vec3& pl, double u1, double u2, Vec3& Y, Vec3& nY, Vec3& o, Vec3& d) const {
        Y = sampleSurface(0, u1, u2, nY);
        d = Y - pl;
        const double dmm = length(d);
        if (!(dmm > 1e-9)) return false;
        d = d / dmm;
        if (!(d.z < 0.0)) return false;                      // light must travel toward the sensor
        const double back = 2.0 * capRadius(0) + 1.0;
        o = (dmm > back) ? Y - d * back : pl;
        Vec3 hit, n;
        if (!hitSurface(0, o, d, hit, n)) return false;
        const Vec3 gap = hit - Y;
        return dot(gap, gap) <= 1e-12;                       // 1 nm off: Y is on the hidden side
    }
    // Camera: from film point `f` (on the sensor plane), the unit direction `d` through a point
    // uniform on the rear element -- its disk when planar, else its clear cap, the point being
    // the first of the cap the ray meets -- and the ray's importance: cos^4 * A_rear / Z^2 for
    // the disk (PBRT's), cosT * cosR * A_cap / r^2 for the cap. False if the point is hidden or
    // the ray does not head into the lens. The caller traces it.
    bool rearDir(const Vec3& f, double u1, double u2, Vec3& d, double& weight) const {
        const int jr = (int)surf.size() - 1;
        if (surf[jr].radius != 0.0) {
            Vec3 nR;
            const Vec3 pR = sampleSurface(jr, u1, u2, nR);
            d = pR - f;
            const double r = length(d);
            if (!(r > 1e-12)) return false;
            d = d / r;
            const double cosT = d.z;
            if (cosT <= 0.0) return false;
            Vec3 hit, n;
            if (!hitSurface(jr, f, d, hit, n)) return false;
            const Vec3 gap = hit - pR;
            if (dot(gap, gap) > 1e-12) return false;         // pR is on the cap's hidden side
            weight = cosT * std::fabs(dot(d, nR)) * surfaceArea(jr) / (r * r);
            return true;
        }
        const double rr = std::sqrt(u1 > 0 ? u1 : 0.0) * rearAperture();
        const double phi = 2.0 * PI * u2;
        d = normalize(Vec3(rr * std::cos(phi), rr * std::sin(phi), rearZ()) - f);
        const double cosT = d.z;                             // d is unit; z = cos to the axis
        if (cosT <= 0.0) return false;
        const double cos4 = (cosT * cosT) * (cosT * cosT);
        const double A = PI * rearAperture() * rearAperture();
        const double Z = rearZ() - filmZ;
        if (Z <= 1e-9) return false;
        weight = cos4 * A / (Z * Z);
        return true;
    }

    // How wide a beam from the scene each surface's aperture admits, per unit of that
    // aperture: a ray parallel to the axis at height h0 reaches surface j at height y_j
    // (paraxially, at the design wavelength, apertures ignored), so an aperture of radius a
    // there makes an entrance pupil of radius a * h0 / |y_j| -- past an internal image too,
    // where the probe has crossed the axis. Element j of the result is h0 / |y_j|, or 0 where
    // the probe is lost (it misses a sphere or is totally reflected) or passes exactly
    // through the axis.
    std::vector<double> entrancePupilScales() const {
        const double lambdaD = 587.6;
        std::vector<double> out(surf.size(), 0.0);
        const double h0 = 1e-3 * std::max(1e-3, surf.empty() ? 1.0 : surf[0].aperture);
        Vec3 o(h0, 0.0, T + 1.0), d(0.0, 0.0, -1.0);
        for (int j = 0; j < (int)surf.size(); ++j) {
            Vec3 hp, n;
            if (!hitSurface(j, o, d, hp, n, /*clip=*/false)) break;
            if (std::fabs(hp.x) > 1e-15) out[(size_t)j] = h0 / std::fabs(hp.x);
            o = hp;
            if (surf[j].radius != 0.0) {
                Vec3 nd;
                if (!refractDir(d, n, iorSceneSide(j, lambdaD) / iorSensorSide(j, lambdaD), nd)) break;
                d = nd;
            }
        }
        return out;
    }
    // Scale the whole prescription (radii, gaps, apertures) so its focal length is exactly
    // `focalMM`: a lens scales without changing its form, its f-number or its aberrations'
    // shapes. Leaves it alone if either focal length is not positive.
    void scaleToFocal(double focalMM) {
        const double f0 = focalLengthMM();
        if (!(f0 > 0.0 && focalMM > 0.0)) return;
        const double k = focalMM / f0;
        for (LensSurface& sf : surf) { sf.radius *= k; sf.thickness *= k; sf.aperture *= k; }
        finalize();
    }
    // The aperture stop: the first surface marked `stop`, else the surface whose entrance pupil
    // is smallest (the textbook definition), given the pupil scales `sc`. -1 for an empty lens.
    int stopIndex(const std::vector<double>& sc) const {
        for (int j = 0; j < (int)surf.size(); ++j) if (surf[j].isStop) return j;
        int stop = surf.empty() ? -1 : 0;
        double best = 1e300;
        for (int j = 0; j < (int)surf.size(); ++j)
            if (sc[(size_t)j] > 0.0 && surf[j].aperture * sc[(size_t)j] < best) {
                best = surf[j].aperture * sc[(size_t)j];
                stop = j;
            }
        return stop;
    }
    int stopIndex() const { return stopIndex(entrancePupilScales()); }
    // The paraxial f-number: focal length over the stop's paraxial entrance pupil diameter. 0 if
    // it cannot be measured. Only a diagnostic: the f-number the presets are sized for is
    // tracedFNumber's.
    double fNumber() const {
        const std::vector<double> sc = entrancePupilScales();
        const int stop = stopIndex(sc);
        if (stop < 0) return 0.0;
        const double f = focalLengthMM(), d = 2.0 * surf[stop].aperture * sc[(size_t)stop];
        return (d > 0.0 && f > 0.0) ? f / d : 0.0;
    }
    // The radius (mm) of the widest pencil parallel to the axis that gets through every
    // aperture at the design wavelength: the real entrance pupil's (bisection on its height).
    double tracedPupilRadius() const {
        if (surf.empty()) return 0.0;
        double lo = 0.0, hi = surf[0].aperture;              // nothing past the front rim gets in
        for (int k = 0; k < 60; ++k) {
            const double h = 0.5 * (lo + hi);
            Ray out;
            if (trace(Vec3(h, 0.0, T + 1.0), Vec3(0.0, 0.0, -1.0), 587.6, /*sensorToScene=*/false, out)) lo = h;
            else hi = h;
        }
        return lo;
    }
    // The f-number by the beam the lens really lets in: the focal length over the real
    // entrance pupil's diameter (tracedPupilRadius). It is the f-number lens designers quote,
    // and the one exposure follows; the paraxial one (fNumber) misjudges it by the pupil's
    // aberration, which a published lens at its full opening makes up to 4%. 0 if nothing gets
    // through.
    double tracedFNumber() const {
        const double r = tracedPupilRadius(), f = focalLengthMM();
        return (r > 0.0 && f > 0.0) ? f / (2.0 * r) : 0.0;
    }
    // Size the stop so the lens lets in exactly an f/`fstop` beam (tracedFNumber), but open
    // it no wider than `maxAperture` (mm) when that is positive. The search starts from the
    // paraxial size and allows up to twice it; when no stop in that range lets the beam in
    // (the cap, the front rim or a fast lens's aberration limits it first), the stop is left at
    // the cap, else at the paraxial size, and tracedFNumber says what the lens does. Leaves
    // the lens alone if the f-number cannot be measured.
    void setFNumber(double fstop, double maxAperture = 0.0) {
        const std::vector<double> sc = entrancePupilScales();
        const int stop = stopIndex(sc);
        if (stop < 0 || !(fstop > 0.0)) return;
        const double f = focalLengthMM();
        if (!(sc[(size_t)stop] > 0.0 && f > 0.0)) return;
        const double want = f / (2.0 * fstop);                // the entrance pupil's radius
        const double paraxial = want / sc[(size_t)stop];
        double& a = surf[(size_t)stop].aperture;
        const double hi = (maxAperture > 0.0) ? maxAperture : 2.0 * paraxial;
        a = hi;
        if (tracedPupilRadius() < want) {
            a = (maxAperture > 0.0) ? maxAperture : paraxial;
            return;
        }
        // The beam widens with the stop until something else limits it: the smallest stop
        // that lets it in.
        double lo = 0.0, up = hi;
        for (int k = 0; k < 60; ++k) {
            a = 0.5 * (lo + up);
            if (tracedPupilRadius() >= want) up = a; else lo = a;
        }
        a = up;
    }

    // Effective focal length (mm), measured from a near-axis parallel probe at the
    // design wavelength: f = h / tan(angle of the emergent ray). Used for reporting.
    double focalLengthMM() const {
        if (surf.empty()) return 0.0;
        const double lambdaD = 587.6;
        // A probe height that is paraxial (spherical aberration moves a finite ray's focal
        // length off the lens's) and set by the lens's length, which no aperture change moves:
        // measuring before and after a generator resizes its stop agrees.
        const double h = 1e-5 * std::fmax(1.0, T);
        Vec3 o(h, 0.0, T + 1.0), d(0.0, 0.0, -1.0);
        Ray out;
        if (!trace(o, d, lambdaD, /*sensorToScene=*/false, out, /*clip=*/false)) return 0.0;
        double slope = -out.d.x / out.d.z;                // dx/dz of the emergent ray
        if (std::fabs(slope) < 1e-12) return 0.0;
        return std::fabs(h / slope);                      // report the magnitude (mm)
    }
};

// ---------------------------------------------------------------------------
// The pupils, for aiming at the stop from either side.
//
// Light tracing picks where light from a scene point enters the lens. Uniform over the front
// element's clear cap (sampleSurface) is exact for any lens, but when the front element is
// much wider than the beam the stop admits -- a fast wide-angle, any lens stopped down -- most
// picks are clipped inside, and for a point outside the field of view every pick is traced
// through the glass only to miss the film. Hanika & Dachsbacher (2014, sec. 5.2) and Schrade
// et al. (2016) aim at the stop instead, solving per sample for the direction that reaches a
// sampled point of it (Newton steps on a fitted polynomial): several traces per sample. These
// tables aim for the price of a lookup. The camera's rays have the mirror problem: genLensRay
// picks a point uniform on the REAR element, of which a stopped-down beam uses a few per cent,
// and pbrt's exit-pupil bounds (RealisticCamera::BoundExitPupil) are the fix this mirrors.
//
// The rays from a scene point that get through to the film cross a plane at the entrance
// pupil (the stop's image seen from the scene) in a region that, a pupil being what it is,
// barely moves with the point's distance and moves slowly with its angle off the axis. The
// entrance table bins scene points by that angle and by inverse distance and keeps, per bin, a
// region on that plane -- an ellipse, or a rectangle where an ellipse fits badly -- holding
// every crossing that reaches the film; lensEntry draws the crossing uniformly in the region
// and aims at it. The exit table does the same for camera rays: bins by the film point's
// distance from the sensor's centre, regions on the exit pupil's plane (the stop's image seen
// from the sensor) holding every crossing of a film-to-scene ray that gets out of the lens;
// genLensRay draws there. The lens is rotationally symmetric, so a bin's region is built for
// points at azimuth 0 and serves every azimuth rotated.
//
// The regions come from tracing, not from paraxial optics, so aberrations, vignetting by any
// element and the film's edge are all in them (lenspupil::fitRegion). Per bin, over points
// from the bin widened by 15% on every side: rays drawn the old way (uniform on the element)
// find where light gets through; rays aimed inside a growing ellipse find its edge; the region
// is fitted and padded; then batches of rays aimed just past its edge, and fresh batches of
// uniform rays, check that nothing getting through falls outside, enlarging the region over
// anything that does, until a whole batch of each finds nothing (the count is kept for
// -checklens). A bin keeps the unaimed sampler where aiming
// would not pay -- the stop already fills the element -- or where its region never comes
// clean, and a bin where no light got through, nor in any neighbouring bin, rejects outright,
// before any tracing. The film edge used is a circle round the sensor's corners, which the
// lens's rotational symmetry needs; a camera whose raster outgrows that circle, or whose
// focus moves, stops using the tables (Camera::pupil).
struct LensPupilCell {
    double cx = 0.0, a = 0.0, b = 0.0;   // region on its plane (mm) for azimuth 0: centred at
                                         // (cx, 0), half-widths a (radial, along x) and b (along y)
    int kind = 0;                        // LensPupil::FRONT / AIM / NONE / AIMBOX
};
// A bin's region, for a point at azimuth (cphi, sphi): AIM is the ellipse with semi-axes a, b;
// AIMBOX the rectangle with half-widths a, b, for a region an ellipse fits badly (a crescent
// at the edge of the field, where the film's rim cuts the beam).
inline Vec3 pupilRegionSample(const LensPupilCell& C, double cphi, double sphi, double u1, double u2,
                              double z) {
    double ex, ey;
    if (C.kind == 3) {                                   // LensPupil::AIMBOX
        ex = C.cx + C.a * (2.0 * u1 - 1.0);
        ey = C.b * (2.0 * u2 - 1.0);
    } else {
        const double rr = std::sqrt(u1), an = 2.0 * PI * u2;
        ex = C.cx + C.a * rr * std::cos(an);
        ey = C.b * rr * std::sin(an);
    }
    return Vec3(ex * cphi - ey * sphi, ex * sphi + ey * cphi, z);
}
// How far (ex, ey), in the bin's own azimuth-0 frame, lies out in the region: <= 1 inside.
inline double pupilRegionNorm(const LensPupilCell& C, double ex, double ey) {
    const double qx = (ex - C.cx) / C.a, qy = ey / C.b;
    return (C.kind == 3) ? std::max(std::fabs(qx), std::fabs(qy)) : std::sqrt(qx * qx + qy * qy);
}
// The same for a point X of the plane, seen at azimuth (cphi, sphi).
inline double pupilRegionNormAt(const LensPupilCell& C, const Vec3& X, double cphi, double sphi) {
    return pupilRegionNorm(C, X.x * cphi + X.y * sphi, -X.x * sphi + X.y * cphi);
}
inline double pupilRegionArea(const LensPupilCell& C) { return (C.kind == 3 ? 4.0 : PI) * C.a * C.b; }

// One side's table: its bins' regions, all on the plane z.
struct LensPupilTable {
    double z = 0.0;                      // the plane (mm): the entrance or the exit pupil's
    std::vector<LensPupilCell> cells;
    int nAim = 0, nFront = 0, nNone = 0;
    long long enlarged = 0;              // uniform rays that got through outside their bin's
                                         // fitted region, each of which enlarged it
    // A crossing uniform in bin c's region, rotated to azimuth (cphi, sphi).
    Vec3 sample(int c, double cphi, double sphi, double u1, double u2) const {
        return pupilRegionSample(cells[c], cphi, sphi, u1, u2, z);
    }
    // Is X (on the plane) inside bin c's region at azimuth (cphi, sphi)? The slack covers the
    // rounding between a crossing drawn and the same crossing recomputed from its ray.
    bool inside(int c, const Vec3& X, double cphi, double sphi) const {
        return pupilRegionNormAt(cells[c], X, cphi, sphi) <= 1.0 + 1e-9;
    }
    double area(int c) const { return pupilRegionArea(cells[c]); }
};

struct LensPupil {
    // FRONT keeps the unaimed sampler (uniform on the front element for light, on the rear
    // element for camera rays); NONE rejects before any trace.
    enum { FRONT = 0, AIM = 1, NONE = 2, AIMBOX = 3 };
    // What the tables were built for. They describe the prescription as it stood then: a
    // LensSystem is to be edited only before its cameras call prepareLensPupil.
    const LensSystem* lens = nullptr;
    double racc2 = 0.0;    // squared film radius (mm^2)
    double filmZ = 0.0;    // the sensor plane
    int stop = -1;         // the surface taken as the stop (for the report)

    // ---- the entrance side: light traced in (Camera::lensEntry); entry.z is zRef ----
    LensPupilTable entry;  // bins [iTheta * nU + iU], iU = floor(nU * dMin / distance)
    double zStart = 0.0;   // above every point of the front cap: where an aimed leg starts
    double dMin = 0.0;     // scene points nearer than this to (0, 0, entry.z) keep the front sampler
    int nTheta = 0, nU = 0;
    double dTheta = 0.0;   // bin width in the angle off the axis, over [0, pi/2)

    // ---- the exit side: camera rays traced out (Camera::genLensRay) ----
    LensPupilTable exit;   // bins [iRho]
    int nRho = 0;
    double dRho = 0.0;     // bin width in the film point's distance from the sensor centre

    // Do the tables serve lens L at its current focus, for a raster reaching sqrt(racc2Needed)
    // mm from the sensor centre? Camera::pupil and the GPU's bake both ask this.
    bool fits(const LensSystem* L, double racc2Needed) const {
        return L == lens && L && filmZ == L->filmZ && racc2 >= racc2Needed;
    }

    // Which sampler light from lens-local point p takes: an entrance bin >= 0 to aim (cphi,
    // sphi then hold p's azimuth), -1 for the front sampler, -2 when no light from p reaches
    // the film. Points beside the glass or near it keep the front sampler.
    int classify(const Vec3& p, double& cphi, double& sphi) const {
        if (!(p.z > zStart)) return -1;
        const double r2 = p.x * p.x + p.y * p.y, qz = p.z - entry.z;
        const double D2 = r2 + qz * qz;
        if (!(D2 >= dMin * dMin)) return -1;
        const double D = std::sqrt(D2);
        int it = (int)(std::acos(std::min(1.0, qz / D)) / dTheta);
        int iu = (int)(dMin / D * nU);
        it = std::min(std::max(it, 0), nTheta - 1);
        iu = std::min(std::max(iu, 0), nU - 1);
        const int c = it * nU + iu;
        const int k = entry.cells[c].kind;
        if (k == NONE) return -2;
        if (k != AIM && k != AIMBOX) return -1;
        const double r = std::sqrt(r2);
        if (r > 0.0) { cphi = p.x / r; sphi = p.y / r; } else { cphi = 1.0; sphi = 0.0; }
        return c;
    }
    // The same for a camera ray from film point f (lens-local, on the sensor plane): an exit
    // bin >= 0 to aim (cphi, sphi = f's azimuth), -1 for the rear-element sampler, -2 when no
    // ray from that far off centre gets out of the lens.
    int classifyFilm(const Vec3& f, double& cphi, double& sphi) const {
        if (exit.cells.empty()) return -1;
        const double r = std::sqrt(f.x * f.x + f.y * f.y);
        int i = (int)(r / dRho);
        i = std::min(std::max(i, 0), nRho - 1);
        const int k = exit.cells[i].kind;
        if (k == NONE) return -2;
        if (k != AIM && k != AIMBOX) return -1;
        if (r > 0.0) { cphi = f.x / r; sphi = f.y / r; } else { cphi = 1.0; sphi = 0.0; }
        return i;
    }
};

// Says once per run that a camera's tables no longer fit it (Camera::pupil): the render is
// still right, only noisier, and nothing else would show why.
inline void lensPupilRefused() {
    static std::atomic<bool> said{false};
    if (!said.load(std::memory_order_relaxed) && !said.exchange(true))
        std::printf("[lens] a camera's focus, film or lens changed after its pupil tables were built: "
                    "it renders without aiming at the stop (the same image, noisier) until "
                    "Camera::prepareLensPupil rebuilds them\n");
}

namespace lenspupil {
// One bin's region, found by tracing. `uniformRay(X)` draws a ray the unaimed way over the
// bin's points and returns -1 (no usable point), 0 (lost in the lens) or 1 (got through, X
// its crossing of the plane with y folded to >= 0 by the mirror symmetry about the x-z
// plane); `aimRay(ex, ey)` draws one through the crossing (ex, ey) the way the sampler aims,
// with the same returns. `scale` (mm) sets the paddings. Fills C (AIM, AIMBOX or FRONT; NONE
// when no light got through, which the caller resolves against the neighbours) and counts
// the verification's enlargements into `enlarged`. Returns whether any light got through.
template <class U, class A>
inline bool fitRegion(Pcg32& rng, U&& uniformRay, A&& aimRay, double scale, LensPupilCell& C,
                      long long& enlarged) {
    std::vector<double> xs, ys;                    // crossings that got through
    long long nUni = 0, okUni = 0;                 // uniform rays tried / through
    auto uniformBatch = [&](int n) {
        for (int k = 0; k < n; ++k) {
            Vec3 X;
            const int r = uniformRay(X);
            if (r < 0) continue;
            ++nUni;
            if (r) { ++okUni; xs.push_back(X.x); ys.push_back(X.y); }
        }
    };
    uniformBatch(1536);
    if (xs.empty()) uniformBatch(2048);
    if (xs.empty()) { C.kind = LensPupil::NONE; return false; }
    auto fit = [&](double& cx, double& a, double& b) {
        double x0 = 1e300, x1 = -1e300, y1 = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) {
            x0 = std::min(x0, xs[i]); x1 = std::max(x1, xs[i]); y1 = std::max(y1, ys[i]);
        }
        cx = 0.5 * (x0 + x1);
        a = std::max(0.5 * (x1 - x0), 1e-6 * scale);
        b = std::max(y1, 1e-6 * scale);
    };
    // Find the edge: aim inside a padded ellipse round what the uniform rays found, and grow it
    // until nothing that gets through comes near its rim.
    double cx, a, b;
    fit(cx, a, b);
    double ra = 1.3 * a + 0.02 * scale, rb = 1.3 * b + 0.02 * scale;
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
    if (!bounded) { C.kind = LensPupil::FRONT; return true; }
    // The region: centred on the crossings' extent. An ellipse when the crossings fit one
    // (scaled to hold every one, it grows by at most 5%), else the extent's rectangle -- an
    // ellipse round a crescent's horns would need its sparse tips sampled to be sure of holding
    // them, a rectangle only the extent. Padded 3%.
    fit(cx, a, b);
    double s = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double qx = (xs[i] - cx) / a, qy = ys[i] / b;
        s = std::max(s, std::sqrt(qx * qx + qy * qy));
    }
    LensPupilCell R;
    R.cx = cx;
    if (s <= 1.05) { R.kind = LensPupil::AIM;    R.a = a * s * 1.03 + 1e-4 * scale; R.b = b * s * 1.03 + 1e-4 * scale; }
    else           { R.kind = LensPupil::AIMBOX; R.a = a * 1.03 + 1e-4 * scale;     R.b = b * 1.03 + 1e-4 * scale; }
    // Check the edge with rays aimed into the band just past it, out to 30% beyond (uniform in
    // how far out, round the whole rim), where anything the fit missed lies, enlarging over
    // whatever gets through, until a whole batch finds nothing. The uniform batches below see
    // the whole element, but where little of it reaches the film (a fisheye's front element
    // passes under 1% of them) a batch holds a few dozen crossings, too few to find a region's
    // sparse extremes. A region whose edge never comes clean keeps the unaimed sampler.
    bool edgeClean = false;
    for (int round = 0; round < 8 && !edgeClean; ++round) {
        double grow = 1.0;
        for (int k = 0; k < 4096; ++k) {
            const double q = 1.0 + 0.3 * rng.uniform(), v = rng.uniform(), w = rng.uniform();
            double ex, ey;
            if (R.kind == LensPupil::AIMBOX) {
                // The rectangle's rim scaled by q, uniform along it: a top or bottom side (length
                // 2a each) or a left or right one (2b each).
                const double s = 2.0 * v - 1.0, side = (w < 0.5) ? -1.0 : 1.0;
                if (rng.uniform() * (R.a + R.b) < R.a) { ex = R.cx + q * R.a * s;    ey = q * R.b * side; }
                else                                   { ex = R.cx + q * R.a * side; ey = q * R.b * s; }
            } else {
                const double an = 2.0 * PI * v;
                ex = R.cx + q * R.a * std::cos(an);
                ey = q * R.b * std::sin(an);
            }
            if (aimRay(ex, ey) != 1) continue;
            const double qq = pupilRegionNorm(R, ex, std::fabs(ey));
            if (qq > 1.0) { ++enlarged; grow = std::max(grow, qq); }
        }
        if (grow > 1.0) { R.a *= grow * 1.03; R.b *= grow * 1.03; }
        else edgeClean = true;
    }
    if (!edgeClean) { C.kind = LensPupil::FRONT; return true; }
    // Then with fresh uniform rays that whatever gets through falls inside, enlarging over any
    // that does not, until a whole batch finds nothing outside. A region that never comes clean
    // keeps the unaimed sampler.
    bool clean = false;
    for (int round = 0; round < 6 && !clean; ++round) {
        const size_t from = xs.size();
        uniformBatch(2048);
        double grow = 1.0;
        for (size_t i = from; i < xs.size(); ++i) {
            const double q = pupilRegionNorm(R, xs[i], ys[i]);
            if (q > 1.0) { ++enlarged; grow = std::max(grow, q); }
        }
        if (grow > 1.0) { R.a *= grow * 1.03; R.b *= grow * 1.03; }
        else clean = true;
    }
    if (!clean) { C.kind = LensPupil::FRONT; return true; }
    // Aim only where it pays: the share of aimed rays that get through against the share of
    // uniform ones, measured over the same points.
    long long nE = 0, okE = 0;
    for (int k = 0; k < 1024; ++k) {
        const Vec3 X = pupilRegionSample(R, 1.0, 0.0, rng.uniform(), rng.uniform(), 0.0);
        const int r = aimRay(X.x, X.y);
        if (r < 0) continue;
        ++nE; okE += r;
    }
    const double fUni = nUni ? (double)okUni / (double)nUni : 0.0;
    const double fAim = nE ? (double)okE / (double)nE : 0.0;
    C.cx = R.cx; C.a = R.a; C.b = R.b;
    C.kind = (fAim > 1.1 * fUni) ? R.kind : LensPupil::FRONT;
    return true;
}

// Where a ray leaving the axial point z0 at a small angle, traced through surfaces j0, j0+dj,
// ..., crosses the axis once it is out: the pupil the stop at z0 makes through them. NaN if
// the ray is clipped or comes out parallel.
inline double pupilImageZ(const LensSystem& L, double z0, int j0, int dj, double lambda) {
    const int M = (int)L.surf.size();
    const double ang = 1e-3;
    Vec3 o(0.0, 0.0, z0), d(std::sin(ang), 0.0, dj < 0 ? std::cos(ang) : -std::cos(ang));
    for (int j = j0; j >= 0 && j < M; j += dj) {
        Vec3 hp, n;
        if (!L.hitSurface(j, o, d, hp, n, /*clip=*/false)) return NAN;
        o = hp;
        if (L.surf[j].radius != 0.0) {
            const double eta = (dj < 0) ? L.iorSensorSide(j, lambda) / L.iorSceneSide(j, lambda)
                                        : L.iorSceneSide(j, lambda) / L.iorSensorSide(j, lambda);
            Vec3 nd;
            if (!LensSystem::refractDir(d, n, eta, nd)) return NAN;
            d = nd;
        }
    }
    if (!(std::fabs(d.x) > 1e-12)) return NAN;
    return o.z - o.x * d.z / d.x;
}

// A bin with no light keeps the unaimed sampler if a neighbour (lit[] true) saw any -- its
// footprint may be too thin for the batches to have hit -- and rejects only when none did.
inline void resolveUnlit(std::vector<LensPupilCell>& cells, const std::vector<char>& lit, int n0, int n1) {
    for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j) {
            if (lit[(size_t)(i * n1 + j)]) continue;
            bool byLit = false;
            for (int di = -1; di <= 1 && !byLit; ++di)
                for (int dj = -1; dj <= 1 && !byLit; ++dj) {
                    const int a = i + di, b = j + dj;
                    if (a >= 0 && a < n0 && b >= 0 && b < n1 && lit[(size_t)(a * n1 + b)]) byLit = true;
                }
            cells[(size_t)(i * n1 + j)].kind = byLit ? LensPupil::FRONT : LensPupil::NONE;
        }
}
// Count a table's bins by kind, for the report.
inline void countKinds(LensPupilTable& T) {
    T.nAim = T.nFront = T.nNone = 0;
    for (const LensPupilCell& C : T.cells) {
        if (C.kind == LensPupil::AIM || C.kind == LensPupil::AIMBOX) ++T.nAim;
        else if (C.kind == LensPupil::NONE) ++T.nNone;
        else ++T.nFront;
    }
}
}   // namespace lenspupil

// Build both tables for lens L focused as it is, for a film reaching sqrt(`racc2`) mm from the
// sensor's centre. The square is kept as given: Camera::pupil compares it with the raster's
// own, and a square root squared back can come out an ulp short. Null if the lens is empty or
// the build was stopped (-stop).
inline std::shared_ptr<const LensPupil> buildLensPupil(const LensSystem& L, double racc2) {
    const int M = (int)L.surf.size();
    if (M == 0 || !(racc2 > 0.0)) return nullptr;
    const double racc = std::sqrt(racc2);
    auto P = std::make_shared<LensPupil>();
    const double lamD = 587.6;
    const int stop = std::max(L.stopIndex(), 0);   // marked, else the smallest entrance pupil
    P->lens = &L;
    P->stop = stop;
    P->racc2 = racc2;
    P->filmZ = L.filmZ;
    const double lamLo = LAMBDA_MIN, lamHi = LAMBDA_MAX; // every wavelength a scene can carry
    constexpr double kWiden = 0.15;                     // a bin's points reach 15% past its edges:
                                                        // a region's extremes sit at those edges,
                                                        // which points drawn only inside approach
                                                        // but never reach

    // ---- the entrance side ----
    // The plane: where the chief ray, leaving the stop's centre for the scene at a small angle,
    // crosses the axis in object space. A plane elsewhere would still be exact, only less tight.
    double zRef = L.zpos[stop];
    if (stop > 0) {
        const double z = lenspupil::pupilImageZ(L, L.zpos[stop], stop - 1, -1, lamD);
        if (std::isfinite(z)) zRef = z;
    }
    const double capR = L.capRadius(0), ap0 = L.surf[0].aperture;
    const double track = L.zpos[0] - L.zpos[M - 1];
    zRef = std::max(std::min(zRef, L.zpos[0]), L.zpos[M - 1] - track - 10.0);
    P->entry.z = zRef;
    P->zStart = L.zpos[0] + capR + 1.0;
    P->dMin = (P->zStart - zRef) + 4.0 * capR;
    P->nTheta = 64; P->nU = 8;
    P->dTheta = 0.5 * PI / P->nTheta;
    const int nCells = P->nTheta * P->nU;
    P->entry.cells.assign((size_t)nCells, LensPupilCell{});
    std::vector<char> lit((size_t)nCells, 0);
    std::vector<long long> enl((size_t)nCells, 0);
    bool done = ft::parallelFor((size_t)nCells, 1, [&](size_t ci) {
        const int it = (int)ci / P->nU, iu = (int)ci % P->nU;
        Pcg32 rng; rng.seed(0x9e3779b97f4a7c15ULL + ci, 0x5e1ec7ULL);
        // A scene point of the (widened) bin at azimuth 0, the +x half of the x-z plane; false
        // where it lies beside the glass, which classify never aims from.
        auto point = [&](Vec3& p) {
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
        // Uniform on the front cap, as lensEntry's front sampler draws (LensSystem::frontLeg).
        auto frontRay = [&](Vec3& X) {
            Vec3 p;
            if (!point(p)) return -1;
            Vec3 Y, nY, o, d;
            if (!L.frontLeg(p, rng.uniform(), rng.uniform(), Y, nY, o, d)) return 0;
            if (!reaches(o, d)) return 0;
            X = Y + d * ((zRef - Y.z) / d.z);
            X.y = std::fabs(X.y);
            return 1;
        };
        // Aimed at crossing (ex, ey), as lensEntry aims.
        auto aimRay = [&](double ex, double ey) {
            Vec3 p;
            if (!point(p)) return -1;
            const Vec3 X(ex, ey, zRef);
            Vec3 d = X - p;
            d = d / length(d);
            const Vec3 o = X - d * ((P->zStart - zRef) / -d.z);
            return reaches(o, d) ? 1 : 0;
        };
        lit[ci] = lenspupil::fitRegion(rng, frontRay, aimRay, ap0, P->entry.cells[ci], enl[ci]) ? 1 : 0;
    });
    if (!done) return nullptr;
    lenspupil::resolveUnlit(P->entry.cells, lit, P->nTheta, P->nU);
    lenspupil::countKinds(P->entry);
    for (long long e : enl) P->entry.enlarged += e;

    // ---- the exit side ----
    // The plane: where the chief ray, leaving the stop's centre for the sensor, crosses the axis
    // once out of the glass. It must lie in front of the sensor, which aims from below it.
    const int jr = M - 1;
    double zExit = L.zpos[stop];
    if (stop < jr) {
        const double z = lenspupil::pupilImageZ(L, L.zpos[stop], stop + 1, +1, lamD);
        if (std::isfinite(z)) zExit = z;
    }
    zExit = std::min(zExit, L.zpos[0]);
    if (!(zExit > L.filmZ + 1.0)) zExit = std::max(L.zpos[jr], L.filmZ + 1.0);
    const double apR = L.surf[jr].aperture;
    P->exit.z = zExit;
    P->nRho = 64;
    P->dRho = racc / P->nRho;
    P->exit.cells.assign((size_t)P->nRho, LensPupilCell{});
    std::vector<char> litE((size_t)P->nRho, 0);
    std::vector<long long> enlE((size_t)P->nRho, 0);
    done = ft::parallelFor((size_t)P->nRho, 1, [&](size_t ci) {
        const int ir = (int)ci;
        Pcg32 rng; rng.seed(0x2545f4914f6cdd1dULL + ci, 0xe417ULL);
        // A film point of the (widened) bin at azimuth 0.
        auto filmPoint = [&]() {
            const double r = std::max((ir - kWiden + (1.0 + 2.0 * kWiden) * rng.uniform()) * P->dRho, 0.0);
            return Vec3(r, 0.0, L.filmZ);
        };
        auto getsOut = [&](const Vec3& f, const Vec3& d) {
            Ray out;
            return L.trace(f, d, lamLo + (lamHi - lamLo) * rng.uniform(), /*sensorToScene=*/true, out);
        };
        // Uniform on the rear element, as genLensRay's unaimed sampler draws (LensSystem::rearDir).
        auto rearRay = [&](Vec3& X) {
            const Vec3 f = filmPoint();
            Vec3 d;
            double w;
            if (!L.rearDir(f, rng.uniform(), rng.uniform(), d, w)) return 0;
            if (!getsOut(f, d)) return 0;
            X = f + d * ((zExit - f.z) / d.z);
            X.y = std::fabs(X.y);
            return 1;
        };
        // Aimed at crossing (ex, ey), as genLensRay aims.
        auto aimRay = [&](double ex, double ey) {
            const Vec3 f = filmPoint();
            const Vec3 d = normalize(Vec3(ex, ey, zExit) - f);
            return getsOut(f, d) ? 1 : 0;
        };
        litE[ci] = lenspupil::fitRegion(rng, rearRay, aimRay, apR, P->exit.cells[ci], enlE[ci]) ? 1 : 0;
    });
    if (!done) return nullptr;
    lenspupil::resolveUnlit(P->exit.cells, litE, P->nRho, 1);
    lenspupil::countKinds(P->exit);
    for (long long e : enlE) P->exit.enlarged += e;
    return P;
}

// The tables for a shared lens and squared film radius (mm^2), built once and kept while the
// lens lives (a flyby's hundreds of frames share one lens). Prints a line when it builds them.
inline std::shared_ptr<const LensPupil> lensPupilFor(const std::shared_ptr<LensSystem>& L, double racc2) {
    if (!L || L->surf.empty()) return nullptr;
    struct Entry { std::weak_ptr<LensSystem> lens; double racc2, filmZ; std::shared_ptr<const LensPupil> tab; };
    static std::mutex mtx;
    static std::vector<Entry> cache;
    std::lock_guard<std::mutex> g(mtx);
    cache.erase(std::remove_if(cache.begin(), cache.end(),
                               [](const Entry& e) { return e.lens.expired(); }), cache.end());
    for (const Entry& e : cache)
        if (e.lens.lock() == L && e.racc2 == racc2 && e.filmZ == L->filmZ) return e.tab;
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<const LensPupil> tab = buildLensPupil(*L, racc2);
    if (!tab) return nullptr;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[lens] '%s': pupil tables aim at the stop (surface %d): light in %d of %d bins "
                "(front sampler %d, rejects %d), camera rays in %d of %d (rear sampler %d, rejects "
                "%d); %.0f ms\n",
                L->name.c_str(), tab->stop, tab->entry.nAim, (int)tab->entry.cells.size(),
                tab->entry.nFront, tab->entry.nNone, tab->exit.nAim, (int)tab->exit.cells.size(),
                tab->exit.nFront, tab->exit.nNone, ms);
    cache.push_back({L, racc2, L->filmZ, tab});
    return tab;
}

// ---------------------------------------------------------------------------
// Generators — build a lens prescription from high-level parameters. All lengths
// in millimetres. The singlet and the achromat are physically derived (lensmaker /
// achromat equations), not fabricated data, so the focal length and (for the
// achromat) the chromatic correction are correct by construction; dispersion during
// rendering comes from the real Sellmeier glass indices. The double Gauss, telephoto,
// wide-angle and fisheye are the other kind: published prescriptions (makePublishedLens),
// scaled and stopped down.
// ---------------------------------------------------------------------------

// Abbe number V_d = (n_d - 1)/(n_F - n_C) from a Sellmeier index function.
inline double abbeNumber(const Spectrum& ior) {
    double nF = ior(486.1), nd = ior(587.6), nC = ior(656.3);
    double den = nF - nC;
    return (std::fabs(den) < 1e-9) ? 0.0 : (nd - 1.0) / den;
}

// How much thicker an element is at height `ap` than on the axis, from its front and rear
// radii (this frame's convention: centre at vertex + radius). Each surface departs from its
// vertex plane by its sag, toward the sensor when its centre is behind it (R < 0). A
// generator gives an element centre thickness `edgeMin - lensEdgeGain(...)` at least, so the
// two faces never cross inside the clear aperture: past where they would, rays pass outside
// the glass and are lost, silently narrowing the lens.
inline double lensEdgeGain(double Rfront, double Rrear, double ap) {
    auto delta = [ap](double R) {
        if (R == 0.0) return 0.0;
        const double a = std::fabs(R);
        const double sag = (ap < a) ? a - std::sqrt(a * a - ap * ap) : a;
        return (R < 0.0) ? -sag : sag;
    };
    return delta(Rfront) - delta(Rrear);
}
// The thinnest edge a generator leaves: 8% of the aperture, relative so a lens keeps its
// form at any scale (a published prescription's edges are all wider than this).
inline double lensEdgeMin(double ap) { return 0.08 * ap; }

// Thicken every element (glass between surfaces j and j+1) whose faces come closer than
// lensEdgeMin at its clear aperture, the larger of its faces'. True if it changed anything.
inline bool lensThickenElements(LensSystem& L) {
    bool changed = false;
    for (int j = 0; j + 1 < (int)L.surf.size(); ++j) {
        if (L.iorSensorSide(j, 587.6) == 1.0) continue;    // air after j: a gap, not an element
        const double ap = std::max(L.surf[j].aperture, L.surf[j + 1].aperture);
        const double need = lensEdgeMin(ap) - lensEdgeGain(L.surf[j].radius, L.surf[j + 1].radius, ap);
        // A hair past the minimum, so the rescale that follows (thicker glass lengthens the
        // focal length, so it shrinks the lens) does not leave the edge just short again.
        if (L.surf[j].thickness < need) { L.surf[j].thickness = need + 1e-3 * lensEdgeMin(ap); changed = true; }
    }
    if (changed) L.finalize();
    return changed;
}
// A generator's last steps: elements thick enough for a real edge (unless `thicken` is false,
// for a published design, physical as it stands), the focal length exactly `focalMM`, the stop
// sized for f/`fstop` (kept when <= 0) and no wider than `maxStopPerMM` * focalMM when that is
// positive. Repeated until thickening changes nothing, since scaling and the stop move the
// apertures the edges are checked at. Then focus at infinity.
inline void lensFinishPreset(LensSystem& L, double focalMM, double fstop, double maxStopPerMM = 0.0,
                             bool thicken = true) {
    for (int pass = 0; pass < 50; ++pass) {
        const bool thickened = thicken && lensThickenElements(L);
        L.scaleToFocal(focalMM);
        L.setFNumber(fstop, maxStopPerMM > 0.0 ? maxStopPerMM * L.focalLengthMM() : 0.0);
        if (!thickened && pass > 0) break;
    }
    L.focusAt(0.0);
}

// A single biconvex glass singlet of focal length `focalMM` at f/`fstop`. The front face's
// clear semi-diameter (= focal/(2*fstop)) doubles as the aperture stop. Strong spherical
// and chromatic aberration on purpose — the simplest "real lens".
inline LensSystem makeSinglet(double focalMM, double fstop, const Spectrum& glass) {
    double nd = glass(587.6);
    double f  = focalMM;
    double R  = 2.0 * (nd - 1.0) * f;                     // equiconvex: 1/f=(n-1)(2/R)
    double semi = std::fmax(1.0, f / (2.0 * std::fmax(0.5, fstop)));
    // The front face is the stop; the rear one clears the beam by 5%, as the achromat's
    // elements do, so it never trims the axial pencil the stop is sized for.
    const double apRear = 1.05 * semi;
    // A plausible centre thickness, and at least enough for an edge at the clear aperture.
    double ct = std::fmax(std::fmax(1.5, 0.12 * semi), lensEdgeMin(apRear) - lensEdgeGain(-R, +R, apRear));
    LensSystem L; L.name = "singlet";
    // The lensmaker radii above are in the object->image sign convention; our lens
    // frame stores curvature as `centre = vertex + radius` with +z toward the scene
    // (the PBRT convention), which is the opposite sign — so the front convex surface
    // takes -R and the rear +R to make a converging element.
    L.surf.push_back({ -R, ct,  glass,            semi });   // front (convex toward scene)
    L.surf.push_back({ +R, f,   iorConstant(1.0), apRear }); // back; rear gap ~ f (autofocus refines)
    L.finalize();
    // The lensmaker equation is the thin lens's: the thickness moves the focal length off it.
    lensFinishPreset(L, focalMM, fstop);
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
    const double ap = semi * 1.05;                       // the elements' clear semi-aperture
    // Centre thicknesses: plausible, and at least enough for an edge at the clear aperture.
    double ct1 = std::fmax(std::fmax(2.0, 0.16 * semi), lensEdgeMin(ap) - lensEdgeGain(R1, R2, ap));
    double ct2 = std::fmax(std::fmax(1.2, 0.08 * semi), lensEdgeMin(ap) - lensEdgeGain(R2, R4, ap));
    double gap = std::fmax(1.0, 0.5 * semi);             // stop stand-off
    LensSystem L; L.name = "achromat";
    L.surf.push_back({ R1, ct1, crown,            ap });          // crown front
    L.surf.push_back({ R2, ct2, flint,            ap });          // cemented crown/flint
    L.surf.push_back({ R4, gap, iorConstant(1.0), ap });          // flint rear -> air
    LensSurface stop; stop.radius = 0.0; stop.thickness = f; stop.ior = iorConstant(1.0);
    stop.aperture = semi; stop.isStop = true;
    L.surf.push_back(stop);                                                 // diaphragm, rear gap ~ f
    L.finalize();
    // The power split is the thin-lens one: the thicknesses move the focal length off it. The
    // stop sits behind the converging doublet, which narrows the beam on its way there, so
    // setFNumber sizes it through its pupil magnification, for an entrance pupil of exactly
    // f/2N, which every element's aperture clears.
    lensFinishPreset(L, focalMM, fstop);
    return L;
}

// ---- published prescriptions ------------------------------------------------------------
// Real lenses, from pbrt's lens files (pbrt-v4-scenes/lenses), each the scaled version of a
// design in Smith, "Modern Lens Design" (MLD). The rows are the files' own, untouched:
// radius in the published sign convention (centre of curvature on the sensor side when
// positive; 0 is the aperture stop), gap to the next surface (the last one's 0: autofocus
// places the sensor), refractive index (0 for air, as the stop rows write it), and aperture
// DIAMETER. Each glass has a single index, so these presets have no dispersion. They are left
// as published -- never thickened -- and scaled as a whole.
struct PublishedLens { const char* name; const double (*rows)[4]; int n; };

// dgauss.50mm.dat: D-GAUSS F/2 22deg HFOV, US patent 2,673,491 (Tronnier), MLD p. 312, scaled
// to 50 mm from 100 mm. The classic normal lens: two meniscus groups round a central stop.
inline const double kLensDGauss[11][4] = {
    { 29.475,  3.76,  1.67,  25.2 }, { 84.83,   0.12,  1.0,   25.2 }, { 19.275,  4.025, 1.67,  23.0 },
    { 40.77,   3.275, 1.699, 23.0 }, { 12.75,   5.705, 1.0,   18.0 }, { 0.0,     4.5,   0.0,   17.1 },
    {-14.495,  1.18,  1.603, 17.0 }, { 40.77,   6.065, 1.658, 20.0 }, {-20.385,  0.19,  1.0,   20.0 },
    { 437.065, 3.22,  1.717, 20.0 }, {-39.73,   0.0,   1.0,   20.0 }};
// telephoto.250mm.dat: SIGLER super achromate telephoto, EFL 254 mm, f/5.6, MLD p. 175, scaled
// to 250 mm from 100 mm. A positive front group, the stop, a negative rear group: the lens is
// shorter than its focal length.
inline const double kLensTelephoto[7][4] = {
    { 54.6275, 12.52,   1.529, 47.5 }, {-86.365,   3.755,  1.599, 44.5 }, { 271.7625, 2.8175, 1.0, 41.5 },
    { 0.0,     67.4125, 0.0,   40.5 }, {-32.13,    3.755,  1.613, 31.5 }, { 49.5325, 12.52,   1.603, 33.5 },
    {-50.945,  0.0,     1.0,   37.0 }};
// wide.22mm.dat: wide-angle (38 degree) lens, Nakamura, MLD p. 360, scaled to 22 mm from 100 mm.
// A negative front meniscus ahead of the stop, the retrofocus layout that buys back focus.
inline const double kLensWide[13][4] = {
    { 35.98738,  1.21638, 1.54,  23.716 }, { 11.69718,  9.9957,  1.0,   17.996 },
    { 13.08714,  5.12622, 1.772, 12.364 }, {-22.63294,  1.76924, 1.617,  9.812 },
    { 71.05802,  0.8184,  1.0,    9.152 }, { 0.0,       2.27766, 0.0,    8.756 },
    { -9.58584,  2.43254, 1.617,  8.184 }, {-11.28864,  0.11506, 1.0,    9.152 },
    {-166.7765,  3.09606, 1.713, 10.648 }, { -7.5911,   1.32682, 1.805, 11.44  },
    {-16.7662,   3.98068, 1.0,   12.276 }, { -7.70286,  1.21638, 1.617, 13.42  },
    {-11.97328,  0.0,     1.0,   17.996 }};
// fisheye.10mm.dat: Muller 16 mm f/4 155.9-degree fisheye, MLD p. 164, scaled to 10 mm from
// 100 mm. Two strongly curved negative menisci bend light from nearly 80 degrees off axis into
// the rest of the lens -- the case a front-element sampler wastes most on (well under 1% of
// its rays get through).
inline const double kLensFisheye[12][4] = {
    { 30.2249,  0.8335, 1.62,  30.34 }, { 11.3931,  7.4136, 1.0,   20.68 }, { 75.2019,  1.0654, 1.639, 17.8 },
    {  8.3349, 11.1549, 1.0,   13.42 }, {  9.5882,  2.0054, 1.654,  9.02 }, { 43.8677,  5.3895, 1.0,    8.14 },
    {  0.0,     1.4163, 0.0,    6.08 }, { 29.4541,  2.1934, 1.517,  5.96 }, { -5.2265,  0.9714, 1.805,  5.84 },
    {-14.2884,  0.0627, 1.0,    5.96 }, {-22.3726,  0.94,   1.673,  5.96 }, {-15.0404,  0.0,    1.0,    6.52 }};

inline const PublishedLens kPublishedLenses[] = {
    { "dgauss",    kLensDGauss,    11 },
    { "telephoto", kLensTelephoto,  7 },
    { "wide",      kLensWide,      13 },
    { "fisheye",   kLensFisheye,   12 },
};

// Build a published lens: scaled so its focal length is exactly `focalMM` (<= 0 keeps the
// published size), its stop sized so it lets in exactly an f/`fstop` beam (<= 0 keeps the
// design's own opening), but never wider than the design's stop (a larger one would only be
// clipped by the elements round it). A row of radius 0 is the stop, as pbrt's files mark it.
inline LensSystem makePublishedLens(const PublishedLens& P, double focalMM, double fstop) {
    LensSystem L;
    L.name = P.name;
    int stop = -1;
    for (int i = 0; i < P.n; ++i) {
        const double* r = P.rows[i];
        LensSurface s;
        s.radius = (r[0] == 0.0) ? 0.0 : -r[0];          // published convention -> this frame
        s.thickness = r[1];
        s.ior = iorConstant(r[2] == 0.0 ? 1.0 : r[2]);    // 0 is air
        s.aperture = 0.5 * r[3];                          // diameter -> semi-aperture
        s.isStop = (r[0] == 0.0);
        if (s.isStop && stop < 0) stop = (int)L.surf.size();
        L.surf.push_back(s);
    }
    L.finalize();
    const double f0 = L.focalLengthMM();
    const double maxStopPerMM = (f0 > 0.0 && stop >= 0) ? L.surf[stop].aperture / f0 : 0.0;
    lensFinishPreset(L, focalMM > 0.0 ? focalMM : f0, fstop, maxStopPerMM, /*thicken=*/false);
    return L;
}
inline const PublishedLens* findPublishedLens(const std::string& key) {
    for (const PublishedLens& P : kPublishedLenses) if (key == P.name) return &P;
    return nullptr;
}

// The canonical name of a preset (`k` already lowercased with spaces, '_' and '-' removed):
// the aliases folded in, "" if unknown.
inline std::string lensPresetName(const std::string& k) {
    if (k == "singlet" || k == "biconvex" || k == "simple") return "singlet";
    if (k == "achromat" || k == "doublet") return "achromat";
    if (k == "dgauss" || k == "doublegauss") return "dgauss";
    if (k == "telephoto" || k == "tele") return "telephoto";
    if (k == "wide" || k == "wideangle") return "wide";
    if (k == "fisheye") return "fisheye";
    return "";
}
// Is the preset a published prescription (true), or designed from formulas (false)?
inline bool lensPresetIsPublished(const std::string& canonical) {
    return findPublishedLens(canonical) != nullptr;
}

// Resolve a named lens preset into a prescription. The formula-built ones (singlet,
// achromat) take `focalMM` (50 when <= 0) and f/`fstop` (2.8 when <= 0); the published ones
// (dgauss, telephoto, wide, fisheye) keep their own size and opening for a value <= 0.
// Returns false on an unknown name.
inline bool resolveLensPreset(const std::string& name, double focalMM, double fstop,
                              LensSystem& out) {
    std::string k;
    for (char c : name) { if (c == ' ' || c == '_' || c == '-') continue; k += (char)std::tolower((unsigned char)c); }
    const std::string canon = lensPresetName(k);
    if (canon.empty()) return false;
    if (const PublishedLens* P = findPublishedLens(canon)) {
        out = makePublishedLens(*P, focalMM, fstop);
        return true;
    }
    if (focalMM <= 0.0) focalMM = 50.0;
    if (fstop   <= 0.0) fstop   = 2.8;
    const Spectrum bk7 = defaultGlass("BK7"), sf10 = defaultGlass("SF10");
    if (canon == "singlet") { out = makeSinglet(focalMM, fstop, bk7); out.name = "singlet"; return true; }
    out = makeAchromat(focalMM, fstop, bk7, sf10); out.name = "achromat";
    return true;
}
