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
#include "linalg.h"
#include "geometry.h"   // Ray, PI
#include "spectrum.h"           // Spectrum, iorConstant, sellmeier/cauchy evaluators
#include "spectral_library.h"   // resolveGlassIor (loads BK7/SF10 dispersion from data/glass)

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
    double denom = (std::fabs(V1 - V2) < 1e-6) ? 1e-6 : (V1 - V2);
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
    // Crown/flint glasses for the singlet/doublet elements, loaded from the spectral
    // library (data/glass/BK7.glass, SF10.glass). Constant-index fallbacks keep the
    // preset usable if the data files are missing.
    Spectrum bk7, sf10;
    if (!resolveGlassIor("BK7",  bk7))  bk7  = iorConstant(1.5168);
    if (!resolveGlassIor("SF10", sf10)) sf10 = iorConstant(1.7283);
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
