// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include <algorithm>
#include "geometry.h"
#include "bvh.h"
#include "spectrum.h"
#include "scene_film.h"

enum class MatType { Diffuse, Dielectric, Mirror, HalfMirror, Glossy, Fluorescent };

struct Material {
    MatType type = MatType::Diffuse;
    // reflect means: diffuse albedo / mirror tint / glossy tint / half-mirror
    // reflect-probability, depending on type. For Fluorescent it is the elastic
    // (wavelength-preserving) diffuse albedo.
    Spectrum reflect = constantSpectrum(0.5);
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    Spectrum ior     = iorConstant(1.5);      // dielectric index vs lambda
    double roughness = 0.1;                    // glossy lobe width [0,1]
    bool isLight = false;

    // --- Fluorescence (MatType::Fluorescent) --------------------------------
    // A photon at lambda excites the dye with probability fluoAbsorb(lambda); the
    // dye then re-radiates (quantum yield fluoYield) at a Stokes-shifted lambda'
    // drawn from the normalized emission SPD fluoEmit. Single-wavelength forward
    // tracing handles this naturally: sample lambda' ~ fluoEmit and the M/pdf
    // ratio cancels, so the throughput weight is just the branch probability.
    Spectrum fluoAbsorb = constantSpectrum(0.0);  // excitation prob epsilon(lambda)
    Spectrum fluoEmit   = constantSpectrum(0.0);  // emission SPD M(lambda') (shape)
    EmissionSampler fluoEmitSampler;              // built from fluoEmit
    double fluoYield = 1.0;                        // quantum yield Q in [0,1]
};

// A classic "green highlighter" fluorophore: absorbs blue/violet strongly, glows
// green (~560 nm). Shared by the fluoro demo scene and the -checkfluoro self-test
// so both exercise the exact same material definition (single source of truth).
inline Material makeFluoroMaterial() {
    Material f;
    f.type = MatType::Fluorescent;
    f.reflect     = constantSpectrum(0.05);         // small elastic base reflectance
    f.fluoAbsorb  = shortPass(480.0, 0.06, 0.85);   // excite below ~480 nm
    f.fluoEmit    = gaussianBand(560.0, 25.0, 1.0); // emit green-yellow
    f.fluoEmitSampler.build(f.fluoEmit, 1.0);
    f.fluoYield   = 0.9;
    return f;
}

// A homogeneous participating medium filling the whole scene (fog / haze). A
// photon travelling a distance travels freely until a collision sampled from
// exp(-sigma_t * t); at the collision it scatters (prob albedo = sigma_s/sigma_t,
// new direction from the Henyey-Greenstein phase function) or is absorbed. Beer-
// Lambert transmittance is captured implicitly by the free-flight sampling (analog
// Monte Carlo), so photon throughput stays unchanged — matching the rest of the
// renderer. Coefficients are spectral, so wavelength-dependent (e.g. Rayleigh
// ~1/lambda^4) fog that scatters blue and transmits red works for free.
struct Medium {
    bool enabled = false;
    Spectrum sigma_a = constantSpectrum(0.0); // absorption coefficient vs lambda
    Spectrum sigma_s = constantSpectrum(0.0); // scattering coefficient vs lambda
    double g = 0.0;                            // HG anisotropy [-1,1] (0 = isotropic)

    double sigmaT(double lambda) const {
        return std::max(0.0, sigma_a(lambda) + sigma_s(lambda));
    }
    double albedo(double lambda) const {       // single-scattering albedo sigma_s/sigma_t
        double s = std::max(0.0, sigma_s(lambda));
        double t = s + std::max(0.0, sigma_a(lambda));
        return t > 0.0 ? s / t : 0.0;
    }
};

// A flat rectangular contact sensor (model A) spanning origin + s*uAxis + t*vAxis.
struct Sensor {
    Vec3 origin, uAxis, vAxis; // uAxis/vAxis are full edge vectors
    Film film;
    void alloc() { film.alloc(); }
};

struct Scene {
    std::vector<Tri> tris;
    std::vector<Sphere> spheres;
    std::vector<Material> mats;
    Sensor sensor;
    Medium medium;   // optional global fog / participating medium (disabled by default)

    // Area light: a quad (two tris) with uniform emission. Cached for sampling.
    Vec3 lightOrigin, lightU, lightV, lightNormal;
    double lightArea = 0.0;
    EmissionSampler lightSpd;
    double lightEmitIntegral = 0.0;

    // Collimated-beam mode: all photons travel along beamDir (for the prism demo).
    bool collimated = false;
    Vec3 beamDir{1, 0, 0};

    Bvh bvh;   // acceleration structure over tris (0..nTris) then spheres.

    // Finalize triangle normals and build the BVH. Call after all geometry is
    // added. Primitive index i: i < tris.size() -> tris[i]; else spheres[i-nTris].
    void build() {
        for (auto& t : tris) t.finalize();
        buildBvh();
    }
    void finalizeTris() { build(); }   // kept for existing call sites

    void buildBvh() {
        const double pad = 1e-6;       // avoid zero-thickness slabs on flat prims
        std::vector<Aabb> boxes;
        boxes.reserve(tris.size() + spheres.size());
        for (const auto& t : tris) {
            Aabb b; b.expand(t.v0); b.expand(t.v1); b.expand(t.v2);
            b.lo = b.lo - Vec3{pad, pad, pad}; b.hi = b.hi + Vec3{pad, pad, pad};
            boxes.push_back(b);
        }
        for (const auto& s : spheres) {
            Aabb b; b.expand(s.c - Vec3{s.r, s.r, s.r}); b.expand(s.c + Vec3{s.r, s.r, s.r});
            boxes.push_back(b);
        }
        bvh.build(boxes);
    }

    Hit closestHit(const Ray& r, double tmin = 1e-6, TraversalStats* stats = nullptr) const {
        Hit h;
        double tMax = DBL_MAX;
        const size_t nT = tris.size();
        bvh.traverseClosest(r, tmin, tMax, [&](int prim, double& tm) {
            if (prim < (int)nT) { if (intersectTri(r, tris[prim], tmin, h)) tm = h.t; }
            else                { if (intersectSphere(r, spheres[prim - nT], tmin, h)) tm = h.t; }
        }, stats);
        return h;
    }

    // Is anything blocking the segment from o toward dir, before maxDist?
    // Used by model-B camera connections (shadow ray to the pinhole).
    // NOTE: dielectrics block connections (can't connect through specular) — the
    // SDS limitation. Glass therefore appears dark in model B; caustics it casts
    // onto diffuse surfaces still render, since those diffuse vertices connect.
    bool occluded(const Vec3& o, const Vec3& dir, double maxDist, double tmin = 1e-6) const {
        Ray r{o, dir};
        const size_t nT = tris.size();
        return bvh.traverseAny(r, tmin, maxDist - tmin, [&](int prim) {
            Hit h; h.t = maxDist - tmin;
            if (prim < (int)nT) return intersectTri(r, tris[prim], tmin, h);
            return intersectSphere(r, spheres[prim - nT], tmin, h);
        });
    }

    // Linear-scan reference (pre-BVH), kept for the -checkbvh self-test.
    Hit closestHitLinear(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        for (const auto& t : tris)    intersectTri(r, t, tmin, h);
        for (const auto& s : spheres) intersectSphere(r, s, tmin, h);
        return h;
    }
};
