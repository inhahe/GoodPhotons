// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include "geometry.h"
#include "spectrum.h"
#include "scene_film.h"

enum class MatType { Diffuse, Dielectric, Mirror, HalfMirror, Glossy };

struct Material {
    MatType type = MatType::Diffuse;
    // reflect means: diffuse albedo / mirror tint / glossy tint / half-mirror
    // reflect-probability, depending on type.
    Spectrum reflect = constantSpectrum(0.5);
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    Spectrum ior     = iorConstant(1.5);      // dielectric index vs lambda
    double roughness = 0.1;                    // glossy lobe width [0,1]
    bool isLight = false;
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

    // Area light: a quad (two tris) with uniform emission. Cached for sampling.
    Vec3 lightOrigin, lightU, lightV, lightNormal;
    double lightArea = 0.0;
    EmissionSampler lightSpd;
    double lightEmitIntegral = 0.0;

    // Collimated-beam mode: all photons travel along beamDir (for the prism demo).
    bool collimated = false;
    Vec3 beamDir{1, 0, 0};

    void finalizeTris() { for (auto& t : tris) t.finalize(); }

    Hit closestHit(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        for (const auto& t : tris)    intersectTri(r, t, tmin, h);
        for (const auto& s : spheres) intersectSphere(r, s, tmin, h);
        return h;
    }

    // Is anything blocking the segment from o toward dir, before maxDist?
    // Used by model-B camera connections (shadow ray to the pinhole).
    // NOTE: dielectrics block connections (can't connect through specular) — the
    // SDS limitation. Glass therefore appears dark in model B; caustics it casts
    // onto diffuse surfaces still render, since those diffuse vertices connect.
    bool occluded(const Vec3& o, const Vec3& dir, double maxDist, double tmin = 1e-6) const {
        Ray r{o, dir};
        Hit h; h.t = maxDist - tmin;
        for (const auto& t : tris)    if (intersectTri(r, t, tmin, h)) return true;
        for (const auto& s : spheres) if (intersectSphere(r, s, tmin, h)) return true;
        return false;
    }
};
