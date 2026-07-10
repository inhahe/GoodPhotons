// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include "geometry.h"
#include "spectrum.h"
#include "scene_film.h"

struct Material {
    Spectrum reflect = constantSpectrum(0.5); // diffuse albedo vs lambda
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
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
    std::vector<Material> mats;
    Sensor sensor;

    // Area light: a quad (two tris) with uniform emission. Cached for sampling.
    Vec3 lightOrigin, lightU, lightV, lightNormal;
    double lightArea = 0.0;
    EmissionSampler lightSpd;
    double lightEmitIntegral = 0.0;

    void finalizeTris() { for (auto& t : tris) t.finalize(); }

    Hit closestHit(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        for (int i = 0; i < (int)tris.size(); ++i)
            if (intersectTri(r, tris[i], tmin, h)) h.tri = i;
        return h;
    }

    // Is anything blocking the segment from o toward dir, before maxDist?
    // Used by model-B camera connections (shadow ray to the pinhole).
    bool occluded(const Vec3& o, const Vec3& dir, double maxDist, double tmin = 1e-6) const {
        Ray r{o, dir};
        Hit h; h.t = maxDist - tmin;
        for (int i = 0; i < (int)tris.size(); ++i)
            if (intersectTri(r, tris[i], tmin, h)) return true;
        return false;
    }
};
