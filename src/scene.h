// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include "geometry.h"
#include "spectrum.h"

struct Material {
    Spectrum reflect = constantSpectrum(0.5); // diffuse albedo vs lambda
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    bool isLight = false;
};

// A flat rectangular sensor spanning origin + s*uAxis + t*vAxis, s,t in [0,1].
struct Sensor {
    Vec3 origin, uAxis, vAxis; // uAxis/vAxis are full edge vectors
    int resX = 256, resY = 256;
    std::vector<Vec3> xyz;     // accumulated XYZ per pixel
    std::vector<double> hits;  // photon hits per pixel (diagnostics)
    void alloc() { xyz.assign((size_t)resX * resY, {}); hits.assign((size_t)resX * resY, 0.0); }
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
};
