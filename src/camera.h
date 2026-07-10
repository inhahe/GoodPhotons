// Pinhole camera + light-tracing "connect to camera" (model B) importance.
// A photon at a surface vertex is connected to the pinhole and splatted onto the
// film. This is orders of magnitude faster than waiting for a photon to fly
// through a physical aperture (model A), while keeping photons independent.
//
// Thin-lens / full physical-lens cameras replace the pinhole later; the pupil
// there is a disk (or lens front element) we importance-sample instead of a point.
#pragma once
#include <cmath>
#include "geometry.h"
#include "scene_film.h"

struct Camera {
    Vec3 eye;
    Vec3 u, v, w;            // right, up, forward (orthonormal)
    double tanHalfX = 0, tanHalfY = 0;
    Film film;

    // Finite-aperture camera-obscura parameters (model A perspective catch).
    // A photon is recorded only if it flies through the aperture disc; smaller
    // aperture -> sharper but darker/grainier, larger -> brighter but blurrier.
    double apertureR = 0.02; // aperture radius (scene units)
    double filmDist  = 1.0;  // aperture->film distance (only ratio to apertureR matters for blur)

    void lookAt(Vec3 eye_, Vec3 target, Vec3 worldUp, double fovYDeg, int rx, int ry) {
        eye = eye_;
        w = normalize(target - eye);
        u = normalize(cross(w, worldUp));
        v = cross(u, w);
        tanHalfY = std::tan(0.5 * fovYDeg * 3.141592653589793 / 180.0);
        tanHalfX = tanHalfY * (double)rx / (double)ry;
        film.resX = rx; film.resY = ry; film.alloc();
    }

    // Project a world point to raster. Returns false if behind camera or off-film.
    // On success sets px,py and cosCam = cosine between forward and eye->point.
    bool project(const Vec3& p, int& px, int& py, double& cosCam, double& dist2) const {
        Vec3 d = p - eye;
        double cz = dot(d, w);
        if (cz <= 1e-9) return false;
        double cx = dot(d, u), cy = dot(d, v);
        double ix = (cx / cz) / tanHalfX;   // [-1,1] in view
        double iy = (cy / cz) / tanHalfY;
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * 0.5 + 0.5) * film.resX);
        py = (int)((iy * 0.5 + 0.5) * film.resY);
        dist2 = dot(d, d);
        cosCam = cz / std::sqrt(dist2);
        return true;
    }

    // Camera importance normaliser: image-plane area at unit distance.
    double imagePlaneArea() const { return 4.0 * tanHalfX * tanHalfY; }

    // Model A perspective catch: does this photon ray pass through the finite
    // aperture disc (before hitting the scene, within hitDist) and land on the
    // film? Pure forward physics — no connect/splat. On success sets px,py.
    // The film sits filmDist behind the aperture, so the image is real (inverted);
    // we un-invert here so the result matches project()'s raster convention.
    bool catchPhoton(const Ray& ray, double hitDist, int& px, int& py) const {
        double dw = dot(ray.d, w);
        if (dw >= -1e-9) return false;                       // not heading toward the film
        double tAp = dot(eye - ray.o, w) / dw;
        if (tAp <= 1e-6 || tAp >= hitDist) return false;     // aperture not the first thing hit
        Vec3 P = ray.o + ray.d * tAp;                        // entry point on aperture plane
        Vec3 r = P - eye;
        if (dot(r, r) > apertureR * apertureR) return false; // missed the aperture disc
        double s = -filmDist / dw;                           // P -> film plane (dw<0 => s>0)
        Vec3 Fcenter = eye - w * filmDist;
        Vec3 Q = P + ray.d * s;
        Vec3 rel = Q - Fcenter;
        double ix = -dot(rel, u) / (filmDist * tanHalfX);    // un-invert real image
        double iy = -dot(rel, v) / (filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * 0.5 + 0.5) * film.resX);
        py = (int)((iy * 0.5 + 0.5) * film.resY);
        return true;
    }
};
