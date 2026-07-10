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
    double lensF     = 0.0;  // thin-lens focal length; 0 => no lens (straight-through
                             // camera obscura: blurred everywhere, no focus plane).

    // Configure a thin lens so the plane at `focusDist` in front of the lens
    // images sharply onto the film. Thin-lens law 1/so + 1/si = 1/f with the
    // image distance si = filmDist gives f = 1/(1/focusDist + 1/filmDist).
    // A larger apertureR then yields a shallower depth of field (more bokeh).
    void setFocus(double focusDist) {
        if (focusDist > 0.0) lensF = 1.0 / (1.0 / focusDist + 1.0 / filmDist);
        else                 lensF = 0.0;
    }

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

    // Generate a pinhole ray through raster pixel (px,py) with in-pixel jitter
    // (jx,jy in [0,1)). Exact inverse of project(): sx,sy are the [-1,1] view
    // coordinates project() would recover. Used by the backward reference tracer.
    Ray genRay(int px, int py, double jx, double jy) const {
        double sx = 2.0 * ((px + jx) / (double)film.resX) - 1.0;
        double sy = 2.0 * ((py + jy) / (double)film.resY) - 1.0;
        Vec3 d = normalize(w + u * (sx * tanHalfX) + v * (sy * tanHalfY));
        return Ray{eye, d};
    }

    // Model A perspective catch: does this photon ray pass through the finite
    // aperture disc (before hitting the scene, within hitDist) and land on the
    // film? Pure forward physics — no connect/splat. On success sets px,py.
    // The film sits filmDist behind the aperture, so the image is real (inverted);
    // we un-invert here so the result matches project()'s raster convention.
    //
    // With a thin lens (lensF > 0) the photon direction is refracted at the lens
    // by the paraxial ray transfer u' = u - rho/f (rho = transverse hit position,
    // u = transverse slope). Rays from a point at the focus distance then all land
    // at one film point (sharp); other depths spread into a blur circle (bokeh).
    bool catchPhoton(const Ray& ray, double hitDist, int& px, int& py) const {
        double dw = dot(ray.d, w);
        if (dw >= -1e-9) return false;                       // not heading toward the film
        double tAp = dot(eye - ray.o, w) / dw;
        if (tAp <= 1e-6 || tAp >= hitDist) return false;     // aperture not the first thing hit
        Vec3 P = ray.o + ray.d * tAp;                        // entry point on lens/aperture plane
        Vec3 rho = P - eye;                                  // transverse offset (perp to w)
        if (dot(rho, rho) > apertureR * apertureR) return false; // missed the disc

        Vec3 nAxis = w * (-1.0);        // propagation axis toward the film (behind eye)
        Vec3 dir = ray.d;
        if (lensF > 0.0) {
            double dax = dot(dir, nAxis);                    // > 0 (dw < 0)
            Vec3 slope = (dir - nAxis * dax) / dax;          // transverse slope u
            Vec3 slopeP = slope - rho * (1.0 / lensF);       // thin-lens: u' = u - rho/f
            dir = normalize(nAxis + slopeP);
        }
        double ddax = dot(dir, nAxis);
        if (ddax <= 1e-9) return false;
        double s = filmDist / ddax;                          // P -> film plane
        Vec3 Fcenter = eye + nAxis * filmDist;               // film centre (= eye - w*filmDist)
        Vec3 Q = P + dir * s;
        Vec3 rel = Q - Fcenter;
        double ix = -dot(rel, u) / (filmDist * tanHalfX);    // un-invert real image
        double iy = -dot(rel, v) / (filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * 0.5 + 0.5) * film.resX);
        py = (int)((iy * 0.5 + 0.5) * film.resY);
        return true;
    }
};
