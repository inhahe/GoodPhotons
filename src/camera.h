// Pinhole camera + light-tracing "connect to camera" (model B) importance.
// A photon at a surface vertex is connected to the pinhole and splatted onto the
// film. This is orders of magnitude faster than waiting for a photon to fly
// through a physical aperture (model A), while keeping photons independent.
//
// Thin-lens / full physical-lens cameras replace the pinhole later; the pupil
// there is a disk (or lens front element) we importance-sample instead of a point.
#pragma once
#include <cmath>
#include <memory>
#include "geometry.h"
#include "scene_film.h"
#include "lens.h"

// Lens projection model: the mapping from a ray's angle-from-axis theta to its
// image radius r (with focal length 1). RECTILINEAR (r = tan theta) is the normal
// perspective lens: straight lines stay straight, but it cannot reach 180 deg and
// stretches the corners. The others are fisheye/panoramic projections that trade
// straight lines for very wide (>= 180 deg) fields:
//   EQUIDISTANT   r = theta            (classic "true" fisheye; angle proportional to radius)
//   EQUISOLID     r = 2 sin(theta/2)   (most consumer fisheyes; preserves area)
//   STEREOGRAPHIC r = 2 tan(theta/2)   ("little planet"; preserves local shape)
//   ORTHOGRAPHIC  r = sin(theta)       (hemispherical, 180 deg max)
enum CameraProjection {
    CAM_RECTILINEAR = 0, CAM_EQUIDISTANT, CAM_EQUISOLID, CAM_STEREOGRAPHIC, CAM_ORTHOGRAPHIC
};

// r(theta): image radius for a ray at angle theta from the optical axis (focal = 1).
inline double projRadius(int proj, double th) {
    switch (proj) {
        case CAM_EQUIDISTANT:   return th;
        case CAM_EQUISOLID:     return 2.0 * std::sin(0.5 * th);
        case CAM_STEREOGRAPHIC: return 2.0 * std::tan(0.5 * th);
        case CAM_ORTHOGRAPHIC:  return std::sin(th);
        default:                return std::tan(th);            // CAM_RECTILINEAR
    }
}
// Inverse: theta for a given image radius r (focal = 1). Clamps the arcsin domain.
inline double projRadiusInv(int proj, double r) {
    auto cl = [](double x){ return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); };
    switch (proj) {
        case CAM_EQUIDISTANT:   return r;
        case CAM_EQUISOLID:     return 2.0 * std::asin(cl(0.5 * r));
        case CAM_STEREOGRAPHIC: return 2.0 * std::atan(0.5 * r);
        case CAM_ORTHOGRAPHIC:  return std::asin(cl(r));
        default:                return std::atan(r);            // CAM_RECTILINEAR
    }
}
// dr/dtheta: used for the per-pixel solid angle (splat importance Jacobian).
inline double projRadiusDeriv(int proj, double th) {
    switch (proj) {
        case CAM_EQUIDISTANT:   return 1.0;
        case CAM_EQUISOLID:     return std::cos(0.5 * th);
        case CAM_STEREOGRAPHIC: { double c = std::cos(0.5 * th); return 1.0 / (c * c); }
        case CAM_ORTHOGRAPHIC:  return std::cos(th);
        default:              { double c = std::cos(th);       return 1.0 / (c * c); } // rect: sec^2
    }
}

struct Camera {
    Vec3 eye;
    Vec3 u, v, w;            // right, up, forward (orthonormal)
    double tanHalfX = 0, tanHalfY = 0;
    Film film;

    // Lens projection (see CameraProjection). RECTILINEAR is the default and keeps
    // the classic pinhole math byte-for-byte; the fisheye/panoramic modes remap the
    // ray angle. `halfFovY` is the vertical half-field in radians and `rEdge` is the
    // image radius at the vertical film edge (= projRadius(projection, halfFovY)),
    // both set by lookAt/setProjection so genRay/project can normalise the film.
    int    projection = CAM_RECTILINEAR;
    double halfFovY   = 0.0;
    double rEdge      = 0.0;

    // Finite-aperture camera-obscura parameters. These define the physical lens
    // shared by the next-event lens splat (model A, the default physical camera)
    // and the brute-force forward catch (model C). Smaller aperture -> deeper focus
    // (sharper), larger -> shallower depth of field (more bokeh). Model B is the
    // apertureR -> 0 pinhole limit and ignores these.
    double apertureR = 0.02; // aperture radius (scene units)
    double filmDist  = 1.0;  // aperture->film distance (only ratio to apertureR matters for blur)
    double lensF     = 0.0;  // thin-lens focal length; 0 => no lens (straight-through
                             // camera obscura: blurred everywhere, no focus plane).

    // Off-axis (asymmetric-frustum) horizontal shift, in normalised [-1,1] view units.
    // Used ONLY by stereoscopic rendering: the two eyes are two PARALLEL cameras offset
    // along the right axis u, each with a horizontally SHEARED frustum so a shared
    // convergence plane has zero parallax (the correct off-axis method — no toe-in, so
    // no vertical parallax / eye strain). project()/genRay()/lensImage() add this shift
    // consistently so they remain exact inverses of one another. Rectilinear only;
    // 0 (the default) keeps every non-stereo render byte-for-byte identical.
    double frustumShiftX = 0.0;

    // Optional physical multi-element lens (the "mesh-lens" camera). When set, the
    // backward reference tracer (mode R) generates rays by tracing them from the film
    // out through the real glass interfaces (genLensRay), superseding the analytic
    // pinhole/thin-lens projection. Shared so copies of a Camera share one lens.
    std::shared_ptr<LensSystem> lens;
    bool hasLens() const { return lens && !lens->surf.empty(); }

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
        halfFovY = 0.5 * fovYDeg * 3.141592653589793 / 180.0;
        setProjection(projection);   // (re)compute rEdge for the current projection
        // Store ONLY the film resolution metadata — project()/genRay()/pixelPlaneArea()/
        // pixelSolidAngle()/genLensRay() read film.resX/resY to normalise the raster, but
        // nothing ever accumulates into THIS camera's embedded xyz/hits buffers (every
        // render path — renderForward, renderPhotonCamera, renderForwardShared's per-thread
        // targets, the backward tracer, checkpoints — owns a SEPARATE Film). Allocating the
        // ~res*32-byte buffer per camera is pure waste and, for a long camera_curve, fatal:
        // a 600-frame flyby built 600 toRender + 600 meter cameras, each carrying a live
        // 960x540 film (~16 MB), OOMing at ~20 GB before a single photon traced. Skipping
        // alloc() here drops each camera to a few hundred bytes.
        film.resX = rx; film.resY = ry;
    }

    // Select the lens projection and cache the vertical-edge image radius. Call
    // after lookAt (which sets halfFovY) so authored fisheye lenses take effect.
    void setProjection(int p) { projection = p; rEdge = projRadius(p, halfFovY); }

    // Project a world point to raster. Returns false if behind camera or off-film.
    // On success sets px,py and cosCam = cosine between forward and eye->point.
    bool project(const Vec3& p, int& px, int& py, double& cosCam, double& dist2) const {
        Vec3 d = p - eye;
        double cz = dot(d, w);
        if (projection == CAM_RECTILINEAR) {
            if (cz <= 1e-9) return false;
            double cx = dot(d, u), cy = dot(d, v);
            double ix = (cx / cz) / tanHalfX - frustumShiftX;   // [-1,1] in view (off-axis shear)
            double iy = (cy / cz) / tanHalfY;
            if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
            px = (int)((ix * 0.5 + 0.5) * film.resX);
            py = (int)((iy * 0.5 + 0.5) * film.resY);
            dist2 = dot(d, d);
            cosCam = cz / std::sqrt(dist2);
            return true;
        }
        // Fisheye/panoramic: map the direction's angle-from-axis theta to an image
        // radius via the projection, then place it along the (u,v) azimuth. A wide
        // lens can see theta > 90 deg (cz <= 0), so we do NOT reject on cz here.
        double len = std::sqrt(dot(d, d));
        if (len < 1e-12) return false;
        double costh = cz / len;
        if (costh < -1) costh = -1; else if (costh > 1) costh = 1;
        double th = std::acos(costh);
        double rho = projRadius(projection, th) / rEdge;      // normalised image radius
        double ru = dot(d, u), rv = dot(d, v);
        double rhoDir = std::sqrt(ru * ru + rv * rv);
        double ix, iy;
        if (rhoDir < 1e-12) { ix = 0.0; iy = 0.0; }
        else { ix = rho * ru / rhoDir; iy = rho * rv / rhoDir; }
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;   // outside the film
        px = (int)((ix * 0.5 + 0.5) * film.resX);
        py = (int)((iy * 0.5 + 0.5) * film.resY);
        dist2 = len * len;
        cosCam = costh;
        return true;
    }

    // Solid angle subtended by one pixel for a connection whose direction makes
    // cosine `cosCam` with the optical axis. This is the projection-general splat
    // normaliser: the mode-B contribution is beta*f*cosSurf/(dist^2 * pixelSolidAngle).
    // For a rectilinear lens this equals pixelPlaneArea()*cosCam^3 (recovering the
    // classic 1/(A_pix cos^4) importance once the geometry's cosCam is folded in).
    double pixelSolidAngle(double cosCam) const {
        if (projection == CAM_RECTILINEAR)
            return pixelPlaneArea() * cosCam * cosCam * cosCam;
        double c = cosCam < -1 ? -1 : (cosCam > 1 ? 1 : cosCam);
        double th = std::acos(c);
        double dr = projRadiusDeriv(projection, th);
        double r  = projRadius(projection, th);
        double denom = dr * r;
        if (denom < 1e-12) denom = 1e-12;
        double aNorm = 4.0 / ((double)film.resX * (double)film.resY);  // pixel area in [-1,1]^2 view
        return aNorm * std::sin(th) * rEdge * rEdge / denom;
    }

    // Camera importance normaliser: image-plane area at unit distance.
    double imagePlaneArea() const { return 4.0 * tanHalfX * tanHalfY; }

    // Per-pixel image-plane area at unit distance. The model-B connect() splats a
    // photon's contribution into a single pixel, so the pinhole importance must be
    // normalised by the area of ONE pixel on the image plane (imagePlaneArea /
    // pixel-count), not the whole plane. Using this makes the forward light tracer
    // measure ABSOLUTE radiance (pixel == L), so it agrees with the backward
    // reference on a unit scale (mode V/P best-fit -> ~1) and the directly-viewed
    // environment background composites without any ad-hoc rescale.
    double pixelPlaneArea() const {
        return imagePlaneArea() / ((double)film.resX * (double)film.resY);
    }

    // Generate a pinhole ray through raster pixel (px,py) with in-pixel jitter
    // (jx,jy in [0,1)). Exact inverse of project(): sx,sy are the [-1,1] view
    // coordinates project() would recover. Used by the backward reference tracer.
    Ray genRay(int px, int py, double jx, double jy) const {
        double sx = 2.0 * ((px + jx) / (double)film.resX) - 1.0;
        double sy = 2.0 * ((py + jy) / (double)film.resY) - 1.0;
        if (projection == CAM_RECTILINEAR) {
            Vec3 d = normalize(w + u * ((sx + frustumShiftX) * tanHalfX) + v * (sy * tanHalfY));
            return Ray{eye, d};
        }
        // Fisheye/panoramic: the normalised film radius rho maps to a ray angle th.
        double rho = std::sqrt(sx * sx + sy * sy);
        if (rho < 1e-12) return Ray{eye, w};
        double th = projRadiusInv(projection, rho * rEdge);
        if (th > 3.141592653589793) th = 3.141592653589793;
        Vec3 radial = (u * sx + v * sy) * (1.0 / rho);   // unit in-film direction
        Vec3 d = normalize(w * std::cos(th) + radial * std::sin(th));
        return Ray{eye, d};
    }

    // Generate a world-space camera ray for pixel (px,py) through the physical lens
    // (requires hasLens()). Samples the film point (with pixel jitter jx,jy) and a
    // point on the rear element disk (u1,u2), then refracts the ray from the film out
    // through every glass interface at wavelength `lambda` (so chromatic aberration is
    // exact). Returns false if the ray is vignetted (clipped by an element or the
    // stop, or total-internal-reflected). On success `weight` is the radiometric
    // importance (cos^4 * pupil-area / Z^2) that makes exposure, depth of field and
    // corner vignetting physically correct; multiply the estimated radiance by it.
    bool genLensRay(int px, int py, double jx, double jy, double u1, double u2,
                    double lambda, Ray& worldRay, double& weight) const {
        weight = 0.0;
        const LensSystem& L = *lens;
        // Film point in lens-local mm. The real image is inverted, so we negate the
        // view coords: a +u/+v scene direction must come from a -x/-y film point,
        // which reproduces the pinhole genRay()'s raster orientation.
        double sx = 2.0 * ((px + jx) / (double)film.resX) - 1.0;
        double sy = 2.0 * ((py + jy) / (double)film.resY) - 1.0;
        // The renderer's film buffer may not share the sensor's aspect (it is square
        // in the current pipeline). Anchor on the sensor width and derive the mapped
        // vertical half-extent from the output pixel aspect so pixels stay square (no
        // stretch); this crops the physical 3:2 sensor to the output frame.
        double halfW = 0.5 * L.filmW_mm;
        double halfH = halfW * ((double)film.resY / (double)film.resX);
        Vec3 pFilm(-sx * halfW, -sy * halfH, L.filmZ);
        // Uniform sample on the rear-element disk (the pupil we trace toward).
        double rr = std::sqrt(u1 > 0 ? u1 : 0.0) * L.rearAperture();
        double phi = 2.0 * PI * u2;
        Vec3 pRear(rr * std::cos(phi), rr * std::sin(phi), L.rearZ());
        Vec3 d0 = normalize(pRear - pFilm);
        Ray outLocal;
        if (!L.trace(pFilm, d0, lambda, /*sensorToScene=*/true, outLocal)) return false;
        // PBRT-style importance: cos^4 of the film-ray axis angle, times the sampled
        // pupil area, over the squared film->pupil axial distance.
        double cosT = d0.z;                       // d0 is unit; z-component = cos to axis
        if (cosT <= 0.0) return false;
        double cos4 = (cosT * cosT) * (cosT * cosT);
        double A = PI * L.rearAperture() * L.rearAperture();
        double Z = L.rearZ() - L.filmZ;
        if (Z <= 1e-9) return false;
        weight = cos4 * A / (Z * Z);
        // Lens-local (mm) -> world. The front-surface vertex plane is pinned at `eye`;
        // transverse offsets and the axial (z-T) offset convert mm -> scene metres.
        Vec3 oW = eye + (u * outLocal.o.x + v * outLocal.o.y) * 1e-3
                      + w * ((outLocal.o.z - L.T) * 1e-3);
        Vec3 dW = u * outLocal.d.x + v * outLocal.d.y + w * outLocal.d.z;
        worldRay = Ray{oW, normalize(dW)};
        return true;
    }

    // Image a photon that enters the lens at pupil point `A` travelling along
    // `dir`, refract it through the thin lens (if any) and find the raster cell it
    // lands on. Shared by the brute-force catch (model C) and the next-event lens
    // splat (model A). The film sits filmDist behind the aperture, so the image is
    // real (inverted); we un-invert here to match project()'s raster convention.
    //
    // With a thin lens (lensF > 0) the direction is refracted by the paraxial ray
    // transfer u' = u - rho/f (rho = transverse pupil position, u = transverse
    // slope). Rays from a point at the focus distance then all land at one film
    // point (sharp); other depths spread into a blur circle (bokeh).
    bool lensImage(const Vec3& A, const Vec3& dir, int& px, int& py) const {
        Vec3 nAxis = w * (-1.0);        // propagation axis toward the film (behind eye)
        Vec3 rho = A - eye;             // transverse offset of the pupil hit (perp to w)
        Vec3 d = dir;
        if (lensF > 0.0) {
            double dax = dot(d, nAxis);
            if (dax <= 1e-9) return false;                   // grazing/behind the lens
            Vec3 slope = (d - nAxis * dax) / dax;            // transverse slope u
            Vec3 slopeP = slope - rho * (1.0 / lensF);       // thin-lens: u' = u - rho/f
            d = normalize(nAxis + slopeP);
        }
        double ddax = dot(d, nAxis);
        if (ddax <= 1e-9) return false;
        double s = filmDist / ddax;                          // A -> film plane
        Vec3 Fcenter = eye + nAxis * filmDist;               // film centre (= eye - w*filmDist)
        Vec3 Q = A + d * s;
        Vec3 rel = Q - Fcenter;
        double ix = -dot(rel, u) / (filmDist * tanHalfX) - frustumShiftX;  // un-invert real image (off-axis shear)
        double iy = -dot(rel, v) / (filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * 0.5 + 0.5) * film.resX);
        py = (int)((iy * 0.5 + 0.5) * film.resY);
        return true;
    }

    // Model C brute-force catch: does this photon ray physically pass through the
    // finite aperture disc (before hitting the scene, within hitDist) and land on
    // the film? Pure forward physics — no connect/splat. On success sets px,py.
    bool catchPhoton(const Ray& ray, double hitDist, int& px, int& py) const {
        double dw = dot(ray.d, w);
        if (dw >= -1e-9) return false;                       // not heading toward the film
        double tAp = dot(eye - ray.o, w) / dw;
        if (tAp <= 1e-6 || tAp >= hitDist) return false;     // aperture not the first thing hit
        Vec3 P = ray.o + ray.d * tAp;                        // entry point on lens/aperture plane
        Vec3 rho = P - eye;                                  // transverse offset (perp to w)
        if (dot(rho, rho) > apertureR * apertureR) return false; // missed the disc
        return lensImage(P, ray.d, px, py);
    }
};
