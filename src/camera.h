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

    // ---- shading footprint (O8 stage 2) -------------------------------------------
    // The coefficient half of `fw`: the world-space width one camera sample covers PER
    // UNIT of distance, at normal incidence. A renderer computes this once per frame and
    // hands it to patShadingFootprint() at each primary hit, which adds the distance and
    // the obliquity; see that function for what the combination means and Hit::fw for who
    // fills it. Two decisions are baked in here:
    //
    //  * THE PIXEL'S ANGULAR SIZE comes from pixelSolidAngle(), not from fov_y/res_y, so
    //    fisheye and panoramic lenses — whose pixels are neither square nor uniform over
    //    the film — need no separate case. The solid angle is turned into the diameter of
    //    the disc SUBTENDING it, `2*sqrt(Om/pi)`, which is the right conversion precisely
    //    because `fw` is defined as a disc diameter. It is evaluated ON AXIS: the
    //    off-axis variation is a cos^3 effect (rectilinear), an order of magnitude below
    //    the obliquity term, and taking the on-axis value keeps `fw` from depending on
    //    which pixel a surface happens to land in.
    //  * SUPERSAMPLING. At `spp` jittered samples per pixel the reconstruction already
    //    averages over the footprint stochastically and the residual falls about as
    //    1/sqrt(spp), so the width one sample should band-limit to shrinks by the same
    //    factor. That is what makes `fw` degrade gracefully rather than lock in a blur:
    //    raise -spp and the filtering backs off on its own, converging to the unfiltered
    //    ground truth, which is exactly the behaviour that lets one scene serve both a
    //    1-spp preview and a converged render.
    //  * A RESOLUTION OVERRIDE (rx, ry). pixelSolidAngle() normalises by this camera's
    //    OWN film.resX/resY, but the preview rasterizer draws the same camera into a
    //    window of its own size. Passing that size rescales the pixel count — exact in
    //    both branches, since pixelSolidAngle is proportional to 1/(resX*resY) in each
    //    (pixelPlaneArea for rectilinear, aNorm for the angular maps). 0 = use the film.
    double footprintPerDist(int spp = 1, int rx = 0, int ry = 0) const {
        double om = pixelSolidAngle(1.0);
        if (rx > 0 && ry > 0 && film.resX > 0 && film.resY > 0)
            om *= ((double)film.resX * (double)film.resY) / ((double)rx * (double)ry);
        double d = 2.0 * std::sqrt((om > 0.0 ? om : 0.0) / 3.141592653589793);
        if (spp > 1) d /= std::sqrt((double)spp);
        return d;
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
    // point on the rear element (u1,u2: its disk, or its clear cap when it is curved),
    // then refracts the ray from the film out
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
        Ray outLocal;
        const int jr = (int)L.surf.size() - 1;
        if (L.surf[jr].radius != 0.0) {
            // A CURVED rear element: a point uniform on its clear cap, so every ray that
            // crosses the clear glass can be drawn (the plane disc below would miss the ones
            // crossing the cap outside the disc's shadow). The point must be the first of the
            // cap the film ray meets. Importance: cos at the film * cos at the cap * cap area
            // over the squared distance, the irradiance estimate for this pupil sample.
            Vec3 nR;
            const Vec3 pR = L.sampleSurface(jr, u1, u2, nR);
            Vec3 d0 = pR - pFilm;
            const double r = length(d0);
            if (!(r > 1e-12)) return false;
            d0 = d0 / r;
            const double cosT = d0.z;
            if (cosT <= 0.0) return false;
            Vec3 hit, n;
            if (!L.hitSurface(jr, pFilm, d0, hit, n)) return false;
            const Vec3 gap = hit - pR;
            if (dot(gap, gap) > 1e-12) return false;        // pR is on the cap's hidden side
            if (!L.trace(pFilm, d0, lambda, /*sensorToScene=*/true, outLocal)) return false;
            weight = cosT * std::fabs(dot(d0, nR)) * L.surfaceArea(jr) / (r * r);
        } else {
            // A planar rear (a stop, or a flat): a uniform sample on its disk.
            double rr = std::sqrt(u1 > 0 ? u1 : 0.0) * L.rearAperture();
            double phi = 2.0 * PI * u2;
            Vec3 pRear(rr * std::cos(phi), rr * std::sin(phi), L.rearZ());
            Vec3 d0 = normalize(pRear - pFilm);
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
        }
        // Lens-local (mm) -> world. The front-surface vertex plane is pinned at `eye`;
        // transverse offsets and the axial (z-T) offset convert mm -> scene metres.
        Vec3 oW = eye + (u * outLocal.o.x + v * outLocal.o.y) * 1e-3
                      + w * ((outLocal.o.z - L.T) * 1e-3);
        Vec3 dW = u * outLocal.d.x + v * outLocal.d.y + w * outLocal.d.z;
        worldRay = Ray{oW, normalize(dW)};
        // A degenerate prescription must not hand the tracer a NaN ray as a success: treat it
        // as vignetted.
        if (!(std::isfinite(worldRay.o.x) && std::isfinite(worldRay.o.y) && std::isfinite(worldRay.o.z) &&
              std::isfinite(worldRay.d.x) && std::isfinite(worldRay.d.y) && std::isfinite(worldRay.d.z) &&
              std::isfinite(weight))) { weight = 0.0; return false; }
        return true;
    }

    // ---- the physical lens traced FORWARD: light tracing into the glass --------------
    // genLensRay above is the camera's own direction: film point -> glass -> scene. The
    // forward modes need the opposite, so a scene point can be imaged through the real
    // optics. Three things use it: the next-event lens splat of modes A and B (lensConnect),
    // the brute-force catch of mode C (lensCatch), and BDPT's light-image strategy, which
    // also needs the density the camera subpath would have drawn the same ray with
    // (lensPdfDir).
    //
    // World <-> lens-local mm, the inverse of genLensRay's mapping: the front vertex plane
    // is pinned at `eye`, the optical axis is +z toward the scene.
    Vec3 lensLocalPoint(const Vec3& p) const {
        const Vec3 d = p - eye;
        return Vec3(dot(d, u) * 1e3, dot(d, v) * 1e3, dot(d, w) * 1e3 + lens->T);
    }
    Vec3 lensLocalDir(const Vec3& d) const { return Vec3(dot(d, u), dot(d, v), dot(d, w)); }
    Vec3 lensWorldPoint(const Vec3& q) const {
        return eye + (u * q.x + v * q.y) * 1e-3 + w * ((q.z - lens->T) * 1e-3);
    }
    // The film raster under the physical lens, as genLensRay lays it out: the sensor width
    // spans resX square pixels, so a pixel is filmW_mm/resX on a side; the image is real,
    // so a scene direction +u/+v lands at -x/-y on the sensor.
    double lensPixelMM() const { return lens->filmW_mm / (double)film.resX; }
    bool lensFilmToPixel(const Vec3& f, int& px, int& py) const {
        const double halfW = 0.5 * lens->filmW_mm;
        const double halfH = halfW * ((double)film.resY / (double)film.resX);
        const double sx = -f.x / halfW, sy = -f.y / halfH;
        if (!(sx >= -1.0 && sx < 1.0 && sy >= -1.0 && sy < 1.0)) return false;
        px = (int)((sx * 0.5 + 0.5) * film.resX);
        py = (int)((sy * 0.5 + 0.5) * film.resY);
        if (px >= film.resX) px = film.resX - 1;
        if (py >= film.resY) py = film.resY - 1;
        return true;
    }
    // 1 / (area of one film cell, in scene m^2): turns flux landing in a cell into the
    // cell's irradiance, which is what the backward lens camera measures (genLensRay's
    // cos^4*A/Z^2 weight is an irradiance per unit radiance). The sensor's mm are 1e-3 of a
    // scene metre, the mapping genLensRay places the glass in the world with.
    double lensCellNorm() const { const double a = lensPixelMM() * 1e-3; return 1.0 / (a * a); }

    // Where light from scene point `p` enters the lens: a point Y uniform by area on the
    // front element's clear cap, the straight leg p -> Y, and the leg in lens-local mm,
    // started just in front of the glass. Nothing here depends on the wavelength, so a hero
    // bundle shares one entry. False if Y is not the FIRST point of the element the leg meets
    // (a convex element's far side is hidden), or the leg does not travel toward the sensor.
    //
    // THE REBASED START. Every point of the clear cap lies within capRadius(0) of the
    // element's vertex, so within 2 * capRadius(0) of Y along any line, and an origin that far
    // before Y (plus a millimetre) is in front of all of them. Starting the sphere solve there
    // instead of at p keeps its operands a few centimetres long: from a scene point a
    // kilometre out (10^6 mm) the solve's b^2 - c would cancel all but a few of its digits,
    // and the entry point would wander off Y by more than the hidden-side test allows.
    struct LensEntry {
        Vec3 Y, wdir;    // world entry point; unit direction p -> Y
        double dist;     // |Y - p| (world)
        Vec3 o, d;       // the leg in lens-local mm: start (just in front of the glass), unit direction
        double cosY;     // cosine between the leg and the element's normal at Y
    };
    bool lensEntry(const Vec3& p, double u1, double u2, LensEntry& e) const {
        const LensSystem& L = *lens;
        Vec3 nY;
        const Vec3 yl = L.sampleSurface(0, u1, u2, nY);
        const Vec3 pl = lensLocalPoint(p);
        Vec3 dl = yl - pl;
        const double dmm = length(dl);
        if (!(dmm > 1e-9)) return false;
        dl = dl / dmm;
        if (!(dl.z < 0.0)) return false;                     // light must travel toward the sensor
        const double back = 2.0 * L.capRadius(0) + 1.0;
        const Vec3 o = (dmm > back) ? yl - dl * back : pl;
        Vec3 hit, n;
        if (!L.hitSurface(0, o, dl, hit, n)) return false;
        const Vec3 gap = hit - yl;
        if (dot(gap, gap) > 1e-12) return false;             // 1 nm off: Y is on the hidden side
        e.Y = lensWorldPoint(yl);
        const Vec3 toY = e.Y - p;
        e.dist = length(toY);
        if (!(e.dist > 1e-12)) return false;
        e.wdir = toY / e.dist;
        e.o = o; e.d = dl;
        e.cosY = std::fabs(dot(dl, nY));
        return true;
    }
    // Trace an entry through the glass at `lambda` onto the sensor: its film cell, and
    //   wgt = cosY * A_front / (dist^2 * A_cell),
    // which times beta * f * cosSurf (or albedo * phase) is the irradiance the connection adds
    // to the cell. The flux through dA at Y is beta * f * cosSurf * cosY * dA / dist^2, and
    // sampling Y with density 1/A_front makes that the estimator; glass conserves radiance, so
    // the flux that leaves the rear element is the flux that entered (Fresnel loss is not
    // modelled, as in genLensRay). False if a stop or element clips the ray, it is totally
    // internally reflected, or it lands off the film.
    bool lensEntryImage(const LensEntry& e, double lambda, int& px, int& py, double& wgt) const {
        const LensSystem& L = *lens;
        Vec3 onFilm, dFilm;
        if (!L.traceToFilm(e.o, e.d, lambda, onFilm, dFilm)) return false;
        if (!lensFilmToPixel(onFilm, px, py)) return false;
        const double aCell = lensPixelMM() * lensPixelMM();
        wgt = e.cosY * (L.frontArea() / aCell) / (e.dist * e.dist);   // the mm^2 of both areas cancel
        return std::isfinite(wgt) && wgt > 0.0;
    }
    // Both stages at once: one next-event connection from `p` through the lens.
    bool lensConnect(const Vec3& p, double lambda, double u1, double u2,
                     Vec3& Y, Vec3& wdir, double& dist, int& px, int& py, double& wgt) const {
        LensEntry e;
        if (!lensEntry(p, u1, u2, e) || !lensEntryImage(e, lambda, px, py, wgt)) return false;
        Y = e.Y; wdir = e.wdir; dist = e.dist;
        return true;
    }

    // Mode C through the physical lens, for a photon flight `ray` that meets the scene after
    // hitDist. Returns
    //   LENS_MISSED   the flight does not cross the front element's clear cap (it flies on:
    //                 the camera body is not modelled, as with the thin lens);
    //   LENS_CAUGHT   it enters and reaches the sensor (px, py set);
    //   LENS_ABSORBED it enters and the glass or the stop clips it, it is totally internally
    //                 reflected, or it lands off the film: the lens absorbs it.
    // Entering is decided by geometry alone, so it is the same for every wavelength of a hero
    // bundle; only where each one lands, or whether the glass clips it, depends on the
    // wavelength. The start is rebased as in lensEntry, behind a closest-approach test that
    // rejects nearly every flight without touching the glass.
    enum { LENS_MISSED = 0, LENS_CAUGHT = 1, LENS_ABSORBED = 2 };
    int lensCatch(const Ray& ray, double hitDist, double lambda, int& px, int& py) const {
        const LensSystem& L = *lens;
        const Vec3 ol = lensLocalPoint(ray.o);
        const Vec3 dl = normalize(lensLocalDir(ray.d));
        if (!(dl.z < 0.0)) return LENS_MISSED;
        const Vec3 v0(0.0, 0.0, L.zpos[0]);
        const double rB = L.capRadius(0);
        const double tc = dot(v0 - ol, dl);                  // closest approach to the vertex
        const Vec3 c = ol + dl * tc - v0;
        if (dot(c, c) > (rB + 1e-6) * (rB + 1e-6)) return LENS_MISSED;
        if (tc < -rB) return LENS_MISSED;                    // the element is behind the start
        const double back = rB + 1.0;
        const Vec3 o = (tc > back) ? ol + dl * (tc - back) : ol;
        Vec3 entry, n;
        if (!L.hitSurface(0, o, dl, entry, n)) return LENS_MISSED;
        if (!(length(entry - ol) * 1e-3 < hitDist)) return LENS_MISSED;   // the scene was in the way
        Vec3 onFilm, dFilm;
        if (!L.traceToFilm(o, dl, lambda, onFilm, dFilm)) return LENS_ABSORBED;
        return lensFilmToPixel(onFilm, px, py) ? LENS_CAUGHT : LENS_ABSORBED;
    }

    // BDPT's density for the camera subpath generating the ray that leaves front-element
    // point `y` (world) in direction `dir` (unit, into the scene), DIVIDED by the density
    // 1/A_front with which lensEntry picks y. The light-image (t=1) strategy picks y
    // uniformly on the front element; the camera subpath gets its exit point from the film
    // and rear-pupil sample instead, so the two strategies give y different densities. The
    // MIS ratio between them is p_light(next) * (1/A_front) / p_cam(y, next), and writing
    // the camera's pdfDir as p_cam(y, dir) * A_front makes the ordinary per-vertex ratio
    // (pdfRev / pdfFwd at the next vertex) come out as exactly that.
    //
    // p_cam(y, dir): genLensRay draws the film point uniformly over the film (BDPT counts the
    // whole film, see cameraPdfDir) and a point on the rear element. On a planar rear, a point
    // uniform on its disk: per film area and film solid angle that is
    // Z^2 / (A_film * A_rear * cos^3 th), th the film ray's angle to the axis. On a curved
    // rear, a point uniform on its clear cap: r^2 / (A_film * A_cap * cosR), r the film point's
    // distance to the cap point and cosR the ray's cosine at the cap. Glass conserves etendue,
    // dA cos dw, so per front area and exit solid angle each gains a factor cosFront / cos th.
    // All lengths in mm, so the result is per steradian. 0 if no camera sample could have
    // made this ray: it is clipped, lands off the film, or (planar rear) crosses the rear
    // plane outside the sampled disk.
    double lensPdfDir(const Vec3& y, const Vec3& dir, double lambda) const {
        const LensSystem& L = *lens;
        const Vec3 yl = lensLocalPoint(y), wl = lensLocalDir(dir);
        if (!(wl.z > 0.0)) return 0.0;                       // a camera ray leaves toward the scene
        const Vec3 o = yl + wl * 1.0;                        // 1 mm out, aimed back at y
        const Vec3 d = wl * -1.0;
        Vec3 onFilm, dFilm, rear;
        if (!L.traceToFilm(o, d, lambda, onFilm, dFilm, &rear)) return 0.0;
        int px, py;
        if (!lensFilmToPixel(onFilm, px, py)) return 0.0;
        const double cosT = -dFilm.z;                        // film ray's cosine to the axis
        if (!(cosT > 1e-9)) return 0.0;
        // Surface 0's normal at the exit point, for the etendue's front cosine.
        const Vec3 n0 = (L.surf[0].radius == 0.0)
                            ? Vec3(0, 0, 1)
                            : normalize(yl - Vec3(0, 0, L.zpos[0] + L.surf[0].radius));
        const double cosF = std::fabs(dot(wl, n0));
        const double halfW = 0.5 * L.filmW_mm;
        const double aFilm = (2.0 * halfW) * (2.0 * halfW * (double)film.resY / (double)film.resX);
        const int jr = (int)L.surf.size() - 1;
        double pdf;
        if (L.surf[jr].radius != 0.0) {
            // The camera reaches `rear` only if it is the first point of the cap seen from
            // the film along this ray.
            const Vec3 up = dFilm * -1.0;
            Vec3 hit, n;
            if (!L.hitSurface(jr, onFilm, up, hit, n)) return 0.0;
            const Vec3 gap = hit - rear;
            if (dot(gap, gap) > 1e-12) return 0.0;
            const double r = length(rear - onFilm);
            const Vec3 nR = normalize(rear - Vec3(0, 0, L.zpos[jr] + L.surf[jr].radius));
            const double cosR = std::fabs(dot(up, nR));
            if (!(cosR > 1e-12)) return 0.0;
            pdf = r * r * cosF * L.frontArea() / (aFilm * L.surfaceArea(jr) * cosR * cosT);
        } else {
            const double Z = L.rearZ() - L.filmZ;
            if (!(Z > 1e-9)) return 0.0;
            // Where the film ray, run back up, crosses the rear plane: inside the sampled disk?
            const double tR = Z / cosT;
            const Vec3 pr = onFilm + dFilm * (-tR);
            const double ra = L.rearAperture();
            // A sliver of tolerance: a ray the camera itself drew at the disk's rim must not
            // read back as outside it through rounding.
            if (pr.x * pr.x + pr.y * pr.y > ra * ra * (1.0 + 1e-9)) return 0.0;
            const double aRear = PI * ra * ra;
            const double c2 = cosT * cosT;
            pdf = Z * Z * cosF * L.frontArea() / (aFilm * aRear * c2 * c2);
        }
        return std::isfinite(pdf) ? pdf : 0.0;
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
