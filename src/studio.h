// studio.h — a procedural photo-studio environment (`light env { kind studio }`, 0.368.0).
//
// The mesh quick-view (`ftrace model.glb`) used to light an object with a uniform grey
// environment and one small "softbox" sun. A uniform environment is invisible in a
// reflection, so a metal had nothing to show but that one highlight -- in the path tracer as
// much as in the preview. A product photographer lights metal with large SOFTBOXES, because
// what a metal looks like is what it reflects: this is that studio, baked into an
// equirectangular image and handed to EnvMap::buildFromRgb like the Preetham sky, so every
// renderer lights with it and every renderer shows it in reflections.
//
//   * a cyclorama: a dark floor rising to a mid-grey horizon and a lighter ceiling, so a
//     reflection always has an "up" and a "down";
//   * KEY  -- a large, slightly warm softbox high on the camera's left (the old quick-view
//     sun's direction, so a model keeps the modelling it had);
//   * FILL -- a broad, dim, slightly cool panel low on the camera's right, to open shadows;
//   * RIM  -- a strip high behind the model (out of the default frame), for edge highlights;
//   * TOP  -- a soft overhead panel.
//
// Directions follow EnvMap's convention: theta from +y (row 0 straight up), phi = atan2(z, x),
// u = phi/(2pi) + 0.5. The quick-view camera sits at +x/+y/+z of the model looking back at it,
// so "the camera's left" is -x/+z. Brightness is relative (the quick-view auto-exposes);
// `intensity` on the light block scales it like any env.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "linalg.h"

namespace studio {

// One rectangular softbox: a flat panel facing the origin, seen through a gnomonic (tangent-
// plane) projection so its outline is a true rectangle rather than a lat-long box. `hx`/`hy`
// are the tangents of its half-angles; `soft` is the fraction of the half-size over which the
// edge rolls off (a real diffuser's edge falloff, and it keeps a sharp mirror reflection from
// aliasing at the map's resolution).
struct Panel { Vec3 c; double hx, hy; Vec3 rgb; double soft; };

inline double smooth01(double t) {
    t = std::min(1.0, std::max(0.0, t));
    return t * t * (3.0 - 2.0 * t);
}

inline Vec3 panelRadiance(const Panel& p, const Vec3& d) {
    const double z = dot(d, p.c);
    if (z <= 1e-6) return Vec3{0, 0, 0};
    Vec3 up{0, 1, 0};
    Vec3 t = cross(up, p.c);
    double tl = std::sqrt(dot(t, t));
    t = (tl > 1e-9) ? t * (1.0 / tl) : Vec3{1, 0, 0};
    const Vec3 b = cross(p.c, t);
    const double x = std::fabs(dot(d, t) / z), y = std::fabs(dot(d, b) / z);
    if (x >= p.hx || y >= p.hy) return Vec3{0, 0, 0};
    const double ex = smooth01((p.hx - x) / (p.hx * p.soft));
    const double ey = smooth01((p.hy - y) / (p.hy * p.soft));
    return p.rgb * (ex * ey);
}

// Linear-RGB radiance of the studio in direction `d` (unit).
inline Vec3 radiance(const Vec3& d) {
    // The cyclorama.
    const double y = d.y;
    double g;
    if (y < 0.0) g = 0.05 + (0.12 - 0.05) * smooth01((y + 0.35) / 0.35);   // floor -> horizon
    else         g = 0.12 + (0.30 - 0.12) * smooth01(y);                    // horizon -> ceiling
    Vec3 L{g, g, g};
    static const Panel kPanels[] = {
        // KEY: the old quick-view sun direction, ~46 x 33 deg, slightly warm.
        {normalize(Vec3{-0.319, 0.785, 0.531}), 0.42, 0.30, Vec3{12.0, 11.6, 11.0}, 0.18},
        // FILL: camera-right and low, broad and dim, slightly cool.
        {normalize(Vec3{0.87, 0.39, 0.32}),     0.55, 0.38, Vec3{2.8, 2.95, 3.1},   0.30},
        // RIM: a strip high behind the model. High enough that the default quick-view frame never
        // sees it (its nearest edge is ~44 deg off the camera axis; the frame's half-diagonal is
        // ~25): seen directly it drags the tracer's auto-exposure and shows as a slab of light.
        {normalize(Vec3{-0.10, 0.62, -0.78}),   0.16, 0.32, Vec3{9.0, 9.0, 9.0},    0.15},
        // TOP: a soft overhead panel.
        {Vec3{0, 1, 0},                          0.35, 0.35, Vec3{3.5, 3.5, 3.5},    0.35},
    };
    for (const Panel& p : kPanels) L += panelRadiance(p, d);
    return L;
}

// The equirectangular bake: row 0 = straight up, EnvMap's u/phi convention.
inline std::vector<Vec3> generate(int w, int h) {
    std::vector<Vec3> img((size_t)w * h);
    for (int row = 0; row < h; ++row) {
        const double theta = (row + 0.5) / h * PI;
        const double st = std::sin(theta), ct = std::cos(theta);
        for (int col = 0; col < w; ++col) {
            const double phi = ((col + 0.5) / w - 0.5) * 2.0 * PI;
            img[(size_t)row * w + col] = radiance(Vec3{st * std::cos(phi), ct, st * std::sin(phi)});
        }
    }
    return img;
}

}  // namespace studio
