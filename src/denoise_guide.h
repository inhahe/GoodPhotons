// Guide images for the learned denoiser (`-oidn`): per pixel, the ALBEDO and the shading
// NORMAL of what the camera sees, which OIDN uses to tell an edge or a texture (present in
// the guides) from Monte-Carlo noise (present only in the render). They are what lets it
// clean a low-sample frame without smearing the floor grid, the marble veining or a
// silhouette.
//
// What a guide ray records follows OIDN's own advice (its README, "Albedos" and "Normals"):
// the first surface that is NOT a perfect specular. A camera ray is followed through clear
// glass, mirrors and filters to the surface behind them, because there is no texture of its
// own on a clear surface to preserve -- what the denoiser must keep sharp through a glass
// orb is the scene refracted in it. Along the way the ray takes each optical element's
// deterministic branch, exactly as mode W does (Renderer::interfaceStep with `whitted`:
// refract unless the Fresnel reflectance is the larger part, total internal reflection
// when there is no refraction), at one wavelength, kGuideLambda. A dispersive fringe is
// therefore not in the guides -- it is a property of the render, which OIDN keeps if it
// is resolved there.
//
// So the guides are DETERMINISTIC: the same camera gives the same images, whatever the
// render's sample count or seed. That is why the denoiser may be told they are clean
// (`cleanAux`, which makes it trust their edges completely), and why they hold still
// across the frames of a flyby -- normals are in WORLD space for the same reason, so a
// static surface keeps its guide while the camera moves. The one stochastic case is a
// physical multi-element lens, whose pupil is sampled; `clean` reports it.
//
// Participating media are ignored: a guide ray sees the surface behind the fog. The fog
// is smooth in the render and needs no edge from the guides.
//
// The pass is CPU-side and runs once per camera before its render starts (main.cpp
// runRender): kGuideSide^2 stratified rays per pixel, so the guides are antialiased the way
// the box-filtered film is, which OIDN requires ("all auxiliary images should use the same
// pixel reconstruction filter as the beauty image").
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>
#include "camera.h"
#include "color.h"
#include "medium_stack.h"
#include "parallel.h"
#include "render.h"
#include "rng.h"
#include "scene.h"

namespace dguide {

// The wavelength a guide ray's optics are evaluated at: the middle of the visible band.
constexpr double kGuideLambda = 550.0;
// Path vertices followed through specular surfaces before giving up (the pixel then keeps
// what it has, albedo 0). The deepest stacks in the gallery are ~8 interfaces.
constexpr int kMaxDepth = 32;
// A Glossy lobe narrower than this is treated as a mirror, and a Dielectric this smooth as
// clear glass: what they show is sharp in the render, so the guide carries that surface. At or
// above it they blur it, and the guide stops at them (their tint, their normal).
constexpr double kMirrorRoughness = 0.05;

struct Images {
    int W = 0, H = 0;
    std::vector<Vec3> albedo;   // linear sRGB in [0,1]
    std::vector<Vec3> normal;   // world-space shading normal, averaged over the pixel
    // What a temporal denoiser reprojects with (ROADMAP 11 stage C), from one ray through the
    // pixel's centre at the frame's instant -- a point, not an average across an edge:
    //   depth       the surface's distance along the camera's axis; 0 where the ray meets nothing
    //   position    its world position (0 where nothing)
    //   prevPosition  where that point sat a frame earlier (carried by its keys), for a
    //               temporal denoiser to tell its own history from another surface's
    //   motionBack  where it sits a frame earlier, in this image's pixels, minus where it sits
    //   motionFwd   ... a frame later. Pixels: x right, y up (the film's rows, bottom first); z 0.
    // A point is carried a frame out by its own motion keys and seen by the camera a frame before
    // or after (Camera::frameNeighbours); a pixel that meets nothing moves with the sky, as the
    // camera turns.
    std::vector<float> depth;
    std::vector<Vec3> position, prevPosition, motionBack, motionFwd;
    bool clean = true;          // deterministic (see the header comment)
    // What the denoiser is handed, made once by seal(): the two images as float RGB rows,
    // UPRIGHT (row 0 = the top of the image; the film is stored bottom-up), and an id unique
    // to this guide set, so the denoiser uploads them once rather than at every write.
    std::vector<float> albedoUp, normalUp;
    uint64_t id = 0;
};

// Fill `g`'s denoiser-side copies and id. Call once the images are final.
inline void seal(Images& g) {
    static std::atomic<uint64_t> nextId{1};
    const size_t N = (size_t)g.W * (size_t)g.H;
    auto pack = [&](const std::vector<Vec3>& v, std::vector<float>& out) {
        out.assign(N * 3, 0.0f);
        if (v.size() != N) return;
        for (int y = 0; y < g.H; ++y)
            for (int x = 0; x < g.W; ++x) {
                const Vec3& p = v[(size_t)(g.H - 1 - y) * g.W + x];
                float* q = &out[((size_t)y * g.W + x) * 3];
                q[0] = (float)p.x; q[1] = (float)p.y; q[2] = (float)p.z;
            }
    };
    pack(g.albedo, g.albedoUp);
    pack(g.normal, g.normalUp);
    g.id = nextId.fetch_add(1);
}

// Spectrum -> linear sRGB for a REFLECTANCE (a value in [0,1] per wavelength): K samples of
// the CIE 1931 observer, weighted so that a flat 1 maps to (1,1,1) exactly. Reflectance
// spectra are smooth, so 17 samples 20 nm apart are plenty for a guide.
struct SpecToRgb {
    static constexpr int K = 17;
    double lam[K];
    Vec3   w[K];
    SpecToRgb() {
        Vec3 white{0, 0, 0};
        for (int k = 0; k < K; ++k) {
            lam[k] = 390.0 + 20.0 * k;
            w[k] = xyzToLinearSrgb(Vec3{cieX(lam[k]), cieY(lam[k]), cieZ(lam[k])});
            white = white + w[k];
        }
        for (int k = 0; k < K; ++k) w[k] = Vec3{w[k].x / white.x, w[k].y / white.y, w[k].z / white.z};
    }
};
inline const SpecToRgb& specToRgb() { static const SpecToRgb t; return t; }

// The RGB of a reflectance or transmittance given as a function of wavelength, clamped to
// [0,1] per channel (an out-of-gamut spectrum lands slightly outside it).
template <class F>
inline Vec3 rgbOf(F&& f) {
    const SpecToRgb& t = specToRgb();
    Vec3 c{0, 0, 0};
    for (int k = 0; k < SpecToRgb::K; ++k) c = c + t.w[k] * f(t.lam[k]);
    return Vec3{std::clamp(c.x, 0.0, 1.0), std::clamp(c.y, 0.0, 1.0), std::clamp(c.z, 0.0, 1.0)};
}

inline Vec3 mul(const Vec3& a, const Vec3& b) { return Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }

// What one guide ray sees, added into `alb` / `nrm`: the albedo of the first non-specular
// surface times the tint of the specular chain that led to it (a mirror's colour, a gel's,
// coloured glass's absorption along the way), and that surface's shading normal. A ray that
// escapes, is absorbed, or runs out of depth adds nothing (albedo 0, normal 0, which is
// what OIDN expects of a background).
inline void traceGuide(const Scene& scene, const Renderer& rr, Ray ray, Vec3& alb, Vec3& nrm,
                       const MediumStack& stk0) {
    Vec3 tint{1, 1, 1};
    MediumStack stk = stk0;   // the camera's (Camera::enclosure)
    Pcg32 rng;   // interfaceStep's signature; the whitted step draws nothing from it
    for (int b = 0; b < kMaxDepth; ++b) {
        // `skipCamHidden` on the camera segment only, as the renderers do: a hide_camera
        // panel is invisible straight on and visible in a reflection.
        const Hit h = closestHitIn(scene, stk, ray, Scene::kTminEps, nullptr, /*skipHair=*/false,
                                   /*skipCamHidden=*/b == 0);
        if (!h.valid) return;
        // Coloured glass: Beer-Lambert over the leg just travelled inside it.
        if (const int top = stk.topMat(); top >= 0) {
            const Material& g = scene.mats[(size_t)top];
            if (g.absorb(kGuideLambda) > 0.0 || g.absorb(450.0) > 0.0 || g.absorb(650.0) > 0.0)
                tint = mul(tint, rgbOf([&](double L) { return std::exp(-g.absorb(L) * h.t); }));
        }
        const Material* mp = &scene.mats[(size_t)h.matId];
        if (mp->type == MatType::Mix || mp->type == MatType::Layered) {
            // The dominant child, as mode W resolves it (a Layered stack's body: the clear coat
            // above it changes the colour little and has no texture).
            const int c = (mp->type == MatType::Mix) ? mixResolveDominant(scene, *mp, h)
                                                      : mixDominantChild(*mp);
            if (c < 0) { nrm = nrm + h.n; return; }   // the absorbing remainder wins: black
            mp = &scene.mats[(size_t)c];
        }
        const Material& m = *mp;
        if (m.reflectsNothing) {                // a lamp seen directly: a flat white patch
            alb = alb + tint; nrm = nrm + h.n; return;
        }
        // Follows the reflection about n, as the render's walks do; false where they absorb it (a
        // reflection about a tilted or tipped n that leaves behind the surface): black there.
        auto mirror = [&](const Vec3& tintHere) {
            tint = mul(tint, tintHere);
            return continueAlong(h, normalize(reflect(ray.d, h.n)), Lobe::Reflect, ray, scene.hitEps(h));
        };
        auto passStraight = [&]() { ray = passThrough(h, ray.d, scene.hitEps(h)); };
        switch (m.type) {
            case MatType::Dielectric:
            case MatType::ThinFilm:
            case MatType::Multilayer: {
                // Frosted glass blurs what is behind it in the render (interfaceStep jitters the
                // lobe; its whitted step, used here, does not), so the guide stops at it, as at a
                // rough Glossy lobe: a clear surface's albedo, 1, and its own normal.
                if (m.type == MatType::Dielectric &&
                    materialRoughness(scene, m, h) >= kMirrorRoughness) {
                    alb = alb + tint; nrm = nrm + h.n; return;
                }
                const Renderer::InterfaceStep st =
                    rr.interfaceStep(scene, m, h, ray.d, kGuideLambda, rng, stk,
                                     Renderer::Transport::Radiance, /*whitted=*/true);
                if (st.absorbed) { nrm = nrm + h.n; return; }
                ray = st.ray;   // clear glass keeps the tint: OIDN wants a clear surface's albedo at 1
                continue;
            }
            case MatType::Mirror:
                if (!mirror(rgbOf([&](double L) { return reflectSlot(scene, m, h, L); }))) { nrm = nrm + h.n; return; }
                continue;
            case MatType::Grating: {            // the zero order: the reflected image
                // About the true normal, as gratingDiffract builds its orders: never behind it.
                tint = mul(tint, rgbOf([&](double L) { return reflectSlot(scene, m, h, L); }));
                const Vec3 g = orientedGeoN(h);
                ray = Ray{h.p + g * scene.hitEps(h), normalize(reflect(ray.d, g)), 0.0, scene.hitEps(h)};
                continue;
            }
            case MatType::HalfMirror: {
                // A beam splitter reflects with probability `reflect`: take the larger branch.
                if (reflectSlot(scene, m, h, kGuideLambda) >= 0.5) {
                    if (!mirror(Vec3{1, 1, 1})) { nrm = nrm + h.n; return; }
                } else {
                    passStraight();
                }
                continue;
            }
            case MatType::Filter:               // a gel: straight through, tinted
                tint = mul(tint, rgbOf([&](double L) { return transmitSlot(scene, m, h, L); }));
                passStraight();
                continue;
            case MatType::Glossy:
                if (materialRoughness(scene, m, h) < kMirrorRoughness) {
                    // The render's glossy walks end a lobe below n's plane before the true
                    // surface is asked (`dot(o, h.n) <= 0`): black there too. An iridescent
                    // lobe tints by its interference reflectance at the mirror direction.
                    const double cv = clamp01(-dot(ray.d, h.n));
                    if (!(dot(reflect(ray.d, h.n), h.n) > 0.0) ||
                        !mirror(rgbOf([&](double L) {
                            return reflectSlot(scene, m, h, L) * iridescenceR(scene, m, &h, L, cv); }))) {
                        nrm = nrm + h.n; return;
                    }
                    continue;
                }
                break;
            default:
                break;
        }
        // A surface with a texture of its own: its colour is the guide.
        // An iridescent glossy's colour is its interference reflectance, here at the view angle.
        const double cvA = clamp01(-dot(ray.d, h.n));
        Vec3 a = rgbOf([&](double L) {
            double r = diffuseReflectance(scene, m, h, L);
            if (m.type == MatType::DiffuseTransmit) r += transmitSlot(scene, m, h, L);
            if (m.iridescence) r *= iridescenceR(scene, m, &h, L, cvA);
            return r;
        });
        alb = alb + mul(tint, a);
        nrm = nrm + h.n;
        return;
    }
}

// Where point `p0`, at the frame's instant on a primitive of motion set `set` (-1: still), sits
// `frames` frames out, from the motion interval's end on that side (or the other end, when the
// interval stops at the instant): when `frames` is a whole number n of that end's time, its
// motion from the instant applied n times (its inverse, for a negative n) -- exact for a steady
// motion, turning included, as a rolling wheel's is -- and otherwise carried linearly past it. An estimate beyond the interval,
// as a frame's own keys are all it knows of its neighbours.
inline Vec3 framePoint(const Scene& scene, int set, const Vec3& p0, double frames) {
    if (set < 0 || !(scene.motionB > scene.motionA)) return p0;
    double end = frames < 0.0 ? scene.motionA : scene.motionB;
    if (end == 0.0) end = frames < 0.0 ? scene.motionB : scene.motionA;
    if (end == 0.0) return p0;
    const Affine D = scene.motionAt(set, scene.motionU(end));
    const double ratio = frames / end, n = std::round(ratio);
    if (std::fabs(n) >= 1.0 && std::fabs(n) <= 16.0 && std::fabs(ratio - n) < 1e-9) {
        // A negative count -- a frame back over an interval that ends at the instant on that side
        // -- applies the motion's inverse.
        const Affine M = n > 0.0 ? D : D.inverse();
        Vec3 p = p0;
        for (int i = 0; i < (int)std::fabs(n); ++i) p = M.apply(p);
        return p;
    }
    return p0 + (D.apply(p0) - p0) * ratio;
}

// Where point `p0`, at the frame's instant on deforming triangle `t`, sits `frames` frames out:
// the same barycentrics on the triangle there, its corners carried from the instant to the motion
// interval's end on that side (or the other, when the interval stops at the instant) and linearly
// past it -- a deformation has no motion to repeat -- and then, for a mesh riding a moving group,
// carried by the group's motion as its rigid children are (framePoint).
inline Vec3 deformedFramePoint(const Scene& scene, const Tri& t, const Vec3& p0, double frames) {
    if (!(scene.motionB > scene.motionA)) return p0;
    double end = frames < 0.0 ? scene.motionA : scene.motionB;
    if (end == 0.0) end = frames < 0.0 ? scene.motionB : scene.motionA;
    if (end == 0.0) return p0;
    const Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0, d = p0 - t.v0;
    const double d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2), den = d11 * d22 - d12 * d12;
    if (!(den > 0.0)) return p0;
    const double b1 = (d22 * dot(d, e1) - d12 * dot(d, e2)) / den;
    const double b2 = (d11 * dot(d, e2) - d12 * dot(d, e1)) / den;
    const double ue = scene.motionU(end);
    Vec3 c[3];
    scene.deformedCorners(t, ue, c, nullptr, /*carried*/false);   // in the group as at the instant
    const Vec3 pe = c[0] * (1.0 - b1 - b2) + c[1] * b1 + c[2] * b2;
    return framePoint(scene, t.motion, p0 + (pe - p0) * (frames / end), frames);
}

// The centre-ray half of the guides (Images::depth and on). False if a stop was requested.
inline bool motionAovs(const Scene& scene, const Camera& cam, int W, int H, Images& g) {
    const size_t N = (size_t)W * (size_t)H;
    g.depth.assign(N, 0.0f);
    g.position.assign(N, Vec3{0, 0, 0});
    g.prevPosition.assign(N, Vec3{0, 0, 0});
    g.motionBack.assign(N, Vec3{0, 0, 0});
    g.motionFwd.assign(N, Vec3{0, 0, 0});
    Camera prev = cam, next = cam;
    if (cam.frameNeighbours) {
        cam.posedAs((*cam.frameNeighbours)[0], prev);
        cam.posedAs((*cam.frameNeighbours)[1], next);
    }
    // At the instant as a time (the moving primitives' own path), so a hit names its motion set.
    const double u0 = scene.hasMotion() ? scene.motionU(0.0) : -1.0;
    return ft::parallelFor((size_t)H, 1, [&](size_t row) {
        const MotionTimeScope at(u0);
        const int py = (int)row;
        for (int px = 0; px < W; ++px) {
            const size_t i = (size_t)py * W + px;
            const Ray r = cam.genRay(px, py, 0.5, 0.5);
            const Hit h = scene.closestHit(r, scene.eps, nullptr, /*skipHair=*/false, /*skipCamHidden=*/true);
            Vec3 pB, pF;
            if (h.valid) {
                g.depth[i] = (float)dot(h.p - cam.eye, cam.w);
                g.position[i] = h.p;
                if (h.deformTri >= 0) {
                    const Tri& t = scene.tris[(size_t)h.deformTri];
                    pB = deformedFramePoint(scene, t, h.p, -1.0);
                    pF = deformedFramePoint(scene, t, h.p, 1.0);
                } else {
                    pB = framePoint(scene, h.motionSet, h.p, -1.0);
                    pF = framePoint(scene, h.motionSet, h.p, 1.0);
                }
                g.prevPosition[i] = pB;
            } else {
                // The sky: a direction, seen from wherever the camera stands.
                pB = prev.eye + r.d * 1e9;
                pF = next.eye + r.d * 1e9;
            }
            const double cx = px + 0.5, cy = py + 0.5;
            double fx, fy;
            if (prev.projectF(pB, fx, fy)) g.motionBack[i] = Vec3{fx - cx, fy - cy, 0.0};
            if (next.projectF(pF, fx, fy)) g.motionFwd[i] = Vec3{fx - cx, fy - cy, 0.0};
        }
    });
}

// The guides for `cam` at W x H, in the film's row order (row 0 = the bottom of the image,
// as Camera::genRay counts). `side`^2 stratified rays per pixel; `aovs`: the centre-ray half
// too (depth and on), else those stay empty. Returns an empty Images (W == 0) if a stop was
// requested part-way.
inline Images render(const Scene& scene, const Camera& cam, int W, int H, int side, bool aovs) {
    Images g;
    if (W <= 0 || H <= 0) return g;
    side = std::max(1, side);
    const size_t N = (size_t)W * (size_t)H;
    g.albedo.assign(N, Vec3{0, 0, 0});
    g.normal.assign(N, Vec3{0, 0, 0});
    g.clean = !cam.hasLens();
    Renderer rr;
    const double inv = 1.0 / ((double)side * side);
    // Under motion blur each guide ray sees the scene at its own time in the shutter, stratified
    // over the pixel's rays, so the guides smear as the image does: a sharp guide would hand the
    // denoiser edges the image does not have. Drawn after the subpixel jitter, so a scene the
    // camera sees still draws exactly what it drew before.
    const bool timed = cam.seesMotion(scene);
    const bool ok = ft::parallelFor((size_t)H, 1, [&](size_t row) {
        const int py = (int)row;
        PosedCamera posed(cam, timed);     // a moving camera, at each guide ray's time
        for (int px = 0; px < W; ++px) {
            const size_t i = (size_t)py * W + px;
            Pcg32 rng;
            rng.seed(i, 0x9e3779b97f4a7c15ULL);   // fixed per pixel: the guides are deterministic
            Vec3 alb{0, 0, 0}, nrm{0, 0, 0};
            for (int sy = 0; sy < side; ++sy)
                for (int sx = 0; sx < side; ++sx) {
                    const double jx = (sx + rng.uniform()) / side, jy = (sy + rng.uniform()) / side;
                    const double ts = timed ? ((sy * side + sx) + rng.uniform()) / ((double)side * side)
                                            : 0.0;
                    const MotionTimeScope at(timed ? scene.motionU(cam.shutterTime(ts)) : -1.0);
                    const Camera& c = posed.at(cam.shutterTime(ts));
                    Ray ray;
                    if (c.hasLens()) {
                        double w = 0.0;
                        if (!c.genLensRay(px, py, jx, jy, rng.uniform(), rng.uniform(),
                                          kGuideLambda, ray, w) || !(w > 0.0))
                            continue;            // vignetted: contributes black, as in the render
                    } else {
                        ray = c.genRay(px, py, jx, jy);
                    }
                    traceGuide(scene, rr, ray, alb, nrm, c.enclosure);
                }
            g.albedo[i] = alb * inv;
            g.normal[i] = nrm * inv;
        }
    });
    if (!ok) return Images{};
    if (aovs && !motionAovs(scene, cam, W, H, g)) return Images{};
    g.W = W; g.H = H;
    seal(g);
    return g;
}

}  // namespace dguide
