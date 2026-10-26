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
inline void traceGuide(const Scene& scene, const Renderer& rr, Ray ray, Vec3& alb, Vec3& nrm) {
    Vec3 tint{1, 1, 1};
    MediumStack stk;
    Pcg32 rng;   // interfaceStep's signature; the whitted step draws nothing from it
    for (int b = 0; b < kMaxDepth; ++b) {
        // `skipCamHidden` on the camera segment only, as the renderers do: a hide_camera
        // panel is invisible straight on and visible in a reflection.
        const Hit h = scene.closestHit(ray, 1e-6, nullptr, /*skipHair=*/false,
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
            return continueAlong(h, normalize(reflect(ray.d, h.n)), Lobe::Reflect, ray);
        };
        auto passStraight = [&]() { ray = Ray{h.p + ray.d * 1e-6, ray.d}; };
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
                ray = Ray{h.p + g * 1e-6, normalize(reflect(ray.d, g))};
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
                    // surface is asked (`dot(o, h.n) <= 0`): black there too.
                    if (!(dot(reflect(ray.d, h.n), h.n) > 0.0) ||
                        !mirror(rgbOf([&](double L) { return reflectSlot(scene, m, h, L); }))) {
                        nrm = nrm + h.n; return;
                    }
                    continue;
                }
                break;
            default:
                break;
        }
        // A surface with a texture of its own: its colour is the guide.
        Vec3 a = rgbOf([&](double L) {
            double r = diffuseReflectance(scene, m, h, L);
            if (m.type == MatType::DiffuseTransmit) r += transmitSlot(scene, m, h, L);
            return r;
        });
        alb = alb + mul(tint, a);
        nrm = nrm + h.n;
        return;
    }
}

// The guides for `cam` at W x H, in the film's row order (row 0 = the bottom of the image,
// as Camera::genRay counts). `side`^2 stratified rays per pixel. Returns an empty Images
// (W == 0) if a stop was requested part-way.
inline Images render(const Scene& scene, const Camera& cam, int W, int H, int side) {
    Images g;
    if (W <= 0 || H <= 0) return g;
    side = std::max(1, side);
    const size_t N = (size_t)W * (size_t)H;
    g.albedo.assign(N, Vec3{0, 0, 0});
    g.normal.assign(N, Vec3{0, 0, 0});
    g.clean = !cam.hasLens();
    Renderer rr;
    const double inv = 1.0 / ((double)side * side);
    const bool ok = ft::parallelFor((size_t)H, 1, [&](size_t row) {
        const int py = (int)row;
        for (int px = 0; px < W; ++px) {
            const size_t i = (size_t)py * W + px;
            Pcg32 rng;
            rng.seed(i, 0x9e3779b97f4a7c15ULL);   // fixed per pixel: the guides are deterministic
            Vec3 alb{0, 0, 0}, nrm{0, 0, 0};
            for (int sy = 0; sy < side; ++sy)
                for (int sx = 0; sx < side; ++sx) {
                    const double jx = (sx + rng.uniform()) / side, jy = (sy + rng.uniform()) / side;
                    Ray ray;
                    if (cam.hasLens()) {
                        double w = 0.0;
                        if (!cam.genLensRay(px, py, jx, jy, rng.uniform(), rng.uniform(),
                                            kGuideLambda, ray, w) || !(w > 0.0))
                            continue;            // vignetted: contributes black, as in the render
                    } else {
                        ray = cam.genRay(px, py, jx, jy);
                    }
                    traceGuide(scene, rr, ray, alb, nrm);
                }
            g.albedo[i] = alb * inv;
            g.normal[i] = nrm * inv;
        }
    });
    if (!ok) return Images{};
    g.W = W; g.H = H;
    seal(g);
    return g;
}

}  // namespace dguide
