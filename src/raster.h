// raster.h — fast solid-shaded PREVIEW rasterizer (z-buffer, no light transport).
//
// This is the "quick taste" viewer: it turns the whole scene into triangles once
// (analytic spheres tessellated, isosurfaces marched to a mesh, instanced meshes
// baked to world space) and rasterizes each authored camera with a plain z-buffer
// and simple diffuse+headlight shading. There is NO transparency, refraction,
// reflection, shadows, caustics or global illumination — a dielectric shows as a
// solid ghost, a mirror as a flat tint. The point is to see the *composition* and
// (for a camera_curve) the *flyby motion* in a fraction of a second per frame,
// exactly the way the isosurface mesher lets you eyeball an implicit.
//
// It reuses the real Camera projection (Camera::project semantics reimplemented for
// triangle clipping), so the pinhole's off-axis elongation and the fisheye/panoramic
// lenses are reproduced faithfully — a sphere near the frame edge stretches just as
// it will in the physical render.
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include "scene.h"
#include "camera.h"
#include "color.h"
#include "isomesh.h"

namespace raster {

// One preview triangle: world-space positions + per-vertex world normals + a solid
// base colour (linear sRGB albedo) and an "emissive" flag (light geometry glows).
struct PTri {
    Vec3 p0, p1, p2;
    Vec3 n0, n1, n2;
    Vec3 color;
    bool emissive = false;
};

// Integrate a reflectance/emission spectrum against the CIE curves under an
// equal-energy illuminant and convert to (unclamped) linear sRGB. For a reflectance
// this yields the perceived surface colour; for an emission SPD, its chromaticity.
inline Vec3 spectrumToLinearRgb(const Spectrum& s) {
    Vec3 xyz{0, 0, 0};
    double wsum = 0.0;
    for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 5.0) {
        double v = s(lam);
        xyz = xyz + Vec3(cieX(lam), cieY(lam), cieZ(lam)) * v;
        wsum += cieY(lam);
    }
    if (wsum > 0.0) xyz = xyz / wsum;
    Vec3 rgb = xyzToLinearSrgb(xyz);
    rgb.x = std::max(0.0, rgb.x);
    rgb.y = std::max(0.0, rgb.y);
    rgb.z = std::max(0.0, rgb.z);
    return rgb;
}

// Solid preview colour for a material. Diffuse/glossy/fluorescent/etc. use their
// reflectance colour; specular materials (mirror/glass/thin-film) get a light tint
// so they read as a solid object instead of vanishing to black.
inline Vec3 materialColor(const Material& m, bool& emissive) {
    emissive = m.isLight;
    if (m.isLight) {
        Vec3 c = spectrumToLinearRgb(m.emit);
        double mx = std::max({c.x, c.y, c.z, 1e-6});
        return c * (1.0 / mx);   // normalise to a bright, correctly-tinted glow
    }
    Vec3 albedo = spectrumToLinearRgb(m.reflect);
    double lum = 0.2126 * albedo.x + 0.7152 * albedo.y + 0.0722 * albedo.z;
    if (isSpecularType(m.type)) {
        if (m.type == MatType::Mirror) {
            // A mirror: bright neutral tint (by its reflect colour) so it looks metallic.
            Vec3 t = (lum > 1e-3) ? albedo * (0.85 / std::max(lum, 1e-3)) : Vec3{0.85, 0.86, 0.9};
            return Vec3{std::min(t.x, 1.0), std::min(t.y, 1.0), std::min(t.z, 1.0)};
        }
        // Dielectric / thin-film / glossy / grating: pale translucent-looking ghost.
        if (lum < 0.04) return Vec3{0.70, 0.76, 0.85};
        return albedo * (0.7 / std::max(lum, 1e-3));
    }
    return albedo;
}

// One positional/spot source distilled from a scene emitter for preview shading.
struct PLight {
    Vec3   pos{0, 0, 0};        // world position of the source
    Vec3   dir{0, 0, 1};        // spot axis (source -> cone centre); unit
    bool   spot = false;        // apply cone falloff via spotFalloff()
    double cosInner = 1.0, cosOuter = 1.0;  // spot penumbra cosines
    double weight = 1.0;        // power-normalised key weight (Σ weights = 1)
    double falloff2 = 0.0;      // squared reference distance for 1/(1+d²/r²) (0 = none)
};

// The scene's lights distilled for shading: every positional/spot emitter shades
// from its own real direction, plus flat ambient + a subtle camera-headlight fill.
struct PreviewLight {
    std::vector<PLight> lights;  // one entry per positional/spot emitter
    double ambient  = 0.12;      // flat fill so nothing is pure black (kept low for contrast)
    double keyScale = 1.15;      // overall multiplier on the summed weighted N·L
    double fill     = 0.08;      // subtle headlight so back faces aren't crushed to black
};

inline PreviewLight deriveLight(const Scene& sc) {
    PreviewLight L;
    const double refR = sc.sceneRadius > 0 ? sc.sceneRadius * 0.6 : 0.0;
    const double fall2 = refR * refR;
    bool anyEnv = false;
    double totalPow = 0.0;

    for (const auto& e : sc.emitters) {
        if (e.shape == EmitterShape::Env) { anyEnv = true; continue; }
        PLight p;
        switch (e.shape) {
            case EmitterShape::Quad:     p.pos = e.origin + (e.u + e.v) * 0.5; break;
            case EmitterShape::Cylinder: p.pos = e.origin + e.v * 0.5;         break;  // tube centre
            default:                     p.pos = e.origin;                     break;  // sphere / spot
        }
        if (e.shape == EmitterShape::Spot) {
            p.spot = true;
            p.dir = normalize(e.beamDir);
            p.cosInner = e.spotCosInner;
            p.cosOuter = e.spotCosOuter;
        }
        p.weight   = std::max(e.power, 0.0);
        p.falloff2 = fall2;
        totalPow  += p.weight;
        L.lights.push_back(p);
    }

    // Normalise weights so total key intensity is stable regardless of light count.
    if (totalPow > 0.0)
        for (auto& p : L.lights) p.weight /= totalPow;
    else
        for (auto& p : L.lights) p.weight = 1.0 / (double)L.lights.size();

    if (L.lights.empty() && anyEnv) {
        // Env-only: lean on the headlight for shape, higher ambient so the far side
        // doesn't read flat-black.
        L.ambient = 0.30; L.keyScale = 0.0; L.fill = 0.75;
    } else if (anyEnv) {
        // Positional/spot keys PLUS an environment fill: moderate ambient, softer key.
        L.ambient = 0.24; L.keyScale = 0.95; L.fill = 0.12;
    } else {
        // Lone bulb / multiple bulbs, no env (the gallery case): low ambient +
        // inverse-square-ish falloff so surfaces shade from each source outward.
        L.ambient = 0.10; L.keyScale = 1.25; L.fill = 0.06;
    }
    return L;
}

// ---- Scene -> world-space preview triangles (done once, reused for every frame) --
inline std::vector<PTri> tessellate(const Scene& sc, int isoRes) {
    std::vector<PTri> out;
    // Precompute one solid colour per material.
    std::vector<Vec3> matCol(sc.mats.size());
    std::vector<char>  matEmit(sc.mats.size(), 0);
    for (size_t i = 0; i < sc.mats.size(); ++i) {
        bool em = false;
        matCol[i] = materialColor(sc.mats[i], em);
        matEmit[i] = em ? 1 : 0;
    }
    auto colOf = [&](int matId) -> Vec3 {
        return (matId >= 0 && matId < (int)matCol.size()) ? matCol[matId] : Vec3{0.6, 0.6, 0.6};
    };
    auto emOf = [&](int matId) -> bool {
        return (matId >= 0 && matId < (int)matEmit.size()) && matEmit[matId];
    };

    // (1) World triangles.
    out.reserve(sc.tris.size() + 4096);
    for (const auto& t : sc.tris) {
        PTri p;
        p.p0 = t.v0; p.p1 = t.v1; p.p2 = t.v2;
        p.n0 = t.n0; p.n1 = t.n1; p.n2 = t.n2;
        p.color = colOf(t.matId); p.emissive = emOf(t.matId);
        out.push_back(p);
    }

    // (2) Analytic spheres -> UV sphere mesh with radial (smooth) normals.
    const int SU = 28, SV = 18;
    for (const auto& s : sc.spheres) {
        auto sp = [&](int iu, int iv) -> Vec3 {
            double phi   = 2.0 * PI * (double)iu / SU;
            double theta = PI * (double)iv / SV;
            return Vec3{std::sin(theta) * std::cos(phi),
                        std::cos(theta),
                        std::sin(theta) * std::sin(phi)};
        };
        Vec3 col = colOf(s.matId); bool em = emOf(s.matId);
        for (int iv = 0; iv < SV; ++iv)
            for (int iu = 0; iu < SU; ++iu) {
                Vec3 d00 = sp(iu, iv),   d10 = sp(iu + 1, iv);
                Vec3 d01 = sp(iu, iv+1), d11 = sp(iu + 1, iv + 1);
                Vec3 v00 = s.c + d00 * s.r, v10 = s.c + d10 * s.r;
                Vec3 v01 = s.c + d01 * s.r, v11 = s.c + d11 * s.r;
                PTri a; a.p0 = v00; a.p1 = v01; a.p2 = v11; a.n0 = d00; a.n1 = d01; a.n2 = d11; a.color = col; a.emissive = em;
                PTri b; b.p0 = v00; b.p1 = v11; b.p2 = v10; b.n0 = d00; b.n1 = d11; b.n2 = d10; b.color = col; b.emissive = em;
                out.push_back(a); out.push_back(b);
            }
    }

    // (3) Isosurfaces / metaballs / CSG -> marching-tetrahedra mesh.
    if (isoRes > 0) {
        isomesh::Options opt; opt.res = isoRes; opt.adaptive = false; opt.refineIters = 3;
        for (const auto& im : sc.implicits) {
            isomesh::Mesh m = isomesh::marchImplicit(im, opt);
            Vec3 col = colOf(im.matId); bool em = emOf(im.matId);
            for (size_t f = 0; f + 2 < m.tri.size(); f += 3) {
                int i0 = m.tri[f], i1 = m.tri[f + 1], i2 = m.tri[f + 2];
                PTri p;
                p.p0 = m.pos[i0]; p.p1 = m.pos[i1]; p.p2 = m.pos[i2];
                p.n0 = m.nrm[i0]; p.n1 = m.nrm[i1]; p.n2 = m.nrm[i2];
                p.color = col; p.emissive = em;
                out.push_back(p);
            }
        }
    }

    // (4) Instanced mesh assets (BLAS) baked into world space.
    for (const auto& inst : sc.instances) {
        if (inst.blasId < 0 || inst.blasId >= (int)sc.blasList.size()) continue;
        const Blas& bl = sc.blasList[inst.blasId];
        for (const auto& t : bl.tris) {
            int matId = (inst.matOverride >= 0) ? inst.matOverride : t.matId;
            PTri p;
            p.p0 = inst.toWorld.apply(t.v0);
            p.p1 = inst.toWorld.apply(t.v1);
            p.p2 = inst.toWorld.apply(t.v2);
            p.n0 = normalize(inst.toWorld.applyNormal(t.n0));
            p.n1 = normalize(inst.toWorld.applyNormal(t.n1));
            p.n2 = normalize(inst.toWorld.applyNormal(t.n2));
            p.color = colOf(matId); p.emissive = emOf(matId);
            out.push_back(p);
        }
    }
    return out;
}

// A vertex after transform to camera space, carrying the attributes we interpolate.
struct VtxCS {
    double x, y, z;   // camera-space coords (x=right, y=up, z=forward)
    Vec3   wpos;      // world position (for per-pixel light direction)
    Vec3   wn;        // world normal
};

// A vertex projected to the raster, with 1/depth for perspective-correct interp.
struct VtxScreen {
    double sx, sy;    // pixel coords (sx in [0,W], sy in [0,H]; sy=0 is image top)
    double invd;      // 1/depth used as the z-buffer key and interp weight
    Vec3   wpos, wn;
};

// Rasterize a single (already screen-projected) triangle into the frame + z-buffer.
// Attributes are interpolated perspective-correctly via invd. Shading is deferred to
// the caller-supplied `shade` lambda so both projection paths share the fill loop.
template <class ShadeFn>
inline void fillTriangle(const VtxScreen& A, const VtxScreen& B, const VtxScreen& C,
                         int W, int H, int y0, int y1,
                         std::vector<float>& zbuf, std::vector<Vec3>& accum,
                         std::vector<uint8_t>& emisMask, bool triEmis,
                         ShadeFn&& shade) {
    double minx = std::floor(std::min({A.sx, B.sx, C.sx}));
    double maxx = std::ceil (std::max({A.sx, B.sx, C.sx}));
    double miny = std::floor(std::min({A.sy, B.sy, C.sy}));
    double maxy = std::ceil (std::max({A.sy, B.sy, C.sy}));
    int xlo = std::max(0, (int)minx), xhi = std::min(W - 1, (int)maxx);
    int ylo = std::max(y0, (int)miny), yhi = std::min(y1 - 1, (int)maxy);
    if (xlo > xhi || ylo > yhi) return;
    double area = (B.sx - A.sx) * (C.sy - A.sy) - (B.sy - A.sy) * (C.sx - A.sx);
    if (std::fabs(area) < 1e-9) return;
    double inv = 1.0 / area;
    for (int y = ylo; y <= yhi; ++y) {
        double py = y + 0.5;
        for (int x = xlo; x <= xhi; ++x) {
            double px = x + 0.5;
            double w0 = ((B.sx - px) * (C.sy - py) - (B.sy - py) * (C.sx - px)) * inv;
            double w1 = ((C.sx - px) * (A.sy - py) - (C.sy - py) * (A.sx - px)) * inv;
            double w2 = 1.0 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            double invd = w0 * A.invd + w1 * B.invd + w2 * C.invd;   // = 1/depth
            size_t idx = (size_t)y * W + x;
            if (invd <= zbuf[idx]) continue;   // farther than (or equal to) stored
            zbuf[idx] = (float)invd;
            emisMask[idx] = triEmis ? 1 : 0;   // emitters excluded from the auto-exposure anchor
            // Perspective-correct attribute recovery.
            double d = 1.0 / std::max(invd, 1e-12);
            Vec3 wpos = (A.wpos * (w0 * A.invd) + B.wpos * (w1 * B.invd) + C.wpos * (w2 * C.invd)) * d;
            Vec3 wn   = (A.wn   * (w0 * A.invd) + B.wn   * (w1 * B.invd) + C.wn   * (w2 * C.invd)) * d;
            accum[idx] = shade(wpos, wn);
        }
    }
}

// Project a camera-space vertex (x=right, y=up, z=fwd) to the raster. For the
// rectilinear pinhole this is the exact inverse of Camera::genRay; for a fisheye/
// panoramic lens it applies the same angular projRadius() map the real camera uses,
// so off-axis stretch matches. sy=0 is image top (+y/up), matching filmToRgb8's flip.
inline VtxScreen projectVtx(const Camera& cam, const VtxCS& v, int W, int H) {
    VtxScreen s;
    s.wpos = v.wpos; s.wn = v.wn;
    double ndcx, ndcy, depth;
    if (cam.projection == CAM_RECTILINEAR) {
        ndcx = (v.x / v.z) / cam.tanHalfX;
        ndcy = (v.y / v.z) / cam.tanHalfY;
        depth = v.z;                          // camera-forward distance
    } else {
        double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        double costh = (len > 1e-12) ? v.z / len : 1.0;
        costh = std::clamp(costh, -1.0, 1.0);
        double th = std::acos(costh);
        double rho = projRadius(cam.projection, th) / std::max(cam.rEdge, 1e-12);
        double rhoDir = std::sqrt(v.x * v.x + v.y * v.y);
        if (rhoDir < 1e-12) { ndcx = 0.0; ndcy = 0.0; }
        else { ndcx = rho * v.x / rhoDir; ndcy = rho * v.y / rhoDir; }
        depth = len;
    }
    s.sx = (ndcx * 0.5 + 0.5) * W;
    s.sy = (0.5 - 0.5 * ndcy) * H;           // +y (up) -> top of image
    s.invd = 1.0 / std::max(depth, 1e-9);
    return s;
}

// Render one camera to an 8-bit RGB image (row 0 = image top), multithreaded by
// horizontal bands (each band owns its slice of the z-buffer, no locking).
//
// Exposure model mirrors the real renderer's `filmToRgb8` so the preview brightness
// tracks the final render instead of drifting off:
//   * `expComp` (the `exposure` arg) is the photographic *compensation* the caller
//     folded together — iso*shutter*exposure-comp, plus the absolute aperture 1/N²
//     term when applicable — with 1.0 = neutral.
//   * When `autoExpose` is true (the default, matching a non-absolute scene) the raw
//     shaded image is anchored by a p99 auto-exposure: the 99th-percentile luminance
//     over z-buffer-hit pixels maps to ~0.9, exactly like `filmToRgb8`. This is why
//     aperture is (correctly) invisible here — auto-exposure divides it back out —
//     while ISO/shutter/exposure still give exact photographic stops via `expComp`.
//   * When `autoExpose` is false (absolute EV) the p99 anchor is bypassed and the raw
//     colour is scaled by `expComp` directly, so aperture/power differences survive.
//   * `lockAnchor` (optional) shares one auto-exposure anchor across a camera_path's
//     frames: >0 reuses the stored anchor (no flicker on a dolly), ==0 writes the
//     freshly-computed one back for later frames, null => per-frame auto-exposure.
inline std::vector<uint8_t> renderFrame(const std::vector<PTri>& tris, const Camera& cam,
                                        int W, int H, const PreviewLight& light,
                                        int nThreads, double exposure = 1.0,
                                        bool autoExpose = true, double* lockAnchor = nullptr) {
    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    const double EMIS_BOOST = 4.0;    // emitters read as bright light sources (clip to white)
    const Vec3 bg{0.06, 0.07, 0.09};                    // background tint (unlit, unexposed)
    std::vector<float>  zbuf((size_t)W * H, 0.0f);      // z-buffer key = 1/depth (bigger=closer)
    std::vector<Vec3>   accum((size_t)W * H, bg);       // raw linear shade (pre-exposure)
    std::vector<uint8_t> emisMask((size_t)W * H, 0);    // 1 where an emitter was drawn

    const double zn = 1e-3;   // near plane (camera-forward) for rectilinear clipping
    const bool rect = (cam.projection == CAM_RECTILINEAR);

    // Each band owns rows [y0,y1); it walks every triangle but only fills pixels in
    // its own row range, so bands never touch the same z-buffer entry (no locking).
    auto band = [&](int y0, int y1) {
        for (const auto& t : tris) {
            Vec3 col = t.color; bool emis = t.emissive;
            auto shade = [&](const Vec3& wp, const Vec3& wn) -> Vec3 {
                if (emis) return col * EMIS_BOOST;   // raw emitter radiance (exposed later)
                Vec3 N = normalize(wn);
                Vec3 V = normalize(cam.eye - wp);           // toward camera
                if (dot(N, V) < 0.0) N = -N;                 // two-sided
                // Sum diffuse contributions from every scene light, each from its own
                // real direction (with distance falloff and spot cone shaping).
                double lit = 0.0;
                for (const auto& lp : light.lights) {
                    Vec3 d = lp.pos - wp;
                    double dist2 = dot(d, d);
                    Vec3 Ld = (dist2 > 1e-12) ? d / std::sqrt(dist2) : V;
                    double ndl = std::max(0.0, dot(N, Ld));
                    if (ndl <= 0.0) continue;
                    double atten = 1.0;
                    if (lp.falloff2 > 0.0) atten = lp.falloff2 / (lp.falloff2 + dist2);
                    double cone = 1.0;
                    if (lp.spot) cone = spotFalloff(dot(lp.dir, -Ld), lp.cosInner, lp.cosOuter);
                    lit += lp.weight * ndl * atten * cone;
                }
                double head = std::max(0.0, dot(N, V));      // headlight fill
                double k = light.ambient + light.keyScale * lit + light.fill * head;
                // Raw linear shade; the p99 auto-exposure + gamma below (mirroring the
                // real renderer's filmToRgb8) handle the tone map, so no local S-curve.
                return col * k;
            };
            auto toCS = [&](const Vec3& P, const Vec3& N) -> VtxCS {
                Vec3 d = P - cam.eye; VtxCS c;
                c.x = dot(d, cam.u); c.y = dot(d, cam.v); c.z = dot(d, cam.w);
                c.wpos = P; c.wn = N; return c;
            };
            VtxCS cs[3] = { toCS(t.p0, t.n0), toCS(t.p1, t.n1), toCS(t.p2, t.n2) };
            if (rect) {
                VtxCS poly[8]; int np = 0;
                auto emit = [&](const VtxCS& a){ if (np < 8) poly[np++] = a; };
                auto lerpV = [&](const VtxCS& a, const VtxCS& b, double s) -> VtxCS {
                    VtxCS r; r.x=a.x+(b.x-a.x)*s; r.y=a.y+(b.y-a.y)*s; r.z=a.z+(b.z-a.z)*s;
                    r.wpos=a.wpos+(b.wpos-a.wpos)*s; r.wn=a.wn+(b.wn-a.wn)*s; return r;
                };
                for (int i = 0; i < 3; ++i) {
                    const VtxCS& A = cs[i]; const VtxCS& B = cs[(i+1)%3];
                    bool inA = A.z > zn, inB = B.z > zn;
                    if (inA) emit(A);
                    if (inA != inB) { double s = (zn - A.z)/(B.z - A.z); emit(lerpV(A,B,s)); }
                }
                if (np < 3) continue;
                VtxScreen sc0 = projectVtx(cam, poly[0], W, H);
                for (int i = 1; i + 1 < np; ++i) {
                    VtxScreen sc1 = projectVtx(cam, poly[i], W, H);
                    VtxScreen sc2 = projectVtx(cam, poly[i+1], W, H);
                    fillTriangle(sc0, sc1, sc2, W, H, y0, y1, zbuf, accum, emisMask, emis, shade);
                }
            } else {
                bool bad = false;
                for (int i = 0; i < 3; ++i) {
                    double len = std::sqrt(cs[i].x*cs[i].x+cs[i].y*cs[i].y+cs[i].z*cs[i].z);
                    if (cs[i].z <= -0.999 * len) bad = true;
                }
                if (bad) continue;
                VtxScreen sc0 = projectVtx(cam, cs[0], W, H);
                VtxScreen sc1 = projectVtx(cam, cs[1], W, H);
                VtxScreen sc2 = projectVtx(cam, cs[2], W, H);
                fillTriangle(sc0, sc1, sc2, W, H, y0, y1, zbuf, accum, emisMask, emis, shade);
            }
        }
    };

    if (nThreads < 1) nThreads = 1;
    if (nThreads == 1) {
        band(0, H);
    } else {
        std::vector<std::thread> pool;
        int rows = (H + nThreads - 1) / nThreads;
        for (int ti = 0; ti < nThreads; ++ti) {
            int y0 = ti * rows, y1 = std::min(H, y0 + rows);
            if (y0 >= y1) break;
            pool.emplace_back(band, y0, y1);
        }
        for (auto& th : pool) th.join();
    }

    // Auto-exposure anchor (mirror filmToRgb8): map the 99th-percentile luminance of the
    // lit surfaces to ~0.9. Background (unhit) pixels are excluded so an empty frame
    // margin can't skew the anchor; emitters are excluded too so the *subject* drives the
    // exposure (they just clip to white, as in the real render, instead of dragging the
    // anchor down when a large light fills the frame). Absolute EV (autoExpose=false)
    // bypasses this so aperture/power brightness differences survive into the preview.
    double eAuto = 1.0;
    if (autoExpose) {
        if (lockAnchor && *lockAnchor > 0.0) {
            eAuto = *lockAnchor;                    // reuse the path's locked anchor
        } else {
            std::vector<double> lum; lum.reserve(accum.size());
            for (size_t i = 0; i < accum.size(); ++i) {
                if (zbuf[i] <= 0.0f || emisMask[i]) continue;   // skip background + emitters
                const Vec3& c = accum[i];
                lum.push_back(std::max({c.x, c.y, c.z, 0.0}));
            }
            if (!lum.empty()) {
                std::sort(lum.begin(), lum.end());
                double p99 = lum[(size_t)(0.99 * (lum.size() - 1))];
                eAuto = (p99 > 0.0) ? 0.9 / p99 : 1.0;
            }
            if (lockAnchor) *lockAnchor = eAuto;    // first frame sets the anchor
        }
    }
    const double finalExp = eAuto * expComp;

    // Tone map: exposed hit pixels through sRGB gamma; background tint left unexposed.
    std::vector<uint8_t> img((size_t)W * H * 3);
    for (size_t i = 0; i < accum.size(); ++i) {
        Vec3 c = accum[i];
        if (zbuf[i] > 0.0f) c = c * finalExp;       // hit pixels get the exposure
        img[i * 3 + 0] = (uint8_t)std::clamp(srgbGamma(std::max(0.0, c.x)) * 255.0 + 0.5, 0.0, 255.0);
        img[i * 3 + 1] = (uint8_t)std::clamp(srgbGamma(std::max(0.0, c.y)) * 255.0 + 0.5, 0.0, 255.0);
        img[i * 3 + 2] = (uint8_t)std::clamp(srgbGamma(std::max(0.0, c.z)) * 255.0 + 0.5, 0.0, 255.0);
    }
    return img;
}

}  // namespace raster
