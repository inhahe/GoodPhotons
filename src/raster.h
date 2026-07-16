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
#include <functional>
#include <array>
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
    bool clear    = false;   // dielectric/thin-film/filter surface (see-through mode dims/hazes it)
};

// A "clear" preview surface for the optional see-through rasterizer: a transmissive
// dielectric-family material. In see-through mode these aren't drawn as solid ghosts;
// instead each such surface between the camera and the opaque background dims what's
// behind it (multiplicative transmittance) and adds a touch of milky haze, cumulative
// with the number of clear surfaces crossed. Mirror/half-mirror/glossy stay solid.
inline bool isClearPreviewType(MatType t) {
    return t == MatType::Dielectric || t == MatType::ThinFilm ||
           t == MatType::Filter     || t == MatType::DiffuseTransmit;
}

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
// `progress`, if set, is called as each heavy implicit (isosurface/CSG/metaball) is
// about to be marched: progress(done, total) where `total` is the implicit count and
// `done` runs 0..total (0 before the first, total after the last). Marching implicits
// is by far the slow part of tessellation, so this drives the "tessellating N/M" UI.
inline std::vector<PTri> tessellate(const Scene& sc, int isoRes,
                                    const std::function<void(int, int)>& progress = {}) {
    std::vector<PTri> out;
    // Precompute one solid colour per material.
    std::vector<Vec3> matCol(sc.mats.size());
    std::vector<char>  matEmit(sc.mats.size(), 0);
    std::vector<char>  matClear(sc.mats.size(), 0);
    for (size_t i = 0; i < sc.mats.size(); ++i) {
        bool em = false;
        matCol[i] = materialColor(sc.mats[i], em);
        matEmit[i] = em ? 1 : 0;
        matClear[i] = (!sc.mats[i].isLight && isClearPreviewType(sc.mats[i].type)) ? 1 : 0;
    }
    auto colOf = [&](int matId) -> Vec3 {
        return (matId >= 0 && matId < (int)matCol.size()) ? matCol[matId] : Vec3{0.6, 0.6, 0.6};
    };
    auto emOf = [&](int matId) -> bool {
        return (matId >= 0 && matId < (int)matEmit.size()) && matEmit[matId];
    };
    auto clearOf = [&](int matId) -> bool {
        return (matId >= 0 && matId < (int)matClear.size()) && matClear[matId];
    };

    // (1) World triangles.
    out.reserve(sc.tris.size() + 4096);
    for (const auto& t : sc.tris) {
        PTri p;
        p.p0 = t.v0; p.p1 = t.v1; p.p2 = t.v2;
        p.n0 = t.n0; p.n1 = t.n1; p.n2 = t.n2;
        p.color = colOf(t.matId); p.emissive = emOf(t.matId); p.clear = clearOf(t.matId);
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
        Vec3 col = colOf(s.matId); bool em = emOf(s.matId); bool cl = clearOf(s.matId);
        for (int iv = 0; iv < SV; ++iv)
            for (int iu = 0; iu < SU; ++iu) {
                Vec3 d00 = sp(iu, iv),   d10 = sp(iu + 1, iv);
                Vec3 d01 = sp(iu, iv+1), d11 = sp(iu + 1, iv + 1);
                Vec3 v00 = s.c + d00 * s.r, v10 = s.c + d10 * s.r;
                Vec3 v01 = s.c + d01 * s.r, v11 = s.c + d11 * s.r;
                PTri a; a.p0 = v00; a.p1 = v01; a.p2 = v11; a.n0 = d00; a.n1 = d01; a.n2 = d11; a.color = col; a.emissive = em; a.clear = cl;
                PTri b; b.p0 = v00; b.p1 = v11; b.p2 = v10; b.n0 = d00; b.n1 = d11; b.n2 = d10; b.color = col; b.emissive = em; b.clear = cl;
                out.push_back(a); out.push_back(b);
            }
    }

    // (3) Isosurfaces / metaballs / CSG -> marching-tetrahedra mesh.
    if (isoRes > 0) {
        isomesh::Options opt; opt.res = isoRes; opt.adaptive = false; opt.refineIters = 3;
        const int nImp = (int)sc.implicits.size();
        int impIdx = 0;
        for (const auto& im : sc.implicits) {
            if (progress) progress(impIdx, nImp);
            ++impIdx;
            isomesh::Mesh m = isomesh::marchImplicit(im, opt);
            Vec3 col = colOf(im.matId); bool em = emOf(im.matId); bool cl = clearOf(im.matId);
            for (size_t f = 0; f + 2 < m.tri.size(); f += 3) {
                int i0 = m.tri[f], i1 = m.tri[f + 1], i2 = m.tri[f + 2];
                PTri p;
                p.p0 = m.pos[i0]; p.p1 = m.pos[i1]; p.p2 = m.pos[i2];
                p.n0 = m.nrm[i0]; p.n1 = m.nrm[i1]; p.n2 = m.nrm[i2];
                p.color = col; p.emissive = em; p.clear = cl;
                out.push_back(p);
            }
        }
        if (progress) progress(nImp, nImp);
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
            p.color = colOf(matId); p.emissive = emOf(matId); p.clear = clearOf(matId);
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

// A screen-space triangle: three projected vertices plus the shared per-triangle
// attributes and a precomputed y-band [iy0,iy1] for O(1) band rejection. Produced once
// by the project-once pass and consumed by the deferred rasterizer, so projection and
// near-plane clipping happen a single time per triangle instead of once per thread.
struct STri {
    VtxScreen v0, v1, v2;
    Vec3   color;
    bool   emissive;
    bool   clear;      // see-through transmissive surface (handled by the clear-accumulation pass)
    int    iy0, iy1;   // inclusive pixel-row span the triangle can touch
};

// Deferred G-buffer: per-pixel geometry captured during rasterization, shaded once in a
// later pass (so overlapping triangles never shade the same covered pixel twice).
struct GBuffer {
    std::vector<float>   zbuf;    // 1/depth key (bigger = closer); 0 = background
    std::vector<Vec3>    wpos;    // world position of the winning surface
    std::vector<Vec3>    wn;      // world normal of the winning surface
    std::vector<Vec3>    color;   // base albedo of the winning triangle
    std::vector<uint8_t> emis;    // 1 where the winning triangle is an emitter
};

// Rasterize one screen-space triangle into the deferred G-buffer over rows [y0,y1).
// Only geometry/albedo is stored here — shading is deferred to a single later pass so
// each covered pixel is shaded exactly once regardless of overdraw. Attributes are
// interpolated perspective-correctly via invd.
inline void fillTriangleG(const STri& t, int W, int H, int y0, int y1, GBuffer& g) {
    const VtxScreen& A = t.v0; const VtxScreen& B = t.v1; const VtxScreen& C = t.v2;
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
    // Incremental barycentric edge functions: w0,w1 are affine in x, so step them along
    // each row with a single add instead of recomputing the full cross products per
    // sample. Rows recompute w0/w1 exactly from px=xlo+0.5, so only the x-derivative is
    // needed: d w0/d px = (B.sy - C.sy)*inv, d w1/d px = (C.sy - A.sy)*inv.
    const double dw0dx = (B.sy - C.sy) * inv;
    const double dw1dx = (C.sy - A.sy) * inv;
    const uint8_t triEmis = t.emissive ? 1 : 0;
    for (int y = ylo; y <= yhi; ++y) {
        double py = y + 0.5, pxL = xlo + 0.5;
        double w0 = ((B.sx - pxL) * (C.sy - py) - (B.sy - py) * (C.sx - pxL)) * inv;
        double w1 = ((C.sx - pxL) * (A.sy - py) - (C.sy - py) * (A.sx - pxL)) * inv;
        size_t row = (size_t)y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row, w0 += dw0dx, w1 += dw1dx) {
            double w2 = 1.0 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            double invd = w0 * A.invd + w1 * B.invd + w2 * C.invd;   // = 1/depth
            if (invd <= g.zbuf[row]) continue;   // farther than (or equal to) stored
            g.zbuf[row] = (float)invd;
            g.emis[row] = triEmis;               // emitters excluded from the auto-exposure anchor
            // Perspective-correct attribute recovery.
            double d = 1.0 / std::max(invd, 1e-12);
            g.wpos[row]  = (A.wpos * (w0 * A.invd) + B.wpos * (w1 * B.invd) + C.wpos * (w2 * C.invd)) * d;
            g.wn[row]    = (A.wn   * (w0 * A.invd) + B.wn   * (w1 * B.invd) + C.wn   * (w2 * C.invd)) * d;
            g.color[row] = t.color;
        }
    }
}

// See-through accumulation for one clear (transmissive) triangle. Instead of writing a
// solid surface, every covered pixel whose clear fragment lies IN FRONT of the opaque
// depth (invd > g.zbuf) multiplies that pixel's running transmittance `clearT` by the
// per-surface transmittance and its milk product `milkT` by (1 - per-surface milk). The
// product form is order-independent (commutative), so no depth sort of the transparent
// fragments is needed — N crossed surfaces just give clarity^N dimming and a growing haze.
// A grazing-angle (Fresnel-like) term adds extra milk at silhouettes so glass edges read.
inline void fillTriangleClear(const STri& t, const Camera& cam, int W, int H, int y0, int y1,
                              const GBuffer& g, std::vector<float>& clearT, std::vector<float>& milkT,
                              double clarity, double milkPerSurface, double rimStrength) {
    const VtxScreen& A = t.v0; const VtxScreen& B = t.v1; const VtxScreen& C = t.v2;
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
    const double dw0dx = (B.sy - C.sy) * inv;
    const double dw1dx = (C.sy - A.sy) * inv;
    const float tau = (float)clarity;
    for (int y = ylo; y <= yhi; ++y) {
        double py = y + 0.5, pxL = xlo + 0.5;
        double w0 = ((B.sx - pxL) * (C.sy - py) - (B.sy - py) * (C.sx - pxL)) * inv;
        double w1 = ((C.sx - pxL) * (A.sy - py) - (C.sy - py) * (A.sx - pxL)) * inv;
        size_t row = (size_t)y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row, w0 += dw0dx, w1 += dw1dx) {
            double w2 = 1.0 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            double invd = w0 * A.invd + w1 * B.invd + w2 * C.invd;   // = 1/depth
            if (invd <= g.zbuf[row]) continue;   // behind (or at) the opaque surface: occluded
            // Grazing term from the interpolated normal for a silhouette milk rim.
            double d = 1.0 / std::max(invd, 1e-12);
            Vec3 wpos = (A.wpos * (w0 * A.invd) + B.wpos * (w1 * B.invd) + C.wpos * (w2 * C.invd)) * d;
            Vec3 wn   = (A.wn   * (w0 * A.invd) + B.wn   * (w1 * B.invd) + C.wn   * (w2 * C.invd)) * d;
            Vec3 Nn = normalize(wn);
            Vec3 V  = normalize(cam.eye - wpos);
            double ndv = std::fabs(dot(Nn, V));
            double graze = 1.0 - ndv;                 // 0 head-on, ->1 at the silhouette
            double perMilk = milkPerSurface + rimStrength * graze * graze * graze;
            if (perMilk > 0.95) perMilk = 0.95;
            clearT[row] *= tau;
            milkT[row]  *= (float)(1.0 - perMilk);
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
                                        bool autoExpose = true, double* lockAnchor = nullptr,
                                        bool seeThrough = false, double glassClarity = 0.85) {
    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    const double EMIS_BOOST = 4.0;    // emitters read as bright light sources (clip to white)
    const Vec3 bg{0.06, 0.07, 0.09};                    // background tint (unlit, unexposed)
    const size_t N = (size_t)W * H;

    const double zn = 1e-3;   // near plane (camera-forward) for rectilinear clipping
    const bool rect = (cam.projection == CAM_RECTILINEAR);

    if (nThreads < 1) nThreads = 1;

    // Tiny parallel-for over [0,n): splits into nThreads contiguous chunks. Used by the
    // shading + tone-map passes (each pixel is independent, so no locking needed).
    auto parallelFor = [&](size_t n, const std::function<void(size_t, size_t)>& body) {
        if (n == 0) return;
        if (nThreads == 1) { body(0, n); return; }
        std::vector<std::thread> pool;
        size_t chunk = (n + nThreads - 1) / nThreads;
        for (int ti = 0; ti < nThreads; ++ti) {
            size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
            if (a >= b) break;
            pool.emplace_back(body, a, b);
        }
        for (auto& th : pool) th.join();
    };

    // -- Pass 1: project every triangle ONCE (parallel over the triangle list). Each
    // thread clips + projects its slice into a local STri buffer; the buffers are then
    // concatenated. This removes the old per-thread redundancy where every rasterizer
    // band re-projected the entire scene (an nThreads-fold projection cost).
    auto projectRange = [&](size_t a, size_t b, std::vector<STri>& out) {
        auto toCS = [&](const Vec3& P, const Vec3& Nn) -> VtxCS {
            Vec3 d = P - cam.eye; VtxCS c;
            c.x = dot(d, cam.u); c.y = dot(d, cam.v); c.z = dot(d, cam.w);
            c.wpos = P; c.wn = Nn; return c;
        };
        auto push = [&](const VtxScreen& s0, const VtxScreen& s1, const VtxScreen& s2,
                        const Vec3& col, bool emis, bool clr) {
            double lo = std::min({s0.sy, s1.sy, s2.sy});
            double hi = std::max({s0.sy, s1.sy, s2.sy});
            int iy0 = std::max(0, (int)std::floor(lo));
            int iy1 = std::min(H - 1, (int)std::ceil(hi));
            if (iy0 > iy1) return;
            out.push_back(STri{s0, s1, s2, col, emis, clr, iy0, iy1});
        };
        for (size_t ti = a; ti < b; ++ti) {
            const PTri& t = tris[ti];
            VtxCS cs[3] = { toCS(t.p0, t.n0), toCS(t.p1, t.n1), toCS(t.p2, t.n2) };
            if (rect) {
                VtxCS poly[8]; int np = 0;
                auto emit = [&](const VtxCS& a2){ if (np < 8) poly[np++] = a2; };
                auto lerpV = [&](const VtxCS& a2, const VtxCS& b2, double s) -> VtxCS {
                    VtxCS r; r.x=a2.x+(b2.x-a2.x)*s; r.y=a2.y+(b2.y-a2.y)*s; r.z=a2.z+(b2.z-a2.z)*s;
                    r.wpos=a2.wpos+(b2.wpos-a2.wpos)*s; r.wn=a2.wn+(b2.wn-a2.wn)*s; return r;
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
                    push(sc0, sc1, sc2, t.color, t.emissive, t.clear);
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
                push(sc0, sc1, sc2, t.color, t.emissive, t.clear);
            }
        }
    };

    std::vector<STri> stris;
    {
        int pT = std::min<int>(nThreads, std::max<size_t>(1, tris.size()));
        if (pT <= 1) {
            stris.reserve(tris.size());
            projectRange(0, tris.size(), stris);
        } else {
            std::vector<std::vector<STri>> parts(pT);
            std::vector<std::thread> pool;
            size_t chunk = (tris.size() + pT - 1) / pT;
            for (int ti = 0; ti < pT; ++ti) {
                size_t a = (size_t)ti * chunk, b = std::min(tris.size(), a + chunk);
                if (a >= b) break;
                parts[ti].reserve((b - a));
                pool.emplace_back([&, ti, a, b]{ projectRange(a, b, parts[ti]); });
            }
            for (auto& th : pool) th.join();
            size_t tot = 0; for (auto& p : parts) tot += p.size();
            stris.reserve(tot);
            for (auto& p : parts) stris.insert(stris.end(), p.begin(), p.end());
        }
    }

    // See-through (clear-glass) preview parameters. Each clear surface between the camera
    // and the opaque background dims what's behind it by `glassClarity` (transmittance) and
    // adds a little milky haze; both accumulate with the number of clear surfaces crossed.
    const double kMilkPerSurface = std::max(0.0, (1.0 - glassClarity)) * 0.55; // haze per surface
    const double kRimStrength    = 0.55;                     // extra silhouette milk (Fresnel-ish)
    const Vec3   kMilkColor{0.52, 0.55, 0.60};               // display-space haze tint

    // Dispatch a per-row-band body across nThreads (each band owns disjoint rows -> no locking).
    auto dispatchBands = [&](const std::function<void(int,int)>& body) {
        if (nThreads == 1) { body(0, H); return; }
        std::vector<std::thread> pool;
        int rows = (H + nThreads - 1) / nThreads;
        for (int ti = 0; ti < nThreads; ++ti) {
            int y0 = ti * rows, y1 = std::min(H, y0 + rows);
            if (y0 >= y1) break;
            pool.emplace_back(body, y0, y1);
        }
        for (auto& th : pool) th.join();
    };

    // -- Pass 2: deferred G-buffer rasterization, parallel by horizontal row-bands. Each
    // band owns rows [y0,y1) so bands never touch the same pixel (no locking). Triangles
    // whose y-span misses the band are skipped in O(1) via the precomputed iy0/iy1.
    GBuffer g;
    g.zbuf.assign(N, 0.0f);
    g.wpos.assign(N, Vec3{0,0,0});
    g.wn.assign(N, Vec3{0,0,0});
    g.color.assign(N, bg);
    g.emis.assign(N, 0);
    dispatchBands([&](int y0, int y1) {
        for (const STri& s : stris) {
            if (s.iy1 < y0 || s.iy0 >= y1) continue;   // triangle can't touch this band
            if (seeThrough && s.clear) continue;       // clear surfaces handled in Pass 2b
            fillTriangleG(s, W, H, y0, y1, g);
        }
    });

    // -- Pass 2b (see-through only): accumulate the clear surfaces' cumulative transmittance
    // (`clearT`, product of glassClarity per crossed surface) and milk product (`milkT`)
    // against the now-complete opaque depth. Order-independent, so no transparent sort.
    std::vector<float> clearT, milkT;
    if (seeThrough) {
        clearT.assign(N, 1.0f);
        milkT.assign(N, 1.0f);
        dispatchBands([&](int y0, int y1) {
            for (const STri& s : stris) {
                if (!s.clear) continue;
                if (s.iy1 < y0 || s.iy0 >= y1) continue;
                fillTriangleClear(s, cam, W, H, y0, y1, g, clearT, milkT,
                                  glassClarity, kMilkPerSurface, kRimStrength);
            }
        });
    }

    // -- Pass 3: shade each covered pixel exactly once (parallel over pixels). Overlapping
    // triangles no longer re-shade the same pixel — only the winning surface is shaded.
    std::vector<Vec3> accum(N, bg);
    parallelFor(N, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            if (g.zbuf[i] <= 0.0f) continue;         // background stays bg tint
            const Vec3& col = g.color[i];
            if (g.emis[i]) { accum[i] = col * EMIS_BOOST; continue; }  // raw emitter radiance
            Vec3 N3 = normalize(g.wn[i]);
            Vec3 V = normalize(cam.eye - g.wpos[i]);     // toward camera
            if (dot(N3, V) < 0.0) N3 = -N3;              // two-sided
            double lit = 0.0;
            for (const auto& lp : light.lights) {
                Vec3 d = lp.pos - g.wpos[i];
                double dist2 = dot(d, d);
                Vec3 Ld = (dist2 > 1e-12) ? d / std::sqrt(dist2) : V;
                double ndl = std::max(0.0, dot(N3, Ld));
                if (ndl <= 0.0) continue;
                double atten = 1.0;
                if (lp.falloff2 > 0.0) atten = lp.falloff2 / (lp.falloff2 + dist2);
                double cone = 1.0;
                if (lp.spot) cone = spotFalloff(dot(lp.dir, -Ld), lp.cosInner, lp.cosOuter);
                lit += lp.weight * ndl * atten * cone;
            }
            double head = std::max(0.0, dot(N3, V));     // headlight fill
            double k = light.ambient + light.keyScale * lit + light.fill * head;
            accum[i] = col * k;
        }
    });

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
            std::vector<double> lum; lum.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                if (g.zbuf[i] <= 0.0f || g.emis[i]) continue;   // skip background + emitters
                const Vec3& c = accum[i];
                lum.push_back(std::max({c.x, c.y, c.z, 0.0}));
            }
            if (!lum.empty()) {
                // Only the 99th-percentile order statistic matters, so partition instead
                // of a full sort (O(n) vs O(n log n)).
                size_t k = (size_t)(0.99 * (lum.size() - 1));
                std::nth_element(lum.begin(), lum.begin() + k, lum.end());
                double p99 = lum[k];
                eAuto = (p99 > 0.0) ? 0.9 / p99 : 1.0;
            }
            if (lockAnchor) *lockAnchor = eAuto;    // first frame sets the anchor
        }
    }
    const double finalExp = eAuto * expComp;

    // sRGB gamma lookup table: the tone map clamps each channel to [0,1] before encoding,
    // and gamma is monotonic (anything >=1 saturates to 255), so a 4096-entry LUT over
    // [0,1] replaces three std::pow calls per pixel with a table read + round.
    static const std::array<uint8_t, 4097> kSrgbLut = [] {
        std::array<uint8_t, 4097> t{};
        for (int i = 0; i <= 4096; ++i)
            t[i] = (uint8_t)std::clamp(srgbGamma(i / 4096.0) * 255.0 + 0.5, 0.0, 255.0);
        return t;
    }();
    auto encode = [&](double c) -> uint8_t {
        if (c <= 0.0) return kSrgbLut[0];
        if (c >= 1.0) return 255;
        return kSrgbLut[(int)(c * 4096.0 + 0.5)];
    };

    // Tone map: exposed hit pixels through sRGB gamma; background tint left unexposed.
    std::vector<uint8_t> img(N * 3);
    parallelFor(N, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            Vec3 c = accum[i];
            if (g.zbuf[i] > 0.0f) c = c * finalExp;   // hit pixels get the exposure
            if (seeThrough) {                          // composite clear glass (display-linear)
                float T = clearT[i], mt = milkT[i];
                if (T < 1.0f || mt < 1.0f)
                    c = c * (double)T + kMilkColor * (1.0 - (double)mt);
            }
            img[i * 3 + 0] = encode(c.x);
            img[i * 3 + 1] = encode(c.y);
            img[i * 3 + 2] = encode(c.z);
        }
    });
    return img;
}

// Draw a red look-at crosshair at world point `target` onto an already-rendered RGB
// frame (W*H*3, row 0 = top). Projects with the same camera math as the triangles;
// if the point is in front and roughly on-screen it stamps a red '+' with a centre
// gap plus a small box, so the exact aim point stays visible. Drawn on top (ignores
// depth) so you can always see where the interactive camera is pointed. This is the
// visible marker for the 6-DOF preview control (eye xyz + this target xyz).
//
// `worldRadius` (>0) makes the crosshair a fixed *world* size rather than a fixed
// screen size: the arm length is the on-screen projection of a `worldRadius`-long
// segment sitting at the target, so the marker SHRINKS as the target is pushed farther
// and GROWS as it's pulled nearer, exactly per the camera's perspective — a visual cue
// for the target's depth. `worldRadius==0` falls back to the old constant-screen size.
inline void drawTargetMarker(std::vector<uint8_t>& img, int W, int H,
                             const Camera& cam, const Vec3& target,
                             double worldRadius = 0.0) {
    Vec3 d = target - cam.eye;
    VtxCS c; c.x = dot(d, cam.u); c.y = dot(d, cam.v); c.z = dot(d, cam.w);
    c.wpos = target; c.wn = Vec3{0, 0, 1};
    if (cam.projection == CAM_RECTILINEAR && c.z <= 1e-6) return;   // behind the camera
    VtxScreen s = projectVtx(cam, c, W, H);
    if (s.sx < -W || s.sx > 2 * W || s.sy < -H || s.sy > 2 * H) return;  // wildly off-screen
    const uint8_t R = 255, G = 40, B = 40;
    auto put = [&](int x, int y) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        size_t i = ((size_t)y * W + x) * 3;
        img[i + 0] = R; img[i + 1] = G; img[i + 2] = B;
    };
    const int icx = (int)std::lround(s.sx), icy = (int)std::lround(s.sy);
    int arm;
    if (worldRadius > 0.0) {
        // Project a point offset from the target by `worldRadius` along camera-right; the
        // pixel gap to the centre is the perspective-correct on-screen size of that world
        // length. Clamp so a very distant target still shows a tiny cross and a very near
        // one doesn't swallow the whole frame.
        Vec3 pw = target + cam.u * worldRadius;
        Vec3 d2 = pw - cam.eye;
        VtxCS o; o.x = dot(d2, cam.u); o.y = dot(d2, cam.v); o.z = dot(d2, cam.w);
        o.wpos = pw; o.wn = Vec3{0, 0, 1};
        VtxScreen so = projectVtx(cam, o, W, H);
        double px = std::hypot(so.sx - s.sx, so.sy - s.sy);
        arm = (int)std::lround(std::clamp(px, 3.0, 0.75 * std::max(W, H)));
    } else {
        arm = std::max(10, W / 36);   // arm length (roughly constant on screen)
    }
    const int gap = std::max(2, arm / 4);   // centre gap so the exact point is unobscured
    const int th  = std::max(1, arm / 40);  // line half-thickness (scales with the cross)
    for (int t = -th; t <= th; ++t)
        for (int a = gap; a <= arm; ++a) {
            put(icx + a, icy + t); put(icx - a, icy + t);   // horizontal arms
            put(icx + t, icy + a); put(icx + t, icy - a);   // vertical arms
        }
    const int bs = gap - 1;                                 // small centre box outline
    for (int a = -bs; a <= bs; ++a) {
        put(icx + a, icy - bs); put(icx + a, icy + bs);
        put(icx - bs, icy + a); put(icx + bs, icy + a);
    }
}

}  // namespace raster
