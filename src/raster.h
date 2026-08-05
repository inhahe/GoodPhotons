// raster.h — fast solid-shaded PREVIEW rasterizer (z-buffer, no light transport).
//
// This is the "quick taste" viewer: it turns the whole scene into triangles once
// (analytic spheres tessellated, isosurfaces marched to a mesh, instanced meshes
// baked to world space) and rasterizes each authored camera with a plain z-buffer
// and simple diffuse+headlight shading.
//
// Everything that gives a surface its LOOK at a single point is previewed:
//   * Image skins (`reflect texture:<name>`) — UVs are interpolated in the deferred
//     G-buffer and the texture's linear RGB sampled per pixel in the shade pass. The
//     UVs come from per-vertex coords, a world triplanar projection, or (for marched
//     implicits, which have no per-vertex UVs) the primitive's own `uv planar/
//     spherical/cylindrical` projection, re-evaluated per marched vertex.
//   * Palette (indexed-spectral) maps — resolved to one linear-sRGB colour per palette
//     entry, so an index map previews as its actual spectra, not as raw indices.
//   * Procedural `pattern` drives on the albedo (`reflect pattern:`/`reflect_map
//     pattern:`) and on the EMISSION (`emit pattern:`/`emit_map pattern:`), evaluated
//     per pixel by the same VM the tracer uses. The emission mask matters most: without
//     it a masked emitter previews as one flat glowing slab instead of its pattern.
//   * Normal maps (`normal_map`) — perturbed through the triangle's UV-derived TBN.
//   * Mix / layered materials — resolved to their dominant child (the same choice
//     deterministic mode W makes), instead of collapsing to the parent's flat colour.
//     A two-child mix carrying a `weight_map` is resolved PER PIXEL instead: the mask is
//     sampled at the shaded point and the whole losing payload (albedo, skin, normal map,
//     pattern drives) is swapped in, so a wear mask / decal / painted blend previews as
//     the spatial A/B pattern it is rather than as one flat winner.
// There is NO transparency, refraction, reflection, shadows, caustics or global
// illumination — a dielectric shows as a solid ghost, a mirror as a flat tint. Glossy
// lobes do not exist here either, so roughness/film-thickness maps are ignored by
// design (they drive nothing a preview can show). The point is to see the *composition*
// and (for a camera_curve) the *flyby motion* in a fraction of a second per frame,
// exactly the way the isosurface mesher lets you eyeball an implicit.
//
// It reuses the real Camera projection (Camera::project semantics reimplemented for
// triangle clipping), so the pinhole's off-axis elongation and the fisheye/panoramic
// lenses are reproduced faithfully — a sphere near the frame edge stretches just as
// it will in the physical render.
#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <array>
#include "scene.h"
#include "camera.h"
#include "color.h"
#include "isomesh.h"

namespace raster {

// Everything the shade pass needs to know about a surface's MATERIAL, and nothing about
// its geometry. Split out of PTri so that a per-hit `mix` (below) can swap the whole
// payload per PIXEL: the two children of a weight-mapped mix differ in albedo, skin and
// pattern drives all at once, so they have to travel together rather than as loose fields.
struct PShade {
    Vec3 color;
    int  tex = -1;           // index into the scene texture table, or -1 (flat `color`)
    double triplanarScale = 0.0;  // >0: sample the texture by world triplanar, not UV
    // Scalar pattern drives (index into Scene::patterns, or -1). Evaluated per pixel in
    // the shade pass and multiplied into the slot they name, exactly as the tracer's
    // slotPatMul does — that is what turns `emit_map pattern:grid_ground` from "the whole
    // floor glows" into the thin grid lines it actually is.
    int  reflectPat = -1;    // scales the albedo (`reflect pattern:` / `reflect_map pattern:`)
    int  emitPat    = -1;    // scales the emission (`emit pattern:` / `emit_map pattern:`)
    int  normalTex  = -1;    // tangent-space normal map, or -1
    double normalStrength = 1.0;
    bool emissive = false;
    bool clear    = false;   // dielectric/thin-film/filter surface (see-through mode dims/hazes it)
};

// A two-child `mix` whose blend is driven per hit by `weight_map pattern:` /
// `weight_map texture:` — a spatial A/B selection, not a constant weight, so it cannot be
// resolved once at bake time. The preview stores the LOSING child's payload here and picks
// between it and the triangle's own (the winning-at-t>=0.5 child) per pixel, which is
// exactly what the deterministic Whitted preview does in mixResolveDominant(): a hard
// threshold at t == 0.5 rather than a stochastic dither, so raster and -mode W agree.
//
// One entry per MATERIAL, not per triangle — the payload is material-derived, so a scene
// with three weight-mapped mixes has three entries no matter how many triangles carry them.
// That keeps PTri one int larger instead of doubling its shading half.
struct PMix {
    int    weightPat = -1;   // Scene::patterns index driving child-0's share, or -1
    int    weightTex = -1;   // ...or a scalar texture (`weight_map texture:`), or -1
    PShade b;                // the child-1 payload, shown where the weight evaluates < 0.5
};

// One preview triangle: world-space positions + per-vertex world normals + the material
// payload it was baked with (inherited, so `t.color` / `t.tex` still read as before).
struct PTri : PShade {
    Vec3 p0, p1, p2;
    Vec3 n0, n1, n2;
    // Per-vertex texture coordinates (u in .x, v in .y). Only meaningful when tex >= 0.
    Vec3 uv0{0, 0, 0}, uv1{0, 0, 0}, uv2{0, 0, 0};
    int  mix = -1;           // index into PreviewGeom::mixes for a per-hit mix, else -1
    // Raw (unnormalized, un-orthogonalized) dP/dU tangent for normal mapping. Constant
    // over the triangle, so tessellate() precomputes it once for every triangle that can
    // shade a normal map (its own material's, or its mix child's) instead of the shade
    // pass re-deriving it from the edge/UV deltas at every covered pixel. Zero when the
    // UV parameterisation is degenerate/absent — the shade pass then falls back to a
    // stable frame about the shading normal, exactly as the old per-pixel path did.
    Vec3 tanRaw{0, 0, 0};
};

// Tessellated preview geometry plus the side tables its triangles index. Bundled so the
// two cannot be handed around separately and fall out of sync: a PTri's `mix` index is
// only meaningful against the `mixes` built in the same tessellate() call.
struct PreviewGeom {
    std::vector<PTri> tris;
    std::vector<PMix> mixes;
    void clear() { tris.clear(); mixes.clear(); }
    bool empty() const { return tris.empty(); }
    size_t size() const { return tris.size(); }
};

// The raw dP/dU tangent of one triangle (the standard UV-gradient construction), or zero
// when the UV parameterisation is degenerate. Called once per triangle by tessellate()'s
// tangent bake; the shade pass finishes the frame per pixel (Gram-Schmidt against the
// interpolated shading normal + normalize), which is the only part that varies per pixel.
inline Vec3 triTangentRaw(const PTri& t) {
    Vec3 e1 = t.p1 - t.p0, e2 = t.p2 - t.p0;
    Vec3 d1 = t.uv1 - t.uv0, d2 = t.uv2 - t.uv0;
    double det = d1.x * d2.y - d2.x * d1.y;
    if (std::fabs(det) > 1e-18) return (e1 * d2.y - e2 * d1.y) * (1.0 / det);
    return Vec3{0, 0, 0};
}

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
        if (e.shape == EmitterShape::Sun) {
            // A distant sun is directional: fake it as a point source parked far up the
            // beam with distance falloff disabled, so every surface shades from the same
            // (to ~1e-4 rad) direction at full strength. No new PLight field, hence no
            // change needed in the two GPU mirrors of this struct.
            p.pos = sc.sceneCenter - normalize(e.beamDir) * (sc.sceneRadius * 1e4 + 1e4);
            p.falloff2 = 0.0;
        }
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
inline PreviewGeom tessellate(const Scene& sc, int isoRes,
                              const std::function<void(int, int)>& progress = {}) {
    PreviewGeom geom;
    std::vector<PTri>& out = geom.tris;
    // One baked shading payload per material (was a fistful of parallel arrays; a single
    // PShade keeps them from drifting apart and lets the mix table below reuse them).
    std::vector<PShade> matSh(sc.mats.size());
    std::vector<int>    matMix(sc.mats.size(), -1);  // index into geom.mixes, or -1
    PShade fallback;                                  // unknown/out-of-range material
    fallback.color = Vec3{0.6, 0.6, 0.6};

    // Bake ONE material's own preview payload. No mix resolution here — the callers below
    // decide which material to bake, so this stays a pure Material -> PShade function.
    auto bakeOwn = [&](const Material& m) -> PShade {
        PShade s;
        bool em = false;
        s.color    = materialColor(m, em);
        s.emissive = em;
        s.clear    = (!m.isLight && isClearPreviewType(m.type));
        // An image skin: a diffuse-albedo texture bound via `reflect texture:<name>`.
        // The preview shades from the texture's linear RGB (Texture::sampleRgb), so no
        // Jakob-Hanika coefficient precompute is needed (that's only for spectral hits).
        // Palette (indexed) maps are included: sampleRgb resolves the index through
        // Texture::paletteRgb rather than shading with the raw index byte.
        // Emitters are still excluded, and deliberately: emission is an SPD plus an
        // optional `emit_map pattern:` — there is NO textured-emission slot in the
        // material model, so a `reflect texture:` on a light means nothing to the tracer
        // and previewing it would invent detail the real render does not have. Spatially
        // varying emission is previewed through emitPat below, which IS the real mechanism.
        int rt = m.reflectTex;
        if (!m.isLight && rt >= 0 && rt < (int)sc.textures.size() && sc.textures[rt].valid()) {
            s.tex = rt;
            s.triplanarScale = m.triplanarScale;
        }
        // Scalar pattern drives. `reflect pattern:` / `reflect_map pattern:` both land in
        // reflectPat and multiply the albedo; `emit pattern:` / `emit_map pattern:` land
        // in emitPat and multiply the emission. Same slots, same clamp, as the tracer.
        s.reflectPat = m.reflectPat;
        s.emitPat    = m.emitPat;
        if (m.normalTex >= 0 && m.normalTex < (int)sc.textures.size() &&
            sc.textures[m.normalTex].valid()) {
            s.normalTex      = m.normalTex;
            s.normalStrength = m.normalStrength;
        }
        return s;
    };
    // A Mix material has no shading of its own — it selects among child materials, so its
    // own `reflect` slot is normally unset and previewing it shows a flat default grey.
    // Resolve to the HEAVIEST child via mixDominantChild, which is exactly what the
    // deterministic Whitted preview (-mode W) does, so raster and mode W agree on what a
    // mix looks like. Layered/Multilayer use the same child list, so they resolve too.
    // Iterated (a child may itself be a mix) with a depth cap against a malformed cycle;
    // a leftover-absorption result (-1) keeps the parent, which previews as its own colour.
    auto resolveMix = [&](size_t i) -> int {
        int cur = (int)i;
        for (int guard = 0; guard < 8; ++guard) {
            if (cur < 0 || cur >= (int)sc.mats.size()) break;
            const Material& m = sc.mats[cur];
            if (m.mixChildren.empty()) break;
            int pick = mixDominantChild(m);
            if (pick < 0 || pick == cur || pick >= (int)sc.mats.size()) break;
            cur = pick;
        }
        return cur;
    };
    // Pass 1 — every material, previewed through its CONSTANT-weight mix chain.
    for (size_t i = 0; i < sc.mats.size(); ++i)
        matSh[i] = bakeOwn(sc.mats[resolveMix(i)]);
    // Pass 2 — upgrade the two-child mixes whose blend is driven per hit by `weight_map`.
    // Pass 1 collapsed these to whichever child had the larger CONSTANT weight (usually a
    // 50/50 tie, so always child 0), which is why a weight-mapped mix previewed as one flat
    // colour while -mode W showed the mask. Now child 0 rides on the triangle and child 1
    // goes in the side table, to be chosen per pixel at the t == 0.5 threshold.
    //
    // A child that is ITSELF a weight-mapped mix still flattens (matSh[child] is that
    // child's own dominant collapse): nesting one spatial mask inside another would need a
    // recursive per-pixel walk, and no scene in the library does it.
    for (size_t i = 0; i < sc.mats.size(); ++i) {
        const Material& m = sc.mats[i];
        if (m.mixChildren.size() != 2) continue;
        if (m.mixWeightPat < 0 && m.mixWeightTex < 0) continue;
        const int c0 = m.mixChildren[0], c1 = m.mixChildren[1];
        const int n  = (int)sc.mats.size();
        if (c0 < 0 || c0 >= n || c1 < 0 || c1 >= n) continue;
        PMix mx;
        mx.weightPat = m.mixWeightPat;
        mx.weightTex = m.mixWeightTex;
        mx.b         = matSh[c1];          // shown where the weight evaluates < 0.5
        matMix[i]    = (int)geom.mixes.size();
        geom.mixes.push_back(mx);
        matSh[i]     = matSh[c0];          // ...and child 0 where it is >= 0.5
    }
    // Stamp EVERY material-derived field onto a triangle in one place. Each geometry kind
    // below (world tris, spheres, implicits, instances) calls exactly this, so adding a
    // per-material preview feature can no longer be wired into three of the four paths and
    // silently dropped on the fourth — which is how marched implicits ended up unable to
    // show a skin at all. Only the UV SOURCE differs per kind, and that stays local.
    auto applyMat = [&](PTri& p, int matId) {
        const bool ok = matId >= 0 && matId < (int)matSh.size();
        static_cast<PShade&>(p) = ok ? matSh[matId] : fallback;
        p.mix = ok ? matMix[matId] : -1;
    };

    // (1) World triangles.
    out.reserve(sc.tris.size() + 4096);
    for (const auto& t : sc.tris) {
        PTri p;
        p.p0 = t.v0; p.p1 = t.v1; p.p2 = t.v2;
        p.n0 = t.n0; p.n1 = t.n1; p.n2 = t.n2;
        applyMat(p, t.matId);
        p.uv0 = t.uv0; p.uv1 = t.uv1; p.uv2 = t.uv2;
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
        // Equirectangular (lat/long) UV per vertex, matching the analytic sphere hit in
        // geometry.h (u = 0.5 + atan2(z,x)/2pi, v = 0.5 - asin(y)/pi) so a skin lines up
        // with the real render. Computed from the unit direction d (== the vertex normal).
        auto uvOf = [](const Vec3& d) -> Vec3 {
            return Vec3{0.5 + std::atan2(d.z, d.x) / (2.0 * PI),
                        0.5 - std::asin(std::clamp(d.y, -1.0, 1.0)) / PI, 0.0};
        };
        for (int iv = 0; iv < SV; ++iv)
            for (int iu = 0; iu < SU; ++iu) {
                Vec3 d00 = sp(iu, iv),   d10 = sp(iu + 1, iv);
                Vec3 d01 = sp(iu, iv+1), d11 = sp(iu + 1, iv + 1);
                Vec3 v00 = s.c + d00 * s.r, v10 = s.c + d10 * s.r;
                Vec3 v01 = s.c + d01 * s.r, v11 = s.c + d11 * s.r;
                // Seam fix: atan2 wraps at u=1->0 across the last column; add 1 turn to the
                // higher-index column's u so the interpolated span stays monotonic.
                Vec3 uv00 = uvOf(d00), uv01 = uvOf(d01), uv10 = uvOf(d10), uv11 = uvOf(d11);
                if (iu == SU - 1) { uv10.x += 1.0; uv11.x += 1.0; }
                PTri a; a.p0 = v00; a.p1 = v01; a.p2 = v11; a.n0 = d00; a.n1 = d01; a.n2 = d11;
                applyMat(a, s.matId);
                a.uv0 = uv00; a.uv1 = uv01; a.uv2 = uv11;
                PTri b; b.p0 = v00; b.p1 = v11; b.p2 = v10; b.n0 = d00; b.n1 = d11; b.n2 = d10;
                applyMat(b, s.matId);
                b.uv0 = uv00; b.uv1 = uv11; b.uv2 = uv10;
                out.push_back(a); out.push_back(b);
            }
    }

    // (3) Isosurfaces / metaballs / CSG -> marching-tetrahedra mesh.
    if (isoRes > 0) {
        isomesh::Options opt; opt.res = isoRes; opt.adaptive = false; opt.refineIters = 3;
        const int nImp = (int)sc.implicits.size();
        // March the implicits in PARALLEL: each marchImplicit(im, opt) is a
        // deterministic pure function of its inputs and the meshes land in a
        // per-implicit slot, so emitting PTris below in the original implicit
        // order yields a triangle list identical to the old sequential loop.
        // Marching is by far the slow part of tessellation (seconds of field
        // evals on a heavy scene) and was single-threaded on the calling thread.
        // Tasks are handed out biggest-lattice-first so one whale implicit
        // doesn't start last and stretch the makespan.
        std::vector<isomesh::Mesh> meshes(sc.implicits.size());
        if (nImp > 0) {
            if (progress) progress(0, nImp);
            std::vector<int> order(nImp);
            for (int i = 0; i < nImp; ++i) order[i] = i;
            auto cellEstimate = [&](const Implicit& im) -> double {
                Vec3 e = im.bounds.hi - im.bounds.lo;
                double maxe = std::max(e.x, std::max(e.y, e.z));
                if (maxe <= 0) return 0.0;
                auto cells = [&](double v) { return std::max(1.0, std::round(opt.res * (v / maxe))); };
                return cells(e.x) * cells(e.y) * cells(e.z);
            };
            std::vector<double> est(nImp);
            for (int i = 0; i < nImp; ++i) est[i] = cellEstimate(sc.implicits[i]);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return est[a] > est[b]; });
            unsigned hw = std::thread::hardware_concurrency(); if (hw == 0) hw = 4;
            int T = (int)std::min<size_t>((size_t)nImp, (size_t)hw);
            std::atomic<int> next{0}, done{0};
            std::mutex progMx;
            // Shared read-only table view (see Scene::patTables): every worker marches with
            // the same one, so a sampled field polygonises identically across threads.
            const PatTables tabs = sc.patTables();
            auto workBody = [&]() {
                for (;;) {
                    int slot = next.fetch_add(1);
                    if (slot >= nImp) break;
                    int i = order[slot];
                    meshes[i] = isomesh::marchImplicit(sc.implicits[i], opt, &tabs);
                    int d = done.fetch_add(1) + 1;
                    if (progress) { std::lock_guard<std::mutex> lk(progMx); progress(d, nImp); }
                }
            };
            if (T <= 1) {
                workBody();
            } else {
                std::vector<std::thread> pool;
                pool.reserve(T);
                for (int t = 0; t < T; ++t) pool.emplace_back(workBody);
                for (auto& th : pool) th.join();
            }
        } else if (progress) {
            progress(nImp, nImp);   // preserve the old progress(0,0) final call
        }
        for (int ii = 0; ii < nImp; ++ii) {
            const auto& im = sc.implicits[ii];
            const isomesh::Mesh& m = meshes[ii];
            // Marching cubes emits no per-vertex UVs, so a skin needs a PROJECTION to land
            // on an implicit. Two independent ones, in priority order (matching the shade
            // pass, which tests tpScale first):
            //   * material `uv triplanar` (triplanarScale > 0) -> sampled from world pos;
            //   * primitive `uv planar|spherical|cylindrical` (im.uvProj) -> the SAME
            //     projectUV() the ray-hit path runs in implicit.h's writeHit, evaluated
            //     per marched vertex here and then barycentrically interpolated. This is
            //     what gallery_rain's marble caps use (`uv planar axis=y`); without it
            //     every cap previewed as its flat pre-texture albedo.
            // Neither present -> no UV source exists, so leave the skin off rather than
            // smear texel (0,0) over the whole surface.
            PTri proto;
            applyMat(proto, im.matId);
            const bool projUV = (im.uvProj != UvProjection::None);
            // A UV-sampled skin needs a projection; a triplanar one does not.
            if (proto.tex >= 0 && proto.triplanarScale <= 0.0 && !projUV) proto.tex = -1;
            // Same reference box and centre the tracer uses, hoisted out of the vertex loop.
            const Aabb& ub = im.uvBoundsSet ? im.uvBounds : im.bounds;
            const Vec3  uctr = (ub.lo + ub.hi) * 0.5;
            // Project once per VERTEX (not per triangle corner): a marched mesh shares
            // vertices between faces, so this is ~6x less work than projecting inline.
            // Patterns and normal maps read (u,v) too — `uv planar` exists on an implicit
            // precisely so pattern/expression materials get coordinates — so any of them
            // being bound is reason enough to project.
            const bool wantUV = projUV &&
                                ((proto.tex >= 0 && proto.triplanarScale <= 0.0) ||
                                 proto.normalTex >= 0 || proto.reflectPat >= 0 ||
                                 proto.emitPat >= 0);
            std::vector<Vec3> pUV;
            if (wantUV) {
                pUV.resize(m.pos.size());
                for (size_t vi = 0; vi < m.pos.size(); ++vi)
                    pUV[vi] = projectUV(m.pos[vi], ub.lo, ub.hi, uctr, im.uvProj, im.uvAxis);
            }
            for (size_t f = 0; f + 2 < m.tri.size(); f += 3) {
                int i0 = m.tri[f], i1 = m.tri[f + 1], i2 = m.tri[f + 2];
                PTri p = proto;
                p.p0 = m.pos[i0]; p.p1 = m.pos[i1]; p.p2 = m.pos[i2];
                p.n0 = m.nrm[i0]; p.n1 = m.nrm[i1]; p.n2 = m.nrm[i2];
                if (!pUV.empty()) {
                    p.uv0 = pUV[i0]; p.uv1 = pUV[i1]; p.uv2 = pUV[i2];
                    // SEAM REPAIR (azimuthal projections only). Spherical/cylindrical u is
                    // an angle normalised to [0,1), so a triangle straddling the -x meridian
                    // gets corners like (0.99, 0.01, 0.02). The ray-hit path never sees this
                    // — it projects AT the hit — but we interpolate, so that triangle would
                    // run u backwards across the entire texture: one garish vertical stripe
                    // of the whole image at the seam. Lift the low corners by one turn so
                    // the triangle stays monotonic (with `wrap repeat` this samples exactly
                    // right; with `clamp` the sliver clamps to the edge texel, still local).
                    if (im.uvProj == UvProjection::Spherical ||
                        im.uvProj == UvProjection::Cylindrical) {
                        double umax = std::max({p.uv0.x, p.uv1.x, p.uv2.x});
                        if (umax - std::min({p.uv0.x, p.uv1.x, p.uv2.x}) > 0.5) {
                            if (umax - p.uv0.x > 0.5) p.uv0.x += 1.0;
                            if (umax - p.uv1.x > 0.5) p.uv1.x += 1.0;
                            if (umax - p.uv2.x > 0.5) p.uv2.x += 1.0;
                        }
                    }
                }
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
            applyMat(p, matId);
            p.uv0 = t.uv0; p.uv1 = t.uv1; p.uv2 = t.uv2;   // UVs are instance-invariant
            out.push_back(p);
        }
    }
    // Tangent bake: precompute the raw dP/dU tangent for every triangle that can shade a
    // normal map — its own material's, or the one its mix's losing child would swap in.
    // Constant over a triangle, so deriving it here (once per SESSION) replaces the shade
    // pass re-deriving it from the edge/UV deltas at every covered pixel of every frame.
    // Runs at the very tail so it sees the FINAL per-vertex UVs (after the azimuthal seam
    // repair above, which lifts individual corners by a full turn and thus changes dUV).
    for (auto& p : out) {
        const bool wantTan = p.normalTex >= 0 ||
            (p.mix >= 0 && p.mix < (int)geom.mixes.size() &&
             geom.mixes[p.mix].b.normalTex >= 0);
        if (wantTan) p.tanRaw = triTangentRaw(p);
    }
    return geom;   // `out` aliases geom.tris; geom.mixes was filled during the material bake
}

// A vertex after transform to camera space, carrying the attributes we interpolate.
struct VtxCS {
    double x, y, z;   // camera-space coords (x=right, y=up, z=forward)
    Vec3   wpos;      // world position (for per-pixel light direction)
    Vec3   wn;        // world normal
    Vec3   uv;        // texture coords (u,v in .x,.y); interpolated for skins
};

// A vertex projected to the raster, with 1/depth for perspective-correct interp.
struct VtxScreen {
    double sx, sy;    // pixel coords (sx in [0,W], sy in [0,H]; sy=0 is image top)
    double invd;      // 1/depth used as the z-buffer key and interp weight
    Vec3   wpos, wn;
    Vec3   uv;        // texture coords (interpolated perspective-correctly for skins)
};

// A screen-space triangle: three projected vertices plus the shared per-triangle
// attributes and a precomputed y-band [iy0,iy1] for O(1) band rejection. Produced once
// by the project-once pass and consumed by the deferred rasterizer, so projection and
// near-plane clipping happen a single time per triangle instead of once per thread.
struct STri {
    VtxScreen v0, v1, v2;
    // Index of the SOURCE PTri rather than a copy of its shading attributes. The shade
    // pass is deferred, so it can fetch colour / texture / pattern / normal-map bindings
    // straight from tris[src] for the one winning fragment. That keeps the rasterizer's
    // innermost loop writing a single int where it used to write a Vec3 + int + float,
    // and means a new per-material preview feature costs no extra G-buffer channel.
    // (This is also how the GPU twin has always worked — see raster_cuda.cu's kShade,
    // which reads its attributes bit-verbatim from the source DPTri.)
    int    src;
    bool   needUV;     // interpolate UVs for this triangle (a skin, pattern or normal map reads them)
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
    std::vector<int>     tri;     // index of the winning source PTri, or -1 (background)
    std::vector<uint8_t> emis;    // 1 where the winning triangle is an emitter
    std::vector<Vec3>    uv;      // interpolated texture coords of the winning surface
};

// A persistent band pool: N workers that sleep on a condition variable and execute one
// broadcast job at a time (each worker gets its own index and derives its slice). The
// frame pipeline runs SEVEN parallel passes back to back (project, zbuf clear, raster,
// shade, two exposure-anchor scans, tonemap); with plain std::thread that was ~7*N
// thread creations PER FRAME — several milliseconds of pure spawn cost plus scheduler
// jitter that showed up directly as the min-to-median spread in -raster-bench. Here a
// pass costs one notify_all and N wakeups instead. Not nestable (run() must not be
// called from inside a job), which the strictly sequential pass structure guarantees.
class BandPool {
public:
    explicit BandPool(int n) : nW_(n < 1 ? 1 : n) {
        workers_.reserve(nW_);
        for (int i = 0; i < nW_; ++i)
            workers_.emplace_back([this, i] {
                uint64_t seen = 0;
                std::unique_lock<std::mutex> lk(m_);
                for (;;) {
                    cvJob_.wait(lk, [&] { return quit_ || gen_ != seen; });
                    if (quit_) return;
                    seen = gen_;
                    const std::function<void(int)>* j = job_;
                    lk.unlock();
                    (*j)(i);
                    lk.lock();
                    if (--pending_ == 0) cvDone_.notify_one();
                }
            });
    }
    ~BandPool() {
        { std::lock_guard<std::mutex> lk(m_); quit_ = true; }
        cvJob_.notify_all();
        for (auto& t : workers_) t.join();
    }
    BandPool(const BandPool&) = delete;
    BandPool& operator=(const BandPool&) = delete;
    int size() const { return nW_; }
    // Run body(workerIndex) on every worker and wait for all of them.
    void run(const std::function<void(int)>& body) {
        std::unique_lock<std::mutex> lk(m_);
        job_ = &body;
        pending_ = nW_;
        ++gen_;
        cvJob_.notify_all();
        cvDone_.wait(lk, [&] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    std::vector<std::thread>       workers_;
    std::mutex                     m_;
    std::condition_variable        cvJob_, cvDone_;
    const std::function<void(int)>* job_ = nullptr;
    uint64_t                       gen_ = 0;
    int                            pending_ = 0;
    int                            nW_ = 1;
    bool                           quit_ = false;
};

// Frame-to-frame scratch for renderFrame. The G-buffer alone is ~85 bytes per pixel
// (~100 MB at 1280x960 counting the HDR accumulator), so allocating and value-filling it
// from scratch EVERY frame — as the old local vectors did — cost more than the entire
// rasterization: freshly mapped pages must be zeroed by the OS and then faulted in,
// twice over per frame. A caller that renders repeatedly (the interactive explorer, a
// flyby, the meter pre-pass) passes one of these to reuse the allocations; only zbuf is
// actually re-cleared per frame (in parallel), because every other channel is written
// before it is read: the shade/encode passes read them solely where zbuf > 0, and any
// pixel with zbuf > 0 had ALL its channels stored by fillTriangleG this same frame.
// The worker pool lives here too, so its threads persist across frames with the buffers.
struct RasterScratch {
    GBuffer                        g;
    std::vector<Vec3>              accum;    // HDR shade target (bg written by the shade pass)
    std::vector<STri>              stris;    // projected triangles (capacity reused)
    std::vector<std::vector<STri>> parts;    // per-thread projection buffers
    std::vector<float>             clearT, milkT;   // see-through products
    std::unique_ptr<BandPool>      pool;     // persistent workers (created on first frame)
};

// --- Watertight coverage: canonical edge functions -------------------------------------
//
// A pixel is inside a triangle when all three edge functions agree in sign. The naive
// version (incremental normalized barycentrics seeded per row from the bbox's left
// column) is NOT watertight: two triangles sharing an edge seed from different `xlo`,
// scale by different 1/area, and derive the third weight as 1-w0-w1, so they compute
// *different* floating-point values for the same shared edge. When a pixel centre lands
// exactly on that edge both can come out a hair negative and BOTH reject — a crack.
// This is not hypothetical: at 800x600 the cornell box's quad diagonals are exactly 45
// degrees, so the edge line passes dead-on through ~220 consecutive pixel centres and
// leaves a visible one-pixel seam of background (measured: 127 holes at 800x600, 0 at
// 801x600 — a pure exact-tie artefact).
//
// The fix is to make both sharers evaluate the SAME expression on the SAME operands, so
// their results are bitwise identical and the sign is guaranteed opposite:
//
//   * the edge's two endpoints are put in a canonical (lexicographic by screen x, then y)
//     order before the coefficients are formed, so both triangles build identical P and
//     Q-P no matter which way round they traverse the edge;
//   * `flip` records whether this triangle traverses the edge in canonical order, and
//     recovers its own signed edge function as flip * E;
//   * E is evaluated as the plain cross product about the canonical endpoint P —
//     `E = dx*(py-Py) - dy*(px-Px)` — never incrementally, so the value at a pixel does
//     not depend on where the scanline span started.
//
// Combined with the triangle's area sign as sf = sign(area) * flip, the two sharers of an
// edge always have OPPOSITE sf (consistent winding flips `flip`; inconsistent winding
// flips `sign(area)` instead), so exactly one of them sees a positive value. The remaining
// exact-zero case is broken deterministically by taking the pixel only when sf > 0, which
// is likewise true for exactly one of the pair. No epsilon, no fixed-point, no top-left
// rule, and it tolerates meshes whose winding disagrees with their vertex normals.
//
// Anchoring at P rather than at the origin matters for CONDITIONING, which is what decides
// how tight the fit is around a shared vertex. The expanded affine form needs the constant
// Px*Qy - Py*Qx, whose magnitude is ~W*H even for a short edge, so its rounding displaces
// the edge line by ~ulp(W*H)/|Q-P|; anchored at P every operand is a local offset instead,
// which on the CUDA side (float) shrinks that displacement by orders of magnitude. Two
// sharers always agree exactly either way, so a shared EDGE is watertight regardless — but
// the three edges meeting at a shared VERTEX are perturbed independently, and a wide
// perturbation can leave a sliver there that no triangle claims.
//
// sf (+-1) is folded straight into the stored dx/dy so the inner loop needs no extra
// multiply. That is still exactly antisymmetric between the two sharers: IEEE negation is
// exact and round-to-nearest is symmetric under negation, so negating every operand
// negates the result bit for bit (true for a contracted FMA as well). v then doubles as
// the unnormalized barycentric weight of the vertex opposite the edge, and w = v / |area|.
struct EdgeFn {
    double Px, Py;      // canonical first endpoint = the evaluation origin
    double dx, dy;      // sf * (Q - P)
    bool   tie;         // this triangle takes the pixel when v is exactly 0
};
inline EdgeFn makeEdge(double Px, double Py, double Qx, double Qy, double s) {
    double flip = 1.0;
    if (Qx < Px || (Qx == Px && Qy < Py)) {                 // canonicalize the endpoint order
        std::swap(Px, Qx); std::swap(Py, Qy); flip = -1.0;
    }
    const double sf = s * flip;
    EdgeFn e;
    e.Px  = Px;  e.Py = Py;
    e.dx  = sf * (Qx - Px);
    e.dy  = sf * (Qy - Py);
    e.tie = sf > 0.0;
    return e;
}

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
    // Watertight coverage (see EdgeFn above): edge i is the one OPPOSITE vertex i, so its
    // value is the unnormalized barycentric weight of that vertex. sf folds in the winding
    // sign so an accepted pixel always has v >= 0, and w = v / |area|.
    const double invA = 1.0 / std::fabs(area);
    const double s = (area > 0.0) ? 1.0 : -1.0;
    const EdgeFn E0 = makeEdge(B.sx, B.sy, C.sx, C.sy, s);
    const EdgeFn E1 = makeEdge(C.sx, C.sy, A.sx, A.sy, s);
    const EdgeFn E2 = makeEdge(A.sx, A.sy, B.sx, B.sy, s);
    const uint8_t triEmis = t.emissive ? 1 : 0;
    for (int y = ylo; y <= yhi; ++y) {
        const double py = y + 0.5;
        const double r0 = E0.dx * (py - E0.Py);   // row constants: identical for both sharers
        const double r1 = E1.dx * (py - E1.Py);
        const double r2 = E2.dx * (py - E2.Py);
        size_t row = (size_t)y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row) {
            const double px = x + 0.5;
            const double v0 = r0 - E0.dy * (px - E0.Px);
            if (v0 < 0.0 || (v0 == 0.0 && !E0.tie)) continue;
            const double v1 = r1 - E1.dy * (px - E1.Px);
            if (v1 < 0.0 || (v1 == 0.0 && !E1.tie)) continue;
            const double v2 = r2 - E2.dy * (px - E2.Px);
            if (v2 < 0.0 || (v2 == 0.0 && !E2.tie)) continue;
            const double w0 = v0 * invA, w1 = v1 * invA, w2 = v2 * invA;
            double invd = w0 * A.invd + w1 * B.invd + w2 * C.invd;   // = 1/depth
            if (invd <= g.zbuf[row]) continue;   // farther than (or equal to) stored
            g.zbuf[row] = (float)invd;
            g.emis[row] = triEmis;               // emitters excluded from the auto-exposure anchor
            // Perspective-correct attribute recovery.
            double d = 1.0 / std::max(invd, 1e-12);
            g.wpos[row]  = (A.wpos * (w0 * A.invd) + B.wpos * (w1 * B.invd) + C.wpos * (w2 * C.invd)) * d;
            g.wn[row]    = (A.wn   * (w0 * A.invd) + B.wn   * (w1 * B.invd) + C.wn   * (w2 * C.invd)) * d;
            g.tri[row]   = t.src;
            if (t.needUV)
                g.uv[row] = (A.uv * (w0 * A.invd) + B.uv * (w1 * B.invd) + C.uv * (w2 * C.invd)) * d;
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
    // Same watertight coverage as fillTriangleG. It matters even more here: the clear pass
    // MULTIPLIES into clearT/milkT, so a shared edge covered by both sharers would darken a
    // seam line twice, and one covered by neither would leave a hairline of un-tinted glass.
    const double invA = 1.0 / std::fabs(area);
    const double s = (area > 0.0) ? 1.0 : -1.0;
    const EdgeFn E0 = makeEdge(B.sx, B.sy, C.sx, C.sy, s);
    const EdgeFn E1 = makeEdge(C.sx, C.sy, A.sx, A.sy, s);
    const EdgeFn E2 = makeEdge(A.sx, A.sy, B.sx, B.sy, s);
    const float tau = (float)clarity;
    for (int y = ylo; y <= yhi; ++y) {
        const double py = y + 0.5;
        const double r0 = E0.dx * (py - E0.Py);
        const double r1 = E1.dx * (py - E1.Py);
        const double r2 = E2.dx * (py - E2.Py);
        size_t row = (size_t)y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row) {
            const double px = x + 0.5;
            const double v0 = r0 - E0.dy * (px - E0.Px);
            if (v0 < 0.0 || (v0 == 0.0 && !E0.tie)) continue;
            const double v1 = r1 - E1.dy * (px - E1.Px);
            if (v1 < 0.0 || (v1 == 0.0 && !E1.tie)) continue;
            const double v2 = r2 - E2.dy * (px - E2.Px);
            if (v2 < 0.0 || (v2 == 0.0 && !E2.tie)) continue;
            const double w0 = v0 * invA, w1 = v1 * invA, w2 = v2 * invA;
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
    s.wpos = v.wpos; s.wn = v.wn; s.uv = v.uv;
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

// Exact k-th smallest of n NON-NEGATIVE doubles — the same value std::nth_element would
// leave at [k] — found with two parallel O(n) scans instead of nth_element's serial
// partition recursion (which was the auto-exposure anchor's dominant cost: ~4 ms alone
// for a 1.2-Mpixel frame). Non-negative IEEE doubles order monotonically as their raw
// bit patterns, so a histogram over the TOP 16 BITS partitions the values into 65536
// order-preserving buckets: find the bucket holding rank k by prefix sum, collect just
// that bucket's members (typically a few dozen), and select within them. Selection is by
// VALUE over a multiset, so neither the pack order nor tie order can change the result.
// May permute v[] (the small-n path selects in place), exactly as nth_element did.
inline double selectKthNonNeg(double* v, size_t n, size_t k,
                              BandPool* pool, int nThreads) {
    if (n == 0) return 0.0;
    if (k >= n) k = n - 1;
    constexpr int B = 1 << 16;
    if (!pool || nThreads <= 1 || n < ((size_t)1 << 15)) {   // small frames: not worth scans
        std::nth_element(v, v + k, v + n);
        return v[k];
    }
    const int nB = pool->size();
    static thread_local std::vector<uint32_t> s_hist;        // per-worker histogram slabs
    if (s_hist.size() < (size_t)nB * B) s_hist.resize((size_t)nB * B);
    uint32_t* histBase = s_hist.data();
    const size_t chunk = (n + nB - 1) / nB;
    pool->run([&](int ti) {
        uint32_t* h = histBase + (size_t)ti * B;
        std::fill(h, h + B, 0u);                             // also zeroes idle workers' slabs
        size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
        for (size_t i = a; i < b; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &v[i], sizeof bits);
            ++h[(int)(bits >> 48)];
        }
    });
    // Merge the per-worker histograms (parallel over bucket ranges), then a serial prefix
    // scan over the 65536 merged counts to locate the bucket holding rank k.
    static thread_local std::vector<uint64_t> s_merged;
    if (s_merged.size() < (size_t)B) s_merged.resize(B);
    uint64_t* merged = s_merged.data();
    {
        const int bchunk = (B + nB - 1) / nB;
        pool->run([&](int ti) {
            int a = ti * bchunk, b = std::min(B, a + bchunk);
            for (int bi = a; bi < b; ++bi) {
                uint64_t c = 0;
                for (int w = 0; w < nB; ++w) c += histBase[(size_t)w * B + bi];
                merged[bi] = c;
            }
        });
    }
    size_t cum = 0; int bkt = 0; size_t inBkt = 0;
    for (int bi = 0; bi < B; ++bi) {
        if (cum + merged[bi] > k) { bkt = bi; inBkt = (size_t)merged[bi]; break; }
        cum += merged[bi];
    }
    // Collect the target bucket's members: each worker owns a disjoint segment of the
    // candidate buffer sized by its own histogram count, so no locking.
    static thread_local std::vector<double> s_cand;
    if (s_cand.size() < inBkt) s_cand.resize(inBkt);
    double* cand = s_cand.data();
    static thread_local std::vector<size_t> s_boff;
    if (s_boff.size() < (size_t)nB) s_boff.resize(nB);
    size_t* boff = s_boff.data();
    {
        size_t o = 0;
        for (int ti = 0; ti < nB; ++ti) { boff[ti] = o; o += histBase[(size_t)ti * B + bkt]; }
    }
    pool->run([&](int ti) {
        size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
        double* dst = cand + boff[ti];
        for (size_t i = a; i < b; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &v[i], sizeof bits);
            if ((int)(bits >> 48) == bkt) *dst++ = v[i];
        }
    });
    const size_t kk = k - cum;
    std::nth_element(cand, cand + kk, cand + inBkt);
    return cand[kk];
}

// sRGB gamma lookup table shared by the CPU tonemap below and the CUDA rasterizer's
// on-device tonemap (raster_cuda.cu uploads these exact bytes once): the tone map clamps
// each channel to [0,1] before encoding, and gamma is monotonic (anything >=1 saturates
// to 255), so a 4096-entry LUT over [0,1] replaces three std::pow calls per pixel with a
// table read + round — and having BOTH backends index the same table is what keeps their
// encoded bytes identical.
inline const std::array<uint8_t, 4097>& srgbLut8() {
    static const std::array<uint8_t, 4097> t = [] {
        std::array<uint8_t, 4097> a{};
        for (int i = 0; i <= 4096; ++i)
            a[i] = (uint8_t)std::clamp(srgbGamma(i / 4096.0) * 255.0 + 0.5, 0.0, 255.0);
        return a;
    }();
    return t;
}

// Shared exposure + tone-map tail (the back half of renderFrame). Given a per-pixel
// HDR `accum` buffer that has already been shaded (background pixels hold the unlit bg
// tint), a `zbuf` hit key (>0 where a surface was drawn), an `emis` mask, and the
// optional see-through transmittance/milk products, this applies the p99 auto-exposure
// anchor and the sRGB tone map exactly as filmToRgb8 does, returning W*H*3 RGB8 (row 0 =
// top). The CUDA rasterizer no longer calls this (it runs a device twin of this exact
// maths — same p99 order statistic, same double-precision tonemap, same srgbLut8()
// table — verified byte-identical); it remains the single host implementation, and any
// change here must be mirrored in raster_cuda.cu's expose kernels.
// Template core: `pixel(i)` must return the exact double-precision Vec3 colour of
// pixel i (kept templated so a non-Vec3 HDR buffer can be adapted without a copy).
template <class FetchVec3>
inline std::vector<uint8_t> exposeAndEncodeT(
        FetchVec3&& pixel, const float* zbuf, const uint8_t* emis,
        int W, int H, int nThreads,
        double expComp, bool autoExpose, double* lockAnchor,
        bool seeThrough, const float* clearT, const float* milkT,
        const Vec3& milkColor, BandPool* pool = nullptr) {
    const size_t N = (size_t)W * H;
    if (nThreads < 1) nThreads = 1;
    if (pool && pool->size() != nThreads) pool = nullptr;   // stale pool: fall back to spawning
    auto parallelFor = [&](size_t n, const std::function<void(size_t, size_t)>& body) {
        if (n == 0) return;
        if (nThreads == 1) { body(0, n); return; }
        size_t chunk = (n + nThreads - 1) / nThreads;
        if (pool) {
            pool->run([&](int ti) {
                size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
                if (a < b) body(a, b);
            });
            return;
        }
        std::vector<std::thread> tp;
        for (int ti = 0; ti < nThreads; ++ti) {
            size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
            if (a >= b) break;
            tp.emplace_back(body, a, b);
        }
        for (auto& th : tp) th.join();
    };

    // Auto-exposure anchor (mirror filmToRgb8): map the 99th-percentile luminance of the
    // lit surfaces to ~0.9. Background (unhit) pixels are excluded so an empty frame
    // margin can't skew the anchor; emitters are excluded too so the *subject* drives the
    // exposure (they just clip to white, as in the real render, instead of dragging the
    // anchor down when a large light fills the frame). Absolute EV (autoExpose=false)
    // bypasses this so aperture/power brightness differences survive into the preview.
    //
    // The collect runs banded across threads into a persistent scratch buffer: band ti
    // fills [off[ti], off[ti]+cnt[ti]) with its qualifying pixels in row-major order, so
    // the packed buffer holds the exact multiset a serial scan would produce and the
    // k-th-smallest selection (selectKthNonNeg) yields the identical 99th-percentile value.
    double eAuto = 1.0;
    if (autoExpose) {
        if (lockAnchor && *lockAnchor > 0.0) {
            eAuto = *lockAnchor;                    // reuse the path's locked anchor
        } else {
            static thread_local std::vector<double> s_lum;   // persistent scratch (per calling thread)
            size_t total = 0;
            if (nThreads == 1) {
                if (s_lum.size() < N) s_lum.resize(N);
                double* dst = s_lum.data();
                for (size_t i = 0; i < N; ++i) {
                    if (zbuf[i] <= 0.0f || emis[i]) continue;   // skip background + emitters
                    const Vec3 c = pixel(i);
                    *dst++ = std::max({c.x, c.y, c.z, 0.0});
                }
                total = (size_t)(dst - s_lum.data());
            } else {
                const size_t bands = (size_t)nThreads;
                const size_t chunk = (N + bands - 1) / bands;
                std::vector<size_t> cnt(bands, 0), off(bands, 0);
                // Band bodies for the two scans; dispatched on the persistent pool when
                // one was passed in, else on freshly spawned threads (identical split
                // either way, so the packed order — and thus the anchor — is unchanged).
                auto countBand = [&](size_t ti) {
                    size_t a = ti * chunk, b = std::min(N, a + chunk);
                    size_t c = 0;
                    for (size_t i = a; i < b; ++i)
                        if (!(zbuf[i] <= 0.0f || emis[i])) ++c;
                    cnt[ti] = c;
                };
                // NB: s_lum is thread_local, and lambdas do NOT capture thread-locals —
                // each worker would resolve the name to its own empty instance. Hand the
                // workers a plain pointer to *this* thread's buffer instead.
                auto packBand = [&](size_t ti, double* lumBase) {
                    size_t a = ti * chunk, b = std::min(N, a + chunk);
                    double* dst = lumBase + off[ti];
                    for (size_t i = a; i < b; ++i) {
                        if (zbuf[i] <= 0.0f || emis[i]) continue;
                        const Vec3 c = pixel(i);
                        *dst++ = std::max({c.x, c.y, c.z, 0.0});
                    }
                };
                auto runBands = [&](const std::function<void(size_t)>& body) {
                    if (pool) {
                        pool->run([&](int ti) {
                            size_t a = (size_t)ti * chunk;
                            if (a < N) body((size_t)ti);
                        });
                        return;
                    }
                    std::vector<std::thread> tp;
                    for (size_t ti = 0; ti < bands; ++ti) {
                        size_t a = ti * chunk;
                        if (a >= N) break;
                        tp.emplace_back(body, ti);
                    }
                    for (auto& th : tp) th.join();
                };
                runBands(countBand);                    // pass 1: count per band
                for (size_t ti = 0; ti < bands; ++ti) { off[ti] = total; total += cnt[ti]; }
                if (s_lum.size() < total) s_lum.resize(total);
                double* lumBase = s_lum.data();
                runBands([&](size_t ti) { packBand(ti, lumBase); });   // pass 2: pack
            }
            if (total > 0) {
                // Only the 99th-percentile order statistic matters, so select instead of
                // sorting — and in parallel (radix-bucket scan) instead of nth_element's
                // serial partition recursion.
                size_t k = (size_t)(0.99 * (total - 1));
                double p99 = selectKthNonNeg(s_lum.data(), total, k, pool, nThreads);
                eAuto = (p99 > 0.0) ? 0.9 / p99 : 1.0;
            }
            if (lockAnchor) *lockAnchor = eAuto;    // first frame sets the anchor
        }
    }
    const double finalExp = eAuto * expComp;

    const std::array<uint8_t, 4097>& kSrgbLut = srgbLut8();
    auto encode = [&](double c) -> uint8_t {
        if (c <= 0.0) return kSrgbLut[0];
        if (c >= 1.0) return 255;
        return kSrgbLut[(int)(c * 4096.0 + 0.5)];
    };

    // Tone map: exposed hit pixels through sRGB gamma; background tint left unexposed.
    std::vector<uint8_t> img(N * 3);
    parallelFor(N, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            Vec3 c = pixel(i);
            if (zbuf[i] > 0.0f) c = c * finalExp;   // hit pixels get the exposure
            if (seeThrough) {                          // composite clear glass (display-linear)
                float T = clearT[i], mt = milkT[i];
                if (T < 1.0f || mt < 1.0f)
                    c = c * (double)T + milkColor * (1.0 - (double)mt);
            }
            img[i * 3 + 0] = encode(c.x);
            img[i * 3 + 1] = encode(c.y);
            img[i * 3 + 2] = encode(c.z);
        }
    });
    return img;
}

// Vector-based wrapper (the original signature): the CPU rasterizer and any other
// Vec3-buffer caller go through here; it simply adapts to the template core above.
inline std::vector<uint8_t> exposeAndEncode(
        const std::vector<Vec3>& accum, const std::vector<float>& zbuf,
        const std::vector<uint8_t>& emis, int W, int H, int nThreads,
        double expComp, bool autoExpose, double* lockAnchor,
        bool seeThrough, const std::vector<float>& clearT, const std::vector<float>& milkT,
        const Vec3& milkColor, BandPool* pool = nullptr) {
    const Vec3* A = accum.data();
    return exposeAndEncodeT([A](size_t i) { return A[i]; },
                            zbuf.data(), emis.data(), W, H, nThreads,
                            expComp, autoExpose, lockAnchor, seeThrough,
                            clearT.empty() ? nullptr : clearT.data(),
                            milkT.empty()  ? nullptr : milkT.data(),
                            milkColor, pool);
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
inline std::vector<uint8_t> renderFrame(const PreviewGeom& geom, const Camera& cam,
                                        int W, int H, const PreviewLight& light,
                                        int nThreads, double exposure = 1.0,
                                        bool autoExpose = true, double* lockAnchor = nullptr,
                                        bool seeThrough = false, double glassClarity = 0.85,
                                        const Scene* scenePtr = nullptr,
                                        RasterScratch* scratch = nullptr) {
    // Geometry and its side tables arrive together (a PTri's `mix` index is only meaningful
    // against the mixes built alongside it), then are aliased for the passes below.
    const std::vector<PTri>& tris  = geom.tris;
    const std::vector<PMix>& mixes = geom.mixes;
    // Frame-to-frame buffer reuse (see RasterScratch): a caller that renders repeatedly
    // passes a scratch; a one-shot caller gets a frame-local one and behaves as before.
    RasterScratch localScratch;
    RasterScratch& S = scratch ? *scratch : localScratch;
    // The shade pass needs more of the Scene than just its textures: scalar patterns are
    // evaluated per pixel and the pattern VM reads the scene's `grid:`/`scatter:` tables
    // through bindPatScene. Passing the Scene (rather than a texture vector) is what lets
    // `emit_map pattern:` mask an emitter instead of the whole surface glowing.
    const std::vector<Texture>* textures = scenePtr ? &scenePtr->textures : nullptr;
    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    const double EMIS_BOOST = 4.0;    // emitters read as bright light sources (clip to white)
    const Vec3 bg{0.06, 0.07, 0.09};                    // background tint (unlit, unexposed)
    const size_t N = (size_t)W * H;

    const double zn = 1e-3;   // near plane (camera-forward) for rectilinear clipping
    const bool rect = (cam.projection == CAM_RECTILINEAR);

    if (nThreads < 1) nThreads = 1;

    // Persistent workers for every parallel pass below (see BandPool). Created on the
    // first frame and reused for the rest of the session; recreated only if the caller
    // changes its thread count. Even a one-shot call (frame-local scratch) wins: one
    // pool spawn serves all seven passes instead of each spawning its own threads.
    if (nThreads > 1 && (!S.pool || S.pool->size() != nThreads))
        S.pool = std::make_unique<BandPool>(nThreads);
    BandPool* pool = (nThreads > 1) ? S.pool.get() : nullptr;

    // Tiny parallel-for over [0,n): splits into nThreads contiguous chunks (the same
    // partition the old spawn-per-pass version used, so band ownership is unchanged).
    // Used by the shading + tone-map passes (each pixel is independent, no locking).
    auto parallelFor = [&](size_t n, const std::function<void(size_t, size_t)>& body) {
        if (n == 0) return;
        if (!pool) { body(0, n); return; }
        size_t chunk = (n + nThreads - 1) / nThreads;
        pool->run([&](int ti) {
            size_t a = (size_t)ti * chunk, b = std::min(n, a + chunk);
            if (a < b) body(a, b);
        });
    };

    // -- Pass 1: project every triangle ONCE (parallel over the triangle list). Each
    // thread clips + projects its slice into a local STri buffer; the buffers are then
    // concatenated. This removes the old per-thread redundancy where every rasterizer
    // band re-projected the entire scene (an nThreads-fold projection cost).
    auto projectRange = [&](size_t a, size_t b, std::vector<STri>& out) {
        auto toCS = [&](const Vec3& P, const Vec3& Nn, const Vec3& UV) -> VtxCS {
            Vec3 d = P - cam.eye; VtxCS c;
            c.x = dot(d, cam.u); c.y = dot(d, cam.v); c.z = dot(d, cam.w);
            c.wpos = P; c.wn = Nn; c.uv = UV; return c;
        };
        auto push = [&](const VtxScreen& s0, const VtxScreen& s1, const VtxScreen& s2,
                        int src, bool needUV, bool emis, bool clr) {
            double lo = std::min({s0.sy, s1.sy, s2.sy});
            double hi = std::max({s0.sy, s1.sy, s2.sy});
            int iy0 = std::max(0, (int)std::floor(lo));
            int iy1 = std::min(H - 1, (int)std::ceil(hi));
            if (iy0 > iy1) return;
            out.push_back(STri{s0, s1, s2, src, needUV, emis, clr, iy0, iy1});
        };
        for (size_t ti = a; ti < b; ++ti) {
            const PTri& t = tris[ti];
            // Two-sided shading, decided ONCE for the whole triangle. A surface whose
            // normals point away from the eye (the cornell box's walls are wound outward
            // and viewed from inside) must be lit as if they faced us; but the test has to
            // be per-TRIANGLE, not per-pixel. Done per pixel on the interpolated normal it
            // inverts a 1-px band at every silhouette, because there dot(N,V) grazes
            // through zero while the surface is still genuinely front-facing.
            // A triangle counts as back-facing only when ALL THREE vertices agree: a
            // silhouette triangle straddles the horizon (some vertices front, some back)
            // and must keep its smooth normals, while geometry truly seen from behind has
            // every vertex facing away and still flips exactly as it did before.
            const bool back = dot(t.n0, cam.eye - t.p0) < 0.0 &&
                              dot(t.n1, cam.eye - t.p1) < 0.0 &&
                              dot(t.n2, cam.eye - t.p2) < 0.0;
            // Interpolate UVs when ANY per-pixel binding reads them: an image skin, a
            // normal map, or a scalar pattern (patterns get u/v in their context, and a
            // `[0 1](u)` ramp is exactly a UV read). Triplanar skins sample from world
            // position instead, but a pattern on the same material may still want UVs.
            // A per-hit `mix` ALWAYS needs them: the mask is sampled at (u,v) whether it
            // is a pattern (u/v live in its context) or a scalar texture, and the child
            // payload it may swap in can carry a skin/normal map/pattern of its own.
            const bool needUV = (t.tex >= 0 && t.triplanarScale <= 0.0) ||
                                t.normalTex >= 0 || t.reflectPat >= 0 || t.emitPat >= 0 ||
                                t.mix >= 0;
            VtxCS cs[3] = { toCS(t.p0, back ? -t.n0 : t.n0, t.uv0),
                            toCS(t.p1, back ? -t.n1 : t.n1, t.uv1),
                            toCS(t.p2, back ? -t.n2 : t.n2, t.uv2) };
            if (rect) {
                VtxCS poly[8]; int np = 0;
                auto emit = [&](const VtxCS& a2){ if (np < 8) poly[np++] = a2; };
                auto lerpV = [&](const VtxCS& a2, const VtxCS& b2, double s) -> VtxCS {
                    VtxCS r; r.x=a2.x+(b2.x-a2.x)*s; r.y=a2.y+(b2.y-a2.y)*s; r.z=a2.z+(b2.z-a2.z)*s;
                    r.wpos=a2.wpos+(b2.wpos-a2.wpos)*s; r.wn=a2.wn+(b2.wn-a2.wn)*s;
                    r.uv=a2.uv+(b2.uv-a2.uv)*s; return r;
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
                    push(sc0, sc1, sc2, (int)ti, needUV, t.emissive, t.clear);
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
                push(sc0, sc1, sc2, (int)ti, needUV, t.emissive, t.clear);
            }
        }
    };

    std::vector<STri>& stris = S.stris;
    stris.clear();                               // keeps capacity across frames
    {
        int pT = std::min<int>(nThreads, std::max<size_t>(1, tris.size()));
        if (pT <= 1 || !pool) {
            stris.reserve(tris.size());
            projectRange(0, tris.size(), stris);
        } else {
            std::vector<std::vector<STri>>& parts = S.parts;
            if ((int)parts.size() < pT) parts.resize(pT);
            for (auto& p : parts) p.clear();     // ALL of them (keeps capacity): a stale
                                                 // buffer past this frame's pT must not
                                                 // leak into the concatenation below
            size_t chunk = (tris.size() + pT - 1) / pT;
            pool->run([&](int ti) {
                if (ti >= pT) return;
                size_t a = (size_t)ti * chunk, b = std::min(tris.size(), a + chunk);
                if (a >= b) return;
                parts[ti].reserve(b - a);
                projectRange(a, b, parts[ti]);
            });
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

    // Dispatch a per-row-band body across nThreads (each band owns disjoint rows -> no
    // locking; the same row split the old spawn-per-pass version used).
    auto dispatchBands = [&](const std::function<void(int,int)>& body) {
        if (!pool) { body(0, H); return; }
        int rows = (H + nThreads - 1) / nThreads;
        pool->run([&](int ti) {
            int y0 = ti * rows, y1 = std::min(H, y0 + rows);
            if (y0 < y1) body(y0, y1);
        });
    };

    // -- Pass 2: deferred G-buffer rasterization, parallel by horizontal row-bands. Each
    // band owns rows [y0,y1) so bands never touch the same pixel (no locking). Triangles
    // whose y-span misses the band are skipped in O(1) via the precomputed iy0/iy1.
    //
    // Only zbuf is cleared (in parallel — a serial fill of these buffers used to dominate
    // the whole frame). Every other channel is write-before-read: the shade and encode
    // passes read them exclusively where zbuf > 0, and a pixel with zbuf > 0 had all its
    // channels stored by fillTriangleG this same frame (uv whenever its triangle's
    // bindings read UVs, which is exactly when the shade pass samples them). resize()
    // value-initializes only on growth, so steady-state frames touch nothing here.
    GBuffer& g = S.g;
    g.zbuf.resize(N);
    g.wpos.resize(N);
    g.wn.resize(N);
    g.tri.resize(N);
    g.emis.resize(N);
    g.uv.resize(N);
    parallelFor(N, [&](size_t a, size_t b) {
        std::fill(g.zbuf.begin() + a, g.zbuf.begin() + b, 0.0f);
    });
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
    // These ARE read at every pixel by the encode pass, so both get a real fill (parallel,
    // reusing the scratch allocation).
    std::vector<float>& clearT = S.clearT;
    std::vector<float>& milkT  = S.milkT;
    if (!seeThrough) { clearT.clear(); milkT.clear(); }
    if (seeThrough) {
        clearT.resize(N);
        milkT.resize(N);
        parallelFor(N, [&](size_t a, size_t b) {
            std::fill(clearT.begin() + a, clearT.begin() + b, 1.0f);
            std::fill(milkT.begin() + a, milkT.begin() + b, 1.0f);
        });
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
    // The background tint is written HERE (rather than pre-filling the whole buffer
    // serially before the pass): every pixel gets exactly one store either way, so the
    // pre-fill was pure extra traffic.
    std::vector<Vec3>& accum = S.accum;
    accum.resize(N);
    parallelFor(N, [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            if (g.zbuf[i] <= 0.0f) { accum[i] = bg; continue; }   // background tint
            const int si = g.tri[i];
            if (si < 0 || si >= (int)tris.size()) { accum[i] = bg; continue; }
            const PTri& pt = tris[si];
            // The unit shading normal, needed by (almost) every path below — normalized
            // ONCE instead of separately by the pattern context and the lighting model.
            const Vec3 N0 = normalize(g.wn[i]);
            // The PatCtx the tracer builds at a hit (world point, oriented normal, u, v —
            // and fieldVal 0, which is exact here because an isosurface's marched vertices
            // lie on the level set). Built at most ONCE per pixel and only when something
            // actually needs it, since most surfaces have neither a mix mask nor a pattern.
            PatCtx pc;
            bool   pcReady = false;
            auto   ctx = [&]() -> const PatCtx& {
                if (!pcReady) {
                    pc = makePatCtx(g.wpos[i], 0.0, N0, g.uv[i].x, g.uv[i].y);
                    bindPatScene(pc, *scenePtr);
                    pcReady = true;
                }
                return pc;
            };
            // A `weight_map`-driven two-child mix selects a WHOLE material payload per
            // pixel — albedo, skin and pattern drives together — so resolve it before
            // reading any of them. Hard threshold at 0.5, exactly as mixResolveDominant()
            // (same pattern/texture evaluation, same clamp). The PatCtx is only built for
            // a PATTERN mask; a texture mask samples straight from the interpolated UV.
            const PShade* sh = &pt;
            if (scenePtr && pt.mix >= 0 && pt.mix < (int)mixes.size()) {
                const PMix& mx = mixes[pt.mix];
                double wt = 0.0;
                if (mx.weightPat >= 0 && mx.weightPat < (int)scenePtr->patterns.size())
                    wt = scenePtr->patterns[mx.weightPat].eval(ctx());
                else if (mx.weightTex >= 0 && mx.weightTex < (int)scenePtr->textures.size())
                    wt = scenePtr->textures[mx.weightTex].scalarAt(g.uv[i].x, g.uv[i].y);
                wt = (wt < 0.0) ? 0.0 : (wt > 1.0 ? 1.0 : wt);
                if (wt < 0.5) sh = &mx.b;
            }
            Vec3 col = sh->color;
            // Image skin: replace the flat albedo with the texture's linear RGB, sampled
            // either at the interpolated per-vertex UV or by world triplanar projection.
            if (textures && sh->tex >= 0 && sh->tex < (int)textures->size()) {
                const Texture& tx = (*textures)[sh->tex];
                col = (sh->triplanarScale > 0.0)
                    ? tx.sampleRgbTriplanar(g.wpos[i], g.wn[i], sh->triplanarScale)
                    : tx.sampleRgb(g.uv[i].x, g.uv[i].y);
            }
            // Scalar pattern drives. A bound pattern multiplies its slot and is clamped to
            // [0,1], mirroring slotPatMul.
            if (scenePtr && (sh->reflectPat >= 0 || sh->emitPat >= 0)) {
                const int slot = g.emis[i] ? sh->emitPat : sh->reflectPat;
                if (slot >= 0 && slot < (int)scenePtr->patterns.size()) {
                    double p = scenePtr->patterns[slot].eval(ctx());
                    col = col * (p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p));
                }
            }
            if (g.emis[i]) { accum[i] = col * EMIS_BOOST; continue; }  // raw emitter radiance
            // No two-sided flip here: it is decided ONCE PER TRIANGLE at projection time
            // (see projectRange). Testing the smoothly-interpolated normal per pixel used
            // to invert it in a 1-px band at every silhouette, where dot(N,V) legitimately
            // grazes through zero — that produced dark speckles on the sphere's rim.
            Vec3 N3 = N0;
            // Tangent-space normal map. The rasterizer has no per-vertex tangents, so the
            // frame comes from the triangle's UV gradient — precomputed by tessellate()'s
            // tangent bake (PTri::tanRaw), since it is constant over the triangle; only
            // the Gram-Schmidt against the interpolated normal is per-pixel work. A zero
            // tanRaw means degenerate/absent UVs: fall back to a stable basis about N,
            // exactly as before. Perturbing here (not in the G-buffer) keeps the pass-2
            // inner loop untouched.
            if (textures && sh->normalTex >= 0 && sh->normalTex < (int)textures->size()) {
                const Texture& nx = (*textures)[sh->normalTex];
                if (nx.valid()) {
                    Vec3 T = pt.tanRaw;
                    if (!(dot(T, T) > 0.0)) {
                        Vec3 ax = (std::fabs(N3.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
                        T = cross(ax, N3);
                    }
                    T = T - N3 * dot(N3, T);     // re-orthogonalize against the shading normal
                    double tl = std::sqrt(dot(T, T));
                    if (tl > 1e-12) {
                        T = T * (1.0 / tl);
                        Vec3 B3 = cross(N3, T);
                        Vec3 tn = nx.sampleNormalTS(g.uv[i].x, g.uv[i].y);
                        double s3 = sh->normalStrength;
                        Vec3 pert = T * (tn.x * s3) + B3 * (tn.y * s3) + N3 * tn.z;
                        double pl = std::sqrt(dot(pert, pert));
                        if (pl > 1e-12) N3 = pert * (1.0 / pl);
                    }
                }
            }
            Vec3 V = normalize(cam.eye - g.wpos[i]);     // toward camera
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

    // Auto-exposure + sRGB tone map: shared with the CUDA rasterizer (see exposeAndEncode),
    // so both backends anchor and encode identically. The see-through buffers are empty when
    // !seeThrough and simply ignored by the helper in that case. Rides the same worker pool
    // as the passes above (its three scans used to spawn their own threads each).
    return exposeAndEncode(accum, g.zbuf, g.emis, W, H, nThreads, expComp, autoExpose,
                           lockAnchor, seeThrough, clearT, milkT, kMilkColor, pool);
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
