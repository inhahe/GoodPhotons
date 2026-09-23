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
#include <string>
#include <cstdio>
#include <cstdlib>
#include "parallel.h"
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
    // RASTER-PBR. `rough < 0` means "no specular lobe" and shades exactly as before, so every
    // diffuse material in every existing scene is untouched. `f0` is the normal-incidence
    // reflectance -- for a metal that IS its measured reflectance (which is why a gold preview
    // must tint its highlight gold, not white), for a dielectric a small achromatic value.
    double rough  = -1.0;
    Vec3   f0{0, 0, 0};
    int    roughPat = -1;    // `roughness pattern:` — the map the header says is ignored today
    int    roughTex = -1;    // `roughness texture:`
    // Per-CROSSING RGB transmittance of a clear surface, from the material itself (see
    // clearTintOf). White for anything that states no colour, so a plain window behaves
    // exactly as it did when this was one global scalar. Meaningless unless `clear`.
    Vec3 clearTint{1, 1, 1};
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
    // Per-vertex COLOUR (linear RGB), or hasVcol=false. The rasterizer is an RGB pipeline,
    // so it consumes the mesh's vertex colours directly — no Jakob-Hanika lift, which is
    // the whole reason Scene::vertColors stores RGB rather than fitted coefficients: the
    // two backends want different things from the same numbers, and RGB is what both can
    // start from.
    bool hasVcol = false;
    Vec3 vc0{1, 1, 1}, vc1{1, 1, 1}, vc2{1, 1, 1};
};

// Tessellated preview geometry plus the side tables its triangles index. Bundled so the
// two cannot be handed around separately and fall out of sync: a PTri's `mix` index is
// only meaningful against the `mixes` built in the same tessellate() call.
struct PreviewGeom {
    std::vector<PTri> tris;
    std::vector<PMix> mixes;
    // Half-diagonal of the tessellated bounds. The see-through pass integrates absorption
    // over PATH LENGTH, so it needs a length scale, and taking it from the geometry itself
    // means the CPU and GPU backends cannot disagree about it (a Scene pointer is optional
    // on the render entry points; this is not) and a 2 cm ring behaves like a 40 m building.
    double radius = 1.0;
    void clear() { tris.clear(); mixes.clear(); radius = 1.0; }
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
// The RGB transmittance ONE crossing of a clear surface applies, derived from the
// material rather than from a single global dial — so a red filter tints what is behind
// it red, and a dense one darkens it more than a clear window does.
//
// Where the number comes from depends on what the material actually states:
//
//   * `Filter` / `DiffuseTransmit` carry `transmit`, a DIMENSIONLESS T(lambda) in [0,1].
//     That is a transmittance already, so both the hue and the magnitude are real and are
//     used as-is.
//   * `Dielectric` / `ThinFilm` colour their interior with `absorb`, a Beer-Lambert
//     coefficient per unit LENGTH. Turning that into a transmittance needs a thickness,
//     and an order-independent rasterizer never pairs a front face with the back face it
//     belongs to — there is no thickness to raise it to. So only the HUE is taken
//     (normalised so the strongest channel is 1) and `-glass-clarity` goes on setting how
//     much each crossing dims. Glass with no absorption is colourless, which is the
//     physically honest answer rather than a guess.
//
// The exponential is taken in WAVELENGTH space and converted afterwards, not the other
// way round: exp() of an RGB-collapsed coefficient is not the RGB of the exponential, and
// the difference is exactly the saturation of a strongly absorbing glass.
inline Vec3 clearTintOf(const Material& m) {
    // WHITE BALANCE. spectrumToLinearRgb of a FLAT spectrum is not neutral: the CIE
    // integral of an equal-energy stimulus lands at linear sRGB ~(1.198, 0.950, 0.908),
    // which is why ftrace's own self-test prints that number. Feeding a transmittance
    // through it raw would give a perfectly colourless window a warm cast — the tint of
    // the illuminant model, not of the glass. Dividing by the flat response measures
    // every transmittance AGAINST no absorption, so `absorb 0` comes out exactly white
    // and a 0.5 grey gel comes out exactly 0.5 grey.
    static const Vec3 kFlat = spectrumToLinearRgb(constantSpectrum(1.0));
    auto balance = [](const Vec3& v) {
        return Vec3{kFlat.x > 1e-9 ? v.x / kFlat.x : v.x,
                    kFlat.y > 1e-9 ? v.y / kFlat.y : v.y,
                    kFlat.z > 1e-9 ? v.z / kFlat.z : v.z};
    };
    auto clamp01 = [](Vec3 v) {
        v.x = std::min(1.0, std::max(0.0, v.x));
        v.y = std::min(1.0, std::max(0.0, v.y));
        v.z = std::min(1.0, std::max(0.0, v.z));
        return v;
    };
    if (m.type == MatType::Filter || m.type == MatType::DiffuseTransmit) {
        const Vec3 t = clamp01(balance(spectrumToLinearRgb(m.transmit)));
        // A material that transmits nothing at all is not tinted glass, it is opaque —
        // leave it white and let the (unchanged) clarity dial dim it, rather than
        // compositing a black hole over the background.
        if (t.x + t.y + t.z <= 1e-6) return Vec3{1, 1, 1};
        return t;
    }
    // If the asset stated the thickness its absorption was authored against
    // (Material::absorbRefDist — glTF's KHR_materials_volume attenuationDistance), believe
    // it: the transmittance is then a real number, not just a hue, and a dense onyx reads
    // near-black instead of white. HALF the distance goes into each crossing, because the
    // authored figure describes light traversing the body once and a closed solid is
    // crossed twice — front and back — so splitting it means the pair multiplies back to
    // what was authored.
    const double d = (m.absorbRefDist > 0.0) ? m.absorbRefDist * 0.5 : 0.0;
    if (d > 0.0) {
        return clamp01(balance(spectrumToLinearRgb(
            [&m, d](double lam) { return std::exp(-std::max(0.0, m.absorb(lam)) * d); })));
    }
    // Nobody said how thick: take the hue only and leave the magnitude to -glass-clarity.
    const Vec3 raw = balance(spectrumToLinearRgb(
        [&m](double lam) { return std::exp(-std::max(0.0, m.absorb(lam))); }));
    // Normalise BEFORE clamping: clipping first would distort the hue of a strongly
    // coloured glass by flattening whichever channel ran over.
    const double mx = std::max({raw.x, raw.y, raw.z});
    if (!(mx > 1e-6)) return Vec3{1, 1, 1};      // opaque or unset: no hue to take
    return clamp01(raw * (1.0 / mx));             // hue only; magnitude stays with clarity
}

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

// One source distilled from a scene emitter for preview shading. CALIBRATED since 0.368.0: it
// carries the emitter's real colour and strength (its XYZ, cieMean * emitIntegral, through the
// renderer's own XYZ -> linear-sRGB matrix, times the emitter's geometry) and falls off as the
// real 1/d^2, where it used to be a colourless power-normalised weight with a made-up falloff.
// Units are "preview units": everything deriveLight() builds is scaled so the brightest side of
// a white surface at the scene centre reads 1.0, which keeps absolute-exposure previews and the
// emitter display (EMIS_BOOST) where they always sat.
struct PLight {
    enum Kind : int { Point = 0, Spot = 1, Area = 2, Sun = 3 };
    int    kind = Point;
    Vec3   pos{0, 0, 0};        // Point / Spot / Area: the source's position
    Vec3   dir{0, 0, 1};        // Spot: the axis light TRAVELS along; Area: its emission normal;
                                // Sun: the unit direction TOWARD the sun
    double cosInner = 1.0, cosOuter = 1.0;  // Spot penumbra cosines
    Vec3   rgb{0, 0, 0};        // Point / Spot / Area: radiant intensity (Area: along its normal);
                                // Sun: the irradiance on a surface facing it
    double size = 0.0;          // Point / Spot / Area: the source's radius -- 1/d^2 is clamped
                                // inside it and it widens the highlight; Sun: angular radius (rad)
};

// ---- RASTER-PBR: the split-sum specular ------------------------------------------------------
// Karis' analytic fit to the split-sum DFG term (SIGGRAPH 2013 "Real Shading in Unreal Engine
// 4"), so the preview needs no LUT texture shipped alongside it. Returns the (A, B) that
// multiply F0: specular_env = prefiltered * (F0*A + B).
inline void envBrdfApprox(double NoV, double rough, double& A, double& B) {
    // Karis, "Physically Based Shading on Mobile" (2014): c0/c1 are the published constants.
    // Until 0.368.0 this body was NOT that fit -- it returned A = (1-r)^4 and B = 2^(-9.28 NoV)(1-r)^3,
    // which at roughness 0.5 scales the environment by 0.06 where the DFG integral gives ~0.72. Under
    // a 4 % dielectric coat that hardly shows; a metal is nothing BUT this term, so every rough
    // metal previewed several times too dark (known-issues RASTER-METAL-LOOK).
    const double r0 = rough * -1.0    + 1.0;
    const double r1 = rough * -0.0275 + 0.0425;
    const double r2 = rough * -0.572  + 1.04;
    const double r3 = rough *  0.022  - 0.04;
    const double a004 = std::min(r0 * r0, std::exp2(-9.28 * NoV)) * r0 + r1;
    A = -1.04 * a004 + r2;
    B =  1.04 * a004 + r3;
}
// Normalised GGX with Smith height-correlated masking and Schlick Fresnel, evaluated for one
// key light. This is the half of the split-sum a preview cannot fake: the moving highlight is
// the cue that says "satin" rather than "chalk", and it is view-dependent by definition.
inline double ggxSpec(const Vec3& N, const Vec3& V, const Vec3& L, double rough) {
    const Vec3 H = normalize(V + L);
    const double NoV = std::max(1e-4, dot(N, V));
    const double NoL = std::max(0.0,  dot(N, L));
    if (NoL <= 0.0) return 0.0;
    const double NoH = std::max(0.0, dot(N, H));
    const double a  = std::max(1e-3, rough * rough);
    const double a2 = a * a;
    const double d  = NoH * NoH * (a2 - 1.0) + 1.0;
    const double D  = a2 / (PI * d * d);
    // Smith height-correlated visibility, which already folds in the 1/(4 NoL NoV).
    const double lv = NoL * std::sqrt(NoV * NoV * (1.0 - a2) + a2);
    const double ll = NoV * std::sqrt(NoL * NoL * (1.0 - a2) + a2);
    const double Vis = (lv + ll > 0.0) ? 0.5 / (lv + ll) : 0.0;
    return D * Vis * NoL;
}

// Irradiance ARRIVING at `P` from one preview light, on a surface facing the light: `Ld` is the
// unit direction toward the light, `E` the irradiance (the caller multiplies by N.L), `srcAng`
// the source's angular radius as seen from `P` (what widens its highlight). False when the light
// contributes nothing there (behind an area light, outside a spot's cone, at its own centre).
inline bool previewLightAt(const PLight& lp, const Vec3& P, Vec3& Ld, Vec3& E, double& srcAng) {
    if (lp.kind == PLight::Sun) { Ld = lp.dir; E = lp.rgb; srcAng = lp.size; return true; }
    const Vec3 d = lp.pos - P;
    const double dist2 = dot(d, d);
    if (!(dist2 > 1e-24)) return false;
    Ld = d * (1.0 / std::sqrt(dist2));
    const double r2 = std::max(dist2, lp.size * lp.size);   // inside the source: stop growing
    double g = 1.0 / r2;
    if (lp.kind == PLight::Spot)      g *= spotFalloff(dot(lp.dir, Ld * -1.0), lp.cosInner, lp.cosOuter);
    else if (lp.kind == PLight::Area) g *= std::max(0.0, -dot(lp.dir, Ld));   // one-sided, Lambertian
    if (!(g > 0.0)) return false;
    E = lp.rgb * g;
    srcAng = lp.size / std::sqrt(r2);
    return true;
}

// A source of angular radius `ang` reflected in a lobe of roughness `rough`: the highlight is the
// lobe convolved with the source, which GGX approximates by widening alpha by half the source's
// angular radius (a half-vector turns half as fast as the reflection). D stays normalised, so a
// big softbox gives a big soft highlight carrying the same energy a small one packs into a glint.
// The tracers' glossy lobe (bsdf_eval.h glossyExponent) is a Phong lobe of exponent 2/r^2 - 2
// about the mirror direction; this preview's is GGX with alpha = rough^2. Matching the two
// lobes' half-widths (Phong 0.833 r, GGX 1.29 alpha, both in the reflected direction) gives
// alpha = 0.646 r -- so a material roughness reaches the preview as sqrt(0.646 r). Used raw, as
// it was before 0.368.0, every preview highlight came out a factor ~2 narrower than the render's.
inline double previewRough(double r) { return std::sqrt(0.646 * std::max(0.0, r)); }

inline double widenRough(double rough, double ang) {
    return std::sqrt(std::min(1.0, rough * rough + 0.5 * ang));
}

// ---- The preview ENVIRONMENT (0.368.0) --------------------------------------------------------
// What a surface sees around it, as a small equirectangular radiance map: the scene's env light
// (constant, image or sky) where a direction escapes, and -- for an FTSL scene -- the preview-lit
// surroundings where it does not (the light probe, in deriveLight). Two things are read from it:
//   * the DIFFUSE ambient, as order-2 spherical harmonics of the irradiance (nine RGB numbers,
//     ~1-3 % from the exact cosine convolution: Ramamoorthi & Hanrahan 2001);
//   * the SPECULAR environment, pre-blurred at kEnvLevels roughnesses and read along the
//     reflection vector -- the split sum's other half, which the preview used to fake with a
//     two-colour gradient anchored on a dim ambient. It is what a metal shows.
// Row 0 is straight up; phi = atan2(z, x), u = phi/(2pi) + 0.5 -- EnvMap's convention.
constexpr int    kEnvW = 64, kEnvH = 32, kEnvLevels = 6;
// The environment drawn BEHIND the scene (PreviewLight::envBg): sharper than the lighting maps,
// because it is looked at directly rather than integrated.
constexpr int    kBgW = 256, kBgH = 128;
// The inverse depth a background pixel showing the environment is stamped with: a surface at
// (effectively) infinity, so the exposure treats it as the tracer treats its env background --
// exposed and METERED -- and anything clear in front of it still composites over it.
constexpr float  kEnvBgInvDepth = 1e-30f;
constexpr double kEnvRough[kEnvLevels] = {0.0, 0.25, 0.4, 0.55, 0.7, 1.0};

inline Vec3 envTexelDir(int col, int row) {
    const double theta = (row + 0.5) / kEnvH * PI;
    const double phi = ((col + 0.5) / kEnvW - 0.5) * 2.0 * PI;
    const double st = std::sin(theta);
    return Vec3{st * std::cos(phi), std::cos(theta), st * std::sin(phi)};
}
// The real SH basis to order 2.
inline void shBasis9(const Vec3& d, double Y[9]);
inline Vec3 shEval9(const Vec3 c[9], const Vec3& N);
inline void shBasis9(const Vec3& d, double Y[9]) {
    Y[0] = 0.282095;
    Y[1] = 0.488603 * d.y; Y[2] = 0.488603 * d.z; Y[3] = 0.488603 * d.x;
    Y[4] = 1.092548 * d.x * d.y; Y[5] = 1.092548 * d.y * d.z;
    Y[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);
    Y[7] = 1.092548 * d.x * d.z; Y[8] = 0.546274 * (d.x * d.x - d.y * d.y);
}

// Cheap inverse trig for the preview environment lookup (0.368.0). The map's texels are 5.6
// degrees wide, so exact acos/atan2 -- the costliest thing a glossy pixel did -- buy nothing.
// acos: Abramowitz & Stegun 4.4.45, |error| <= 6.8e-5 rad. atan: a minimax odd polynomial on
// [-1, 1], |error| <= 1e-5 rad, folded to the full circle. The device twins are identical.
inline double previewAcos(double x) {
    const double ax = std::fabs(x);
    const double r = std::sqrt(std::max(0.0, 1.0 - ax)) *
                     (1.5707288 + ax * (-0.2121144 + ax * (0.0742610 + ax * -0.0187293)));
    return (x >= 0.0) ? r : PI - r;
}
inline double previewAtan2(double y, double x) {
    const double ax = std::fabs(x), ay = std::fabs(y);
    const double mx = std::max(ax, ay);
    if (!(mx > 0.0)) return 0.0;
    const double z = std::min(ax, ay) / mx, z2 = z * z;
    double a = z * (0.99997726 + z2 * (-0.33262347 + z2 * (0.19354346 + z2 * (-0.11643287 +
                    z2 * (0.05265332 + z2 * -0.01172120)))));
    if (ay > ax) a = 0.5 * PI - a;
    if (x < 0.0) a = PI - a;
    return (y < 0.0) ? -a : a;
}

inline Vec3 shEval9(const Vec3 c[9], const Vec3& N) {
    double Y[9]; shBasis9(N, Y);
    Vec3 s{0, 0, 0};
    for (int i = 0; i < 9; ++i) s += c[i] * Y[i];
    return Vec3{std::max(0.0, s.x), std::max(0.0, s.y), std::max(0.0, s.z)};
}

struct PreviewLight {
    std::vector<PLight> lights;   // every emitter but the env, calibrated (see PLight)
    // Diffuse ambient: E(N)/pi as order-2 SH -- the radiance a white Lambertian surface facing N
    // returns from the surroundings. Evaluate with ambientAt().
    Vec3   sh[9] = {};
    double fill = 0.03;           // a faint camera headlight, so an unlit side is never pure black
    // Specular environment: kEnvLevels maps of kEnvW x kEnvH, level k blurred for roughness
    // kEnvRough[k]. Empty = nothing to reflect. Evaluate with envSpecularAt().
    std::vector<Vec3> envSpec;
    // BOX PROJECTION (0.368.0). When the map holds the scene's own surfaces (the light probe)
    // a reflection must not be looked up along R from the PROBE -- the probe stands elsewhere,
    // and a mirror by a wall would read the far side of the room. The standard correction:
    // follow the reflection ray from the shading point to where it leaves the scene's box and
    // look the probe up toward THAT point. Off for an env-only map, which is infinitely far.
    // The scene's environment light as the BACKGROUND (kBgW x kBgH, env only -- never the probe's
    // surfaces), or empty when the scene has none and the preview keeps its slate backdrop. The
    // tracer shows its env behind the scene and meters it with everything else; so must the
    // preview, or a model that is all metal meters on its own softbox glints and goes dark.
    std::vector<Vec3> envBg;
    Vec3 envBgAt(const Vec3& d) const {
        const double v = previewAcos(std::min(1.0, std::max(-1.0, d.y))) * (1.0 / PI);
        double u = previewAtan2(d.z, d.x) * (0.5 / PI) + 0.5;
        if (u >= 1.0) u -= 1.0;
        if (u < 0.0) u += 1.0;
        const double x = u * kBgW - 0.5, y = v * kBgH - 0.5;
        int x0 = (int)x; if ((double)x0 > x) --x0;
        int y0 = (int)y; if ((double)y0 > y) --y0;
        const double fx = x - x0, fy = y - y0;
        int x1 = x0 + 1, y1 = y0 + 1;
        if (x0 < 0) x0 += kBgW;
        if (x1 >= kBgW) x1 -= kBgW;
        y0 = std::max(0, y0); y1 = std::min(kBgH - 1, y1);
        const Vec3* m = envBg.data();
        return m[y0 * kBgW + x0] * ((1.0 - fx) * (1.0 - fy)) + m[y0 * kBgW + x1] * (fx * (1.0 - fy)) +
               m[y1 * kBgW + x0] * ((1.0 - fx) * fy) + m[y1 * kBgW + x1] * (fx * fy);
    }
    bool boxProj = false;
    Vec3 boxLo{0, 0, 0}, boxHi{0, 0, 0}, probePos{0, 0, 0};
    Vec3 lookupDir(const Vec3& P, const Vec3& R) const {
        if (!boxProj) return R;
        double tExit = 1e300;
        const double p[3] = {P.x, P.y, P.z}, r[3] = {R.x, R.y, R.z};
        const double lo[3] = {boxLo.x, boxLo.y, boxLo.z}, hi[3] = {boxHi.x, boxHi.y, boxHi.z};
        for (int a = 0; a < 3; ++a) {
            if (p[a] < lo[a] || p[a] > hi[a]) return R;           // outside the box: no proxy
            if (r[a] > 1e-12)       tExit = std::min(tExit, (hi[a] - p[a]) / r[a]);
            else if (r[a] < -1e-12) tExit = std::min(tExit, (lo[a] - p[a]) / r[a]);
        }
        if (!(tExit < 1e299)) return R;
        const Vec3 d = (P + R * tExit) - probePos;
        const double l = std::sqrt(dot(d, d));
        return (l > 1e-12) ? d * (1.0 / l) : R;
    }
    Vec3 ambientAt(const Vec3& N) const {
        double Y[9]; shBasis9(N, Y);
        Vec3 c{0, 0, 0};
        for (int i = 0; i < 9; ++i) c += sh[i] * Y[i];
        return Vec3{std::max(0.0, c.x), std::max(0.0, c.y), std::max(0.0, c.z)};
    }
    // Bilinear in (u, v) -- u wraps, v clamps -- on two adjacent roughness levels, blended by `t`.
    // The four texel positions and weights are shared by both levels: this is the costliest thing a
    // glossy pixel does, so it avoids std::floor and the integer modulo (u is in [0,1), so the left
    // column is at worst -1).
    Vec3 envSpecularAt(const Vec3& R, double rough) const {
        if (envSpec.empty()) return Vec3{0, 0, 0};
        const double v = previewAcos(std::min(1.0, std::max(-1.0, R.y))) * (1.0 / PI);
        double u = previewAtan2(R.z, R.x) * (0.5 / PI) + 0.5;
        if (u >= 1.0) u -= 1.0;
        if (u < 0.0) u += 1.0;
        int k = 0;
        while (k + 1 < kEnvLevels - 1 && rough > kEnvRough[k + 1]) ++k;
        const double t = std::min(1.0, std::max(0.0, (rough - kEnvRough[k]) / (kEnvRough[k + 1] - kEnvRough[k])));
        const double x = u * kEnvW - 0.5, y = v * kEnvH - 0.5;
        int x0 = (int)x; if ((double)x0 > x) --x0;
        int y0 = (int)y; if ((double)y0 > y) --y0;
        const double fx = x - x0, fy = y - y0;
        int x1 = x0 + 1, y1 = y0 + 1;
        if (x0 < 0) x0 += kEnvW;
        if (x1 >= kEnvW) x1 -= kEnvW;
        y0 = std::max(0, y0); y1 = std::min(kEnvH - 1, y1);
        const double w00 = (1.0 - fx) * (1.0 - fy), w10 = fx * (1.0 - fy);
        const double w01 = (1.0 - fx) * fy,         w11 = fx * fy;
        const size_t i00 = (size_t)y0 * kEnvW + x0, i10 = (size_t)y0 * kEnvW + x1;
        const size_t i01 = (size_t)y1 * kEnvW + x0, i11 = (size_t)y1 * kEnvW + x1;
        const Vec3* a = envSpec.data() + (size_t)k * kEnvW * kEnvH;
        const Vec3* b = a + (size_t)kEnvW * kEnvH;
        const Vec3 ca = a[i00] * w00 + a[i10] * w10 + a[i01] * w01 + a[i11] * w11;
        const Vec3 cb = b[i00] * w00 + b[i10] * w10 + b[i01] * w01 + b[i11] * w11;
        return ca * (1.0 - t) + cb * t;
    }
};

// The albedo a probe ray sees on a surface: the material's preview colour, a mix resolved to
// its dominant child as everywhere else in this file, an image skin sampled at the hit.
inline Vec3 probeAlbedo(const Scene& sc, const Hit& h, bool& emissive) {
    int id = h.matId;
    for (int guard = 0; guard < 8; ++guard) {
        if (id < 0 || id >= (int)sc.mats.size()) break;
        const Material& m = sc.mats[id];
        if (m.mixChildren.empty()) break;
        const int pick = mixDominantChild(m);
        if (pick < 0 || pick == id || pick >= (int)sc.mats.size()) break;
        id = pick;
    }
    emissive = false;
    if (id < 0 || id >= (int)sc.mats.size()) return Vec3{0.6, 0.6, 0.6};
    const Material& m = sc.mats[id];
    bool em = false;
    Vec3 c = materialColor(m, em);
    emissive = em || m.isLight;
    if (!emissive && m.reflectTex >= 0 && m.reflectTex < (int)sc.textures.size() &&
        sc.textures[m.reflectTex].valid())
        c = sc.textures[m.reflectTex].sampleRgb(h.u, h.v);
    return c;
}

// Distil the scene's lights for the preview (0.368.0). `probePos` is where the light probe
// stands -- the first camera's eye, which is in free space by construction -- and
// `probeGeometry` says whether it looks at the scene's surfaces at all: false for the mesh
// quick-view, whose only geometry is the model being viewed (a probe would see the model's own
// inside). Runs once per scene load; every per-pixel cost it adds is two SH/map lookups.
// `withSpecular` false skips the pre-blurred reflection levels -- for a diffuse-only consumer
// (the GPU implicit preview) that derives per frame, where the blur is the one costly step.
// `reflectPos`: where the REFLECTION probe stands (the camera's eye; null = probePos). The ambient
// probe stands at probePos, where the camera looks.
inline PreviewLight deriveLight(const Scene& sc, const Vec3& probePos, bool probeGeometry,
                                bool withSpecular = true, const Vec3* reflectPos = nullptr) {
    PreviewLight L;
    const double R = sc.sceneRadius > 0.0 ? sc.sceneRadius : 1.0;

    // ---- 1. The lights, in physical units (normalised at the end). -----------------------
    for (const Emitter& e : sc.emitters) {
        if (e.shape == EmitterShape::Env) continue;
        const Vec3 rgb = xyzToLinearSrgb(e.cieMean * e.emitIntegral);   // radiance (sun: /Omega)
        PLight p;
        switch (e.shape) {
            case EmitterShape::Sun:
                p.kind = PLight::Sun;
                p.dir  = normalize(e.beamDir) * -1.0;
                p.rgb  = rgb * e.spotOmega;                  // back to irradiance
                p.size = std::acos(std::min(1.0, std::max(-1.0, e.spotCosOuter)));
                break;
            case EmitterShape::Spot:
                p.kind = PLight::Spot;
                p.pos = e.origin; p.dir = normalize(e.beamDir);
                p.cosInner = e.spotCosInner; p.cosOuter = e.spotCosOuter;
                p.rgb  = rgb;                                // emitIntegral is the on-axis intensity
                p.size = std::max(e.radius, 1e-3 * R);
                break;
            case EmitterShape::Quad:
                p.kind = PLight::Area;
                p.pos = e.origin + (e.u + e.v) * 0.5;
                p.dir = e.collimated ? normalize(e.beamDir) : normalize(e.normal);
                p.rgb  = rgb * e.area;                       // I(w) = L A cos -- the cos is per point
                p.size = std::sqrt(std::max(0.0, e.area) / PI);
                break;
            case EmitterShape::Cylinder:
                p.kind = PLight::Point;
                p.pos = e.origin + e.v * 0.5;
                p.rgb  = rgb * (e.area * 0.25);              // mean projected area of a convex body: S/4
                p.size = std::max(e.radius, 0.5 * length(e.v));
                break;
            case EmitterShape::Mesh: {
                p.kind = PLight::Point;
                Vec3 cen{0, 0, 0}; double wsum = 0.0, prev = 0.0;
                for (const EmitTri& t : e.meshTris) {
                    const double a = t.cumArea - prev; prev = t.cumArea;
                    cen += (t.v0 + (t.e1 + t.e2) * (1.0 / 3.0)) * a; wsum += a;
                }
                p.pos = (wsum > 0.0) ? cen * (1.0 / wsum) : e.origin;
                p.rgb  = rgb * (e.area * 0.25);
                p.size = std::sqrt(std::max(0.0, e.area) / (4.0 * PI));
                break;
            }
            default:                                         // Sphere
                p.kind = PLight::Point;
                p.pos = e.origin;
                p.rgb  = rgb * (e.area * 0.25);              // == pi r^2 L
                p.size = e.radius;
                break;
        }
        if (p.rgb.x > 0.0 || p.rgb.y > 0.0 || p.rgb.z > 0.0) L.lights.push_back(p);
    }

    // ---- 2. The environment map: the env light where a direction escapes. -----------------
    const int NT = kEnvW * kEnvH;
    std::vector<Vec3> dirs(NT);
    std::vector<double> dOmega(NT);
    for (int row = 0; row < kEnvH; ++row)
        for (int col = 0; col < kEnvW; ++col) {
            const int t = row * kEnvW + col;
            dirs[t] = envTexelDir(col, row);
            const double theta = (row + 0.5) / kEnvH * PI;
            dOmega[t] = (2.0 * PI / kEnvW) * (PI / kEnvH) * std::sin(theta);
        }
    // The env light on a W x H lat-long grid (box-filtered from an image map; constant otherwise).
    auto bakeEnv = [&](int GW, int GH) {
        std::vector<Vec3> out((size_t)GW * GH, Vec3{0, 0, 0});
        if (sc.envIndex < 0) return out;
        if (sc.envMap) {
            const EnvMap& m = *sc.envMap;
            std::vector<int> colMap(m.w), rowMap(m.h);
            std::vector<double> rowW(m.h);
            for (int c = 0; c < m.w; ++c) {
                double u = (c + 0.5) / m.w + m.rotOffset; u -= std::floor(u);
                colMap[c] = std::min(GW - 1, (int)(u * GW));
            }
            for (int r = 0; r < m.h; ++r) {
                rowMap[r] = std::min(GH - 1, (int)((r + 0.5) / m.h * GH));
                rowW[r] = std::sin((r + 0.5) / m.h * PI);
            }
            std::vector<Vec3> acc(out.size(), Vec3{0, 0, 0});
            std::vector<double> wsum(out.size(), 0.0);
            for (int r = 0; r < m.h; ++r) {
                const int base = rowMap[r] * GW;
                const double w = rowW[r];
                const Vec3* src = m.xyzT.data() + (size_t)r * m.w;
                for (int c = 0; c < m.w; ++c) { acc[base + colMap[c]] += src[c] * w; wsum[base + colMap[c]] += w; }
            }
            for (int row = 0; row < GH; ++row)
                for (int col = 0; col < GW; ++col) {
                    const size_t t = (size_t)row * GW + col;
                    if (wsum[t] > 0.0) { out[t] = xyzToLinearSrgb(acc[t] * (1.0 / wsum[t])); continue; }
                    const double th = (row + 0.5) / GH * PI, ph = ((col + 0.5) / GW - 0.5) * 2.0 * PI;
                    out[t] = xyzToLinearSrgb(m.xyz(Vec3{std::sin(th) * std::cos(ph), std::cos(th),
                                                        std::sin(th) * std::sin(ph)}));
                }
        } else {
            const Vec3 c = xyzToLinearSrgb(sc.envXYZ);
            for (Vec3& v : out) v = c;
        }
        for (Vec3& v : out) v = Vec3{std::max(0.0, v.x), std::max(0.0, v.y), std::max(0.0, v.z)};
        return out;
    };
    std::vector<Vec3> envL = bakeEnv(kEnvW, kEnvH);
    if (withSpecular && sc.envIndex >= 0) L.envBg = bakeEnv(kBgW, kBgH);
    auto projectSH = [&](const std::vector<Vec3>& Lmap, Vec3 out[9]) {
        // Irradiance/pi from radiance: band l scaled by A_l/pi = {1, 2/3, 1/4}.
        static const double kBand[9] = {1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25};
        for (int i = 0; i < 9; ++i) out[i] = Vec3{0, 0, 0};
        double Y[9];
        for (int t = 0; t < NT; ++t) {
            shBasis9(dirs[t], Y);
            for (int i = 0; i < 9; ++i) out[i] += Lmap[t] * (Y[i] * dOmega[t]);
        }
        for (int i = 0; i < 9; ++i) out[i] = out[i] * kBand[i];
    };

    // ---- 3. The light probe: the preview-lit surroundings where a direction is blocked. ----
    // Stood at the first camera's eye (free space by construction), it takes the first surface
    // each direction meets, either side -- ftrace shades surfaces two-sided, and a floor quad
    // authored facing down is still the floor (skipping back faces looked straight through
    // one) -- and gives an emitter's own surface nothing, because the lights reach every
    // shading point analytically already. Each surface it finds is lit exactly as the preview
    // lights it (the calibrated lights, unshadowed, plus the env's ambient), so a metal in a room
    // reflects that room: the walls and their colours. One bounce from one position -- box
    // projection (PreviewLight::lookupDir) corrects the parallax of looking it up elsewhere.
    // TWO PROBES. The AMBIENT probe stands where the camera looks (probePos): inside the room the
    // camera is looking into, which is what gives a closed box its level of bounce light. The
    // REFLECTION probe stands at the camera's eye (reflectPos): free space by construction, and
    // nothing in its view dominates it -- at probePos the object in view filled half the probe's
    // sky, and a gold gyroid reflected mostly itself. Box projection (lookupDir) keeps the walls
    // where they are when a reflection is looked up from the eye.
    std::vector<Vec3> surL = envL;
    const bool anyGeo = probeGeometry && (!sc.tris.empty() || !sc.spheres.empty() ||
                                          !sc.implicits.empty() || !sc.instances.empty());
    Vec3 shEnv[9]; projectSH(envL, shEnv);
    // Trace one probe: what each direction's first surface is (0 escaped, 1 surface, 2 an emitter's
    // own), its albedo and normal, and the calibrated lights' irradiance on it -- UNSHADOWED, like the
    // preview's own lighting (a probe must show a surface as the preview draws it; shadow rays also
    // treat glass as opaque, which blacked out everything a bulb in a glass envelope lights).
    struct Probe { std::vector<char> kind; std::vector<Vec3> alb, nrm, E; };
    const double eps = 1e-5 * R;
    auto trace = [&](const Vec3& pos) {
        Probe P;
        P.kind.assign(NT, 0); P.alb.resize(NT); P.nrm.resize(NT); P.E.resize(NT);
        (void)ft::parallelFor((size_t)NT, 32, [&](size_t t) {   // a clean stop just ends the probe early
            const Hit h = sc.closestHit(Ray{pos, dirs[t]}, eps);
            if (!h.valid) return;                                            // escaped: the env
            bool emissive = false;
            const Vec3 alb = probeAlbedo(sc, h, emissive);
            if (emissive) { P.kind[t] = 2; return; }
            const Vec3 n = h.n;                                              // faces the probe
            Vec3 E{0, 0, 0};
            for (const PLight& lp : L.lights) {
                Vec3 Ld, Ei; double ang;
                if (!previewLightAt(lp, h.p, Ld, Ei, ang)) continue;
                const double ndl = dot(n, Ld);
                if (ndl > 0.0) E += Ei * ndl;
            }
            P.kind[t] = 1; P.alb[t] = alb; P.nrm[t] = n; P.E[t] = E;
        });
        return P;
    };
    // Shade a traced probe with `amb` as every surface's ambient.
    auto shade = [&](const Probe& P, const Vec3 amb[9], std::vector<Vec3>& out) {
        for (int t = 0; t < NT; ++t) {
            if (P.kind[t] == 0) { out[t] = envL[t]; continue; }
            if (P.kind[t] == 2) { out[t] = Vec3{0, 0, 0}; continue; }
            const Vec3 a = shEval9(amb, P.nrm[t]);
            const Vec3& al = P.alb[t]; const Vec3& E = P.E[t];
            out[t] = Vec3{al.x * (E.x / PI + a.x), al.y * (E.y / PI + a.y), al.z * (E.z / PI + a.z)};
        }
    };
    if (anyGeo) {
        // The AMBIENT: the probed surfaces' BRIGHTNESS but not their hue (the env keeps its colour).
        // One probe overweights whatever it stands beside, and it was the colour that contaminated:
        // a gold gyroid tinted a whole gallery orange, a few coloured curves tinted a white room pink.
        // Each pass lights the surfaces with the previous pass's surroundings, so a room is lit by
        // its own walls: THREE bounces, which the reference renders favour for closed rooms (the
        // curve room, the spot box) now that the colour cannot feed back.
        const Probe A = trace(probePos);
        std::vector<Vec3> shaded(NT), grey(NT);
        Vec3 shPrev[9];
        for (int i = 0; i < 9; ++i) shPrev[i] = shEnv[i];
        int passes = 3;
        if (const char* ev = std::getenv("FTRACE_PREVIEW_BOUNCES")) passes = std::max(1, std::min(8, std::atoi(ev)));
        for (int pass = 0; pass < passes; ++pass) {
            shade(A, shPrev, shaded);
            for (int t = 0; t < NT; ++t) {
                if (A.kind[t] != 1) { grey[t] = shaded[t]; continue; }
                const double y = 0.2126 * shaded[t].x + 0.7152 * shaded[t].y + 0.0722 * shaded[t].z;
                grey[t] = Vec3{y, y, y};
            }
            projectSH(grey, shPrev);
        }
        for (int i = 0; i < 9; ++i) L.sh[i] = shPrev[i];
        // The REFLECTIONS: from the eye, in full colour -- a thing reflected is supposed to show its
        // colour -- each surface lit by the lights and the ambient just found.
        if (withSpecular) {
            const Probe Rp = trace(reflectPos ? *reflectPos : probePos);
            shade(Rp, L.sh, surL);
        }
    } else {
        for (int i = 0; i < 9; ++i) L.sh[i] = shEnv[i];                         // env only
    }
    if (probeGeometry && sc.sceneBoxLo.x <= sc.sceneBoxHi.x) {
        // A hair of margin so a surface ON the box face still counts as inside it.
        const Vec3 m = (sc.sceneBoxHi - sc.sceneBoxLo) * 1e-4 + Vec3{1e-9, 1e-9, 1e-9};
        L.boxProj = true; L.boxLo = sc.sceneBoxLo - m; L.boxHi = sc.sceneBoxHi + m;
        // An env light means an open sky: the box has no lid, or an upward reflection would
        // leave through a ceiling at the top of the tallest object and read the horizon.
        if (sc.envIndex >= 0) L.boxHi.y = L.boxLo.y + 1e4 * R;
        L.probePos = reflectPos ? *reflectPos : probePos;
    }

    // ---- 4. The specular levels: surL blurred by a normalised cosine-power lobe per level. ----
    if (withSpecular) {
    L.envSpec.assign((size_t)kEnvLevels * NT, Vec3{0, 0, 0});
    std::copy(surL.begin(), surL.end(), L.envSpec.begin());
    for (int k = 1; k < kEnvLevels; ++k) {
        const double a = kEnvRough[k] * kEnvRough[k];
        const double nPhong = std::max(1.0, (2.0 / (a * a) - 2.0) * 0.25);   // Blinn exponent / 4
        Vec3* out = L.envSpec.data() + (size_t)k * NT;
        (void)ft::parallelFor((size_t)NT, 16, [&](size_t t) {
            const Vec3 R0 = dirs[t];
            Vec3 acc{0, 0, 0}; double wsum = 0.0;
            for (int s2 = 0; s2 < NT; ++s2) {
                const double c = dot(R0, dirs[s2]);
                if (c <= 0.0) continue;
                const double w = std::exp(nPhong * std::log(c)) * dOmega[s2];
                acc += surL[s2] * w; wsum += w;
            }
            out[t] = (wsum > 0.0) ? acc * (1.0 / wsum) : Vec3{0, 0, 0};
        });
    }
    }

    // ---- 5. Preview units: the brightest side of a white surface at the centre reads 1. -----
    double T = 0.0;
    static const Vec3 kAxes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (const Vec3& N : kAxes) {
        Vec3 v = L.ambientAt(N);
        for (const PLight& lp : L.lights) {
            Vec3 Ld, E; double ang;
            if (!previewLightAt(lp, sc.sceneCenter, Ld, E, ang)) continue;
            const double ndl = dot(N, Ld);
            if (ndl > 0.0) v += E * (ndl / PI);
        }
        T = std::max(T, 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z);
    }
    if (T > 0.0 && std::isfinite(T)) {
        const double S = 1.0 / T;
        for (PLight& lp : L.lights) lp.rgb = lp.rgb * S;
        for (Vec3& c : L.sh) c = c * S;
        for (Vec3& c : L.envSpec) c = c * S;
        for (Vec3& c : L.envBg) c = c * S;
    } else {
        // Nothing lights the scene (or it cannot be measured): a flat grey studio, so the preview
        // still shows the shapes -- the old env-only look.
        L.lights.clear();
        for (Vec3& c : L.sh) c = Vec3{0, 0, 0};
        L.sh[0] = Vec3{1, 1, 1} * (0.35 / 0.282095);
        for (Vec3& c : L.envSpec) c = Vec3{0.35, 0.35, 0.35};
        L.fill = 0.5;
    }
    return L;
}
// Scene-centre overload for callers with no camera to stand the probe at (and no probe).
inline PreviewLight deriveLight(const Scene& sc, bool withSpecular = true) {
    return deriveLight(sc, sc.sceneCenter, false, withSpecular);
}

// Deepest `ccap` (rings per spherical cap) any curve LOD may ask for — sizes the
// fixed ring buffer in the curve sweep below.
constexpr int kMaxCurveCap = 2;

// How many preview triangles the curve/fiber sweep may spend in total. Each PTri is
// ~320 B, so this caps fur preview geometry at roughly 3.8 GB; past it the sweep drops
// to a coarser tube and finally thins whole strands (see section 2b). Overridable per
// call (`-raster-curve-budget`), because a machine with room to spare may prefer the
// full 80-tris/segment cone on a groomed pelt.
//
// 12 M is chosen so the reference pelts (scenes/fur_creature.ftsl and the creature in
// scenes/gallery_rain.ftsl, ~1.79 M segments each) land on the 3-sided uncapped tube
// rather than the flat ribbon below it: the ribbon samples only two azimuths, so its
// shading reads noticeably darker and patchier, while the 3-sided tube is visually
// indistinguishable from the full cone at preview resolution for ~1/13 the triangles.
constexpr size_t kDefaultCurveBudget = 12u * 1000u * 1000u;

// ---- Scene -> world-space preview triangles (done once, reused for every frame) --
// `progress`, if set, is called as each heavy implicit (isosurface/CSG/metaball) is
// about to be marched: progress(done, total) where `total` is the implicit count and
// `done` runs 0..total (0 before the first, total after the last). Marching implicits
// is by far the slow part of tessellation, so this drives the "tessellating N/M" UI.
// `curveBudget` (0 = kDefaultCurveBudget) caps the triangles spent on curve/fiber
// strands; see section (2b).
// Re-shade a tessellation to NEUTRAL CLAY — the viewer's "Color" toggle off.
//
// Done to the baked geometry rather than as a flag inside the shade pass, so both
// backends get it for free: the CPU rasterizer reads these fields directly and the GPU
// one uploads them, and neither needs to know the mode exists. Everything that could
// re-introduce a colour has to go with the albedo, not just the albedo itself — an image
// skin, a triplanar projection, a `reflect`/`emit` pattern drive, a normal map (which is
// shape, but shape read out of a coloured texture), and both children of a per-hit mix
// plus the mask that chooses between them. What is left is form and lighting alone,
// which is what you want when the question is "what shape is this?" rather than "what
// does it look like?" — and, for an N-D warp, the shape is the whole question.
inline void stripColor(PreviewGeom& g, const Vec3& neutral = Vec3{0.72, 0.72, 0.72}) {
    auto flatten = [&](PShade& s) {
        s.color          = neutral;
        s.tex            = -1;
        s.triplanarScale = 0.0;
        s.reflectPat     = -1;
        s.emitPat        = -1;
        s.normalTex      = -1;
        // A glossy surface is all lobe (bakeOwn's negative f0), which is right for "what does it
        // look like" and wrong for "what shape is it": a mirror-like surface shows its
        // surroundings, not its form. Clay it is -- the neutral albedo as a diffuse body with a
        // neutral highlight on top, so the lobe's shape still reads.
        if (s.f0.x < 0.0) s.f0 = neutral;
        // A clear surface never reaches the shade pass at all — see-through hands it to
        // the clear-accumulation pass, which reads ONLY this tint. Leaving it coloured
        // meant "Color off" did nothing whatsoever on an all-glass model: every triangle
        // took the clear path, so the two renders came out bit-identical.
        s.clearTint      = Vec3{1, 1, 1};
    };
    for (PTri& t : g.tris) {
        flatten(t);
        // A per-vertex colour is a colour like any other: leaving it multiplied in would
        // let a vertex-coloured scan keep its tint with "Color" off, which is exactly the
        // question the toggle exists to answer.
        t.hasVcol = false;
    }
    for (PMix& m : g.mixes) {
        // Both children are now the same colour, so the mask decides nothing; dropping it
        // also spares the shade pass a per-pixel pattern/texture evaluation per hit.
        m.weightPat = -1;
        m.weightTex = -1;
        flatten(m.b);
    }
}

inline PreviewGeom tessellate(const Scene& sc, int isoRes,
                              const std::function<void(int, int)>& progress = {},
                              size_t curveBudget = 0) {
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
        if (s.clear) {
            s.clearTint = clearTintOf(m);
            // Tint the SOLID ghost by the same glass colour. Without see-through a clear
            // material previews from `reflect`, which for imported glass is routinely pure
            // white — so a tray of differently coloured gems came out as a tray of
            // identical pale balls. The tint is hue-normalised, so this colours the ghost
            // without darkening it.
            s.color = Vec3{s.color.x * s.clearTint.x,
                           s.color.y * s.clearTint.y,
                           s.color.z * s.clearTint.z};
        }
        // RASTER-PBR: the specular description. Only Glossy has a lobe the preview can show;
        // Mirror is a delta the rasterizer has no reflection to fill it with, and the clear
        // family is already handled by see-through. `reflect` IS the normal-incidence
        // reflectance for a metal preset, which is why the highlight has to be tinted by it
        // rather than white -- a white highlight on gold is the single most obvious tell.
        // Since 0.368.0 a Mirror is a lobe too: with a real environment to reflect (the light
        // probe, PreviewLight::envSpec) a near-zero roughness shows what a mirror shows, where
        // before it could only be a flat bright tint. Both are tinted by their TRUE reflectance:
        // materialColor lifts every glossy material to luminance 0.7 so it reads as a pale ghost
        // in a flat-shaded preview, which for a surface that is all reflection would brighten a
        // 40 % polished floor to 70 %.
        if (!m.isLight && (m.type == MatType::Glossy || m.type == MatType::Mirror)) {
            const bool mirror = (m.type == MatType::Mirror);
            s.rough    = mirror ? 0.02 : previewRough(m.roughness);
            s.color    = spectrumToLinearRgb(m.reflect);
            s.roughPat = mirror ? -1 : m.roughnessPat;
            s.roughTex = mirror ? -1 : m.roughnessTex;
        }
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
        // A GLOSSY (or mirror) material is ALL LOBE: the tracers give it no diffuse term -- a metal's
        // whole appearance is reflection -- so the preview must not invent one. Its lobe is tinted
        // by the albedo the pixel shades with, texture, pattern drive and vertex colour included,
        // which is the tracer's reflectSlot chain since 0.367.2 (a textured glTF metal keeps its
        // colour in the texture and leaves the constant white). Both facts ride ONE marker, a
        // NEGATIVE f0, which no real reflectance is, so it travels every existing path (the
        // near-clip store, the mix table, the device copy) without a new field; the shaders
        // resolve it to the albedo and drop the diffuse term. 0.367.2 set it for textured glossy
        // only (the tint). 0.368.0 sets it for every glossy material, now that the preview has a
        // calibrated environment for such a surface to reflect: 0.368.0's first attempt, against
        // the old dim sky/ground pair, turned gold and chrome dark brown and grey.
        if (s.rough >= 0.0) s.f0 = Vec3{-1.0, -1.0, -1.0};
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
    // Pass 1.5 — RASTER-PBR through a MIX (0.316.0). Pass 1 collapses a mix to its DOMINANT
    // child, which for a specular-over-diffuse stack is the body: the lobe is in the light
    // child and was thrown away, so an imported glTF dielectric (gltf.h builds exactly that
    // stack: a 4 % uncoloured glossy lobe over the diffuse body) previewed as chalk while the
    // viewer it came from showed satin -- the reported gallery_rain/Alice case. Take the first
    // glossy child's lobe, with F0 scaled by its selection weight, on top of the dominant
    // child's colour. Only when pass 1 found no lobe of its own, so a mix whose dominant child
    // IS the glossy one keeps its full-strength highlight.
    for (size_t i = 0; i < sc.mats.size(); ++i) {
        const Material& m = sc.mats[i];
        if (m.mixChildren.empty() || matSh[i].rough >= 0.0) continue;
        if (m.type == MatType::Layered) {
            // A layered stack's coat is FIELDS on the parent rather than a child lobe, so
            // there is nothing for the loop below to find: read it here. F0 comes from the
            // coat's own index (0.04 at 1.5), or straight from `specular` in manual mode.
            const double n = m.ior ? m.ior(550.0) : 1.5;
            const double f0 = (m.coatModel == 2 && m.coatSpecular >= 0.0)
                                  ? std::min(1.0, std::max(0.0, m.coatSpecular))
                                  : ((n - 1.0) / (n + 1.0)) * ((n - 1.0) / (n + 1.0));
            matSh[i].rough    = previewRough(m.roughness);
            matSh[i].f0       = Vec3{f0, f0, f0};
            matSh[i].roughPat = m.roughnessPat;
            matSh[i].roughTex = m.roughnessTex;
            continue;
        }
        for (size_t k = 0; k < m.mixChildren.size(); ++k) {
            const int c = m.mixChildren[k];
            if (c < 0 || c >= (int)sc.mats.size()) continue;
            const Material& cm = sc.mats[c];
            if (cm.isLight || cm.type != MatType::Glossy) continue;
            double w = (k < m.mixWeights.size()) ? m.mixWeights[k] : 0.0;
            w = (w < 0.0) ? 0.0 : (w > 1.0 ? 1.0 : w);
            if (!(w > 0.0)) continue;
            const Vec3 cf0 = spectrumToLinearRgb(cm.reflect);   // the lobe's true F0 (not the 0.7 ghost)
            matSh[i].rough    = previewRough(cm.roughness);
            matSh[i].f0       = Vec3{cf0.x * w, cf0.y * w, cf0.z * w};
            matSh[i].roughPat = cm.roughnessPat;
            matSh[i].roughTex = cm.roughnessTex;
            break;
        }
    }
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
    // Per-vertex colour, flat triangles and instanced BLAS triangles alike. A BLAS keeps
    // its own triangle array but indexes the SCENE's colour table (see Blas::intersectLocal),
    // so one helper serves both -- and having one is the point: the instanced path is
    // exactly where an attribute added to the flat path gets forgotten.
    auto copyVcol = [&](PTri& p, const Tri& t) {
        if (t.vcol < 0) return;
        const size_t i = (size_t)t.vcol * 3;
        if (i + 8 >= sc.vertColors.size()) return;
        const float* c = sc.vertColors.data() + i;
        p.hasVcol = true;
        p.vc0 = Vec3{c[0], c[1], c[2]};
        p.vc1 = Vec3{c[3], c[4], c[5]};
        p.vc2 = Vec3{c[6], c[7], c[8]};
    };
    out.reserve(sc.tris.size() + 4096);
    // Per scene triangle, in order. Tried threading this (resize + index fill) and it came
    // out SLOWER -- 604 ms to ~750 -- because vector::resize value-initialises 4.26M PTri
    // before the fill overwrites every field, and that zeroing costs more than the loop it
    // was meant to save. Left serial deliberately; if this is ever worth revisiting it needs
    // uninitialised storage, not a resize.
    for (const auto& t : sc.tris) {
        PTri p;
        p.p0 = t.v0; p.p1 = t.v1; p.p2 = t.v2;
        p.n0 = t.n0; p.n1 = t.n1; p.n2 = t.n2;
        applyMat(p, t.matId);
        p.uv0 = t.uv0; p.uv1 = t.uv1; p.uv2 = t.uv2;
        copyVcol(p, t);
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

    // (2b) Curve / fiber segments -> a round-cone mesh (lateral tangent band + both
    // spherical caps). The rasterizer draws triangles, so without this a scene of
    // strands previews EMPTY — and unlike the CUDA path, which gates the whole scene
    // to the CPU rather than render a furred subject bald, the preview has no gate to
    // fall back to. "Geometry I can't draw is geometry that isn't there" is the exact
    // failure the applyMat comment above warns about, so the fix is to draw it.
    //
    // The surface is swept as a stack of RINGS about the segment axis, walking the same
    // three pieces the analytic intersector knows (curve.h): the back cap of sphere(p0,r0),
    // the tangent lateral band, and the front cap of sphere(p1,r1). With
    // `a = (r0-r1)/|p1-p0|` the tangent circles sit at polar angle `acos(a)` on BOTH end
    // spheres, which is what makes one angular sweep cover all three pieces continuously
    // — so the preview mesh is closed, exactly like the surface it approximates.
    //
    // TRIANGLE BUDGET. A fiber is a few pixels wide at preview resolution, so the full
    // 80-tris/segment cone is only affordable on light strand counts. A groomed pelt is
    // NOT light: `scenes/gallery_rain.ftsl` carries 1.79 M fur segments, which at 80 tris
    // and sizeof(PTri) == 320 B is ~46 GB of preview geometry (~92 GB while the vector
    // doubles) — `-explore` on it used to sit at "tessellating (0/33)" forever, thrashing
    // the page file, because this loop ran before the implicits it was reporting progress
    // for. So the sweep now picks the coarsest ring/azimuth LOD that fits `curveBudget`
    // triangles, and only if even the cheapest one busts the budget does it thin whole
    // STRANDS (by curveId, so a kept strand stays continuous rather than dashed).
    if (!sc.curveSegs.empty()) {
        // LOD ladder, richest first: {azimuthal divisions, rings per spherical cap}.
        // ccap == 0 drops the end caps and sweeps the lateral band alone (an open cone) —
        // the caps are sub-pixel on fur. cu == 2 degenerates that band into a double-sided
        // flat ribbon, the cheapest thing that still shows every strand.
        struct CurveLod { int cu, ccap; };
        static const CurveLod kCurveLods[] = {{10,2},{6,1},{4,1},{3,1},{4,0},{3,0},{2,0}};
        auto lodTris = [](const CurveLod& L) -> size_t {
            const int rings = L.ccap > 0 ? 2 * L.ccap + 2 : 2;
            size_t t = (size_t)2 * L.cu * (rings - 1);
            if (L.ccap > 0) t -= (size_t)2 * L.cu;   // the two pole spans collapse to fans
            return t;
        };
        const size_t nSeg = sc.curveSegs.size();
        const size_t budget = curveBudget ? curveBudget : kDefaultCurveBudget;
        size_t lod = 0;
        while (lod + 1 < sizeof(kCurveLods) / sizeof(kCurveLods[0]) &&
               nSeg * lodTris(kCurveLods[lod]) > budget) ++lod;
        const int CU   = kCurveLods[lod].cu;
        const int CCAP = kCurveLods[lod].ccap;
        const size_t perSeg = lodTris(kCurveLods[lod]);
        // Still over budget at the cheapest LOD: keep one strand in `stride`.
        size_t stride = 1;
        if (nSeg * perSeg > budget) stride = (nSeg * perSeg + budget - 1) / budget;
        if (lod > 0 || stride > 1) {
            const std::string thin = stride > 1
                ? ", thinned to 1 strand in " + std::to_string(stride) : std::string();
            std::printf("[raster] %zu curve segments over the %zu-triangle preview budget: "
                        "%d-sided%s tube, %zu tris/segment%s (%zu tris)\n",
                        nSeg, budget, CU, CCAP ? "" : " uncapped", perSeg, thin.c_str(),
                        (nSeg / stride) * perSeg);
            std::fflush(stdout);
        }
        out.reserve(out.size() + (nSeg / stride + 1) * perSeg);
        size_t segIdx = 0;
        for (const auto& s : sc.curveSegs) {
            const size_t key = (s.curveId >= 0) ? (size_t)s.curveId : segIdx;
            ++segIdx;
            if (stride > 1 && key % stride) continue;
            Vec3 ba = s.p1 - s.p0;
            double l = length(ba);
            if (l <= 1e-12) continue;                 // coincident ends: tessellateCurve drops these
            Vec3 ax = ba * (1.0 / l);
            double a = (s.r0 - s.r1) / l;
            if (a > 1.0) a = 1.0; else if (a < -1.0) a = -1.0;   // one ball swallows the other
            Vec3 T, B; onb(ax, T, B);
            const double alpha0 = std::acos(a);       // polar angle of BOTH tangent circles

            // Ring stations, back pole -> front pole. The two tangent rings are shared, so
            // the strip is seamless. Held in a fixed stack buffer rather than a vector:
            // this runs once per segment and a furred scene has millions of them, so a
            // heap allocation here would dominate the sweep.
            struct Ring { Vec3 c; double rad, nAx; double u; };
            Ring rings[2 * kMaxCurveCap + 2];
            int nRings = 0;
            auto push = [&](const Vec3& org, double r, double alpha, double u) {
                const double ca = std::cos(alpha), sa = std::sin(alpha);
                rings[nRings++] = Ring{org + ax * (r * ca), r * sa, ca, u};
            };
            if (CCAP > 0) {
                for (int i = 0; i <= CCAP; ++i)       // cap at p0: alpha pi -> alpha0
                    push(s.p0, s.r0, PI + (alpha0 - PI) * (double)i / CCAP, (double)s.u0);
                push(s.p1, s.r1, alpha0, (double)s.u1);   // the OTHER tangent circle
                for (int i = 1; i <= CCAP; ++i)       // cap at p1: alpha0 -> 0
                    push(s.p1, s.r1, alpha0 * (1.0 - (double)i / CCAP), (double)s.u1);
            } else {
                // Capless LOD: the lateral band alone, tangent circle to tangent circle.
                push(s.p0, s.r0, alpha0, (double)s.u0);
                push(s.p1, s.r1, alpha0, (double)s.u1);
            }

            auto vert = [&](const Ring& rg, int iu, Vec3& p, Vec3& n) {
                const double phi = 2.0 * PI * (double)iu / CU;
                const Vec3 rad = T * std::cos(phi) + B * std::sin(phi);
                n = rad * std::sqrt(std::max(0.0, 1.0 - rg.nAx * rg.nAx)) + ax * rg.nAx;
                p = rg.c + rad * rg.rad;
            };
            // v matches the analytic hit's azimuth (curve.h uses the same onb(axis)), so a
            // pattern reading u/v previews where it will actually land.
            auto uvAt = [&](const Ring& rg, int iu) {
                return Vec3{rg.u, (double)iu / CU, 0.0};
            };
            for (int k = 0; k + 1 < nRings; ++k) {
                const Ring& r0 = rings[k];
                const Ring& r1 = rings[k + 1];
                const bool degen0 = r0.rad <= 1e-12, degen1 = r1.rad <= 1e-12;
                if (degen0 && degen1) continue;
                for (int iu = 0; iu < CU; ++iu) {
                    Vec3 p00, n00, p10, n10, p01, n01, p11, n11;
                    vert(r0, iu, p00, n00); vert(r0, iu + 1, p10, n10);
                    vert(r1, iu, p01, n01); vert(r1, iu + 1, p11, n11);
                    Vec3 uv00 = uvAt(r0, iu), uv10 = uvAt(r0, iu + 1);
                    Vec3 uv01 = uvAt(r1, iu), uv11 = uvAt(r1, iu + 1);
                    if (!degen0) {   // pole rings collapse the quad to a single fan triangle
                        PTri t; t.p0 = p00; t.p1 = p01; t.p2 = p11;
                        t.n0 = n00; t.n1 = n01; t.n2 = n11;
                        applyMat(t, s.matId);
                        t.uv0 = uv00; t.uv1 = uv01; t.uv2 = uv11;
                        out.push_back(t);
                    }
                    if (!degen1) {
                        PTri t; t.p0 = p00; t.p1 = p11; t.p2 = p10;
                        t.n0 = n00; t.n1 = n11; t.n2 = n10;
                        applyMat(t, s.matId);
                        t.uv0 = uv00; t.uv1 = uv11; t.uv2 = uv10;
                        out.push_back(t);
                    }
                }
            }
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
            copyVcol(p, t);                                // so are vertex colours
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
    // Length scale for the see-through path integral, from the geometry actually built.
    {
        Vec3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
        for (const PTri& p : out) {
            for (const Vec3& v : {p.p0, p.p1, p.p2}) {
                lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
                hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
            }
        }
        const Vec3 d = (hi - lo) * 0.5;
        const double r = std::sqrt(dot(d, d));
        geom.radius = (r > 1e-12 && std::isfinite(r)) ? r : 1.0;
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
    // NB the vertex COLOURS are read from the source PTri in the raster pass rather than
    // copied here: STri is written once per visible triangle per frame and read by every
    // covered scanline, so it stays as small as the `src` indirection allows (the same
    // reason it holds `src` instead of a copy of the shading attributes).
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
    bool   vcol;       // interpolate this triangle's per-vertex colour into the albedo
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
    // Interpolated per-vertex colour. Left EMPTY (and untouched by the raster pass)
    // unless the tessellation actually contains a vertex-coloured triangle, so a scene
    // without any pays neither the 24 B/pixel nor the per-pixel write — which is why
    // this is a separate channel rather than another field on a fat per-pixel struct.
    std::vector<Vec3>    vcol;
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
    // clearT is SIX floats per pixel while accumulating (signed optical depth + the
    // open-surface fallback); clearRGB is the folded 3-float transmittance the composite
    // reads. They must be separate buffers: folding 6->3 in place races, because writing
    // slot 3i lands on slot 6(i/2), which another thread may not have consumed yet.
    std::vector<float>             clearT, clearRGB, milkT;
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
inline void fillTriangleG(const STri& t, int W, int H, int y0, int y1, GBuffer& g,
                          const PTri* srcTris = nullptr) {
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
    // Vertex colour is read off the source PTri once per triangle, not per pixel.
    const bool wantVcol = t.vcol && srcTris && !g.vcol.empty();
    const Vec3 C0 = wantVcol ? srcTris[t.src].vc0 : Vec3{1, 1, 1};
    const Vec3 C1 = wantVcol ? srcTris[t.src].vc1 : Vec3{1, 1, 1};
    const Vec3 C2 = wantVcol ? srcTris[t.src].vc2 : Vec3{1, 1, 1};
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
            if (wantVcol)
                g.vcol[row] = (C0 * (w0 * A.invd) + C1 * (w1 * B.invd) + C2 * (w2 * C.invd)) * d;
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
// `clearT` holds THREE floats per pixel (the running RGB transmittance product), `milkT`
// one (the haze is untinted — it is frosting, not glass colour). `tint` is this surface's
// own per-crossing transmittance; `clarity` is the master dial that still multiplies it.
inline void fillTriangleClear(const STri& t, const Camera& cam, int W, int H, int y0, int y1,
                              const GBuffer& g, std::vector<float>& clearT, std::vector<float>& milkT,
                              double clarity, const Vec3& tint,
                              double milkPerSurface, double rimStrength, double invL0) {
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
    // Rounded to float and multiplied IN float, which is what the device twin does
    // (kClearAccum takes a float clarity and a float3 tint). Doing the product in double
    // and rounding once would land a ULP away from the GPU on some pixels, and the two
    // backends are supposed to agree bit for bit.
    const float tauR = (float)clarity * (float)tint.x;
    const float tauG = (float)clarity * (float)tint.y;
    const float tauB = (float)clarity * (float)tint.z;
    // BEER-LAMBERT, NOT PER-SURFACE. `tau` used to be applied once per crossed triangle,
    // which makes the result depend on how finely the glass happens to be tessellated --
    // two panes and one finely-diced pane absorbed differently for no physical reason. The
    // physical quantity is optical depth over the PATH LENGTH through the medium, so keep
    // the same `tau` but reinterpret it as the transmittance of one reference length L0
    // and accumulate sigma * length. `g = -ln(tau)` is that sigma, times L0.
    const float gR = -std::log(std::max(tauR, 1e-6f));
    const float gG = -std::log(std::max(tauG, 1e-6f));
    const float gB = -std::log(std::max(tauB, 1e-6f));
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
            // THE RIM IS A CUE, NOT AN OPTICAL DEPTH. The grazing term exists so a glass
            // object's SILHOUETTE reads; it is a screen-space hint, not a physical
            // quantity, and physical thickness is already accounted for by the clearT
            // product (more glass = more crossings = darker). Multiplying (1 - 0.55) into
            // the haze for every grazing surface therefore double-counts, and worse, it
            // compounds: a sphere crosses 2 surfaces and gets the intended rim, but an -nd
            // extrude crosses 20+ whose side walls are edge-on for GEOMETRIC reasons --
            // they are interior sweeps parallel to the view, not silhouettes -- so after
            // ~10 of them the haze saturates and every pixel of the model turns to flat
            // frost. Combine the rim by MAX instead (stored as 1-max, so a min, which is
            // as order-independent as the product it replaces) and let only the physical
            // per-surface milk compound.
            double rim = rimStrength * graze * graze * graze;
            if (rim > 0.95) rim = 0.95;
            // Front or back? The normal's sign against the view direction says which side
            // of the medium this fragment is, and the signed sum of depths over a CLOSED
            // surface is exactly the path length inside it -- order-independent, like the
            // product it replaces, so still no depth sort.
            // Geometric normal, not the shading one: see the device twin. The winding is
            // a fact about the surface; a shading normal is authored data.
            const Vec3 fnG = cross(B.wpos - A.wpos, C.wpos - A.wpos);
            const float sgn = (dot(fnG, cam.eye - wpos) > 0.0) ? -1.0f : 1.0f;
            const float sd  = sgn * (float)d;
            const float k   = sd * (float)invL0;
            const size_t NPX = (size_t)W * H;
            clearT[row * 6 + 0] += k * gR;
            clearT[row * 6 + 1] += k * gG;
            clearT[row * 6 + 2] += k * gB;
            // Open-surface fallback: a single-sided sheet (a `filter` gel, a one-quad
            // window) has a front and no back, so there is no path length to integrate.
            // Accumulating the FRONT faces' -ln(tau) reproduces the old per-crossing model
            // exactly, and the fold below uses it whenever the signed depth came out empty.
            if (sgn < 0.0f) {
                clearT[row * 6 + 3] += gR;
                clearT[row * 6 + 4] += gG;
                clearT[row * 6 + 5] += gB;
            }
            milkT[row] += sd;                       // signed thickness
            float& r = milkT[NPX + row];
            if ((float)(1.0 - rim) < r) r = (float)(1.0 - rim);
            (void)milkPerSurface;
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
        // `clearT` is THREE floats per pixel (RGB transmittance product), `milkT` one.
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
                const float Tr = clearT[i * 3 + 0], Tg = clearT[i * 3 + 1], Tb = clearT[i * 3 + 2];
                const float mt = milkT[i];
                if (Tr < 1.0f || Tg < 1.0f || Tb < 1.0f || mt < 1.0f) {
                    const double m = 1.0 - (double)mt;
                    // TINT THE HAZE BY THE GLASS IT IS IN. The haze stands for light
                    // scattered inside the crossed surfaces, so it is seen through the
                    // same tint the background is — an untinted one is white light
                    // arriving from nowhere. On a single window this changes almost
                    // nothing (T is near white); on a deep pile of coloured glass it is
                    // the difference between reading the colours and reading a white
                    // veil, because the haze is what dominates there.
                    //
                    // The HUE of the accumulated transmittance, not its magnitude:
                    // scaling by T itself would multiply the haze by the same absorption
                    // twice, and dark glass would lose its frost entirely.
                    const double tmax = std::max({(double)Tr, (double)Tg, (double)Tb});
                    // When the stack is too DENSE to carry a hue, the haze must survive on its own.
                    // `hz` tints the frost by the HUE of the accumulated transmittance, which
                    // needs T normalized by its own max. Falling back to inv = 1 when tmax is
                    // ~0 does not do that -- it leaves hz = milkColor * T, i.e. ~0 -- so both
                    // terms of the composite vanish and the pixel comes out EXACTLY black.
                    // That is wrong in precisely the case the haze exists for: it stands for
                    // light scattered inside the glass, so it is what is left when you can no
                    // longer see through. A deep stack (an -nd extrude multiplies the crossed
                    // surfaces several-fold) turned the whole model into a black silhouette.
                    // With no hue to take, take none: untinted frost.
                    // The threshold is deliberately near the bottom of the double range, not a
                    // comfortable epsilon. `inv` only ever divides T by its own largest
                    // channel, so the result is a hue in [0,1] for ANY strictly positive
                    // tmax, and the division is done in double -- a tmax of 1e-30 is as
                    // safe as one of 0.5. Using 1e-6 threw the hue away far too early: a
                    // ruby glass crossed ~130 times has tmax ~ 0.9^130 = 1e-6, which is
                    // nothing unusual once an -nd extrude multiplies the crossed surfaces,
                    // and the model then jumped from ruby frost to a flat untinted white
                    // while less-crossed parts of the same image (the gems) stayed
                    // coloured. Only a genuine underflow to zero has no hue left to take.
                    const bool   hasHue = tmax > 1e-300;
                    const double inv = hasHue ? 1.0 / tmax : 1.0;
                    const Vec3 hz = hasHue
                        ? Vec3{milkColor.x * (double)Tr * inv,
                               milkColor.y * (double)Tg * inv,
                               milkColor.z * (double)Tb * inv}
                        : milkColor;
                    c = Vec3{c.x * (double)Tr + hz.x * m,
                             c.y * (double)Tg + hz.y * m,
                             c.z * (double)Tb + hz.z * m};
                }
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
                                        RasterScratch* scratch = nullptr,
                                        double hazeCap = 1.0) {
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
    // O8 stage 2: the shading-footprint coefficient for `fw`. W/H override the camera's own
    // film resolution because the preview draws into a window of its own size, and spp is 1
    // (one un-jittered sample per pixel — which is exactly why the preview needs `fw` most).
    // Kept bit-identical to the CUDA preview, which computes the same number in kShade.
    const double fwPerDist = cam.footprintPerDist(1, W, H);

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
            out.push_back(STri{s0, s1, s2, src, needUV, emis, clr,
                               tris[(size_t)src].hasVcol, iy0, iy1});
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
    // The reference length the per-crossing dials are reinterpreted against: a slab this
    // thick absorbs and hazes exactly as one crossing used to, so existing scenes keep
    // their look while the result stops depending on tessellation. Tied to the scene so a
    // 2 cm ring and a 40 m building behave the same.
    const double L0    = (geom.radius > 0.0 ? geom.radius : 1.0) * 0.05;
    const double invL0 = 1.0 / L0;
    const double milkPer = kMilkPerSurface;
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
    // The vertex-colour channel is allocated only when the geometry has one, so a scene
    // without vertex colours pays nothing for the feature: no buffer, no per-pixel write,
    // and the shade pass's `g.vcol.empty()` test folds away to a single predictable branch.
    {
        bool anyVcol = false;
        for (const PTri& t : tris) if (t.hasVcol) { anyVcol = true; break; }
        if (anyVcol) { g.vcol.resize(N); }
        else if (!g.vcol.empty()) g.vcol.clear();     // scratch is reused across frames
    }
    parallelFor(N, [&](size_t a, size_t b) {
        std::fill(g.zbuf.begin() + a, g.zbuf.begin() + b, 0.0f);
    });
    if (!g.vcol.empty())
        parallelFor(N, [&](size_t a, size_t b) {
            std::fill(g.vcol.begin() + a, g.vcol.begin() + b, Vec3{1, 1, 1});
        });
    dispatchBands([&](int y0, int y1) {
        for (const STri& s : stris) {
            if (s.iy1 < y0 || s.iy0 >= y1) continue;   // triangle can't touch this band
            if (seeThrough && s.clear) continue;       // clear surfaces handled in Pass 2b
            fillTriangleG(s, W, H, y0, y1, g, tris.data());
        }
    });

    // -- Pass 2b (see-through only): accumulate the clear surfaces' cumulative transmittance
    // (`clearT`, product of glassClarity per crossed surface) and milk product (`milkT`)
    // against the now-complete opaque depth. Order-independent, so no transparent sort.
    // These ARE read at every pixel by the encode pass, so both get a real fill (parallel,
    // reusing the scratch allocation).
    std::vector<float>& clearT = S.clearT;
    std::vector<float>& milkT  = S.milkT;
    std::vector<float>& clearRGB = S.clearRGB;
    if (!seeThrough) { clearT.clear(); clearRGB.clear(); milkT.clear(); }
    if (seeThrough) {
        // clearT holds SIX floats while accumulating: [0,3) the signed optical depth over
        // path length, [3,6) the front-face-only per-crossing depth used when the geometry
        // turns out to be open. The fold below collapses it back to the three
        // transmittances everything downstream expects, in place.
        clearT.resize(N * 6);
        clearRGB.resize(N * 3);
        // milkT: [0,N) signed thickness (a SUM, so it starts at 0), [N,2N) the silhouette
        // rim as (1 - max), which starts at 1.
        milkT.resize(N * 2);
        parallelFor(N, [&](size_t a, size_t b) {
            std::fill(clearT.begin() + a * 6, clearT.begin() + b * 6, 0.0f);
            std::fill(milkT.begin() + a, milkT.begin() + b, 0.0f);
            std::fill(milkT.begin() + N + a, milkT.begin() + N + b, 1.0f);
        });
        dispatchBands([&](int y0, int y1) {
            for (const STri& s : stris) {
                if (!s.clear) continue;
                if (s.iy1 < y0 || s.iy0 >= y1) continue;
                // The tint rides on the source PTri, so it costs no G-buffer channel and
                // no extra STri field — the same reason STri carries `src` and not a copy
                // of the shading attributes.
                const Vec3& tint = (s.src >= 0 && s.src < (int)tris.size())
                                 ? tris[(size_t)s.src].clearTint : Vec3{1, 1, 1};
                fillTriangleClear(s, cam, W, H, y0, y1, g, clearT, milkT,
                                  glassClarity, tint, kMilkPerSurface, kRimStrength, invL0);
            }
        });
        // -glass-haze: cap how much of a pixel the frost may take, however many surfaces
        // were crossed. The per-surface term is a product, so a sight line through a
        // dozen faceted gems drives it to 1 and the cluster whites out however dark or
        // colourful the glass is.
        //
        // Applied here, once, on the finished product rather than inside the accumulation:
        // clamping a decreasing product at every step and clamping it at the end give
        // exactly the same number (once it is at the floor, further multiplies clamp back
        // to the floor), and doing it here keeps the hot inner loop and the atomics on the
        // device twin untouched. Costs one pass over the buffer, and only when asked for.
        // Fold: optical depth -> transmittance, thickness -> haze, rim applied, all
        // collapsed into the 3-float clearT and 1-float milkT the composite reads. Writing
        // clearT[i*3+c] from clearT[i*6+c] is safe in place because 3i <= 6i.
        const float floorT = (hazeCap < 1.0) ? (float)std::max(0.0, 1.0 - hazeCap) : 0.0f;
        const double kHaze = (milkPer > 0.0 && milkPer < 1.0)
                           ? -std::log(1.0 - milkPer) * invL0 : 0.0;
        parallelFor(N, [&](size_t a, size_t b) {
            for (size_t i = a; i < b; ++i) {
                // Blend rather than branch -- see the device twin's note. The thickness is
                // a sum whose exact value near zero is order-dependent, so a hard
                // threshold there would flip pixels between the two models.
                const float L  = milkT[i];
                const float w  = std::min(std::max(L / (0.02f * (float)L0), 0.0f), 1.0f);
                const double Lh = std::max((double)L, 0.02 * L0);
                for (int c = 0; c < 3; ++c) {
                    float tauP = clearT[i * 6 + c] * (float)invL0 * (float)L0;
                    if (!(tauP > 0.0f)) tauP = 0.0f;
                    const float tauO = clearT[i * 6 + 3 + c];
                    clearRGB[i * 3 + c] = std::exp(-(w * tauP + (1.0f - w) * tauO));
                }
                float m = (float)std::exp(-kHaze * Lh) * milkT[N + i];
                if (m < floorT) m = floorT;
                milkT[i] = m;
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
            if (g.zbuf[i] <= 0.0f) {
                // The environment behind the scene when there is one (pinhole cameras), stamped
                // as a surface at infinity so it is exposed and metered as the tracer's is.
                if (!light.envBg.empty() && cam.projection == CAM_RECTILINEAR) {
                    const int px = (int)(i % (size_t)W), py = (int)(i / (size_t)W);
                    const double ndcx = 2.0 * (px + 0.5) / W - 1.0, ndcy = 1.0 - 2.0 * (py + 0.5) / H;
                    accum[i] = light.envBgAt(normalize(cam.w + cam.u * (ndcx * cam.tanHalfX) +
                                                       cam.v * (ndcy * cam.tanHalfY)));
                    g.zbuf[i] = kEnvBgInvDepth;
                } else {
                    accum[i] = bg;                           // background tint
                }
                continue;
            }
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
                    // curv/cavity stay 0 (see the CUDA twin: no per-face curvature and no
                    // BVH to probe), but `fw` IS known here — a rasterizer knows its own
                    // pixel footprint exactly.
                    const Vec3 dv = g.wpos[i] - cam.eye;
                    const double dist = length(dv);
                    const double cs = dist > 0.0 ? dot(dv, N0) / dist : 0.0;
                    pc = makePatCtx(g.wpos[i], 0.0, N0, g.uv[i].x, g.uv[i].y, 0.0, 0.0,
                                    patShadingFootprint(fwPerDist, dist, cs));
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
            // Per-vertex colour multiplies the material albedo, the same rule the
            // spectral path uses in diffuseReflectance (and glTF's rule for COLOR_0).
            if (!g.vcol.empty()) {
                const Vec3& vc = g.vcol[i];
                col = Vec3{col.x * vc.x, col.y * vc.y, col.z * vc.z};
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
            // RASTER-PBR: the surface's own roughness, honouring the maps the header used to
            // say were "ignored by design". `rough < 0` = no lobe, and the whole specular block
            // below is skipped. A map holds the MATERIAL's roughness, so it goes through the same
            // previewRough mapping bakeOwn applies to the constant.
            double rough = sh ? sh->rough : -1.0;
            if (rough >= 0.0 && sh) {
                if (scenePtr && sh->roughPat >= 0 && sh->roughPat < (int)scenePtr->patterns.size())
                    rough = previewRough(scenePtr->patterns[sh->roughPat].eval(ctx()));
                else if (scenePtr && sh->roughTex >= 0 && sh->roughTex < (int)scenePtr->textures.size())
                    rough = previewRough(scenePtr->textures[sh->roughTex].scalarAt(g.uv[i].x, g.uv[i].y));
                rough = (rough < 0.02) ? 0.02 : (rough > 1.0 ? 1.0 : rough);
            }
            const bool spec = (rough >= 0.0);
            // The highlight colour: the constant f0 of a lobe laid over a diffuse body, or -- for a
            // glossy surface, marked by a negative f0 in bakeOwn -- this pixel's albedo. Such a
            // surface IS its lobe, so it also gets no diffuse term below (0.368.0).
            const bool allLobe = spec && (sh->f0.x < 0.0);
            const Vec3 f0 = (sh->f0.x < 0.0) ? col : sh->f0;
            // The calibrated lights (0.368.0): real colour and strength, 1/d^2, and a highlight
            // widened by each source's angular size. Diffuse radiance is albedo * E / pi and the
            // lobe's is BRDF * E, in the same units -- before 0.368.0 the highlight used the diffuse
            // term's weight without its 1/pi, and so came out a factor pi too weak beside it.
            Vec3 diffE{0, 0, 0}, specAcc{0, 0, 0};
            for (const auto& lp : light.lights) {
                Vec3 Ld, E; double ang;
                if (!previewLightAt(lp, g.wpos[i], Ld, E, ang)) continue;
                const double ndl = dot(N3, Ld);
                if (ndl <= 0.0) continue;
                diffE += E * ndl;
                if (spec) {
                    // The direct lobe: this is the half of the split-sum a preview cannot fake,
                    // because the moving highlight is what reads as "satin" rather than "chalk".
                    const double gg = ggxSpec(N3, V, Ld, widenRough(rough, ang));
                    if (gg > 0.0) {
                        const double m1 = 1.0 - std::max(0.0, dot(V, normalize(V + Ld)));
                        const double m2 = m1 * m1, f = m2 * m2 * m1;   // Schlick's (1-cos)^5
                        specAcc += Vec3{(f0.x + (1.0 - f0.x) * f) * E.x,
                                        (f0.y + (1.0 - f0.y) * f) * E.y,
                                        (f0.z + (1.0 - f0.z) * f) * E.z} * gg;
                    }
                }
            }
            if (allLobe) {
                // A glossy surface has no diffuse term: all of it is the lobe added below -- the
                // lights as highlights, the surroundings (PreviewLight::envSpec) as the reflection.
                accum[i] = Vec3{0.0, 0.0, 0.0};
            } else {
                // Diffuse: the lights, the surroundings' irradiance (order-2 SH of the environment
                // map: the env light and, in a scene, the light probe), and a faint headlight.
                const double head = std::max(0.0, dot(N3, V));
                const Vec3 amb = light.ambientAt(N3);
                accum[i] = Vec3{col.x * (diffE.x * (1.0 / PI) + amb.x + light.fill * head),
                                col.y * (diffE.y * (1.0 / PI) + amb.y + light.fill * head),
                                col.z * (diffE.z * (1.0 / PI) + amb.z + light.fill * head)};
            }
            if (spec) {
                // The ENVIRONMENT half of the split sum: the surroundings pre-blurred for this
                // roughness, read along the reflection of the view -- the part that makes a metal
                // look like a metal. Before 0.368.0 this was a two-colour sky/ground gradient
                // anchored on a dim ambient, with nothing in it to reflect.
                double A = 0.0, B = 0.0;
                envBrdfApprox(std::max(1e-4, dot(N3, V)), rough, A, B);
                const Vec3 Rv = N3 * (2.0 * dot(N3, V)) - V;
                const Vec3 env = light.envSpecularAt(light.lookupDir(g.wpos[i], Rv), rough);
                accum[i] += Vec3{(f0.x * A + B) * env.x, (f0.y * A + B) * env.y, (f0.z * A + B) * env.z}
                          + specAcc;
            }
        }
    });

    // Auto-exposure + sRGB tone map: shared with the CUDA rasterizer (see exposeAndEncode),
    // so both backends anchor and encode identically. The see-through buffers are empty when
    // !seeThrough and simply ignored by the helper in that case. Rides the same worker pool
    // as the passes above (its three scans used to spawn their own threads each).
    return exposeAndEncode(accum, g.zbuf, g.emis, W, H, nThreads, expComp, autoExpose,
                           lockAnchor, seeThrough, clearRGB, milkT, kMilkColor, pool);
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
