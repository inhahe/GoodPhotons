#pragma once
// ============================================================================================
//  ggx.h -- the glossy lobe: a GGX (Trowbridge-Reitz) microfacet reflector, host side.
// ============================================================================================
//
// Every rough reflection the tracers sample and evaluate -- a `glossy` material, a rough
// interference reflector, a `layered` material's rough coat, the stochastic coat model's glossy
// body (layered.h) -- is this lobe. The device's twins are in render_cuda.cu (dGgx*).
//
// THE LOBE. For the pair (a, b), both unit and above the shading normal n, h their half vector:
//     f(a, b) = r D(h) G2(a, b) / (4 cos_a cos_b)
// with the isotropic GGX distribution D, the height-correlated Smith masking-shadowing G2, and r
// the material's reflectance (times an interference term at h, or a coat's Fresnel at h, both the
// caller's). With A(w) = sqrt(cos^2 + alpha^2 sin^2), Smith's Lambda(w) = (A / cos - 1) / 2, so
//     G1(w) = 2 cos / (cos + A),    G2 = 2 cos_a cos_b / (cos_b A_a + cos_a A_b),
//     f = r D / (2 (cos_b A_a + cos_a A_b)),
// which stays finite at grazing. f is symmetric in the pair: the lobe is RECIPROCAL, so a walk from
// the light and a walk from the camera evaluate the same function, and neither carries an adjoint
// ratio (the Phong lobe this replaced divided by the light side's cosine, and a light walk's
// factor cos(out) / cos(in) had no bound at grazing arrival -- known-issues GLOSSY-ADJ-VARIANCE).
//
// SAMPLING draws the distribution of normals VISIBLE from the arrival a (Heitz 2018, by the
// spherical caps of Dupuy & Benyoub 2023) and reflects a about the normal drawn. Its density is
//     pdf(b | a) = G1(a) D(h) / (4 cos_a) = D / (2 (cos_a + A_a)),
// so a draw's f cos / pdf is r times
//     w(a, b) = G2 / G1(a) = cos_b (cos_a + A_a) / (cos_b A_a + cos_a A_b),
// never more than 1, from either end of a path. What w falls short of 1 is the light the single
// scattering model loses to masking (more as roughness and grazing grow).
//
// MULTIPLE SCATTERING (Kulla & Conty 2017). What w falls short of 1 is light the model loses to the
// microfacets' masking, and it is not small for a rough lobe: a white reflector of roughness 0.5
// returns 0.86 at normal incidence, of roughness 1 0.55 -- where the Phong lobe returned all of it.
// Light really bounces between microfacets and leaves, so a glossy lobe adds what its single
// scattering misses as a second lobe,
//     f_ms(a, b) = (1 - E(cos_a)) (1 - E(cos_b)) / (pi (1 - Eavg)),
// E the single-scattering albedo by the arrival's cosine and Eavg its cosine-weighted mean
// (tabulated: ggx_ms_table.h, tools/gen_ggx_ms_table.py): reciprocal, and with it a white lobe
// returns exactly what it receives at every angle. The lobe's reflectance r tints the sum, so its
// albedo is r, as the Phong lobe's was (the colour of light that bounces twice off a coloured metal is
// r^2, not r: a refinement left out, so a draw's factor stays r times a wavelength-free weight). A
// draw picks the single-scattering lobe with probability E(cos_a) -- its share -- else a cosine draw
// for f_ms, and its weight is (f_ss + f_ms) cos / the mixture's density. A lobe whose reflectance is
// taken at the half vector -- a rough interference reflector, a layered material's coat -- has no
// f_ms: it has no half vector to take it at, and a dielectric coat's facets bounce a few percent of
// a few percent.
//
// ROUGHNESS. A material's `roughness` r keeps the meaning it had under the Phong lobe -- the
// lobe's width -- so a scene renders its highlights as wide as before: alpha = 0.646 r, the GGX
// alpha whose half-width at half maximum in the reflected direction (1.29 alpha) equals the Phong
// lobe's (0.833 r, exponent 2 / r^2 - 2) for a small r. It is the factor the preview rasterizer
// (raster.h previewRough) and the glTF importer (gltf.h gltfRoughness: glTF's perceptual roughness
// rho is alpha = rho^2) already used for the same equivalence, so both are exact now. A rough
// dielectric's cone jitter (render.h sampleRoughCone) still reads the Phong width of the same r.

#include "linalg.h"
#include "ggx_ms_table.h"

#include <cmath>

namespace ggx {

constexpr double kPi = 3.14159265358979323846;
// alpha for one unit of roughness (see ROUGHNESS above).
constexpr double kAlphaPerRoughness = 0.646;
// The least roughness a lobe is given: the Phong lobe's own clamp, so a near-mirror stays finite.
constexpr double kMinRoughness = 1e-3;

// The GGX alpha of a material roughness.
inline double alpha(double roughness) {
    return kAlphaPerRoughness * (roughness < kMinRoughness ? kMinRoughness : roughness);
}

// A(w) for a direction at cosine `c` (> 0) to the normal: sqrt(c^2 + alpha^2 (1 - c^2)).
inline double smithA(double c, double a2) {
    const double s2 = 1.0 - c * c;
    return std::sqrt(c * c + a2 * (s2 > 0.0 ? s2 : 0.0));
}

// The lobe's terms for the pair (a, b) about the unit normal n. `ok` false (and the rest
// unset) unless both lie above n. D is read from the half vector's cosine and its squared sine
// taken by a cross product, so a half vector near n keeps its precision at a small alpha.
struct Pair {
    bool ok = false;
    double ca = 0, cb = 0;   // the pair's cosines to n
    double Aa = 0, Ab = 0;   // their A terms
    double D = 0;            // the distribution at their half vector
    double alpha = 0;        // the lobe's
};
inline Pair pair(const Vec3& n, const Vec3& a, const Vec3& b, double alpha) {
    Pair p;
    p.alpha = alpha;
    p.ca = dot(a, n);
    p.cb = dot(b, n);
    if (!(p.ca > 0.0) || !(p.cb > 0.0)) return p;
    const Vec3 hv = a + b;
    const double l = length(hv);
    if (!(l > 0.0)) return p;
    const Vec3 h = hv * (1.0 / l);
    const double ch = dot(h, n);
    const Vec3 x = cross(n, h);
    const double s2 = dot(x, x);
    const double a2 = alpha * alpha;
    const double d = a2 * ch * ch + s2;
    if (!(d > 0.0)) return p;
    p.D = a2 / (kPi * d * d);
    p.Aa = smithA(p.ca, a2);
    p.Ab = smithA(p.cb, a2);
    p.ok = true;
    return p;
}

// f(a, b) / r: the BSDF value without its reflectance. Symmetric in the pair.
inline double f(const Pair& p) { return p.ok ? p.D / (2.0 * (p.cb * p.Aa + p.ca * p.Ab)) : 0.0; }
// The density, in solid angle, with which sample() draws b given the arrival a.
inline double pdf(const Pair& p) { return p.ok ? p.D / (2.0 * (p.ca + p.Aa)) : 0.0; }
// f cos(b) / pdf over r for that draw: G2 / G1(a), in (0, 1].
inline double weight(const Pair& p) {
    return p.ok ? p.cb * (p.ca + p.Aa) / (p.cb * p.Aa + p.ca * p.Ab) : 0.0;
}

// A direction drawn from the lobe about the unit normal n for the arrival a (unit, pointing away
// from the surface toward where the walk came from), from two canonical uniforms in [0, 1).
// u1 == 1 returns the exact mirror direction: a deterministic caller's lattice (mode W) puts its
// sample 0 there, so a 1-spp preview reflects along the mirror; no rng draw reaches it. An arrival
// below n has no visible normal: the mirror is returned, below n, and the caller ends the path
// there as for any draw below the surface.
inline Vec3 sampleUV(const Vec3& n, const Vec3& a, double alpha, double u1, double u2) {
    const double ca = dot(a, n);
    if (u1 >= 1.0 || !(ca > 0.0)) return reflect(a * -1.0, n);
    Vec3 t, bt;
    onb(n, t, bt);
    // The arrival in the frame (t, bt, n), stretched to the unit-roughness configuration.
    const Vec3 vs = normalize(Vec3{alpha * dot(a, t), alpha * dot(a, bt), ca});
    // A point uniform on the spherical cap z in (-vs.z, 1]; with vs it spans the visible normals.
    const double phi = 2.0 * kPi * u2;
    const double z = u1 * (1.0 + vs.z) - vs.z;
    const double s2 = 1.0 - z * z;
    const double st = std::sqrt(s2 > 0.0 ? s2 : 0.0);
    const Vec3 hs{st * std::cos(phi) + vs.x, st * std::sin(phi) + vs.y, z + vs.z};
    // Back to alpha: the microfacet normal, then a reflected about it.
    const Vec3 hl = normalize(Vec3{alpha * hs.x, alpha * hs.y, hs.z > 0.0 ? hs.z : 0.0});
    const Vec3 h = t * hl.x + bt * hl.y + n * hl.z;
    return normalize(h * (2.0 * dot(a, h)) - a);
}

// --- Multiple scattering (see MULTIPLE SCATTERING above) ------------------------------------------
inline constexpr float kMsE[GGX_MS_NA * GGX_MS_NM] = GGX_MS_E_INIT;
inline constexpr float kMsEavg[GGX_MS_NA] = GGX_MS_EAVG_INIT;

// The table's place for `alpha`: row `i`, weight `t` toward row i + 1.
inline void msRow(double alpha, int& i, double& t) {
    double x = alpha / GGX_MS_ALPHA_MAX;
    x = (x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)) * (GGX_MS_NA - 1);
    i = (int)x;
    if (i > GGX_MS_NA - 2) i = GGX_MS_NA - 2;
    t = x - (double)i;
}
// E(c, alpha): a white lobe's single-scattering albedo for an arrival at cosine `c`, bilinear in
// (alpha, sqrt(c)).
inline double msE(double c, double alpha) {
    int i; double t;
    msRow(alpha, i, t);
    double y = std::sqrt(c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c)) * (GGX_MS_NM - 1);
    int j = (int)y;
    if (j > GGX_MS_NM - 2) j = GGX_MS_NM - 2;
    const double u = y - (double)j;
    const float* r0 = kMsE + i * GGX_MS_NM;
    const float* r1 = r0 + GGX_MS_NM;
    const double e0 = r0[j] + (r0[j + 1] - r0[j]) * u, e1 = r1[j] + (r1[j + 1] - r1[j]) * u;
    return e0 + (e1 - e0) * t;
}
// Eavg(alpha) = 2 int_0^1 E(c) c dc.
inline double msEavg(double alpha) {
    int i; double t;
    msRow(alpha, i, t);
    return kMsEavg[i] + (kMsEavg[i + 1] - kMsEavg[i]) * t;
}
// f_ms for the pair: 0 where the single scattering keeps all (a near-mirror lobe).
inline double fMs(const Pair& p) {
    if (!p.ok) return 0.0;
    const double ea = msEavg(p.alpha);
    if (!(1.0 - ea > 1e-6)) return 0.0;
    return (1.0 - msE(p.ca, p.alpha)) * (1.0 - msE(p.cb, p.alpha)) / (kPi * (1.0 - ea));
}

// THE LOBE A MATERIAL SHADES WITH: single scattering, and its multiple scattering where `ms`. A draw
// takes the single-scattering lobe with probability E(cos_a) (msShare), else a cosine draw for f_ms.
// Device twins dGgxFAll / dGgxPdfAll / dGgxWeightAll / dGgxSampleUVAll.
inline double msShare(const Pair& p) { return p.ok ? msE(p.ca, p.alpha) : 1.0; }
inline double fAll(const Pair& p, bool ms) { return ms ? f(p) + fMs(p) : f(p); }
inline double pdfAll(const Pair& p, bool ms) {
    if (!ms || !p.ok) return pdf(p);
    const double ps = msShare(p);
    return ps * pdf(p) + (1.0 - ps) * p.cb / kPi;
}
// f cos(b) / pdfAll for that draw, over r: in (0, 1] without `ms`; with it a white lobe's mean is 1.
inline double weightAll(const Pair& p, bool ms) {
    if (!ms) return weight(p);
    if (!p.ok) return 0.0;
    const double d = pdfAll(p, true);
    return (d > 0.0) ? fAll(p, true) * p.cb / d : 0.0;
}
// sampleUV for the lobe with its multiple scattering where `ms`: u1 splits at the single scattering's
// share (u1 == 1 still the mirror; an arrival below n still returns it).
inline Vec3 sampleUVAll(const Vec3& n, const Vec3& a, double alpha, bool ms, double u1, double u2) {
    const double ca = dot(a, n);
    if (!ms || u1 >= 1.0 || !(ca > 0.0)) return sampleUV(n, a, alpha, u1, u2);
    const double ps = msE(ca, alpha);
    if (!(ps < 1.0) || u1 < ps) return sampleUV(n, a, alpha, (ps < 1.0) ? u1 / ps : u1, u2);
    const double v = (u1 - ps) / (1.0 - ps);          // a cosine draw about n
    const double r = std::sqrt(v), phi = 2.0 * kPi * u2;
    Vec3 t, bt;
    onb(n, t, bt);
    return normalize(t * (r * std::cos(phi)) + bt * (r * std::sin(phi)) + n * std::sqrt(1.0 - v > 0.0 ? 1.0 - v : 0.0));
}

}  // namespace ggx
