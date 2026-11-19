#pragma once
// ============================================================================================
//  ggx.h -- the glossy lobe: a GGX (Trowbridge-Reitz) microfacet reflector, host and device.
// ============================================================================================
//
// Every rough reflection the tracers sample and evaluate -- a `glossy` material, a rough
// interference reflector, a `layered` material's rough coat, the stochastic coat model's glossy
// body (layered.h) -- is this lobe, and this one file is it on both backends: the scalar math is
// templated on its precision (double on the host; on the device double where a connection or MIS
// reads the value, the device's working precision where a draw takes its weight), the vector math on
// any type with .x .y .z, and the tables are read through one lookup that picks the host's array or
// the device's copy.
//
// THE LOBE. For the pair (a, b), both unit and above the shading normal n, h their half vector:
//     f(a, b) = r D(h) G2(a, b) / (4 cos_a cos_b)
// with the isotropic GGX distribution D, the height-correlated Smith masking-shadowing G2, and r
// the material's reflectance (times an interference term at h, or a coat's Fresnel at h, both the
// caller's). With A(w) = sqrt(cos^2 + alpha^2 sin^2), Smith's Lambda(w) = (A / cos - 1) / 2, so
//     G1(w) = 2 cos / (cos + A),    G2 = 2 cos_a cos_b / (cos_b A_a + cos_a A_b),
//     f = r D / (2 (cos_b A_a + cos_a A_b)),
// which stays finite at grazing. f is symmetric in the pair: the lobe is RECIPROCAL, so a walk from
// the light and a walk from the camera evaluate the same function, and neither carries an adjoint.
//
// SAMPLING draws the distribution of normals VISIBLE from the arrival a (Heitz 2018, by the
// spherical caps of Dupuy & Benyoub 2023) and reflects a about the normal drawn. Its density is
//     pdf(b | a) = G1(a) D(h) / (4 cos_a) = D / (2 (cos_a + A_a)),
// so a draw's f cos / pdf is r times
//     w(a, b) = G2 / G1(a) = cos_b (cos_a + A_a) / (cos_b A_a + cos_a A_b),
// never more than 1, from either end of a path.
//
// MULTIPLE SCATTERING (Kulla & Conty 2017). What w falls short of 1 is light the model loses to the
// microfacets' masking, and it is not small for a rough lobe: a white reflector of roughness 0.5
// returns 0.86 of it at normal incidence, of roughness 1 0.55. Light really bounces between
// microfacets and leaves, so a glossy lobe adds what its single scattering misses as a second lobe,
//     f_ms(a, b) = (1 - E(cos_a)) (1 - E(cos_b)) / (pi (1 - Eavg)),
// E the single-scattering albedo by the arrival's cosine and Eavg its cosine-weighted mean
// (tabulated: ggx_ms_table.h, tools/gen_ggx_ms_table.py): reciprocal, and with it a white lobe
// returns what it receives at every angle. The lobe's reflectance r tints the sum, so its albedo is
// r (the colour of light that bounces twice off a coloured metal is r^2, not r: a refinement left
// out, so a draw's factor stays r times a wavelength-free weight). A draw takes the
// single-scattering lobe with probability E(cos_a) -- its share -- else f_ms's own draw, from the
// tabulated shares of the cosine grid's cells under (1 - E) cos (the two rows about its alpha, mixed),
// in proportion to the cosine within a cell, whose weight is therefore close to 1 - E(cos_a)
// wherever it leaves. The draw's weight is (f_ss + f_ms) cos / the mixture's density. A lobe whose reflectance is
// taken at the half vector -- a rough interference reflector, a Fresnel or thin-film coat -- has no
// f_ms: it has no half vector to take it at, and a dielectric coat's facets bounce a few percent of
// a few percent.
//
// ROUGHNESS. alpha = 0.646 r: the GGX alpha as wide at half maximum in the reflected direction
// (1.29 alpha) as a Phong lobe of exponent 2 / r^2 - 2 (0.833 r) for a small r, the width a roughness
// gives a rough dielectric's cone jitter (render.h sampleRoughCone) too, and the factor the preview
// rasterizer (raster.h previewRough) and the glTF importer (gltf.h gltfRoughness: glTF's perceptual
// roughness rho is alpha = rho^2) take. Clamped to [0.001, 2], the multiple-scattering table's span.

#include "ggx_ms_table.h"

#include <cmath>

// GGX_V: before a template over the vector (or rng) type. Host code instantiates it with the host's
// Vec3 and Pcg32, whose members nvcc's device pass sees as host-only; that instantiation is never
// called from device code, so its execution-space check is off.
#if defined(__CUDACC__)
  #if defined(_MSC_VER)
    #define GGX_V __pragma(nv_exec_check_disable)
  #else
    #define GGX_V _Pragma("nv_exec_check_disable")
  #endif
  #define GGX_HD __host__ __device__ static inline
  #define GGX_SQRT(x) ::sqrt(x)
  #define GGX_COS(x) ::cos(x)
  #define GGX_SIN(x) ::sin(x)
#else
  #define GGX_V
  #define GGX_HD static inline
  #define GGX_SQRT(x) std::sqrt(x)
  #define GGX_COS(x) std::cos(x)
  #define GGX_SIN(x) std::sin(x)
#endif

namespace ggx {

constexpr double kPi = 3.14159265358979323846;
// alpha for one unit of roughness (see ROUGHNESS above).
constexpr double kAlphaPerRoughness = 0.646;
// The roughness a lobe is held to: the least keeps a near-mirror finite; the most is the
// multiple-scattering table's last row.
constexpr double kMinRoughness = 1e-3;
constexpr double kMaxRoughness = GGX_MS_ALPHA_MAX / kAlphaPerRoughness;

// --- The tables (ggx_ms_table.h): the host's arrays, and on the device its own copies ---------------
inline constexpr float kMsE[GGX_MS_NA * GGX_MS_NM] = GGX_MS_E_INIT;
inline constexpr float kMsEavg[GGX_MS_NA] = GGX_MS_EAVG_INIT;
inline constexpr float kMsCdf[GGX_MS_NA * (GGX_MS_NT + 1)] = GGX_MS_CDF_INIT;
#if defined(__CUDACC__)
static __device__ const float dMsE[GGX_MS_NA * GGX_MS_NM] = GGX_MS_E_INIT;
static __device__ const float dMsEavg[GGX_MS_NA] = GGX_MS_EAVG_INIT;
static __device__ const float dMsCdf[GGX_MS_NA * (GGX_MS_NT + 1)] = GGX_MS_CDF_INIT;
#endif
GGX_HD const float* msETable() {
#if defined(__CUDA_ARCH__)
    return dMsE;
#else
    return kMsE;
#endif
}
GGX_HD const float* msEavgTable() {
#if defined(__CUDA_ARCH__)
    return dMsEavg;
#else
    return kMsEavg;
#endif
}
GGX_HD const float* msCdfTable() {
#if defined(__CUDA_ARCH__)
    return dMsCdf;
#else
    return kMsCdf;
#endif
}

// --- Vector helpers over any type with .x .y .z (the host's Vec3, the device's DVec3) ---------------
// copysign(1, z), the device's by splicing the sign bit: nvcc's device pass compiles against the host
// compiler's <cmath>, whose float copysign (MSVC 14.5x) came out of it returning 0 for every input
// (render_cuda.cu, the copysign overloads at the top of namespace gpu).
GGX_HD float signOf(float z) {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0x3f800000 | (__float_as_int(z) & (int)0x80000000));
#else
    return std::copysign(1.0f, z);
#endif
}
GGX_HD double signOf(double z) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(0x3ff0000000000000LL | (__double_as_longlong(z) & (long long)0x8000000000000000ULL));
#else
    return std::copysign(1.0, z);
#endif
}
GGX_V template <class V> GGX_HD auto vdot(const V& a, const V& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
GGX_V template <class V> GGX_HD V vnorm(const V& v) {
    const auto l = GGX_SQRT(vdot(v, v));
    return V{v.x / l, v.y / l, v.z / l};
}
// An orthonormal frame (t, b) about the unit n (linalg.h onb's construction).
GGX_V template <class V> GGX_HD void vonb(const V& n, V& t, V& b) {
    using T = decltype(n.x);
    const T sign = signOf(n.z);
    const T a = (T)-1 / (sign + n.z);
    const T d = n.x * n.y * a;
    t = V{(T)1 + sign * n.x * n.x * a, sign * d, -sign * n.x};
    b = V{d, sign + n.y * n.y * a, -n.y};
}
// The mirror of the arrival a about n: reflect(-a, n), as linalg.h computes it.
GGX_V template <class V> GGX_HD V vmirror(const V& n, const V& a) {
    using T = decltype(n.x);
    const V d{-a.x, -a.y, -a.z};
    const T k = (T)2 * vdot(d, n);
    return V{d.x - n.x * k, d.y - n.y * k, d.z - n.z * k};
}
// A direction at cosine `c` to n and azimuth `phi` in the frame (t, b, n).
GGX_V template <class V, class T> GGX_HD V vframe(const V& t, const V& b, const V& n, T c, T phi) {
    const T s2 = (T)1 - c * c;
    const T s = GGX_SQRT(s2 > (T)0 ? s2 : (T)0);
    const T x = s * GGX_COS(phi), y = s * GGX_SIN(phi);
    return vnorm(V{t.x * x + b.x * y + n.x * c, t.y * x + b.y * y + n.y * c, t.z * x + b.z * y + n.z * c});
}

// The GGX alpha of a material roughness.
template <class T> GGX_HD T alpha(T roughness) {
    const T r = !(roughness > (T)kMinRoughness) ? (T)kMinRoughness   // NaN too
                                                : (roughness > (T)kMaxRoughness ? (T)kMaxRoughness : roughness);
    return (T)kAlphaPerRoughness * r;
}

// A(w) for a direction at cosine `c` (> 0) to the normal: sqrt(c^2 + alpha^2 (1 - c^2)).
template <class T> GGX_HD T smithA(T c, T a2) {
    const T s2 = (T)1 - c * c;
    return GGX_SQRT(c * c + a2 * (s2 > (T)0 ? s2 : (T)0));
}

// The lobe's terms for the pair (a, b) about the unit normal n, in precision T. `ok` false (and the
// rest unset) unless both lie above n. D is read from the half vector's cosine and its squared sine
// taken by a cross product, so a half vector near n keeps its precision at a small alpha.
template <class T> struct PairT {
    bool ok = false;
    T ca = 0, cb = 0;   // the pair's cosines to n
    T Aa = 0, Ab = 0;   // their A terms
    T D = 0;            // the distribution at their half vector
    T alpha = 0;        // the lobe's
};
using Pair = PairT<double>;
template <class T, class V> GGX_HD PairT<T> pairT(const V& n, const V& a, const V& b, T alpha) {
    PairT<T> p;
    p.alpha = alpha;
    const T nx = (T)n.x, ny = (T)n.y, nz = (T)n.z;
    const T ax = (T)a.x, ay = (T)a.y, az = (T)a.z, bx = (T)b.x, by = (T)b.y, bz = (T)b.z;
    p.ca = ax * nx + ay * ny + az * nz;
    p.cb = bx * nx + by * ny + bz * nz;
    if (!(p.ca > (T)0) || !(p.cb > (T)0)) return p;
    T hx = ax + bx, hy = ay + by, hz = az + bz;
    const T l = GGX_SQRT(hx * hx + hy * hy + hz * hz);
    if (!(l > (T)0)) return p;
    hx /= l; hy /= l; hz /= l;
    const T ch = hx * nx + hy * ny + hz * nz;
    const T cx = ny * hz - nz * hy, cy = nz * hx - nx * hz, cz = nx * hy - ny * hx;
    const T s2 = cx * cx + cy * cy + cz * cz;
    const T a2 = alpha * alpha;
    const T d = a2 * ch * ch + s2;
    if (!(d > (T)0)) return p;
    p.D = a2 / ((T)kPi * d * d);
    p.Aa = smithA(p.ca, a2);
    p.Ab = smithA(p.cb, a2);
    p.ok = true;
    return p;
}
GGX_V template <class V> GGX_HD Pair pair(const V& n, const V& a, const V& b, double alpha) {
    return pairT<double>(n, a, b, alpha);
}

// f(a, b) / r: the single-scattering BSDF value without its reflectance. Symmetric in the pair.
template <class T> GGX_HD T f(const PairT<T>& p) { return p.ok ? p.D / ((T)2 * (p.cb * p.Aa + p.ca * p.Ab)) : (T)0; }
// The density, in solid angle, with which sampleUV draws b given the arrival a.
template <class T> GGX_HD T pdf(const PairT<T>& p) { return p.ok ? p.D / ((T)2 * (p.ca + p.Aa)) : (T)0; }
// f cos(b) / pdf over r for that draw: G2 / G1(a), in (0, 1].
template <class T> GGX_HD T weight(const PairT<T>& p) {
    return p.ok ? p.cb * (p.ca + p.Aa) / (p.cb * p.Aa + p.ca * p.Ab) : (T)0;
}

// A direction drawn from the single-scattering lobe about the unit normal n for the arrival a (unit,
// pointing away from the surface toward where the walk came from), from two canonical uniforms in
// [0, 1). u1 == 1 returns the exact mirror direction: a deterministic caller's lattice (mode W) puts
// its sample 0 there; no rng draw reaches it. An arrival below n has no visible normal: the mirror is
// returned, below n, and the caller ends the path there as for any draw below the surface.
GGX_V template <class V, class T> GGX_HD V sampleUV(const V& n, const V& a, T alpha, T u1, T u2) {
    const T ca = vdot(a, n);
    if (u1 >= (T)1 || !(ca > (T)0)) return vmirror(n, a);
    V t, bt;
    vonb(n, t, bt);
    // The arrival in the frame (t, bt, n), stretched to the unit-roughness configuration.
    const V vs = vnorm(V{alpha * vdot(a, t), alpha * vdot(a, bt), ca});
    // A point uniform on the spherical cap z in (-vs.z, 1]; with vs it spans the visible normals.
    const T phi = (T)2 * (T)kPi * u2;
    const T z = u1 * ((T)1 + vs.z) - vs.z;
    const T s2 = (T)1 - z * z;
    const T st = GGX_SQRT(s2 > (T)0 ? s2 : (T)0);
    const V hs{st * GGX_COS(phi) + vs.x, st * GGX_SIN(phi) + vs.y, z + vs.z};
    // Back to alpha: the microfacet normal, then a reflected about it. The cap's edge (u1 == 0) seen
    // straight down (vs == n) is the one point with no normal there: hs is 0. A set of no measure,
    // which an rng's exact 0 reaches once in 2^24 draws; it takes the mirror rather than 0 / 0.
    const V hv{alpha * hs.x, alpha * hs.y, hs.z > (T)0 ? hs.z : (T)0};
    if (!(vdot(hv, hv) > (T)0)) return vmirror(n, a);
    const V hl = vnorm(hv);
    const V h{t.x * hl.x + bt.x * hl.y + n.x * hl.z, t.y * hl.x + bt.y * hl.y + n.y * hl.z,
              t.z * hl.x + bt.z * hl.y + n.z * hl.z};
    const T k = (T)2 * vdot(a, h);
    return vnorm(V{h.x * k - a.x, h.y * k - a.y, h.z * k - a.z});
}

// --- Multiple scattering (see MULTIPLE SCATTERING above) -------------------------------------------
#if !defined(GGX_MS_ROWS_SQRT_ALPHA)
  #error "ggx_ms_table.h predates rows uniform in sqrt(alpha): regenerate it (tools/gen_ggx_ms_table.py)"
#endif
// The table's place for `alpha`: row `i`, weight `t` toward row i + 1, the rows uniform in sqrt(alpha)
// (a near-mirror lobe's E is a function of cos / alpha near grazing, so the small alphas need rows).
// E, Eavg and f_ms's draw are all linear in t, and one place serves a draw's every lookup.
template <class T> struct MsAt {
    int i = 0;
    T t = 0;
};
template <class T> GGX_HD MsAt<T> msAt(T alpha) {
    T x = alpha / (T)GGX_MS_ALPHA_MAX;
    x = GGX_SQRT(!(x > (T)0) ? (T)0 : (x > (T)1 ? (T)1 : x)) * (T)(GGX_MS_NA - 1);   // NaN: row 0
    MsAt<T> at;
    at.i = (int)x;
    if (at.i > GGX_MS_NA - 2) at.i = GGX_MS_NA - 2;
    at.t = x - (T)at.i;
    return at;
}
// E(c): a white lobe's single-scattering albedo for an arrival at cosine `c`, bilinear in
// (sqrt(alpha), sqrt(c)).
template <class T> GGX_HD T msE(T c, const MsAt<T>& at) {
    const T y = GGX_SQRT(!(c > (T)0) ? (T)0 : (c > (T)1 ? (T)1 : c)) * (T)(GGX_MS_NM - 1);   // NaN: 0
    int j = (int)y;
    if (j > GGX_MS_NM - 2) j = GGX_MS_NM - 2;
    const T u = y - (T)j;
    const float* r0 = msETable() + at.i * GGX_MS_NM;
    const float* r1 = r0 + GGX_MS_NM;
    const T e0 = (T)r0[j] + ((T)r0[j + 1] - (T)r0[j]) * u, e1 = (T)r1[j] + ((T)r1[j + 1] - (T)r1[j]) * u;
    return e0 + (e1 - e0) * at.t;
}
template <class T> GGX_HD T msE(T c, T alpha) { return msE(c, msAt(alpha)); }
// Eavg = 2 int_0^1 E(c) c dc, of the tabulated E itself (so f_ms integrates to 1 - E(cos_a) exactly).
template <class T> GGX_HD T msEavg(const MsAt<T>& at) {
    const float* e = msEavgTable();
    return (T)e[at.i] + ((T)e[at.i + 1] - (T)e[at.i]) * at.t;
}
// f_ms for the pair given E at its arrival `ea`: 0 where the single scattering keeps all (a
// near-mirror lobe seen steep).
template <class T> GGX_HD T fMsAt(const PairT<T>& p, const MsAt<T>& at, T ea) {
    const T eavg = msEavg(at);
    if (!((T)1 - eavg > (T)1e-6)) return (T)0;
    return ((T)1 - ea) * ((T)1 - msE(p.cb, at)) / ((T)kPi * ((T)1 - eavg));
}
template <class T> GGX_HD T fMs(const PairT<T>& p) {
    if (!p.ok) return (T)0;
    const MsAt<T> at = msAt(p.alpha);
    return fMsAt(p, at, msE(p.ca, at));
}
// f_ms's draw. E is linear in t between the table's rows, so f_ms cos's density in the cosine,
// (1 - E) cos over its mass (1 - Eavg) / 2, is the two neighbouring rows' densities mixed by their
// masses: the draw takes the upper row with probability `p1` (msRows) and that row's tabulated cell
// shares (ggx_ms_table.h), cell k spanning the cosines (mu0, mu1) = ((k / NT)^2, ((k + 1) / NT)^2),
// and within a cell a density in proportion to the cosine -- f_ms cos's own shape there but for
// (1 - E), which a cell barely moves. (Row 0, alpha 0, has no mass: it is never taken.)
template <class T> GGX_HD void msRows(const MsAt<T>& at, const float*& c0, const float*& c1, T& p1) {
    const float* e = msEavgTable();
    const T m0 = ((T)1 - (T)e[at.i]) * ((T)1 - at.t), m1 = ((T)1 - (T)e[at.i + 1]) * at.t;
    p1 = (m0 + m1 > (T)0) ? m1 / (m0 + m1) : (T)1;
    c0 = msCdfTable() + at.i * (GGX_MS_NT + 1);
    c1 = c0 + (GGX_MS_NT + 1);
}
// mu1^2 - mu0^2 for cell k: the span of the squared cosine a draw is uniform in.
template <class T> GGX_HD T msCellSpan(int k, T& mu0sq) {
    const T t0 = (T)k / (T)GGX_MS_NT, t1 = (T)(k + 1) / (T)GGX_MS_NT;
    const T m0 = t0 * t0, m1 = t1 * t1;
    mu0sq = m0 * m0;
    return m1 * m1 - mu0sq;
}
// The density, in solid angle, of f_ms's draw for an exit at cosine `c`: in each row, the cell's
// share times 2c / (mu1^2 - mu0^2) in the cosine, over the azimuth's 2 pi.
template <class T> GGX_HD T msPdf(T c, const MsAt<T>& at) {
    if (!(c > (T)0)) return (T)0;
    const float *c0, *c1;
    T p1;
    msRows(at, c0, c1, p1);
    int k = (int)(GGX_SQRT(c > (T)1 ? (T)1 : c) * (T)GGX_MS_NT);
    if (k > GGX_MS_NT - 1) k = GGX_MS_NT - 1;
    T mu0sq;
    const T span = msCellSpan(k, mu0sq);
    const T share = ((T)1 - p1) * ((T)c0[k + 1] - (T)c0[k]) + p1 * ((T)c1[k + 1] - (T)c1[k]);
    return share * c / ((T)kPi * span);
}
template <class T> GGX_HD T msPdf(T c, T alpha) { return msPdf(c, msAt(alpha)); }
// The exit cosine f_ms's draw takes for the uniform `v` in [0, 1): v below p1 picks the upper row,
// and is stretched to a fresh uniform for its cells.
template <class T> GGX_HD T msSampleCos(T v, const MsAt<T>& at) {
    const float *c0, *c1;
    T p1;
    msRows(at, c0, c1, p1);
    const float* cdf;
    if (v < p1) { cdf = c1; v = v / p1; }
    else        { cdf = c0; v = (v - p1) / ((T)1 - p1); }
    v = v < (T)0 ? (T)0 : (v >= (T)1 ? (T)0.99999994 : v);
    int lo = 0, hi = GGX_MS_NT;           // cdf[lo] <= v < cdf[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if ((T)cdf[mid] <= v) lo = mid; else hi = mid;
    }
    const T share = (T)cdf[lo + 1] - (T)cdf[lo];
    T frac = share > (T)0 ? (v - (T)cdf[lo]) / share : (T)0.5;
    frac = frac < (T)0 ? (T)0 : (frac > (T)1 ? (T)1 : frac);
    T mu0sq;
    const T span = msCellSpan(lo, mu0sq);
    return GGX_SQRT(mu0sq + span * frac);
}
template <class T> GGX_HD T msSampleCos(T v, T alpha) { return msSampleCos(v, msAt(alpha)); }

// THE LOBE A MATERIAL SHADES WITH: single scattering, and its multiple scattering where `ms`. A draw
// takes the single-scattering lobe with probability E(cos_a), else f_ms's draw. The density is the
// mixture's (pdfMix, given the table's place and E(cos_a)); a draw reads both once (drawUV).
template <class T> GGX_HD T pdfMix(const PairT<T>& p, const MsAt<T>& at, T ea) {
    const T ps = ea > (T)1 ? (T)1 : ea;
    return ps * pdf(p) + ((T)1 - ps) * msPdf(p.cb, at);
}
template <class T> GGX_HD T fAll(const PairT<T>& p, bool ms) { return ms ? f(p) + fMs(p) : f(p); }
template <class T> GGX_HD T pdfAll(const PairT<T>& p, bool ms) {
    if (!ms || !p.ok) return pdf(p);
    const MsAt<T> at = msAt(p.alpha);
    return pdfMix(p, at, msE(p.ca, at));
}
// f cos(b) / pdfAll for a drawn pair, over r: in (0, 1] without `ms`; with it about 1 for a white
// lobe. weightAt is the same given the table's place and E(cos_a).
template <class T> GGX_HD T weightAt(const PairT<T>& p, const MsAt<T>& at, T ea) {
    if (!p.ok) return (T)0;
    const T d = pdfMix(p, at, ea);
    return (d > (T)0) ? (f(p) + fMsAt(p, at, ea)) * p.cb / d : (T)0;
}
// sampleUV for the lobe with its multiple scattering where `ms`, given the table's place and
// E(cos_a): u1 splits at the single scattering's share (u1 == 1 still the mirror; an arrival below n
// still returns it). `msOut` (may be null) says whether the draw took f_ms's branch: a scattering
// that keeps nothing of the arrival's direction, as a diffuse bounce does (the photon walks' caustic
// split reads it).
GGX_V template <class V, class T> GGX_HD V sampleUVAt(const V& n, const V& a, T alpha, const MsAt<T>& at, T ea,
                                                    T u1, T u2, bool* msOut) {
    if (msOut) *msOut = false;
    if (!(ea < (T)1) || u1 < ea) return sampleUV(n, a, alpha, (ea < (T)1) ? u1 / ea : u1, u2);
    if (msOut) *msOut = true;
    V t, bt;
    vonb(n, t, bt);
    return vframe(t, bt, n, msSampleCos((u1 - ea) / ((T)1 - ea), at), (T)2 * (T)kPi * u2);
}
// A draw from the lobe, with its multiple scattering where `ms` (sampleUVAt; the single scattering's
// for an arrival below n), and its weight `w`, f cos / pdf over r (weightAt), in precision T; `msOut`
// as sampleUVAt's. Mode W's sample 0 (u1 == 1, the exact mirror, standing for the whole lobe at one
// sample) takes the lobe's albedo there instead: 1 with the multiple scattering, E(cos_a) without.
GGX_V template <class V, class T> GGX_HD V drawUV(const V& n, const V& a, T alpha, bool ms, T u1, T u2, T& w,
                                                bool* msOut = nullptr) {
    if (msOut) *msOut = false;
    const T ca = vdot(a, n);
    if (u1 >= (T)1 && ca > (T)0) {
        w = ms ? (T)1 : msE(ca, alpha);
        return vmirror(n, a);
    }
    if (!ms || !(ca > (T)0)) {
        const V o = sampleUV(n, a, alpha, u1, u2);
        w = weight(pairT<T>(n, a, o, alpha));
        return o;
    }
    const MsAt<T> at = msAt(alpha);
    const T ea = msE(ca, at);
    const V o = sampleUVAt(n, a, alpha, at, ea, u1, u2, msOut);
    w = weightAt(pairT<T>(n, a, o, alpha), at, ea);
    return o;
}

// THE DRAW THE TRACERS TAKE: a direction from the lobe of material `roughness` about the shading
// normal n for the arrival wo (pointing back along the incoming ray), with its multiple scattering
// where `ms`, and `w` its weight -- from two canonical uniforms (an rng's, or mode W's lattice), or off
// the caller's stream `rng` (anything with uniform(): two draws, in that order). `msOut` (may be null)
// says whether it took the multiple scattering's branch. The caller ends the path where the direction
// is not above n.
GGX_V template <class V, class T> GGX_HD V drawRough(const V& n, const V& wo, T roughness, bool ms, T u1, T u2, T& w,
                                                   bool* msOut = nullptr) {
    return drawUV(n, wo, alpha(roughness), ms, u1, u2, w, msOut);
}
GGX_V template <class V, class T, class R> GGX_HD V drawRough(const V& n, const V& wo, T roughness, bool ms, R& rng,
                                                        T& w, bool* msOut = nullptr) {
    const T u1 = (T)rng.uniform();
    const T u2 = (T)rng.uniform();
    return drawRough(n, wo, roughness, ms, u1, u2, w, msOut);
}

}  // namespace ggx
