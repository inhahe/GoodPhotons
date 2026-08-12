// Histogram-preserving stochastic tiling (O7) — Heitz & Neyret, "High-Performance
// By-Example Noise using a Histogram-Preserving Blending Operator" (HPG 2018).
//
// THE PROBLEM. A photographic texture repeated over a large surface reads as a grid:
// the eye locks onto any distinctive feature and sees it recur on a lattice. The fix
// everyone reaches for first — sample the image at three randomly offset positions and
// cross-fade — destroys the very thing that made it a photograph. Averaging N samples
// of a field with variance s^2 has variance s^2/N, so the blend regions come out visibly
// washed out and low-contrast against the un-blended interiors, which reads as *worse*
// than the grid it was meant to hide.
//
// THE OPERATOR. Heitz & Neyret's answer is to blend in a space where the variance loss
// is exactly computable, then undo it:
//
//   1. Precompute T, the per-channel transform that maps the image's histogram onto a
//      Gaussian N(1/2, 1/6) — a rank transform, so it is exact and monotone regardless of
//      what the input histogram looks like. Store T(I) (call it G) and the inverse T^-1
//      as a 1-D lookup table per channel.
//   2. At lookup time, take the three samples from G, blend with barycentric weights w_i,
//      and RESTORE THE VARIANCE: a weighted sum of independent unit-variance Gaussians has
//      standard deviation sqrt(sum w_i^2), so dividing the centred blend by that puts it
//      back on N(1/2, 1/6) exactly.
//   3. Map back through T^-1.
//
// Because step 2 lands on the *same* Gaussian the input was mapped to, step 3 returns a
// value drawn from the *input's own histogram*. The blend regions therefore have the same
// contrast, the same tonal distribution and the same extremes as the source image — the
// property the naive cross-fade throws away, and the reason this is worth ~200 lines.
//
// THE LATTICE. The three samples come from the vertices of a triangle lattice over UV
// space (a skewed simplex grid, so exactly three tiles overlap anywhere and the weights
// are the barycentrics — no fourth sample, no seams). Each lattice vertex hashes to a
// constant UV offset, so the content under a given vertex is a random crop of the input,
// and the crop changes from vertex to vertex.
//
// WHAT IS *NOT* IMPLEMENTED, AND WHY. The paper's LOD correction (a 2-D invT LUT indexed
// by mip level, compensating for the variance a mipmap has already removed) is absent
// because this renderer has no mip pyramid: every texture fetch is a full-resolution
// nearest/bilinear tap. There is consequently nothing to compensate for. If mipmapping is
// ever added, this is the one piece that has to come with it.
//
// HOST/DEVICE. Everything here is one `STOCH_HD` implementation shared by all three
// texture backends (host `Texture`, render_cuda's spectral `DTexture`, raster_cuda's
// preview `DTex`), so a stochastic texture cannot look different on the CPU, in the
// megakernel and in the preview. The lattice hash is INTEGER, not the paper's
// `fract(sin(...)*43758.5)`: a large-argument `sin` is exactly the kind of thing whose
// last bits differ between a CPU libm and a GPU, which would put the three backends on
// different crops of the image.
#pragma once
#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
  #ifndef STOCH_HD
  #define STOCH_HD __host__ __device__
  #endif
  #define STOCH_FLOOR(x) ::floor(x)
  #define STOCH_SQRT(x)  ::sqrt(x)
#else
  #ifndef STOCH_HD
  #define STOCH_HD
  #endif
  #define STOCH_FLOOR(x) std::floor(x)
  #define STOCH_SQRT(x)  std::sqrt(x)
#endif

// The Gaussian the histogram is mapped onto: mean 1/2, standard deviation 1/6 (the paper's
// choice, which puts +-3 sigma at 0 and 1).
#define STOCH_G_MEAN  0.5
#define STOCH_G_SIGMA (1.0 / 6.0)

// ...but the stored range and the LUT's DOMAIN are +-6 sigma, not +-3. The paper stores G
// in a unorm texture and therefore has to clamp at +-3 sigma, which costs it two things
// this implementation does not have to pay:
//
//   * the round trip stops being exact in the tails. Every texel past 3 sigma collapses
//     onto the same clamped G, so T^-1(T(x)) returns one value for all of them. Measured
//     on a 96x96 test image that was a 0.052 error on a [0,1] image — the brightest few
//     texels all read back as the same grey.
//   * the BLEND clips. Restoring the variance multiplies the deviation by up to sqrt(3)
//     (at a triangle's centroid), so a perfectly ordinary tap at the 96th percentile lands
//     outside [0,1] and gets pinned to the image's extreme, piling up mass at both ends of
//     the output histogram — the exact defect the operator exists to avoid.
//
// The planes here are float, so there is no reason to accept either: storing the true
// quantile and widening the LUT's domain to match costs one extra multiply in stochInvT.
#define STOCH_G_LO   (STOCH_G_MEAN - 6.0 * STOCH_G_SIGMA)
#define STOCH_G_HI   (STOCH_G_MEAN + 6.0 * STOCH_G_SIGMA)

// Entries per channel in the inverse-histogram LUT. 2048 over a domain twice the paper's
// leaves the same resolution per sigma as its 1024 over [0,1], which is already finer than
// the 256 distinct levels an 8-bit source can hold; linear interpolation covers the rest.
#define STOCH_LUT_N 2048

// ---------------------------------------------------------------------------
// Linear RGB -> Jakob-Hanika coefficients, tabulated.
// ---------------------------------------------------------------------------
// Why the spectral path needs this at all: the blend below produces a COLOUR that is
// not any texel's colour, so its spectrum was never fitted at load. The obvious
// alternative — Gaussianize and blend the coefficient planes directly, skipping RGB —
// was implemented first and is wrong. Coefficient space is not a colour space: mixing
// the c0 of one crop with the c2 of another yields a spectrum off the manifold the
// image occupies, which shows up as saturated speckle (the demo scene grew blue-cyan
// fringes on the spectral path that the RGB raster preview did not have). Measured on
// scenes/lichen.ppm, as distance from the nearest colour the source image actually
// contains:
//
//     variant                      mean      worst    % further than 0.02
//     plain tiling (the floor)     0.00086   0.0145    0.00
//     blend RGB, then this LUT     0.00340   0.0504    4.68
//     blend coefficients           0.01283   0.6181   11.44
//
// Blending in RGB is 12x better in the worst case, and it has a second, bigger payoff:
// the raster preview and the spectral render now run the SAME operator on the SAME
// planes, so they agree by construction instead of merely resembling one another.
//
// (Heitz & Neyret additionally decorrelate the channels by PCA before the transform.
// Tried and measured here. In *coefficient* space it helps a great deal — it is what
// removes the blue entirely. Once the blend moved to RGB it was a wash: worst case
// 0.1465 -> 0.0505, but mean 0.0027 -> 0.0034, decile match 0.0027 -> 0.0134 and the
// off-gamut fraction 2.4% -> 4.7%, all worse. Left out, since a per-plane-set 3x3 basis
// to store, upload to two device backends and keep in sync is not worth a tail that
// small. Revisit if a texture whose channels are strongly but obliquely correlated
// shows artifacts.)
// Grid points per axis. 64^3 * 3 floats = 3.1 MB, built in 0.6 s (threaded) the first
// time a stochastic texture loads. Measured round-trip |XYZ - target| over 4000 colours,
// half of them pushed toward black, at |p| <= 60: 48^3 mean 5.0e-4 / worst 0.036,
// 64^3 mean 2.7e-4 / worst 0.019, 80^3 mean 1.8e-4 / worst 0.016. The worst case stops
// improving there because it is no longer the table — the sigmoid-of-a-quadratic model
// itself is 0.019 off at pure white — so 64 is where the resolution stops buying anything.
#ifndef STOCH_JH_N
#define STOCH_JH_N 64
#endif
// Bound on |p| (the quadratic, not the coefficients) that the tabulated fits are held to.
// The sigmoid is within 1.4e-4 of its limit by |p| = 60, so this costs nothing in
// reflectance, and it is what makes the entries small enough to interpolate between at
// all: an unbounded fit reaches |c| = 1.6e6 at the white corner. See
// upsample::clampSaturated, which enforces it, and coeffLut(), which applies it.
#define STOCH_JH_PMAX 60.0

// Trilinear lookup in the sqrt-warped grid. `lut` is STOCH_JH_N^3 * 3 floats, ordered
// [r][g][b][coeff]. Clamps to the cube: an out-of-gamut blend (possible, since the
// operator amplifies deviations) gets the nearest representable colour's spectrum
// rather than an extrapolated one.
STOCH_HD inline void stochJhCoeff(const float* lut, double r, double g, double b,
                                  double* c /*[3]*/) {
    const int N = STOCH_JH_N;
    int i0[3], i1[3];
    double t[3];
    const double v[3] = {r, g, b};
    for (int k = 0; k < 3; ++k) {
        double x = v[k] < 0.0 ? 0.0 : (v[k] > 1.0 ? 1.0 : v[k]);
        x = STOCH_SQRT(x) * (double)(N - 1);          // the warp
        double f = STOCH_FLOOR(x);
        int    i = (int)f;
        if (i > N - 2) { i = N - 2; f = (double)i; }
        if (i < 0)     { i = 0;     f = 0.0; }
        i0[k] = i; i1[k] = i + 1; t[k] = x - f;
    }
    c[0] = c[1] = c[2] = 0.0;
    for (int a = 0; a < 2; ++a) {
        const double wa = a ? t[0] : 1.0 - t[0];
        for (int bb = 0; bb < 2; ++bb) {
            const double wb = wa * (bb ? t[1] : 1.0 - t[1]);
            for (int d = 0; d < 2; ++d) {
                const double w = wb * (d ? t[2] : 1.0 - t[2]);
                const size_t idx = (((size_t)(a ? i1[0] : i0[0]) * N + (bb ? i1[1] : i0[1]))
                                    * N + (d ? i1[2] : i0[2])) * 3;
                c[0] += w * (double)lut[idx];
                c[1] += w * (double)lut[idx + 1];
                c[2] += w * (double)lut[idx + 2];
            }
        }
    }
}

// Reflectance at one wavelength from coefficients — the device twin of
// upsample::reflAt, duplicated here (three lines) so the device path does not have to
// include the host-only upsampling header.
STOCH_HD inline double stochReflAt(const double* c, double lambda) {
    const double t = (lambda - 595.0) / 235.0;
    const double p = c[0] * t * t + c[1] * t + c[2];
    return 0.5 + 0.5 * p / STOCH_SQRT(1.0 + p * p);
}

// Per-texture tiling parameters. POD, so the identical record ships to both device
// backends. `on == 0` means the texture samples the ordinary way and nothing else here
// is read (and no Gaussianized planes are built or uploaded).
struct StochTile {
    int      on    = 0;
    // Size of one randomly-offset patch, in texture repeats. 1.0 is the paper's density
    // (lattice scale 2*sqrt(3)); larger means bigger patches, so more of the source shows
    // through un-blended but the repeat becomes easier to spot.
    double   patch = 1.0;
    unsigned seed  = 0u;
};

// ---------------------------------------------------------------------------
// The rank transform's two directions.
// ---------------------------------------------------------------------------

// Inverse standard-normal CDF. Acklam's rational approximation (|err| < 1.15e-9)
// followed by one Halley step against erfc, which takes it to full double precision —
// this is a LOAD-TIME function (it builds T), so accuracy is worth more than speed, and
// -checkstochtile pins it against std::erf.
inline double stochNormalQuantile(double p) {
    if (!(p > 0.0)) return -8.0;
    if (!(p < 1.0)) return  8.0;
    static const double a[6] = {-3.969683028665376e+01,  2.209460984245205e+02,
                                -2.759285104469687e+02,  1.383577518672690e+02,
                                -3.066479806614716e+01,  2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01,  1.615858368580409e+02,
                                -1.556989798598866e+02,  6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                 2.445134137142996e+00,  3.754408661907416e+00};
    const double pl = 0.02425, ph = 1.0 - pl;
    double x;
    if (p < pl) {
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p > ph) {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else {
        double q = p - 0.5, r = q * q;
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }
    // Halley refinement: e = Phi(x) - p, u = e*sqrt(2pi)*exp(x^2/2).
    double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    double u = e * std::sqrt(2.0 * 3.14159265358979323846) * std::exp(x * x / 2.0);
    x -= u / (1.0 + x * u / 2.0);
    return x;
}

// Standard-normal CDF (host only; used to place the LUT's sample points).
inline double stochNormalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Forward transform for a texel whose value has rank `rank` among `count` sorted values.
// (rank + 1/2)/count is the mid-rank plotting position, so the transform never asks for
// the quantile at exactly 0 or 1. The +-6 sigma clamp is a guard, not a design point: it
// only engages past ~5e8 texels, an order of magnitude beyond any image this can load.
inline double stochForward(size_t rank, size_t count) {
    double p = ((double)rank + 0.5) / (double)count;
    double g = STOCH_G_MEAN + STOCH_G_SIGMA * stochNormalQuantile(p);
    return g < STOCH_G_LO ? STOCH_G_LO : (g > STOCH_G_HI ? STOCH_G_HI : g);
}

// ---------------------------------------------------------------------------
// The triangle lattice.
// ---------------------------------------------------------------------------

// Two independent uniforms in [0,1) from a lattice vertex + seed. Same integer-mix shape
// as pattern.h's patHash3 (deliberately: this project's noise all agrees bit-for-bit
// across backends because none of it hashes through transcendentals).
STOCH_HD inline void stochHash2(int ix, int iy, unsigned seed, double& a, double& b) {
    uint32_t h = (uint32_t)ix * 374761393u + (uint32_t)iy * 668265263u
               + seed * 2147483647u + 2166136261u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    uint32_t k = (h + 2654435761u) * 2246822519u;
    k = (k ^ (k >> 15)) * 3266489917u;
    k ^= (k >> 13);
    a = (double)h * (1.0 / 4294967296.0);
    b = (double)k * (1.0 / 4294967296.0);
}

// Barycentric weights and the three random UV offsets at (u,v). Exactly three lattice
// cells cover any point, and the weights sum to 1, which is what makes the variance
// restoration below a closed form rather than a fit.
STOCH_HD inline void stochTriGrid(double u, double v, double patch, unsigned seed,
                                  double w[3], double offU[3], double offV[3]) {
    const double s = 3.4641016151377544 / (patch > 1e-6 ? patch : 1e-6);   // 2*sqrt(3)
    const double su = u * s, sv = v * s;
    // Skew into the simplex grid: mat2(1, 0, -1/sqrt(3), 2/sqrt(3)) applied to (su,sv).
    const double kx = su - 0.5773502691896258 * sv;
    const double ky = 1.1547005383792515 * sv;
    const double fkx = STOCH_FLOOR(kx), fky = STOCH_FLOOR(ky);
    const int bx = (int)fkx, by = (int)fky;
    const double tx = kx - fkx, ty = ky - fky;
    const double tz = 1.0 - tx - ty;
    int vx[3], vy[3];
    if (tz > 0.0) {
        w[0] = tz;   w[1] = ty;         w[2] = tx;
        vx[0] = bx;     vy[0] = by;
        vx[1] = bx;     vy[1] = by + 1;
        vx[2] = bx + 1; vy[2] = by;
    } else {
        w[0] = -tz;  w[1] = 1.0 - ty;   w[2] = 1.0 - tx;
        vx[0] = bx + 1; vy[0] = by + 1;
        vx[1] = bx + 1; vy[1] = by;
        vx[2] = bx;     vy[2] = by + 1;
    }
    for (int i = 0; i < 3; ++i) stochHash2(vx[i], vy[i], seed, offU[i], offV[i]);
}

// Variance-restoring blend of three Gaussianized samples. sum(w) == 1 by construction, so
// the mean is already right; only the spread has to be undone.
STOCH_HD inline double stochBlend(const double g[3], const double w[3]) {
    const double m = w[0] * g[0] + w[1] * g[1] + w[2] * g[2];
    const double n = STOCH_SQRT(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    return (n > 0.0) ? ((m - STOCH_G_MEAN) / n + STOCH_G_MEAN) : m;
}

// T^-1 from the per-channel LUT, linearly interpolated over [STOCH_G_LO, STOCH_G_HI].
// `g` outside that clamps to the image's own extremes — 6 sigma out, which is where the
// source image had no data either.
STOCH_HD inline double stochInvT(const float* lut, double g) {
    if (!lut) return g;
    const double t = (g - STOCH_G_LO) * ((double)(STOCH_LUT_N - 1) / (STOCH_G_HI - STOCH_G_LO));
    if (!(t > 0.0)) return (double)lut[0];
    if (!(t < (double)(STOCH_LUT_N - 1))) return (double)lut[STOCH_LUT_N - 1];
    const double f = STOCH_FLOOR(t);
    const int i = (int)f;
    const double fr = t - f;
    return (double)lut[i] * (1.0 - fr) + (double)lut[i + 1] * fr;
}

// ---------------------------------------------------------------------------
// Sampling a Gaussianized plane set.
// ---------------------------------------------------------------------------

// Always Repeat: the lattice offsets push the fetch UV arbitrarily far outside [0,1], so
// a stochastic texture has no border to clamp or mirror against. The texture's own `wrap`
// is meaningless here and the loader warns if it was set to anything else.
STOCH_HD inline int stochWrapIdx(int i, int n) { int m = i % n; return (m < 0) ? m + n : m; }

// Bilinear (or nearest) fetch of `nch` interleaved float channels, with the same v flip
// and texel centring as Texture::bilerpSetup, so the Gaussianized planes are addressed
// exactly like the images they came from.
STOCH_HD inline void stochFetch(const float* data, int nch, int w, int h, int filter,
                                double u, double v, double* out) {
    if (filter == 0) {
        int x = stochWrapIdx((int)STOCH_FLOOR(u * w), w);
        int y = stochWrapIdx((int)STOCH_FLOOR((1.0 - v) * h), h);
        const float* p = data + (size_t)(y * (size_t)w + x) * nch;
        for (int c = 0; c < nch; ++c) out[c] = (double)p[c];
        return;
    }
    double tu = u * w - 0.5, tv = (1.0 - v) * h - 0.5;
    double flx = STOCH_FLOOR(tu), fly = STOCH_FLOOR(tv);
    double fx = tu - flx, fy = tv - fly;
    int x0 = stochWrapIdx((int)flx, w), x1 = stochWrapIdx((int)flx + 1, w);
    int y0 = stochWrapIdx((int)fly, h), y1 = stochWrapIdx((int)fly + 1, h);
    const float* p00 = data + (size_t)((size_t)y0 * w + x0) * nch;
    const float* p10 = data + (size_t)((size_t)y0 * w + x1) * nch;
    const float* p01 = data + (size_t)((size_t)y1 * w + x0) * nch;
    const float* p11 = data + (size_t)((size_t)y1 * w + x1) * nch;
    for (int c = 0; c < nch; ++c) {
        double a = (double)p00[c] * (1 - fx) + (double)p10[c] * fx;
        double b = (double)p01[c] * (1 - fx) + (double)p11[c] * fx;
        out[c] = a * (1 - fy) + b * fy;
    }
}

// The whole operator: three offset fetches of the Gaussianized plane set, one
// variance-restoring blend per channel, one LUT inversion per channel. `lut` is
// nch * STOCH_LUT_N floats, channel-major. Writes nch values (nch <= 4).
STOCH_HD inline void stochSample(const StochTile& st, const float* gauss, const float* lut,
                                 int nch, int w, int h, int filter,
                                 double u, double v, double* out) {
    double wgt[3], offU[3], offV[3];
    stochTriGrid(u, v, st.patch, st.seed, wgt, offU, offV);
    double s[3][4];
    for (int i = 0; i < 3; ++i)
        stochFetch(gauss, nch, w, h, filter, u + offU[i], v + offV[i], s[i]);
    for (int c = 0; c < nch; ++c) {
        const double g[3] = {s[0][c], s[1][c], s[2][c]};
        out[c] = stochInvT(lut + (size_t)c * STOCH_LUT_N, stochBlend(g, wgt));
    }
}
