// gabor.h — anisotropic, band-limited Gabor noise (sparse convolution), host + device.
//
// This is the O4 answer to "noise stretched and steered along a direction field":
// wood grain following a trunk, brushed metal, muscle striation, hair/fur flow. It is
// also the O8 answer to "noise you can antialias", because unlike lattice value noise
// its power spectrum is a narrow band you choose, rather than everything up to the
// lattice Nyquist.
//
// WHY NOT JUST ROTATE THE COORDINATES. The obvious way to steer noise is
// `noise(R(p) * p)` with a spatially varying rotation R. That is the same trap as
// spatially varying FREQUENCY (see the non-stationary write-up in REFERENCE.md): the
// Jacobian of `R(p)*p` is `R + (dR/dp)*p`, whose second term grows with |p|, so the
// texture shears more and more the farther the shading point is from the (arbitrary)
// origin, and the local orientation is not the R you asked for. Gabor noise has no
// such term: orientation is a parameter of each KERNEL, and a kernel only ever sees
// the OFFSET `u = p - p_i` from its own centre, which is bounded by the kernel radius.
// Letting the direction vary with p leaves a residual `(dw/dp) . u`, bounded by the
// local turning rate times ONE cell — independent of position, and zero for a constant
// direction. That difference is the whole point of the primitive.
//
// CONSTRUCTION. Sparse convolution (Lewis 1989) of Gabor kernels (Lagae et al. 2009),
// in the "phase-augmented" form (Lagae & Drettakis 2011):
//
//     G(p) = sum_i  w_i * E(|p - p_i|) * cos(2*pi * (f * ((p - p_i) . w^) + phi_i))
//
//   * {p_i} is a homogeneous Poisson process of intensity `lambda` points per unit
//     volume, generated per integer cell: each cell independently draws Poisson(lambda)
//     points uniform inside itself. The union of independent Poisson processes on
//     disjoint regions IS a homogeneous Poisson process, so the point set — and hence
//     the noise — is EXACTLY stationary, invariant under arbitrary (not just integer)
//     translation. Lattice noises are only integer-stationary; this one is not tied to
//     its grid at all, and `-checkgabor` §6 measures that.
//   * w_i ~ U[-1, 1) (zero mean, E[w^2] = 1/3) and phi_i ~ U[0, 1) turns.
//   * w^ is the unit steering direction, supplied by the CALLER, so it can be any
//     expression — a flow field, a warped gradient, a tangent. A zero-length direction
//     means "isotropic": each impulse then draws its own uniform direction on the
//     sphere, giving isotropic BAND-PASS noise (a useful thing in its own right, and
//     the one thing lattice noise cannot do at all).
//
// THE ENVELOPE, AND WHY IT IS NOT A GAUSSIAN. Lagae's kernel uses a Gaussian truncated
// where it falls to 5% of its peak, and the standard 3x3x3 neighbourhood search is
// therefore APPROXIMATE — it silently drops the tails. We use instead a compactly
// supported C2 envelope of radius exactly one cell,
//
//     E(r) = (1 - r^2)^3   for r < 1,   0 otherwise
//
// whose value, first and second derivatives all vanish at r = 1. Three consequences:
// (a) the 3x3x3 search is MATHEMATICALLY exact — a cell two rings out is > 1 away in
// the infinity norm, hence > 1 in Euclid, hence outside every kernel it can hold
// (`-checkgabor` §1 pins this against a +-4-block brute force); (b) no `exp` is needed,
// and the support test is on the SQUARED distance, so ~85% of candidate impulses are
// rejected with three multiplies; (c) the spectral tail decays polynomially rather than
// exponentially, which is invisible next to the cosine's own bandwidth.
//
// NORMALISATION IS ANALYTIC, NOT MEASURED. Because each impulse carries an independent
// uniform PHASE, E_phi[cos^2(theta + phi)] = 1/2 pointwise, so Campbell's theorem gives
//
//     Var[G] = lambda * E[w^2] * (1/2) * INT E(|u|)^2 du
//            = lambda * (1/3)  * (1/2) * 4*pi*1024/45045
//
// with NO dependence on f or on the direction — the frequency term integrates out
// exactly. (Without the random phase there would be a residual Fourier term in f, and
// the normalisation would drift as the frequency operand moved.) The result is mapped
// to the [0,1] / mean-0.5 convention every other pattern primitive obeys by
// `0.5 + 0.5 * G / (3 sigma)`, clamped; ~0.3% of samples clip, which is invisible and
// costs the [0,1] contract nothing.
//
// DETERMINISM. Everything here is IEEE arithmetic plus `floor`/`sqrt` (both correctly
// rounded) — including the cosine, which is our own range-reduced Taylor evaluation
// rather than libm's. That is deliberate: `cos` is NOT correctly rounded, and CUDA's
// differs from the host's by up to 2 ulp, which would break the "a pattern evaluates
// bit-for-bit identically on every backend" contract that patWorley/povDNoise keep.
#pragma once

#if defined(__CUDACC__)
  #ifndef GABOR_HD
  #define GABOR_HD __host__ __device__
  #endif
#else
  #ifndef GABOR_HD
  #define GABOR_HD
  #endif
#endif

#include <math.h>

// Impulse density, points per unit cell. 6 gives lambda * (4/3)pi = 25 kernels
// overlapping any query point (Lagae calls ~16 "reasonable", ~64 "high quality"), at a
// cost of ~162 candidate impulses drawn and ~25 surviving the support test per lookup.
#define PAT_GABOR_LAMBDA        6.0
#define PAT_GABOR_EXP_NEG_LAM   2.4787521766663585e-3   // exp(-6), the Knuth threshold
// Hard cap on impulses in one cell. P(Poisson(6) > 32) ~ 1e-12, so this never binds in
// practice; it exists so a corrupted hash stream cannot spin forever.
#define PAT_GABOR_NMAX          32
// 1 / (3 sigma), sigma^2 = lambda/3 * 1/2 * 4*pi*1024/45045 = 0.2856690747 at lambda 6.
// sigma = 0.5344802851, 3 sigma = 1.6034408553. Checked empirically by `-checkgabor` §3.
#define PAT_GABOR_INV3SIGMA     0.62365862055

// murmur3 fmix32: full-avalanche 32-bit finalizer (same finalizer worley.h uses; the
// SEEDS below are different, so the two noises are statistically independent even when
// evaluated at identical coordinates).
GABOR_HD inline unsigned int patGaborMix(unsigned int h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Cell -> base hash. Large odd constants, distinct from worley.h's, so permuted cells
// don't collide and the two primitives don't correlate.
GABOR_HD inline unsigned int patGaborCellHash(int cx, int cy, int cz) {
    unsigned int h = (unsigned int)cx * 2654435761u
                   ^ (unsigned int)cy * 1597334677u
                   ^ (unsigned int)cz * 3812015801u;
    return patGaborMix(h ^ 0x51ed270bu);
}

// cos(2*pi*t) for arbitrary t, evaluated identically on every backend.
//
// Reduce t to [-0.5, 0.5) by `t - floor(t + 0.5)` (exact for |t| below ~2^52), fold by
// evenness to [0, 0.5] and by cos(pi - x) = -cos(x) to [0, 0.25] turns, i.e. an
// argument in [0, pi/2]. Then a plain Taylor series in y = x^2 through y^11 = x^22:
// the omitted term is x^24/24! < 1e-19, and the worst intermediate cancellation
// (x^2/2 ~ 1.23 against a result near 0 at the interval end) costs ~3e-16 absolute.
// That is orders of magnitude tighter than anything a noise field can care about, and
// unlike libm it is the SAME sequence of IEEE operations on host and device.
GABOR_HD inline double patGaborCosTurns(double t) {
    t = t - floor(t + 0.5);                       // [-0.5, 0.5)
    double a = t < 0.0 ? -t : t;                  // cos is even -> [0, 0.5]
    double sgn = 1.0;
    if (a > 0.25) { a = 0.5 - a; sgn = -1.0; }    // cos(2pi(0.5-a)) = -cos(2pi a)
    const double x = a * 6.283185307179586477;    // [0, pi/2]
    const double y = x * x;
    double p =        8.8967913924505732e-22;     // 1/22!
    p = 4.1103176233121648e-19 - y * p;           // 1/20!
    p = 1.5619206968586225e-16 - y * p;           // 1/18!
    p = 4.7794773323873853e-14 - y * p;           // 1/16!
    p = 1.1470745597729725e-11 - y * p;           // 1/14!
    p = 2.0876756987868099e-09 - y * p;           // 1/12!
    p = 2.7557319223985893e-07 - y * p;           // 1/10!
    p = 2.4801587301587302e-05 - y * p;           // 1/8!
    p = 1.3888888888888889e-03 - y * p;           // 1/6!
    p = 4.1666666666666664e-02 - y * p;           // 1/4!
    p = 5.0000000000000000e-01 - y * p;           // 1/2!
    p = 1.0                    - y * p;
    return sgn * p;
}

// Raw Gabor field at (x, y, z): frequency `f` in cycles per unit of the caller's
// (already scaled) space, steered along (wx, wy, wz). Zero-mean, unit-ish variance is
// NOT applied here — this is the unnormalised sum, exposed so the self-test can measure
// the analytic variance directly. Use patGabor() for the [0,1] shading value.
GABOR_HD inline double patGaborRaw(double x, double y, double z,
                                   double f, double wx, double wy, double wz) {
    const double wl = wx * wx + wy * wy + wz * wz;
    const bool iso = !(wl > 1e-24);      // written so a NaN direction also falls back
    double ox = 0.0, oy = 0.0, oz = 0.0;
    if (!iso) { const double inv = 1.0 / sqrt(wl); ox = wx * inv; oy = wy * inv; oz = wz * inv; }

    const int bx = (int)floor(x), by = (int)floor(y), bz = (int)floor(z);
    double acc = 0.0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int cx = bx + dx, cy = by + dy, cz = bz + dz;
        const unsigned int h0 = patGaborCellHash(cx, cy, cz);
        // Poisson(lambda) impulse count for this cell, by Knuth's product method on an
        // independent hash chain. Expected lambda+1 iterations.
        int n = 0;
        {
            double p = 1.0;
            unsigned int hc = patGaborMix(h0 ^ 0x2f6a5d21u);
            for (;;) {
                p *= (double)hc * (1.0 / 4294967296.0);
                if (p <= PAT_GABOR_EXP_NEG_LAM) break;
                if (++n >= PAT_GABOR_NMAX) break;
                hc = patGaborMix(hc + 0x9e3779b9u);
            }
        }
        for (int i = 0; i < n; ++i) {
            // Each impulse gets an INDEXED base hash and derives its slots from that,
            // so the support rejection below can skip the remaining draws for free
            // instead of having to advance a sequential stream.
            const unsigned int hb = patGaborMix(h0 + 0x9e3779b9u * (unsigned int)(i + 1));
            const unsigned int r0 = patGaborMix(hb + 0x85ebca6bu);
            const unsigned int r1 = patGaborMix(hb + 0xc2b2ae35u);
            const unsigned int r2 = patGaborMix(hb + 0x27d4eb2fu);
            const double ux = x - ((double)cx + (double)r0 * (1.0 / 4294967296.0));
            const double uy = y - ((double)cy + (double)r1 * (1.0 / 4294967296.0));
            const double uz = z - ((double)cz + (double)r2 * (1.0 / 4294967296.0));
            const double d2 = ux * ux + uy * uy + uz * uz;
            if (d2 >= 1.0) continue;                  // outside this kernel's support
            const double e1 = 1.0 - d2;
            const double env = e1 * e1 * e1;          // (1 - r^2)^3, C2 at r = 1
            const unsigned int r3 = patGaborMix(hb + 0x165667b1u);
            const unsigned int r4 = patGaborMix(hb + 0x9e3779b1u);
            const double w  = (double)r3 * (2.0 / 4294967296.0) - 1.0;   // [-1, 1)
            const double ph = (double)r4 * (1.0 / 4294967296.0);         // [0, 1) turns
            double proj;
            if (iso) {                                 // per-impulse uniform direction
                const unsigned int r5 = patGaborMix(hb + 0x85ebca77u);
                const unsigned int r6 = patGaborMix(hb + 0xc2b2ae3du);
                const double cw = (double)r5 * (2.0 / 4294967296.0) - 1.0;  // cos(theta)
                const double sw = sqrt(1.0 - cw * cw);
                const double th = (double)r6 * (1.0 / 4294967296.0);        // turns
                const double ct = patGaborCosTurns(th);
                const double st = patGaborCosTurns(th - 0.25);              // = sin(2pi th)
                proj = ux * (sw * ct) + uy * (sw * st) + uz * cw;
            } else {
                proj = ux * ox + uy * oy + uz * oz;
            }
            acc += w * env * patGaborCosTurns(f * proj + ph);
        }
    }
    return acc;
}

// Shading value in [0, 1] with mean 0.5, matching the convention of `noise`.
GABOR_HD inline double patGabor(double x, double y, double z,
                                double f, double wx, double wy, double wz) {
    const double g = 0.5 + 0.5 * PAT_GABOR_INV3SIGMA *
                     patGaborRaw(x, y, z, f, wx, wy, wz);
    return g < 0.0 ? 0.0 : (g > 1.0 ? 1.0 : g);
}
