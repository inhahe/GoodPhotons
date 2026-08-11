// reaction.h — Gray-Scott reaction-diffusion, baked at load into a texture.
//
// Two chemicals diffuse and react on a periodic grid:
//
//   du/dt = Du L(u) - u v^2 + F (1 - u)
//   dv/dt = Dv L(v) + u v^2 - (F + k) v
//
// U is replenished at rate F and V is drained at rate F + k; the autocatalytic
// term u v^2 converts the first into the second wherever V already exists.
// Turing's point, and the reason this is here, is that the *uniform* solution of
// such a system can be unstable to spatial perturbation while remaining stable in
// time — so a featureless sheet spontaneously organises into spots, labyrinths or
// travelling fronts with an intrinsic wavelength that appears nowhere in the
// equations. That is a different kind of texture from everything else in this
// renderer: `noise`, `worley`, `gabor` and `bnoise` all place features by fiat,
// whereas here the features are the outcome of a process, so their spacing, their
// branch points and their defects are correlated the way a real coat pattern's
// are. It is also the standard model of exactly that — animal markings, coral,
// fingerprints, chemical Turing patterns in a gel.
//
// WHY THIS IS A BAKE AND NOT A PATTERN OP. Every other procedural here answers
// "what is the value at p?" in O(1) with no state. Reaction-diffusion cannot: the
// value at a point is the endpoint of a trajectory of the entire field, so there
// is no local closed form to evaluate per hit. The honest shape for it is an
// offline solve producing an image — which is precisely what `texture` already is.
// A reaction texture is therefore baked once at load and then flows through the
// unmodified texture pipeline: UV wrap, Jakob-Hanika spectral upsampling,
// triplanar, `reflect texture:`, `tex:<name>(u,v)` as a term inside a pattern
// formula, GPU upload, raster preview. No renderer path knows it exists.
//
// PERIODIC BY CONSTRUCTION. The Laplacian wraps on both axes, so the solve runs on
// a torus and the baked image tiles seamlessly under `wrap repeat`. This is not a
// detail that could be deferred: a reaction-diffusion field cannot be made tileable
// after the fact, because blending its edges destroys exactly the long-range
// correlations that distinguish it from noise. The topology has to be chosen up
// front, and the seed is periodic for the same reason (see rdSeed).
//
// THE STENCIL is the 9-point one (0.2 orthogonal, 0.05 diagonal, -1 centre) that
// Pearson used and that every published (F, k) map since is calibrated against, so
// parameter values from the literature transfer directly. It is also markedly more
// isotropic than the 5-point stencil, whose axis-aligned bias shows up as a square
// grain in the labyrinth regimes.
//
// THE DEFAULT DIFFUSION IS A RESCALE OF THE CLASSIC ONE, and this matters more than
// it sounds. A Gray-Scott feature is a fixed number of GRID CELLS wide — order
// 2*pi*sqrt(D/F) — so at the textbook Du = 0.16 a spot is about four cells across
// and comes out visibly SQUARE, pixel-locked to the lattice. Multiplying both
// coefficients by s^2 is a pure spatial rescale of the same continuum problem: the
// (F, k) physics, and hence every published parameter value, is untouched, while
// each feature becomes s times wider in cells and rounds off properly. The defaults
// below are the classic 0.16 / 0.08 at s = 2.5. The ceiling is stability, not
// taste: dt * D * 1.6 <= 2 caps s at 2.79.
//
// STABILITY. Explicit Euler on this stencil is stable while dt * D * 1.6 <= 2,
// since the stencil's Fourier symbol bottoms out at -1.6 (the corner mode
// a = b = pi). The defaults leave 20% margin. The coefficients are authorable, so
// the bound is checked and a violation is a load error rather than a texture full
// of NaNs.
#pragma once

#include <math.h>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "parallel.h"    // ft::stopRequested — cooperative `-stop` during a long bake

// The 9-point Laplacian's weights, and the magnitude of its most negative Fourier
// eigenvalue (at a = b = pi): -1 + 0.2*(-4) + 0.05*(4) = -1.6.
#define RD_W_ORTHO 0.20
#define RD_W_DIAG  0.05
#define RD_LAMBDA_MAX 1.6

// The classic coefficients, and the spatial rescale applied to them by default.
#define RD_DU_CLASSIC 0.16
#define RD_DV_CLASSIC 0.08
#define RD_ZOOM       2.5

// The seed's coarse blocks: edge length in grid cells (before the sqrt(du) rescale),
// and the hash threshold a block must reach to be switched on — 0xC0000000 is the top
// quarter of the 32-bit range, i.e. 25% fill. See rdSeed for why both are what they are.
#define RD_SEED_BLOCK 5.0
#define RD_SEED_FILL  0xC0000000u

struct RDParams {
    double feed  = 0.038;                        // F — the rate U is replenished at
    double kill  = 0.065;                        // k — the extra rate V is drained at
    double du    = RD_DU_CLASSIC * RD_ZOOM * RD_ZOOM;   // 1.00
    double dv    = RD_DV_CLASSIC * RD_ZOOM * RD_ZOOM;   // 0.50
    double dt    = 1.0;                          // explicit Euler step
    int    steps = 6000;                         // how long to run
    int    sim   = 256;                          // solve grid — since a feature is a
                                                 // fixed number of cells wide, this is
                                                 // the features-across-the-texture knob
    unsigned seed = 1u;                          // which realisation
};

// murmur3 fmix32, the same finalizer worley.h / bluenoise.h use.
inline unsigned int rdMix(unsigned int h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
inline unsigned int rdHash2(int x, int y, unsigned int salt) {
    unsigned int h = (unsigned int)x * 2135587861u
                   ^ (unsigned int)y * 805459861u
                   ^ salt * 3266489917u;
    return rdMix(h ^ 0x7feb352du);
}
inline double rdUnit(unsigned int h) { return (double)h * (1.0 / 4294967296.0); }

// The stencil's Fourier symbol at wavenumber (a, b) radians per cell. Exposed
// because it is what `-checkreaction` pins the Laplacian against: applying the
// stencil to cos(a x + b y) must return exactly this multiple of it, which tests
// the weights and the toroidal indexing at once — a stencil that is subtly
// asymmetric, or that mis-wraps one edge, cannot satisfy it at every wavenumber.
inline double rdLapSymbol(double a, double b) {
    return -1.0 + RD_W_ORTHO * (2.0 * cos(a) + 2.0 * cos(b))
                + RD_W_DIAG  * (4.0 * cos(a) * cos(b));
}

// Reference periodic 9-point Laplacian, written for clarity rather than speed.
// The solver's inner loop is the fast twin of this; `-checkreaction` pins the two
// against each other so the optimised version cannot drift.
inline void rdLaplacian(const double* a, int N, double* out) {
    for (int y = 0; y < N; ++y) {
        const int ya = ((y + N - 1) % N) * N, y0 = y * N, yb = ((y + 1) % N) * N;
        for (int x = 0; x < N; ++x) {
            const int xa = (x + N - 1) % N, xb = (x + 1) % N;
            const double o = a[y0 + xa] + a[y0 + xb] + a[ya + x] + a[yb + x];
            const double d = a[ya + xa] + a[ya + xb] + a[yb + xa] + a[yb + xb];
            out[y0 + x] = RD_W_ORTHO * o + RD_W_DIAG * d - a[y0 + x];
        }
    }
}

// True if explicit Euler on this stencil is stable for the given coefficients.
inline bool rdStable(const RDParams& p) {
    double d = p.du > p.dv ? p.du : p.dv;
    return p.dt > 0.0 && p.du >= 0.0 && p.dv >= 0.0 && p.dt * d * RD_LAMBDA_MAX <= 2.0;
}

// The uniform state (u = 1, v = 0) is a fixed point AND is linearly stable for every
// F, k > 0 — Gray-Scott's pattern formation is subcritical, so infinitesimal noise
// dies out and a bake seeded with it returns a blank sheet. The perturbation has to
// be finite-amplitude and, for a texture, present over the whole domain. The seed is
// therefore a coarse random binary field, a quarter of its blocks switched to
// (u, v) = (0.5, 0.25), plus 1% per-cell jitter to break ties between identical blocks.
//
// The two numbers below were both measured, and both matter. BLOCK SIZE has to be near
// one feature: a per-cell seed (block = 1) is smoothed away by diffusion before it can
// nucleate anything and five of the six presets decay to a blank sheet, while blocks
// several features wide behave like a uniform patch. FILL has to be well below the
// site-percolation threshold: at 50% the ON blocks merge into one domain-spanning
// region, and then whether the sheet patterns at all comes down to whether that single
// region survives — at sim = 128 the `spots` preset lost it and left one small nucleus
// creeping outward across an otherwise blank texture. At 25% the blocks stay isolated
// nuclei that seed the whole sheet independently, which is what makes every preset
// pattern reliably at every `sim` and `seed`. (`-checkreaction` §7 is exactly this
// assertion.) Finally `bx = x * nb / N` partitions the axis exactly, so the seed is
// periodic and leaves no defect line at the wrap, and the block size tracks sqrt(du)
// so it stays one feature wide however the diffusion is scaled.
inline void rdSeed(const RDParams& p, std::vector<float>& u, std::vector<float>& v) {
    const int N = p.sim;
    u.assign((size_t)N * N, 1.0f);
    v.assign((size_t)N * N, 0.0f);
    double zoom = sqrt((p.du > 0.0 ? p.du : RD_DU_CLASSIC) / RD_DU_CLASSIC);
    int nb = (int)(N / (RD_SEED_BLOCK * zoom));
    if (nb < 1) nb = 1; else if (nb > N) nb = N;
    for (int y = 0; y < N; ++y) {
        const int by = (int)((long long)y * nb / N);
        for (int x = 0; x < N; ++x) {
            const int bx = (int)((long long)x * nb / N);
            if (rdHash2(bx, by, p.seed) < RD_SEED_FILL) continue;     // a quarter of them
            const double j = rdUnit(rdHash2(x, y, p.seed ^ 0x9e3779b9u));
            const size_t i = (size_t)y * N + x;
            u[i] = (float)(0.50 + 0.01 * (j - 0.5));
            v[i] = (float)(0.25 + 0.01 * (j - 0.5));
        }
    }
}

// A sense-reversing barrier. The solve is thousands of tiny steps over one grid, so
// ft::parallelFor is the wrong tool — it spawns a thread pool per call, and paying
// that 6000 times costs more than the arithmetic. Threads are instead started once
// and rendezvous here between steps. Every cell of a step reads only the PREVIOUS
// buffer, so the result is bit-identical regardless of how the rows were split, and
// no reduction is involved to spoil that.
struct RDBarrier {
    std::atomic<unsigned> count{0};
    std::atomic<unsigned> gen{0};
    unsigned n;
    explicit RDBarrier(unsigned nt) : n(nt) {}
    void wait() {
        const unsigned g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            gen.store(g + 1, std::memory_order_release);
            return;
        }
        for (int spins = 0; gen.load(std::memory_order_acquire) == g; ++spins)
            if (spins > 2048) { std::this_thread::yield(); spins = 0; }
    }
};

// Advance (u, v) by p.steps. Returns false if a `-stop` landed mid-solve, in which
// case the state is a partial result and the caller must abandon the load.
inline bool rdRun(const RDParams& p, std::vector<float>& u, std::vector<float>& v) {
    const int N = p.sim;
    if (N < 1 || p.steps < 0) return true;
    std::vector<float> ub[2] = {u, std::vector<float>(u.size())};
    std::vector<float> vb[2] = {v, std::vector<float>(v.size())};

    // Wrap tables: one L1-resident indirection instead of a modulo or a branch in
    // the innermost loop, which is where every cycle of this bake is spent.
    std::vector<int> xm(N), xp(N), ym(N), yp(N);
    for (int i = 0; i < N; ++i) {
        xm[i] = (i + N - 1) % N; xp[i] = (i + 1) % N;
        ym[i] = (i + N - 1) % N; yp[i] = (i + 1) % N;
    }

    const double F = p.feed, K = p.kill, DU = p.du, DV = p.dv, DT = p.dt;
    const double WO = RD_W_ORTHO, WD = RD_W_DIAG;

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned nt = (unsigned)(N / 16); if (nt < 1) nt = 1;
    if (nt > hw) nt = hw;

    RDBarrier bar(nt);
    std::atomic<bool> stop{false};

    auto band = [&](unsigned tid) {
        const int y0 = (int)((long long)N * tid / nt);
        const int y1 = (int)((long long)N * (tid + 1) / nt);
        for (int s = 0; s < p.steps; ++s) {
            // Only thread 0 polls, and it polls BEFORE its compute phase, so the
            // store is published by this step's barrier and every thread reads the
            // same value after it. Polling independently per thread would let two
            // threads disagree about whether to run step s+1.
            if (tid == 0 && (s & 63) == 0)
                stop.store(ft::stopRequested(), std::memory_order_release);
            const std::vector<float>& us = ub[s & 1]; std::vector<float>& ud = ub[1 - (s & 1)];
            const std::vector<float>& vs = vb[s & 1]; std::vector<float>& vd = vb[1 - (s & 1)];
            for (int y = y0; y < y1; ++y) {
                const int r0 = y * N, ra = ym[y] * N, rb = yp[y] * N;
                for (int x = 0; x < N; ++x) {
                    const int xa = xm[x], xb = xp[x];
                    const double uc = us[r0 + x], vc = vs[r0 + x];
                    const double uo = (double)us[r0 + xa] + us[r0 + xb] + us[ra + x] + us[rb + x];
                    const double ud8 = (double)us[ra + xa] + us[ra + xb] + us[rb + xa] + us[rb + xb];
                    const double vo = (double)vs[r0 + xa] + vs[r0 + xb] + vs[ra + x] + vs[rb + x];
                    const double vd8 = (double)vs[ra + xa] + vs[ra + xb] + vs[rb + xa] + vs[rb + xb];
                    const double lu = WO * uo + WD * ud8 - uc;
                    const double lv = WO * vo + WD * vd8 - vc;
                    const double reac = uc * vc * vc;
                    ud[r0 + x] = (float)(uc + DT * (DU * lu - reac + F * (1.0 - uc)));
                    vd[r0 + x] = (float)(vc + DT * (DV * lv + reac - (F + K) * vc));
                }
            }
            if (nt > 1) bar.wait();
            if (stop.load(std::memory_order_acquire)) return;
        }
    };

    if (nt == 1) {
        // No barrier, but keep the same poll cadence so a stop still lands quickly.
        band(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt - 1);
        for (unsigned t = 1; t < nt; ++t) pool.emplace_back(band, t);
        band(0);
        for (auto& t : pool) t.join();
    }
    if (stop.load(std::memory_order_acquire)) return false;
    u.swap(ub[p.steps & 1]);
    v.swap(vb[p.steps & 1]);
    return true;
}

// Seed, solve, and normalise V to [0,1] by its own measured range. The raw scale of
// V depends strongly on F and k, so a fixed mapping would make half the usable
// parameter space come out nearly black; smoothstep the result to choose contrast.
// `washedOut`, if given, reports that the field settled to a near-constant. That is
// not an error — but the overwhelmingly likely cause is (F, k) outside the
// pattern-forming crescent, worth saying out loud because a blank texture is
// otherwise indistinguishable from a wiring mistake.
inline bool rdSimulate(const RDParams& p, std::vector<float>& out, bool* washedOut = nullptr) {
    std::vector<float> u, v;
    rdSeed(p, u, v);
    if (!rdRun(p, u, v)) return false;

    double lo = 1e300, hi = -1e300;
    for (size_t i = 0; i < v.size(); ++i) {
        double t = v[i];
        if (!(t == t)) t = 0.0;                  // a NaN needs an unstable dt, which
        if (t < lo) lo = t;                      // rdStable rejects before we get here
        if (t > hi) hi = t;
    }
    const bool flat = !(hi - lo > 1e-3);
    if (washedOut) *washedOut = flat;
    out.resize(v.size());
    if (flat) {
        for (size_t i = 0; i < v.size(); ++i) out[i] = 0.5f;
    } else {
        const double inv = 1.0 / (hi - lo);
        for (size_t i = 0; i < v.size(); ++i) {
            const double t = (v[i] - lo) * inv;
            out[i] = (float)(t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t));
        }
    }
    return true;
}

// Bilinear resample of a periodic sim grid, so the bake resolution is free of the
// solve resolution. Periodic, so the seam stays seamless; res == sim is exact.
inline double rdSampleWrapped(const std::vector<float>& g, int N, double fx, double fy) {
    const double gx = fx * N - 0.5, gy = fy * N - 0.5;
    const int x0 = (int)floor(gx), y0 = (int)floor(gy);
    const double tx = gx - x0, ty = gy - y0;
    const int xa = ((x0 % N) + N) % N, xb = (((x0 + 1) % N) + N) % N;
    const int ya = ((y0 % N) + N) % N, yb = (((y0 + 1) % N) + N) % N;
    const double a = g[(size_t)ya * N + xa], b = g[(size_t)ya * N + xb];
    const double c = g[(size_t)yb * N + xa], d = g[(size_t)yb * N + xb];
    return (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty;
}

// Named (F, k) points. Gray-Scott's parameter plane is mostly *not* interesting —
// outside a thin crescent every seed decays back to the uniform state — and the
// crescent is only about 0.01 wide in k, so plausible-looking numbers picked by hand
// will usually give a blank texture. Presets are therefore not a convenience here,
// they are the difference between the feature being usable and not. Each of these
// was chosen by rendering the plane and looking; each is checked by `-checkreaction`
// for real contrast and a plausible feature wavelength, so the names cannot rot
// silently if the solver changes.
struct RDPreset { const char* name; double feed, kill; const char* what; };
inline const RDPreset* rdPresets(int& n) {
    static const RDPreset t[] = {
        {"spots",   0.038, 0.065, "round, evenly spaced dots on a dark field"},
        {"holes",   0.030, 0.055, "the inverse — dark pits in a light field"},
        {"maze",    0.030, 0.057, "labyrinth of equal-width corridors, no dots"},
        {"coral",   0.058, 0.063, "thicker rounded ridges, brain-coral"},
        {"worms",   0.046, 0.065, "short strokes mixed with dots"},
        // mitosis is much the coarsest and the only one whose wavelength varies a lot
        // with the seed (21-64 cells against everything else's steady 16-18), so it wants
        // a larger `sim` than the rest to get the same number of features across.
        {"mitosis", 0.018, 0.051, "sparse soft blobs caught mid-division"},
    };
    n = (int)(sizeof(t) / sizeof(t[0]));
    return t;
}
inline bool rdPresetLookup(const std::string& name, double& feed, double& kill) {
    int n = 0; const RDPreset* t = rdPresets(n);
    for (int i = 0; i < n; ++i)
        if (name == t[i].name) { feed = t[i].feed; kill = t[i].kill; return true; }
    return false;
}
inline std::string rdPresetNames() {
    int n = 0; const RDPreset* t = rdPresets(n);
    std::string s;
    for (int i = 0; i < n; ++i) { if (i) s += "|"; s += t[i].name; }
    return s;
}
