// Local majorant grid for participating media — the control/residual decomposition that
// makes transmittance through a thick heterogeneous volume converge.
//
// WHY THIS EXISTS
// ---------------
// A heterogeneous medium cannot use the analytic exp(-sigma*d); it needs a stochastic
// estimator. The engine used plain RATIO TRACKING with ONE global majorant
// sigma_max = sigmaT * densityMax:
//
//     Tr = PROD over candidate collisions at rate sigma_max of (1 - sigma(x_i)/sigma_max)
//
// which is unbiased but whose variance explodes with optical depth. Measured on
// gallery_rain's cloud (sigma_t 2.78, density ~0.3..1.14 from a noise field, so a 2 m chord
// is tau ~ 4-8): the relative standard deviation of ONE transmittance estimate is 3.6 at
// tau = 4 and 41 at tau = 16. The beam gather multiplies TWO of those together (once on the
// beam side, once on the camera side), so a single beam sample carried a relative variance
// in the hundreds. With every beam being MONOCHROMATIC, that heavy tail let one beam
// dominate a pixel and the cloud came out as fully-saturated RGB salt-and-pepper — the
// "iridescent cloud" bug, in a medium whose sigma_t/albedo/phase have no wavelength
// dependence at all.
//
// The instinct — "tighten the majorant" — is exactly WRONG for ratio tracking. As the
// majorant approaches the local sigma, every factor (1 - sigma/sigma_max) approaches zero
// and the estimator degenerates to a binary hit/miss; as the majorant grows, the product
// converges to the deterministic exp(-tau). Ratio tracking wants a LOOSE majorant and pays
// for it in steps. So a local-majorant grid ALONE does nothing for transmittance variance
// (verified numerically before this was written: 128^3 local majorants left the relative
// standard deviation unchanged).
//
// WHAT ACTUALLY FIXES IT: RESIDUAL ratio tracking (Novak et al. 2014). Split the extinction
// into a piecewise-constant CONTROL term that is integrated ANALYTICALLY and a small
// RESIDUAL that is tracked stochastically:
//
//     Tr = exp(-INT sigma_c) * E[ PROD (1 - (sigma(x_i) - sigma_c)/sigma_r) ]
//
// with candidate collisions at rate sigma_r >= sup|sigma - sigma_c| over the cell. The bulk
// of the attenuation moves into the deterministic exponential; what is left is a residual
// whose factors sit near 1. Measured on the same cloud model, per-cell control at a 6 cm
// cell size: relative standard deviation 13.1 -> 0.39 at tau = 8 (a 1100x variance
// reduction) and 41 -> 0.55 at tau = 16 (5700x). It is also FASTER, because sigma_r is a
// small fraction of the global majorant so far fewer candidate collisions are drawn.
//
// The same grid pays for itself a second time on the COLLISION side: delta (Woodcock)
// tracking is unbiased for any majorant, so it can use the tight per-cell sup
// (ctrl + res) instead of the global one, which cuts the null-collision count, and a cell
// whose density is identically zero is skipped outright with no RNG draws at all.
//
// CONSERVATIVENESS
// ----------------
// The grid is estimated by SAMPLING, like the global `density_max` it refines, so it
// carries the same class of assumption. Two things keep it honest:
//
//   * SUPERSAMPLING. Each cell is probed on an (S+1)^3 lattice of its own corners
//     (S = kSuper), i.e. at a spacing S times finer than the cell — far denser than the
//     25^3 whole-bound probe that seeds `density_max`.
//   * DILATION. Each cell's [min, max] is then widened to the union over its 3x3x3
//     neighbourhood, so a feature that falls between one cell's probe points is still
//     covered by the majorant of the cell it lands in, with a full cell of slack. This is
//     also what makes "this cell is empty" safe: a cell reads zero only when it and all 26
//     neighbours probed zero everywhere, i.e. a 3-cell-wide margin of vacuum.
//
// A `safety` factor (default 1.15) then scales the residual half-range. The control term is
// NOT required to bracket the density — only `res >= sup|density - ctrl|` matters for
// unbiasedness, and that is what the dilated min/max gives.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "linalg.h"
#include "parallel.h"

// Per-cell control + residual majorant over a medium's bound AABB. Both fields are
// DIMENSIONLESS density multipliers (the same units as Medium::densityAt); a caller
// multiplies by sigmaT(lambda) to get extinction. `ctrl + res` is a valid local sup of the
// density and is what delta tracking uses; `res` alone is the residual ratio-tracking rate.
struct MajorantGrid {
    Vec3  wmin{0, 0, 0}, wmax{0, 0, 0};
    Vec3  cell{0, 0, 0};       // cell size per axis
    Vec3  invCell{0, 0, 0};    // 1/cell, so the lookup costs no divide
    int   nx = 0, ny = 0, nz = 0;
    std::vector<float> ctrl;   // control density, integrated analytically
    std::vector<float> res;    // residual majorant: sup|density - ctrl| over the cell

    bool   valid() const { return nx > 0 && ny > 0 && nz > 0 && ctrl.size() == res.size()
                                  && ctrl.size() == (size_t)nx * ny * nz; }
    size_t cells() const { return (size_t)nx * ny * nz; }
    size_t idx(int ix, int iy, int iz) const { return ((size_t)iz * ny + iy) * nx + ix; }
};

// Build `g` over [wmin,wmax] by probing `density(p)` (a callable returning the dimensionless
// multiplier at a world point, membership carve included). `targetLongAxis` sets the cell
// count on the longest axis; the others scale with their extent. `capCells` bounds the total
// so a very elongated bound cannot allocate unboundedly.
//
// Returns false if a stop was requested mid-build (ft::parallelFor cancelled) — the caller
// must then discard `g`, exactly as for any other cancelled load pass.
template <class DensityFn>
[[nodiscard]] inline bool buildMajorantGrid(MajorantGrid& g, const Vec3& wmin, const Vec3& wmax,
                                            DensityFn&& density,
                                            int targetLongAxis = 128, int kSuper = 2,
                                            double safety = 1.15, size_t capCells = 4u << 20) {
    g = MajorantGrid{};
    Vec3 ext = wmax - wmin;
    if (!(ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0)) return true;   // degenerate bound: no grid
    double m = std::max(ext.x, std::max(ext.y, ext.z));
    auto axisN = [&](double e) {
        int n = (int)std::lround((double)targetLongAxis * (e / m));
        return std::max(1, std::min(targetLongAxis, n));
    };
    int nx = axisN(ext.x), ny = axisN(ext.y), nz = axisN(ext.z);
    while ((size_t)nx * ny * nz > capCells) {          // shrink uniformly until it fits
        nx = std::max(1, nx / 2); ny = std::max(1, ny / 2); nz = std::max(1, nz / 2);
        if (nx == 1 && ny == 1 && nz == 1) break;
    }
    const int S = std::max(1, kSuper);

    g.wmin = wmin; g.wmax = wmax;
    g.nx = nx; g.ny = ny; g.nz = nz;
    g.cell = Vec3(ext.x / nx, ext.y / ny, ext.z / nz);
    g.invCell = Vec3(1.0 / g.cell.x, 1.0 / g.cell.y, 1.0 / g.cell.z);

    // Pass 1: per-cell [min, max] of the probed density, on an (S+1)^3 corner lattice.
    // Parallel over z-slabs; each slab probes its own S+1 lattice planes (the one plane it
    // shares with its neighbour is recomputed, which is cheaper than synchronising).
    std::vector<float> lo((size_t)nx * ny * nz, 0.0f), hi((size_t)nx * ny * nz, 0.0f);
    const int lxN = nx * S + 1, lyN = ny * S + 1;
    const double sx = ext.x / (nx * S), sy = ext.y / (ny * S), sz = ext.z / (nz * S);
    bool ok = ft::parallelFor((size_t)nz, 1, [&](size_t izs) {
        const int iz = (int)izs;
        std::vector<float> plane((size_t)lxN * lyN * (S + 1));
        for (int k = 0; k <= S; ++k) {
            const double z = wmin.z + sz * (iz * S + k);
            for (int j = 0; j < lyN; ++j) {
                const double y = wmin.y + sy * j;
                float* row = &plane[((size_t)k * lyN + j) * lxN];
                for (int i = 0; i < lxN; ++i)
                    row[i] = (float)density(Vec3(wmin.x + sx * i, y, z));
            }
        }
        for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
            float a = 3.4e38f, b = -3.4e38f;
            for (int k = 0; k <= S; ++k)
            for (int j = 0; j <= S; ++j) {
                const float* row = &plane[((size_t)k * lyN + (iy * S + j)) * lxN + ix * S];
                for (int i = 0; i <= S; ++i) { a = std::min(a, row[i]); b = std::max(b, row[i]); }
            }
            lo[g.idx(ix, iy, iz)] = a; hi[g.idx(ix, iy, iz)] = b;
        }
    });
    if (!ok) { g = MajorantGrid{}; return false; }

    // Pass 2: dilate by one cell (3x3x3 min/max), then split into control + residual.
    g.ctrl.assign((size_t)nx * ny * nz, 0.0f);
    g.res.assign((size_t)nx * ny * nz, 0.0f);
    ok = ft::parallelFor((size_t)nz, 1, [&](size_t izs) {
        const int iz = (int)izs;
        for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
            float a = 3.4e38f, b = -3.4e38f;
            for (int dz = -1; dz <= 1; ++dz) { int z = iz + dz; if (z < 0 || z >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) { int y = iy + dy; if (y < 0 || y >= ny) continue;
            for (int dx = -1; dx <= 1; ++dx) { int x = ix + dx; if (x < 0 || x >= nx) continue;
                size_t k = g.idx(x, y, z);
                a = std::min(a, lo[k]); b = std::max(b, hi[k]);
            }}}
            if (a < 0.0f) a = 0.0f;
            if (b < a) b = a;
            const size_t k = g.idx(ix, iy, iz);
            const double c = 0.5 * ((double)a + (double)b);
            const double r = 0.5 * ((double)b - (double)a) * safety;
            // A cell that probed identically zero over its whole dilated neighbourhood is
            // vacuum: both terms zero, and the traversal skips it with no RNG draws.
            g.ctrl[k] = (b > 0.0f) ? (float)c : 0.0f;
            g.res[k]  = (b > 0.0f) ? (float)r : 0.0f;
        }
    });
    if (!ok) { g = MajorantGrid{}; return false; }
    return true;
}
