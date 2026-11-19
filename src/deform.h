// Mesh deformers (ROADMAP 11 stage E): a mesh's vertices moved by handles. Each deformer is a
// function of its handles' places at one time, evaluated at every motion key on the same rest
// mesh, so every vertex has a path of its own through the shutter (Scene::vertexKeys), and frames
// stay independent -- except the dynamic elastic, which is simulated in frame order once into a
// cache the frames then read (elastic_cache.h).
//
//   falloff   each handle moves the vertices within its radius by its own move, weighted by a
//             smooth profile of the distance; the handles' moves add.
//   rbf       the smooth displacement field that takes every handle's point exactly to its target:
//             a radial basis function interpolant plus an affine part, so a rigid or affine move
//             of the handles moves the mesh exactly so.
//   elastic   as-rigid-as-possible (Sorkine & Alexa 2007): the handles' regions are carried with
//             them and the rest of the surface follows like rubber, every vertex's one-ring kept
//             as close to a rotation of its rest shape as the handles allow. With inertia (the
//             dynamic form) the same energy plus a mass term, stepped through time.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "geometry.h"
#include "parallel.h"

namespace deform {

// A handle: the point `at` on the rest mesh, moved to `to` (at one time). `radius` is how far its
// pull reaches (falloff), or which vertices it holds (elastic: those within it).
struct Handle {
    Vec3 at, to;
    double radius = 0.0;
};

// ---- falloff -------------------------------------------------------------------------------
enum class Profile { Smooth, Linear, Gaussian };

// The weight at distance fraction s in [0, 1] of the radius: 1 at the handle, 0 at the rim, with
// a smooth (C2) rim for Smooth and Gaussian.
inline double profileWeight(Profile p, double s) {
    if (!(s < 1.0)) return 0.0;
    switch (p) {
        case Profile::Linear: return 1.0 - s;
        case Profile::Gaussian: {
            const double e4 = std::exp(-4.0);
            const double g = (std::exp(-4.0 * s * s) - e4) / (1.0 - e4);
            const double t = 1.0 - s * s;           // fades the cut to C2 at the rim
            return g * t * t;
        }
        default: { const double t = 1.0 - s * s; return t * t * t; }
    }
}

inline void falloff(std::vector<Vec3>& p, const std::vector<Handle>& hs, Profile prof) {
    std::vector<Vec3> d(p.size(), Vec3{0, 0, 0});
    for (const Handle& h : hs) {
        if (!(h.radius > 0.0)) continue;
        const Vec3 mv = h.to - h.at;
        for (size_t i = 0; i < p.size(); ++i) {
            const double w = profileWeight(prof, length(p[i] - h.at) / h.radius);
            if (w > 0.0) d[i] = d[i] + mv * w;
        }
    }
    for (size_t i = 0; i < p.size(); ++i) p[i] = p[i] + d[i];
}

// ---- rbf -----------------------------------------------------------------------------------
enum class Kernel { Biharmonic, Triharmonic, Gaussian, Multiquadric };

inline double kernelAt(Kernel k, double r, double eps) {
    switch (k) {
        case Kernel::Triharmonic: return r * r * r;
        case Kernel::Gaussian: return std::exp(-(r / eps) * (r / eps));
        case Kernel::Multiquadric: return std::sqrt(r * r + eps * eps);
        default: return r;                          // biharmonic in 3-D
    }
}

// Solve the dense system A x = B (n x n, m right-hand sides, row-major) in place by Gaussian
// elimination with partial pivoting. False if it is singular.
inline bool solveDense(std::vector<double>& A, std::vector<double>& B, int n, int m) {
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[(size_t)r * n + c]) > std::fabs(A[(size_t)piv * n + c])) piv = r;
        if (!(std::fabs(A[(size_t)piv * n + c]) > 1e-300)) return false;
        if (piv != c) {
            for (int k = 0; k < n; ++k) std::swap(A[(size_t)c * n + k], A[(size_t)piv * n + k]);
            for (int k = 0; k < m; ++k) std::swap(B[(size_t)c * m + k], B[(size_t)piv * m + k]);
        }
        const double inv = 1.0 / A[(size_t)c * n + c];
        for (int r = c + 1; r < n; ++r) {
            const double f = A[(size_t)r * n + c] * inv;
            if (f == 0.0) continue;
            for (int k = c; k < n; ++k) A[(size_t)r * n + k] -= f * A[(size_t)c * n + k];
            for (int k = 0; k < m; ++k) B[(size_t)r * m + k] -= f * B[(size_t)c * m + k];
        }
    }
    for (int c = n - 1; c >= 0; --c) {
        for (int k = 0; k < m; ++k) {
            double s = B[(size_t)c * m + k];
            for (int j = c + 1; j < n; ++j) s -= A[(size_t)c * n + j] * B[(size_t)j * m + k];
            B[(size_t)c * m + k] = s / A[(size_t)c * n + c];
        }
    }
    return true;
}

// The eigenvectors of symmetric 3x3 `a` (Jacobi), as the columns of `V`, and its eigenvalues on
// `a`'s diagonal.
inline void symEigen3(double a[3][3], double V[3][3]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) V[r][c] = r == c ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 24; ++sweep) {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        const double on = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (off <= 1e-30 * on || off < 1e-300) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                const double th = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
                }
            }
    }
}

// Move `p` by the RBF field through the handles: the kernel's interpolant plus an affine part
// over the space the handles span -- a constant, and a linear term along each of their principal
// axes they truly spread along (all three for handles in general position, two for handles in a
// plane, as on a flat sheet, one for handles on a line). So a move of the handles that is affine
// within their span moves the mesh exactly so.
inline bool rbf(std::vector<Vec3>& p, const std::vector<Handle>& hs, Kernel k, double eps, std::string& err) {
    const int n = (int)hs.size();
    if (n == 0) return true;
    Vec3 c{0, 0, 0};
    for (const Handle& h : hs) c = c + h.at;
    c = c * (1.0 / n);
    double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, V[3][3];
    double ext = 0.0;
    for (const Handle& h : hs) {
        const Vec3 d = h.at - c;
        const double dv[3] = {d.x, d.y, d.z};
        for (int r = 0; r < 3; ++r)
            for (int q = 0; q < 3; ++q) cov[r][q] += dv[r] * dv[q];
        ext = std::max(ext, dot(d, d));
    }
    symEigen3(cov, V);
    std::vector<Vec3> axes;                         // the principal axes the handles spread along
    for (int a = 0; a < 3; ++a)
        if (cov[a][a] > 1e-12 * ext * n && (int)axes.size() + 1 < n)
            axes.push_back(Vec3{V[0][a], V[1][a], V[2][a]});
    const int nA = 1 + (int)axes.size();
    const int N = n + nA;
    const auto poly = [&](const Vec3& x, double* out) {
        out[0] = 1.0;
        for (size_t a = 0; a < axes.size(); ++a) out[a + 1] = dot(x - c, axes[a]);
    };
    std::vector<double> A((size_t)N * N, 0.0), B((size_t)N * 3, 0.0);
    double pv[4];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            A[(size_t)i * N + j] = kernelAt(k, length(hs[(size_t)i].at - hs[(size_t)j].at), eps);
        poly(hs[(size_t)i].at, pv);
        for (int a = 0; a < nA; ++a) {
            A[(size_t)i * N + n + a] = pv[a];
            A[(size_t)(n + a) * N + i] = pv[a];
        }
        const Vec3 d = hs[(size_t)i].to - hs[(size_t)i].at;
        B[(size_t)i * 3 + 0] = d.x; B[(size_t)i * 3 + 1] = d.y; B[(size_t)i * 3 + 2] = d.z;
    }
    if (!solveDense(A, B, N, 3)) {
        err = "the RBF handles' system is singular (two handles at one point, or a kernel that "
              "cannot tell them apart)";
        return false;
    }
    for (Vec3& x : p) {
        Vec3 d{0, 0, 0};
        for (int i = 0; i < n; ++i) {
            const double w = kernelAt(k, length(x - hs[(size_t)i].at), eps);
            d = d + Vec3{B[(size_t)i * 3], B[(size_t)i * 3 + 1], B[(size_t)i * 3 + 2]} * w;
        }
        poly(x, pv);
        for (int a = 0; a < nA; ++a)
            d = d + Vec3{B[(size_t)(n + a) * 3], B[(size_t)(n + a) * 3 + 1], B[(size_t)(n + a) * 3 + 2]} * pv[a];
        x = x + d;
    }
    return true;
}

// ---- elastic (as-rigid-as-possible) --------------------------------------------------------
// 3x3 matrices, row-major.
using M3 = std::array<double, 9>;

// The rotation closest to M (the polar factor, reflection-free): R = U diag(1, 1, det(U V^T)) V^T
// for M = U S V^T, the SVD from a Jacobi eigen-decomposition of M^T M.
inline M3 closestRotation(const M3& M) {
    // M^T M, symmetric
    double a[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += M[k * 3 + r] * M[k * 3 + c];
            a[r][c] = s;
        }
    double V[3][3];
    symEigen3(a, V);
    // U's columns: M v_i / sigma_i, orthonormalised (a degenerate one completed by a cross
    // product). Degenerate is judged against the largest singular value, so the answer does not
    // depend on the mesh's scale; a one-ring with no extent at all (M = 0) has no rotation to
    // read and keeps the identity.
    double U[3][3];
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int x, int y) { return a[x][x] > a[y][y]; });
    const double sigmaMax = std::sqrt(std::max(0.0, a[order[0]][order[0]]));
    if (!(sigmaMax > 1e-150) || !std::isfinite(sigmaMax)) return M3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Vec3 u[3];
    for (int i = 0; i < 3; ++i) {
        const int col = order[i];
        const Vec3 v{V[0][col], V[1][col], V[2][col]};
        Vec3 mv{M[0] * v.x + M[1] * v.y + M[2] * v.z, M[3] * v.x + M[4] * v.y + M[5] * v.z,
                M[6] * v.x + M[7] * v.y + M[8] * v.z};
        for (int j = 0; j < i; ++j) mv = mv - u[j] * dot(mv, u[j]);
        double l = length(mv);
        if (l < 1e-9 * sigmaMax) {
            mv = i == 2 ? cross(u[0], u[1]) : (std::fabs(u[0].x) < 0.9 ? cross(u[0], Vec3{1, 0, 0})
                                                                       : cross(u[0], Vec3{0, 1, 0}));
            for (int j = 0; j < i; ++j) mv = mv - u[j] * dot(mv, u[j]);
            l = length(mv);
        }
        u[i] = mv * (1.0 / l);
        U[0][i] = u[i].x; U[1][i] = u[i].y; U[2][i] = u[i].z;
    }
    double Vs[3][3];
    for (int i = 0; i < 3; ++i)
        for (int r = 0; r < 3; ++r) Vs[r][i] = V[r][order[i]];
    // det(U V^T): flip the smallest singular direction if it is a reflection
    const auto det3 = [](double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    if (det3(U) * det3(Vs) < 0.0)
        for (int r = 0; r < 3; ++r) U[r][2] = -U[r][2];
    M3 R{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += U[r][k] * Vs[c][k];
            R[r * 3 + c] = s;
        }
    return R;
}

inline Vec3 mul(const M3& R, const Vec3& v) {
    return Vec3{R[0] * v.x + R[1] * v.y + R[2] * v.z, R[3] * v.x + R[4] * v.y + R[5] * v.z,
                R[6] * v.x + R[7] * v.y + R[8] * v.z};
}

class Arap {
public:
    // The rest mesh: its vertices and triangles. Builds the cotangent weights of its edges.
    void build(const std::vector<Vec3>& rest, const std::vector<std::array<int, 3>>& tris) {
        rest_ = rest;
        const int n = (int)rest.size();
        std::vector<std::vector<std::pair<int, double>>> nb((size_t)n);
        const auto add = [&](int i, int j, double w) {
            for (auto& e : nb[(size_t)i]) if (e.first == j) { e.second += w; return; }
            nb[(size_t)i].push_back({j, w});
        };
        mass_.assign((size_t)n, 0.0);
        for (const auto& t : tris) {
            for (int k = 0; k < 3; ++k) {
                const int i = t[(size_t)k], j = t[(size_t)((k + 1) % 3)], o = t[(size_t)((k + 2) % 3)];
                const Vec3 a = rest[(size_t)i] - rest[(size_t)o], b = rest[(size_t)j] - rest[(size_t)o];
                const double s = length(cross(a, b));
                const double cot = s > 1e-20 ? dot(a, b) / s : 0.0;   // the angle at o faces edge ij
                add(i, j, 0.5 * cot); add(j, i, 0.5 * cot);
            }
            const double area = 0.5 * length(cross(rest[(size_t)t[1]] - rest[(size_t)t[0]],
                                                   rest[(size_t)t[2]] - rest[(size_t)t[0]]));
            for (int k = 0; k < 3; ++k) mass_[(size_t)t[(size_t)k]] += area / 3.0;
        }
        start_.assign((size_t)n + 1, 0);
        nbr_.clear(); w_.clear();
        for (int i = 0; i < n; ++i) {
            for (const auto& e : nb[(size_t)i]) {
                nbr_.push_back(e.first);
                // An obtuse pair can make a cotangent weight negative, and the system indefinite:
                // kept just above zero, as is usual.
                w_.push_back(std::max(e.second, 1e-6));
            }
            start_[(size_t)i + 1] = (int)nbr_.size();
        }
    }
    const std::vector<double>& mass() const { return mass_; }

    // Deform: the vertices `held` marks sit at `target`; the rest are solved, from `x` (the
    // starting guess, overwritten by the result), in `iters` rounds of local rotations and a
    // global solve. `inertia` > 0 adds inertia * mass_i * |x_i - y_i|^2 to the energy (the dynamic
    // form's mass term, y the inertial guess). `fromRest`: `x` is the rest mesh, so the first
    // round takes every rotation as the identity -- the plain Laplacian deform, the usual start,
    // exact at once for a translation -- rather than reading rotations off the held vertices'
    // jump alone. `rest` (null: the mesh build() took) is the rest shape for this solve only -- a
    // dynamic elastic after other deformers rests on their output, which moves with time -- while
    // the edges' weights and the masses stay build()'s.
    // False when a stop arrived part-way (ft::parallelFor): `x` is then half-solved, not to be used.
    [[nodiscard]] bool solve(const std::vector<char>& held, const std::vector<Vec3>& target, std::vector<Vec3>& x,
                             int iters, double inertia = 0.0, const std::vector<Vec3>* y = nullptr,
                             bool fromRest = false, const std::vector<Vec3>* rest = nullptr) const {
        const std::vector<Vec3>& r0 = rest ? *rest : rest_;
        const int n = (int)rest_.size();
        for (int i = 0; i < n; ++i) if (held[(size_t)i]) x[(size_t)i] = target[(size_t)i];
        std::vector<M3> R((size_t)n, M3{1, 0, 0, 0, 1, 0, 0, 0, 1});
        std::vector<Vec3> b((size_t)n);
        for (int it = 0; it < iters; ++it) {
            // local: each one-ring's closest rotation
            if (!(fromRest && it == 0) &&
                !forBlocks(n, [&](int lo, int hi, size_t) {
                    for (int i = lo; i < hi; ++i) {
                        M3 M{};
                        for (int e = start_[(size_t)i]; e < start_[(size_t)i + 1]; ++e) {
                            const int j = nbr_[(size_t)e];
                            const Vec3 r = r0[(size_t)i] - r0[(size_t)j], d = x[(size_t)i] - x[(size_t)j];
                            const double w = w_[(size_t)e];
                            const double rv[3] = {r.x, r.y, r.z}, dv[3] = {d.x, d.y, d.z};
                            for (int p = 0; p < 3; ++p)
                                for (int q = 0; q < 3; ++q) M[(size_t)(p * 3 + q)] += w * dv[p] * rv[q];
                        }
                        R[(size_t)i] = closestRotation(M);
                    }
                }))
                return false;
            // global: L x = b over the free vertices, the held ones on the right-hand side
            if (!forBlocks(n, [&](int lo, int hi, size_t) {
                    for (int i = lo; i < hi; ++i) {
                        Vec3 sum{0, 0, 0};
                        for (int e = start_[(size_t)i]; e < start_[(size_t)i + 1]; ++e) {
                            const int j = nbr_[(size_t)e];
                            const Vec3 r = r0[(size_t)i] - r0[(size_t)j];
                            sum = sum + (mul(R[(size_t)i], r) + mul(R[(size_t)j], r)) * (0.5 * w_[(size_t)e]);
                        }
                        if (inertia > 0.0 && y) sum = sum + (*y)[(size_t)i] * (inertia * mass_[(size_t)i]);
                        b[(size_t)i] = sum;
                    }
                }))
                return false;
            if (!cg(held, b, x, inertia)) return false;
        }
        return true;
    }

private:
    // Vertices in blocks of kBlock, `fn(lo, hi, block)` each, over ft::parallelFor. A block's
    // bounds do not depend on the threads, so a sum gathered per block and then over the blocks
    // in order comes out the same on any machine -- which keeps a bake the same run to run. A
    // solve runs this several times every CG iteration, so a mesh of more than one block keeps
    // its workers (pool_) rather than making threads each time.
    static constexpr int kBlock = 2048;
    template <class Fn>
    bool forBlocks(int n, Fn&& fn) const {
        const size_t nb = (size_t)((n + kBlock - 1) / kBlock);
        const auto block = [&](size_t bi) {
            const int lo = (int)bi * kBlock;
            fn(lo, std::min(n, lo + kBlock), bi);
        };
        if (nb < 2) return ft::parallelFor(nb, 1, block);
        if (!pool_) {
            const unsigned hw = ft::workerCount();   // -t's cap (parallel.h)
            pool_ = std::make_unique<ft::BandPool>((int)std::min<size_t>(hw ? hw : 4, nb));
        }
        return ft::parallelFor(*pool_, nb, 1, block);
    }

    // (L + inertia * M) x = b on the free vertices by Jacobi-preconditioned conjugate gradients,
    // warm-started from x (the held vertices fixed, their terms moved to the right-hand side).
    // False if a stop arrived part-way.
    bool cg(const std::vector<char>& held, const std::vector<Vec3>& b, std::vector<Vec3>& x,
            double inertia) const {
        const int n = (int)rest_.size();
        const size_t nb = (size_t)((n + kBlock - 1) / kBlock);
        std::vector<double> diag((size_t)n, 0.0);
        for (int i = 0; i < n; ++i) {
            double d = inertia * mass_[(size_t)i];
            for (int e = start_[(size_t)i]; e < start_[(size_t)i + 1]; ++e) d += w_[(size_t)e];
            diag[(size_t)i] = d > 0.0 ? d : 1.0;
        }
        // out = the free rows of A v (rhsOnly: minus the held neighbours' terms alone)
        const auto apply = [&](const std::vector<Vec3>& v, std::vector<Vec3>& out, bool rhsOnly) {
            return forBlocks(n, [&](int lo, int hi, size_t) {
                for (int i = lo; i < hi; ++i) {
                    if (held[(size_t)i]) { out[(size_t)i] = Vec3{0, 0, 0}; continue; }
                    Vec3 sum = rhsOnly ? Vec3{0, 0, 0} : v[(size_t)i] * diag[(size_t)i];
                    for (int e = start_[(size_t)i]; e < start_[(size_t)i + 1]; ++e) {
                        const int j = nbr_[(size_t)e];
                        if ((held[(size_t)j] != 0) == rhsOnly) sum = sum - v[(size_t)j] * w_[(size_t)e];
                    }
                    out[(size_t)i] = sum;
                }
            });
        };
        std::vector<double> part1(nb), part2(nb);
        const auto total = [](const std::vector<double>& v) { double t = 0.0; for (double d : v) t += d; return t; };
        // r = b - A x (the held neighbours' terms included through the full product)
        std::vector<Vec3> heldPart((size_t)n), r((size_t)n), z((size_t)n), p((size_t)n), Ap((size_t)n);
        if (!apply(x, heldPart, true) || !apply(x, Ap, false)) return false;
        if (!forBlocks(n, [&](int lo, int hi, size_t bi) {
                double rz = 0.0, r0 = 0.0;
                for (int i = lo; i < hi; ++i) {
                    if (held[(size_t)i]) { r[(size_t)i] = z[(size_t)i] = p[(size_t)i] = Vec3{0, 0, 0}; continue; }
                    r[(size_t)i] = b[(size_t)i] - heldPart[(size_t)i] - Ap[(size_t)i];
                    z[(size_t)i] = r[(size_t)i] * (1.0 / diag[(size_t)i]);
                    p[(size_t)i] = z[(size_t)i];
                    rz += dot(r[(size_t)i], z[(size_t)i]);
                    r0 += dot(r[(size_t)i], r[(size_t)i]);
                }
                part1[bi] = rz; part2[bi] = r0;
            }))
            return false;
        double rz = total(part1);
        const double tol = 1e-20 + 1e-16 * total(part2);
        for (int it = 0; it < 4 * n + 64 && rz > 0.0; ++it) {
            if (!apply(p, Ap, false)) return false;
            if (!forBlocks(n, [&](int lo, int hi, size_t bi) {
                    double s = 0.0;
                    for (int i = lo; i < hi; ++i) s += dot(p[(size_t)i], Ap[(size_t)i]);
                    part1[bi] = s;
                }))
                return false;
            const double pAp = total(part1);
            if (!(pAp > 0.0)) break;
            const double alpha = rz / pAp;
            if (!forBlocks(n, [&](int lo, int hi, size_t bi) {
                    double rr = 0.0, rzNew = 0.0;
                    for (int i = lo; i < hi; ++i) {
                        if (held[(size_t)i]) continue;
                        x[(size_t)i] = x[(size_t)i] + p[(size_t)i] * alpha;
                        r[(size_t)i] = r[(size_t)i] - Ap[(size_t)i] * alpha;
                        z[(size_t)i] = r[(size_t)i] * (1.0 / diag[(size_t)i]);
                        rr += dot(r[(size_t)i], r[(size_t)i]);
                        rzNew += dot(r[(size_t)i], z[(size_t)i]);
                    }
                    part1[bi] = rr; part2[bi] = rzNew;
                }))
                return false;
            const double rr = total(part1), rzNew = total(part2);
            if (rr < tol) break;
            const double beta = rzNew / rz;
            rz = rzNew;
            if (!forBlocks(n, [&](int lo, int hi, size_t) {
                    for (int i = lo; i < hi; ++i)
                        if (!held[(size_t)i]) p[(size_t)i] = z[(size_t)i] + p[(size_t)i] * beta;
                }))
                return false;
        }
        return true;
    }

    std::vector<Vec3> rest_;
    std::vector<int> start_, nbr_;
    std::vector<double> w_, mass_;
    mutable std::unique_ptr<ft::BandPool> pool_;   // forBlocks' workers (one solve at a time)
};

}  // namespace deform
