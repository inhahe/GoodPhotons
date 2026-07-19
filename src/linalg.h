// Core 3D math. Doubles for Phase 0 — correctness over speed while we validate physics.
#pragma once
#include <cmath>

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    // Indexed component access (0=x, 1=y, 2=z) — used by the watertight ray-triangle
    // test's axis permutation. No bounds check (hot path); callers pass 0..2.
    double  operator[](int i) const { return (&x)[i]; }
    double& operator[](int i)       { return (&x)[i]; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) { return a / length(a); }

// Mirror reflection of incident direction d about unit normal n.
inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0 * dot(d, n)); }

// Orthonormal basis around unit normal n (Duff et al. 2017, branchless).
inline void onb(const Vec3& n, Vec3& t, Vec3& b) {
    double sign = std::copysign(1.0, n.z);
    double a = -1.0 / (sign + n.z);
    double d = n.x * n.y * a;
    t = Vec3(1.0 + sign * n.x * n.x * a, sign * d, -sign * n.x);
    b = Vec3(d, sign + n.y * n.y * a, -n.y);
}

// A general affine map: world = M*p + t, with M a 3x3 linear part (row-major
// m[0..8]) and t a translation. Unlike a scale+Euler+translate triple, an affine
// is *closed under composition*, so it can represent an arbitrary nesting of
// translate/rotate/scale nodes — i.e. a scene-graph group hierarchy that bakes
// down to world-space primitives at load time. Points use apply(); directions
// (which ignore translation) use applyDir().
struct Affine {
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Vec3 t{0, 0, 0};

    static Affine identity() { return Affine{}; }

    Vec3 apply(const Vec3& p) const {
        return Vec3{m[0] * p.x + m[1] * p.y + m[2] * p.z + t.x,
                    m[3] * p.x + m[4] * p.y + m[5] * p.z + t.y,
                    m[6] * p.x + m[7] * p.y + m[8] * p.z + t.z};
    }
    Vec3 applyDir(const Vec3& v) const {
        return Vec3{m[0] * v.x + m[1] * v.y + m[2] * v.z,
                    m[3] * v.x + m[4] * v.y + m[5] * v.z,
                    m[6] * v.x + m[7] * v.y + m[8] * v.z};
    }
    // this ∘ child: transform by `c` first, then by `*this` (i.e. parent.compose(child)).
    Affine compose(const Affine& c) const {
        Affine r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i * 3 + j] = m[i * 3 + 0] * c.m[0 * 3 + j] +
                                 m[i * 3 + 1] * c.m[1 * 3 + j] +
                                 m[i * 3 + 2] * c.m[2 * 3 + j];
        r.t = apply(c.t);   // M*c.t + t
        return r;
    }
    // Inverse affine: (M,t)^-1 = (M^-1, -M^-1 t). Returns identity if the linear
    // part is singular (degenerate scale). Used to turn a primitive's local->world
    // transform into the world->local map an SDF leaf stores.
    Affine inverse() const {
        double a = m[0], b = m[1], c = m[2];
        double d = m[3], e = m[4], f = m[5];
        double g = m[6], h = m[7], i = m[8];
        double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        Affine r;
        if (std::fabs(det) < 1e-30) return r;   // identity fallback
        double id = 1.0 / det;
        r.m[0] = (e * i - f * h) * id; r.m[1] = (c * h - b * i) * id; r.m[2] = (b * f - c * e) * id;
        r.m[3] = (f * g - d * i) * id; r.m[4] = (a * i - c * g) * id; r.m[5] = (c * d - a * f) * id;
        r.m[6] = (d * h - e * g) * id; r.m[7] = (b * g - a * h) * id; r.m[8] = (a * e - b * d) * id;
        r.t = Vec3{-(r.m[0] * t.x + r.m[1] * t.y + r.m[2] * t.z),
                   -(r.m[3] * t.x + r.m[4] * t.y + r.m[5] * t.z),
                   -(r.m[6] * t.x + r.m[7] * t.y + r.m[8] * t.z)};
        return r;
    }

    // Transform a surface normal: normals map by the INVERSE-TRANSPOSE of the
    // linear part (not the affine itself), so they stay perpendicular to the
    // surface under non-uniform scale. Result is NOT normalized (the caller
    // renormalizes). Falls back to applyDir when the linear part is singular.
    Vec3 applyNormal(const Vec3& n) const {
        Affine inv = inverse();
        // (M^-1)^T * n  — transpose means read inv.m in column order.
        return Vec3{inv.m[0] * n.x + inv.m[3] * n.y + inv.m[6] * n.z,
                    inv.m[1] * n.x + inv.m[4] * n.y + inv.m[7] * n.z,
                    inv.m[2] * n.x + inv.m[5] * n.y + inv.m[8] * n.z};
    }

    // Uniform-scale factor of the linear part (column norms; rotation preserves
    // length so a column's norm is its axis scale). Sets `nonUniform` when the
    // three axis scales differ beyond a small tolerance — the caller uses this to
    // reject non-uniform scale on primitives that can't represent it (spheres).
    double uniformScale(bool& nonUniform) const {
        double sx = std::sqrt(m[0] * m[0] + m[3] * m[3] + m[6] * m[6]);
        double sy = std::sqrt(m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
        double sz = std::sqrt(m[2] * m[2] + m[5] * m[5] + m[8] * m[8]);
        double mx = std::fmax(sx, std::fmax(sy, sz));
        double mn = std::fmin(sx, std::fmin(sy, sz));
        nonUniform = (mx - mn) > 1e-9 * (mx > 0 ? mx : 1.0);
        return (sx + sy + sz) / 3.0;
    }
};
