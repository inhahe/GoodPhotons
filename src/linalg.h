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
