// Minimal Wavefront OBJ loader. Reads v / f (triangulating polygons via a fan);
// ignores vt/vn/mtl for now (surface appearance comes from the assigned material,
// and geometric normals are recomputed in Scene::build). Dependency-free — enough
// to drop real meshes into a scene and stress the BVH. A richer loader (normals,
// per-face materials, .mtl) can replace this later.
#pragma once
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <array>
#include <algorithm>
#include <cmath>
#include "geometry.h"
#include "scene.h"

// Parse the leading vertex index of an OBJ face token ("12", "12/3", "12/3/4",
// "12//4"). OBJ indices are 1-based; negative means relative to the current end.
inline int objVertexIndex(const std::string& tok, int vertexCount) {
    int idx = std::atoi(tok.c_str());
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return vertexCount + idx;   // relative
    return -1;
}

// Parse the texture-coordinate index (the 2nd field) of an OBJ face token
// ("12/3", "12/3/4"). Returns -1 when the token carries no vt ("12" or "12//4").
inline int objTexIndex(const std::string& tok, int texCount) {
    auto p = tok.find('/');
    if (p == std::string::npos) return -1;
    auto q = tok.find('/', p + 1);
    std::string field = (q == std::string::npos) ? tok.substr(p + 1)
                                                  : tok.substr(p + 1, q - p - 1);
    if (field.empty()) return -1;              // "12//4" has no vt
    int idx = std::atoi(field.c_str());
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return texCount + idx;       // relative
    return -1;
}

// A local->world affine transform for a loaded mesh: world = translate + R*(scale⊙local),
// with R = Rz(rz)·Ry(ry)·Rx(rx) built from Euler angles in DEGREES. Uniform scale is
// just scale={k,k,k}; identity is the default (translate 0, scale 1, no rotation).
struct MeshXform {
    Vec3 translate{0, 0, 0};
    Vec3 scale{1, 1, 1};
    Vec3 rotDeg{0, 0, 0};      // Euler XYZ, degrees

    // Linear part only (scale + rotation, no translation): scale first
    // (component-wise), then rotate X, then Y, then Z.
    Vec3 applyLinear(const Vec3& p) const {
        const double d2r = 3.14159265358979323846 / 180.0;
        double cx = std::cos(rotDeg.x * d2r), sx = std::sin(rotDeg.x * d2r);
        double cy = std::cos(rotDeg.y * d2r), sy = std::sin(rotDeg.y * d2r);
        double cz = std::cos(rotDeg.z * d2r), sz = std::sin(rotDeg.z * d2r);
        Vec3 v{p.x * scale.x, p.y * scale.y, p.z * scale.z};
        v = Vec3{v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z};        // Rx
        v = Vec3{cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z};       // Ry
        v = Vec3{cz * v.x - sz * v.y, sz * v.x + cz * v.y, v.z};        // Rz
        return v;
    }
    Vec3 apply(const Vec3& p) const { return translate + applyLinear(p); }

    // Bake this scale+Euler+translate triple into a general Affine (whose columns
    // are the transformed basis vectors, so the linear math is bit-identical to
    // applyLinear). Lets a mesh's own transform compose with a parent group's.
    Affine toAffine() const {
        Vec3 cx = applyLinear({1, 0, 0});
        Vec3 cy = applyLinear({0, 1, 0});
        Vec3 cz = applyLinear({0, 0, 1});
        Affine a;
        a.m[0] = cx.x; a.m[1] = cy.x; a.m[2] = cz.x;
        a.m[3] = cx.y; a.m[4] = cy.y; a.m[5] = cz.y;
        a.m[6] = cx.z; a.m[7] = cy.z; a.m[8] = cz.z;
        a.t = translate;
        return a;
    }
};

// Build an Affine from a translate / Euler-XYZ-degrees rotate / scale triple,
// matching MeshXform's order (scale, Rx, Ry, Rz, translate). Free helper so the
// FTSL loader can turn a `group { translate/rotate/scale }` node into a
// composable transform without constructing a throwaway MeshXform at each site.
inline Affine affineFromTRS(const Vec3& translate, const Vec3& rotDeg, const Vec3& scale) {
    return MeshXform{translate, scale, rotDeg}.toAffine();
}

// Resolve an OBJ `usemtl <name>` group to a scene material index (>=0), or -1 when
// the name is unknown (the caller then keeps the mesh's default material).
using MtlResolver = std::function<int(const std::string&)>;

// Procedural UV projection for meshes that carry no `vt` coordinates (spec §9.2).
// Applied at load time from the WORLD-space vertex positions, so the generated UVs
// travel through the exact same per-vertex `Tri.uv{0,1,2}` slots that `use_mesh`
// fills — no shading/GPU change is needed (the tracers already interpolate stored
// UVs). `axis` (0=x,1=y,2=z) is the projection/up axis. Coordinates are normalised
// to [0,1] across the mesh AABB so the map wraps once over the object by default;
// `repeat` wrap + a texture-space `scale` tile it further.
enum class UvProjection { None = 0, Planar, Spherical, Cylindrical };

inline UvProjection parseUvProjection(const std::string& s) {
    if (s == "planar")      return UvProjection::Planar;
    if (s == "spherical")   return UvProjection::Spherical;
    if (s == "cylindrical") return UvProjection::Cylindrical;
    return UvProjection::None;
}

// Project one world-space point to (u,v) given the mesh AABB (lo..hi), its centre,
// the projection kind and the up/projection axis (0/1/2). Returns {u,v,0}.
inline Vec3 projectUV(const Vec3& p, const Vec3& lo, const Vec3& hi,
                      const Vec3& ctr, UvProjection proj, int axis) {
    auto comp = [](const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); };
    // The two axes orthogonal to `axis`, in a stable (right-handed-ish) order.
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    auto norm01 = [&](double val, int i) {
        double l = comp(lo, i), h = comp(hi, i);
        double d = h - l;
        return d > 1e-12 ? (val - l) / d : 0.5;
    };
    if (proj == UvProjection::Planar) {
        return Vec3{norm01(comp(p, a0), a0), norm01(comp(p, a1), a1), 0};
    }
    // Direction from the mesh centre (spherical/cylindrical share the azimuth).
    Vec3 d = p - ctr;
    double dz = comp(d, axis);
    double dx = comp(d, a0), dy = comp(d, a1);
    double azim = 0.5 + std::atan2(dy, dx) / (2.0 * PI);   // [0,1)
    if (proj == UvProjection::Cylindrical) {
        return Vec3{azim, norm01(comp(p, axis), axis), 0};
    }
    // Spherical: polar angle from the up axis.
    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    double v = (r > 1e-12) ? std::acos(std::max(-1.0, std::min(1.0, dz / r))) / PI : 0.5;
    return Vec3{azim, v, 0};
}

// Load an OBJ into the scene as triangles of material `matId`, applying the full
// affine transform `xf` (translate + rotation + non-uniform scale). When `loadUV`
// is set, per-vertex texture coordinates are read from `vt` lines and assigned to
// the triangles (for textured materials); otherwise the Tri default UVs are kept.
// When `matResolver` is non-null, OBJ `usemtl <name>` records switch the active
// material for subsequent faces to `matResolver(name)` (falling back to `matId`
// when the name is unknown) — this is the per-face `usemtl use_names` path.
// Returns the number of triangles added (0 on failure). Call before Scene::build().
inline int loadObj(Scene& s, const char* path, int matId, const Affine& xf,
                   bool loadUV = false, const MtlResolver* matResolver = nullptr,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "loadObj: cannot open %s\n", path); return 0; }

    std::vector<Vec3> verts;
    std::vector<Vec3> texcoords;   // (u,v,0) per `vt`
    int curMat = matId;            // active material (switched by `usemtl` when resolving)
    int added = 0;
    std::string line;
    // For a procedural UV projection we need the whole mesh AABB, so record each
    // added triangle's source vertex indices and fill UVs in a second pass.
    const bool proceduralUV = (uvProj != UvProjection::None) && !loadUV;
    std::vector<std::array<int, 3>> triVI;   // vertex indices per added tri
    size_t triStart = s.tris.size();
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            Vec3 v; ss >> v.x >> v.y >> v.z;
            verts.push_back(xf.apply(v));
        } else if (loadUV && line[0] == 'v' && line[1] == 't') {
            std::istringstream ss(line.substr(2));
            double u = 0, v = 0; ss >> u >> v;
            texcoords.push_back(Vec3{u, v, 0});
        } else if (matResolver && line.rfind("usemtl", 0) == 0) {
            std::istringstream ss(line.substr(6));
            std::string name; ss >> name;
            int resolved = name.empty() ? -1 : (*matResolver)(name);
            curMat = (resolved >= 0) ? resolved : matId;
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::vector<int> idx, tidx;
            std::string tok;
            while (ss >> tok) {
                int vi = objVertexIndex(tok, (int)verts.size());
                if (vi < 0 || vi >= (int)verts.size()) continue;
                idx.push_back(vi);
                tidx.push_back(loadUV ? objTexIndex(tok, (int)texcoords.size()) : -1);
            }
            // Fan-triangulate the polygon (0, k, k+1).
            auto uvAt = [&](int ti) -> Vec3 {
                return (ti >= 0 && ti < (int)texcoords.size()) ? texcoords[ti] : Vec3{0, 0, 0};
            };
            for (size_t k = 1; k + 1 < idx.size(); ++k) {
                Tri t{verts[idx[0]], verts[idx[k]], verts[idx[k + 1]], curMat, -1, {}};
                if (loadUV) {
                    t.uv0 = uvAt(tidx[0]); t.uv1 = uvAt(tidx[k]); t.uv2 = uvAt(tidx[k + 1]);
                }
                s.tris.push_back(t);
                if (proceduralUV) triVI.push_back({idx[0], idx[k], idx[k + 1]});
                ++added;
            }
        }
    }
    // Second pass: assign procedural UVs from the world-space mesh AABB.
    if (proceduralUV && !verts.empty()) {
        Vec3 lo = verts[0], hi = verts[0];
        for (const Vec3& v : verts) {
            lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
        }
        Vec3 ctr = (lo + hi) * 0.5;
        int ax = (uvAxis >= 0 && uvAxis <= 2) ? uvAxis : 1;
        for (size_t i = 0; i < triVI.size(); ++i) {
            Tri& t = s.tris[triStart + i];
            t.uv0 = projectUV(verts[triVI[i][0]], lo, hi, ctr, uvProj, ax);
            t.uv1 = projectUV(verts[triVI[i][1]], lo, hi, ctr, uvProj, ax);
            t.uv2 = projectUV(verts[triVI[i][2]], lo, hi, ctr, uvProj, ax);
        }
    }
    std::printf("loadObj: %s -> %d verts, %d tris (mat %d)%s\n",
                path, (int)verts.size(), added, matId,
                proceduralUV ? " [procedural UVs]" : "");
    return added;
}

// MeshXform overload: a single scale+Euler+translate transform (the common case).
inline int loadObj(Scene& s, const char* path, int matId, const MeshXform& xf,
                   bool loadUV = false, const MtlResolver* matResolver = nullptr,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1) {
    return loadObj(s, path, matId, xf.toAffine(), loadUV, matResolver, uvProj, uvAxis);
}

// Backward-compatible convenience overload: translate + uniform scale only.
inline int loadObj(Scene& s, const char* path, int matId,
                   Vec3 translate = {0, 0, 0}, double scale = 1.0) {
    return loadObj(s, path, matId, MeshXform{translate, {scale, scale, scale}, {0, 0, 0}});
}
