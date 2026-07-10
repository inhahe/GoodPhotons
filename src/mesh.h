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

    Vec3 apply(const Vec3& p) const {
        const double d2r = 3.14159265358979323846 / 180.0;
        double cx = std::cos(rotDeg.x * d2r), sx = std::sin(rotDeg.x * d2r);
        double cy = std::cos(rotDeg.y * d2r), sy = std::sin(rotDeg.y * d2r);
        double cz = std::cos(rotDeg.z * d2r), sz = std::sin(rotDeg.z * d2r);
        // Scale first (component-wise), then rotate X, then Y, then Z.
        Vec3 v{p.x * scale.x, p.y * scale.y, p.z * scale.z};
        // Rx
        v = Vec3{v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z};
        // Ry
        v = Vec3{cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z};
        // Rz
        v = Vec3{cz * v.x - sz * v.y, sz * v.x + cz * v.y, v.z};
        return translate + v;
    }
};

// Resolve an OBJ `usemtl <name>` group to a scene material index (>=0), or -1 when
// the name is unknown (the caller then keeps the mesh's default material).
using MtlResolver = std::function<int(const std::string&)>;

// Load an OBJ into the scene as triangles of material `matId`, applying the full
// affine transform `xf` (translate + rotation + non-uniform scale). When `loadUV`
// is set, per-vertex texture coordinates are read from `vt` lines and assigned to
// the triangles (for textured materials); otherwise the Tri default UVs are kept.
// When `matResolver` is non-null, OBJ `usemtl <name>` records switch the active
// material for subsequent faces to `matResolver(name)` (falling back to `matId`
// when the name is unknown) — this is the per-face `usemtl use_names` path.
// Returns the number of triangles added (0 on failure). Call before Scene::build().
inline int loadObj(Scene& s, const char* path, int matId, const MeshXform& xf,
                   bool loadUV = false, const MtlResolver* matResolver = nullptr) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "loadObj: cannot open %s\n", path); return 0; }

    std::vector<Vec3> verts;
    std::vector<Vec3> texcoords;   // (u,v,0) per `vt`
    int curMat = matId;            // active material (switched by `usemtl` when resolving)
    int added = 0;
    std::string line;
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
                ++added;
            }
        }
    }
    std::printf("loadObj: %s -> %d verts, %d tris (mat %d)\n",
                path, (int)verts.size(), added, matId);
    return added;
}

// Backward-compatible convenience overload: translate + uniform scale only.
inline int loadObj(Scene& s, const char* path, int matId,
                   Vec3 translate = {0, 0, 0}, double scale = 1.0) {
    return loadObj(s, path, matId, MeshXform{translate, {scale, scale, scale}, {0, 0, 0}});
}
