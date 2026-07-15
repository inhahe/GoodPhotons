// Minimal Wavefront OBJ loader. Reads v / vt / vn / f (triangulating polygons via
// a fan). Per-vertex `vn` normals are loaded as smooth SHADING normals (transformed
// by the mesh transform's inverse-transpose); a mesh without `vn` stays exactly
// flat-shaded (Tri::finalize falls the per-vertex normals back to the geometric
// normal) UNLESS crease-angle auto-smoothing is requested (see `creaseAngleDeg`),
// which synthesizes smooth per-corner normals from the face normals. Dependency-free
// — enough to drop real meshes into a scene and stress the BVH. A richer loader
// (per-face materials, full .mtl) can replace this later.
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
#include <map>
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

// Parse the normal index (the 3rd field) of an OBJ face token ("12/3/4",
// "12//4"). Returns -1 when the token carries no vn ("12" or "12/3").
inline int objNormalIndex(const std::string& tok, int normCount) {
    auto p = tok.find('/');
    if (p == std::string::npos) return -1;
    auto q = tok.find('/', p + 1);
    if (q == std::string::npos) return -1;     // "12/3" has no vn
    std::string field = tok.substr(q + 1);
    if (field.empty()) return -1;
    int idx = std::atoi(field.c_str());
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return normCount + idx;      // relative
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
// (UvProjection / parseUvProjection / projectUV now live in geometry.h so native
// primitives can reuse the identical wrap — see there.)

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
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                   double creaseAngleDeg = -1.0) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "loadObj: cannot open %s\n", path); return 0; }

    std::vector<Vec3> verts;
    std::vector<Vec3> texcoords;   // (u,v,0) per `vt`
    std::vector<Vec3> normals;     // per `vn`, already in WORLD space (inv-transpose)
    int curMat = matId;            // active material (switched by `usemtl` when resolving)
    int added = 0;
    std::string line;
    // For a procedural UV projection we need the whole mesh AABB, so record each
    // added triangle's source vertex indices and fill UVs in a second pass. Crease
    // smoothing needs the same per-tri vertex indices, so record them for either.
    const bool proceduralUV = (uvProj != UvProjection::None) && !loadUV;
    const bool wantSmooth = (creaseAngleDeg >= 0.0);
    const bool recordVI = proceduralUV || wantSmooth;
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
        } else if (line[0] == 'v' && line[1] == 'n') {
            // Smooth shading normal: transform object->world by the inverse-transpose
            // of the mesh transform's linear part, then normalize (renormalized again
            // in finalize()). Stored world-space so it feeds Tri.n{0,1,2} directly.
            std::istringstream ss(line.substr(2));
            Vec3 vn; ss >> vn.x >> vn.y >> vn.z;
            Vec3 wn = xf.applyNormal(vn);
            double l = std::sqrt(dot(wn, wn));
            normals.push_back(l > 1e-18 ? wn * (1.0 / l) : Vec3{0, 0, 0});
        } else if (matResolver && line.rfind("usemtl", 0) == 0) {
            std::istringstream ss(line.substr(6));
            std::string name; ss >> name;
            int resolved = name.empty() ? -1 : (*matResolver)(name);
            curMat = (resolved >= 0) ? resolved : matId;
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::vector<int> idx, tidx, nidx;
            std::string tok;
            while (ss >> tok) {
                int vi = objVertexIndex(tok, (int)verts.size());
                if (vi < 0 || vi >= (int)verts.size()) continue;
                idx.push_back(vi);
                tidx.push_back(loadUV ? objTexIndex(tok, (int)texcoords.size()) : -1);
                nidx.push_back(objNormalIndex(tok, (int)normals.size()));
            }
            // Fan-triangulate the polygon (0, k, k+1).
            auto uvAt = [&](int ti) -> Vec3 {
                return (ti >= 0 && ti < (int)texcoords.size()) ? texcoords[ti] : Vec3{0, 0, 0};
            };
            auto nAt = [&](int ni) -> Vec3 {
                return (ni >= 0 && ni < (int)normals.size()) ? normals[ni] : Vec3{0, 0, 0};
            };
            for (size_t k = 1; k + 1 < idx.size(); ++k) {
                Tri t{verts[idx[0]], verts[idx[k]], verts[idx[k + 1]], curMat, -1, {}};
                if (loadUV) {
                    t.uv0 = uvAt(tidx[0]); t.uv1 = uvAt(tidx[k]); t.uv2 = uvAt(tidx[k + 1]);
                }
                // Per-vertex shading normals (zero => finalize() falls back to gn,
                // preserving exact flat-shading for meshes without `vn`).
                t.n0 = nAt(nidx[0]); t.n1 = nAt(nidx[k]); t.n2 = nAt(nidx[k + 1]);
                s.tris.push_back(t);
                if (recordVI) triVI.push_back({idx[0], idx[k], idx[k + 1]});
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
    // Crease-angle auto-smoothing: only when the mesh carries NO `vn` (authored
    // normals always win). Synthesize a per-corner shading normal by averaging the
    // FACE normals of the triangles incident to that vertex, but merge two faces only
    // when the angle between their normals is below `creaseAngleDeg` — so soft edges
    // smooth while genuine creases (a cube's 90° edges) stay faceted. Angle-weighted
    // (Thürmer & Wüthrich) to avoid tessellation bias. Vertices are welded by POSITION
    // first, so smoothing works even on exporters that split a shared position into
    // several OBJ vertex indices. Opt-in — a mesh without `smooth` is untouched.
    bool didSmooth = false;
    if (wantSmooth && normals.empty() && !triVI.empty() && !verts.empty()) {
        const size_t nt = triVI.size();
        // Face normals (world space) per added triangle.
        std::vector<Vec3> fn(nt);
        for (size_t i = 0; i < nt; ++i) {
            const Vec3& a = verts[triVI[i][0]];
            const Vec3& b = verts[triVI[i][1]];
            const Vec3& c = verts[triVI[i][2]];
            Vec3 n = cross(b - a, c - a);
            double l = std::sqrt(dot(n, n));
            fn[i] = (l > 1e-18) ? n * (1.0 / l) : Vec3{0, 0, 0};
        }
        // Weld vertices by quantized position (epsilon relative to the mesh size),
        // mapping every OBJ vertex index to a canonical welded id.
        Vec3 lo = verts[0], hi = verts[0];
        for (const Vec3& v : verts) {
            lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
        }
        Vec3 ext = hi - lo;
        double diag = std::sqrt(dot(ext, ext));
        double eps = (diag > 0.0 ? diag : 1.0) * 1e-6;
        double inv = 1.0 / eps;
        std::map<std::array<long long, 3>, int> weldMap;
        std::vector<int> weld(verts.size());
        int nWeld = 0;
        for (size_t v = 0; v < verts.size(); ++v) {
            std::array<long long, 3> key{
                (long long)std::llround(verts[v].x * inv),
                (long long)std::llround(verts[v].y * inv),
                (long long)std::llround(verts[v].z * inv)};
            auto it = weldMap.find(key);
            if (it == weldMap.end()) { weldMap.emplace(key, nWeld); weld[v] = nWeld++; }
            else                     { weld[v] = it->second; }
        }
        // Welded-vertex -> incident triangle list.
        std::vector<std::vector<int>> vtris(nWeld);
        for (size_t i = 0; i < nt; ++i)
            for (int c = 0; c < 3; ++c) vtris[weld[triVI[i][c]]].push_back((int)i);
        const double cosThresh = std::cos(creaseAngleDeg * 3.14159265358979323846 / 180.0);
        auto cornerAngle = [&](int tri, int weldedVid) -> double {
            const std::array<int, 3>& vi = triVI[tri];
            int c = (weld[vi[0]] == weldedVid) ? 0 : (weld[vi[1]] == weldedVid) ? 1 : 2;
            const Vec3& P = verts[vi[c]];
            Vec3 e1 = verts[vi[(c + 1) % 3]] - P, e2 = verts[vi[(c + 2) % 3]] - P;
            double l1 = std::sqrt(dot(e1, e1)), l2 = std::sqrt(dot(e2, e2));
            if (l1 < 1e-18 || l2 < 1e-18) return 0.0;
            double ca = dot(e1, e2) / (l1 * l2);
            ca = ca < -1.0 ? -1.0 : (ca > 1.0 ? 1.0 : ca);
            return std::acos(ca);
        };
        for (size_t i = 0; i < nt; ++i) {
            Tri& t = s.tris[triStart + i];
            for (int c = 0; c < 3; ++c) {
                int wv = weld[triVI[i][c]];
                Vec3 sum{0, 0, 0};
                for (int j : vtris[wv])
                    if (dot(fn[i], fn[j]) >= cosThresh) sum += fn[j] * cornerAngle(j, wv);
                double l = std::sqrt(dot(sum, sum));
                Vec3 sn = (l > 1e-12) ? sum * (1.0 / l) : fn[i];
                if (c == 0) t.n0 = sn; else if (c == 1) t.n1 = sn; else t.n2 = sn;
            }
        }
        didSmooth = true;
    }
    std::printf("loadObj: %s -> %d verts, %d tris (mat %d)%s%s\n",
                path, (int)verts.size(), added, matId,
                proceduralUV ? " [procedural UVs]" : "",
                didSmooth ? " [crease-smoothed]" : "");
    return added;
}

// MeshXform overload: a single scale+Euler+translate transform (the common case).
inline int loadObj(Scene& s, const char* path, int matId, const MeshXform& xf,
                   bool loadUV = false, const MtlResolver* matResolver = nullptr,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                   double creaseAngleDeg = -1.0) {
    return loadObj(s, path, matId, xf.toAffine(), loadUV, matResolver, uvProj, uvAxis,
                   creaseAngleDeg);
}

// Backward-compatible convenience overload: translate + uniform scale only.
inline int loadObj(Scene& s, const char* path, int matId,
                   Vec3 translate = {0, 0, 0}, double scale = 1.0) {
    return loadObj(s, path, matId, MeshXform{translate, {scale, scale, scale}, {0, 0, 0}});
}
