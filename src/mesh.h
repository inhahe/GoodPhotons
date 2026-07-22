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
#include <cstring>
#include <charconv>
#include <string>
#include <vector>
#include <functional>
#include <array>
#include <algorithm>
#include <cmath>
#include <map>
#include "geometry.h"
#include "scene.h"

// --- Fast OBJ text scanning (Opt 8) -----------------------------------------
// The loader slurps the whole file once and walks it with the helpers below
// instead of constructing a per-line std::istringstream: stream construction,
// locale-imbued num_get and per-token std::string allocation dominated load
// time for real meshes (>1M lines). std::from_chars and stream `>>` are both
// correctly rounded, so every parsed double is bit-identical to the old path.

// Whitespace `>>` would skip inside a line ('\n' is consumed by the line split).
inline bool objIsWs(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}
inline const char* objSkipWs(const char* p, const char* e) {
    while (p < e && objIsWs(*p)) ++p;
    return p;
}

// One double with istream semantics: skip leading whitespace, accept a leading
// '+' (std::from_chars doesn't), store 0.0 on failure (what a failed C++11
// stream extraction assigns to its target).
inline const char* objParseDouble(const char* p, const char* e, double& out, bool& ok) {
    p = objSkipWs(p, e);
    const char* q = (p < e && *p == '+') ? p + 1 : p;
    auto r = std::from_chars(q, e, out);
    if (r.ec != std::errc{}) { out = 0.0; ok = false; return p; }
    return r.ptr;
}

// Up to n whitespace-separated doubles, stopping at the first failure — matching
// chained `ss >> a >> b >> c`, where failbit halts the later extractions (every
// call site pre-zeroes the outputs, and the failing slot itself reads 0).
inline void objParseDoubles(const char* p, const char* e, double* out, int n) {
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) p = objParseDouble(p, e, out[i], ok);
}

// std::atoi on a bounded range: optional sign, decimal digits, stop at the first
// non-digit (an OBJ face field puts '/' or the token end there); no digits -> 0.
inline int objParseInt(const char* p, const char* e) {
    bool neg = false;
    if (p < e && (*p == '+' || *p == '-')) neg = (*p++ == '-');
    long long v = 0;
    while (p < e && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return (int)(neg ? -v : v);
}

// Resolve a raw OBJ index against a running element count: 1-based when
// positive, relative-to-end when negative ("-1" = last), 0/absent -> -1.
inline int objResolveIndex(int raw, int count) {
    if (raw > 0) return raw - 1;
    if (raw < 0) return count + raw;
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
    // Slurp the whole file (binary; CRLF is handled at the line split below, so
    // the parsed lines are exactly what text-mode getline used to produce).
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::fprintf(stderr, "loadObj: cannot open %s\n", path); return 0; }
    std::string buf;
    {
        char tmp[1 << 16];
        size_t got;
        while ((got = std::fread(tmp, 1, sizeof tmp, fp)) > 0) buf.append(tmp, got);
    }
    std::fclose(fp);

    std::vector<Vec3> verts;
    std::vector<Vec3> texcoords;   // (u,v,0) per `vt`
    std::vector<Vec3> normals;     // per `vn`, already in WORLD space (inv-transpose)
    int curMat = matId;            // active material (switched by `usemtl` when resolving)
    int added = 0;
    // For a procedural UV projection we need the whole mesh AABB, so record each
    // added triangle's source vertex indices and fill UVs in a second pass. Crease
    // smoothing needs the same per-tri vertex indices, so record them for either.
    const bool proceduralUV = (uvProj != UvProjection::None) && !loadUV;
    const bool wantSmooth = (creaseAngleDeg >= 0.0);
    const bool recordVI = proceduralUV || wantSmooth;
    std::vector<std::array<int, 3>> triVI;   // vertex indices per added tri
    size_t triStart = s.tris.size();
    std::vector<int> fIdx, fTidx, fNidx;     // per-face scratch (capacity reused)
    const char* p = buf.data();
    const char* const bend = p + buf.size();
    while (p < bend) {
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', (size_t)(bend - p)));
        const char* ls = p;                   // line = [ls, le)
        const char* le = nl ? nl : bend;
        p = nl ? nl + 1 : bend;
        if (le > ls && le[-1] == '\r') --le;  // CRLF: what text-mode getline stripped
        if (le - ls < 2) continue;
        const char c0 = ls[0], c1 = ls[1];
        if (c0 == 'v' && c1 == ' ') {
            double d[3] = {0, 0, 0};
            objParseDoubles(ls + 2, le, d, 3);
            verts.push_back(xf.apply(Vec3{d[0], d[1], d[2]}));
        } else if (loadUV && c0 == 'v' && c1 == 't') {
            double d[2] = {0, 0};
            objParseDoubles(ls + 2, le, d, 2);
            texcoords.push_back(Vec3{d[0], d[1], 0});
        } else if (c0 == 'v' && c1 == 'n') {
            // Smooth shading normal: transform object->world by the inverse-transpose
            // of the mesh transform's linear part, then normalize (renormalized again
            // in finalize()). Stored world-space so it feeds Tri.n{0,1,2} directly.
            double d[3] = {0, 0, 0};
            objParseDoubles(ls + 2, le, d, 3);
            Vec3 wn = xf.applyNormal(Vec3{d[0], d[1], d[2]});
            double l = std::sqrt(dot(wn, wn));
            normals.push_back(l > 1e-18 ? wn * (1.0 / l) : Vec3{0, 0, 0});
        } else if (matResolver && le - ls >= 6 && std::memcmp(ls, "usemtl", 6) == 0) {
            const char* q = objSkipWs(ls + 6, le);
            const char* qe = q;
            while (qe < le && !objIsWs(*qe)) ++qe;
            std::string name(q, qe);
            int resolved = name.empty() ? -1 : (*matResolver)(name);
            curMat = (resolved >= 0) ? resolved : matId;
        } else if (c0 == 'f' && c1 == ' ') {
            fIdx.clear(); fTidx.clear(); fNidx.clear();
            const char* q = ls + 2;
            for (;;) {
                q = objSkipWs(q, le);
                if (q >= le) break;
                const char* t = q;                        // token = [t, q)
                while (q < le && !objIsWs(*q)) ++q;
                int vi = objResolveIndex(objParseInt(t, q), (int)verts.size());
                if (vi < 0 || vi >= (int)verts.size()) continue;
                // Fields: "v", "v/vt", "v/vt/vn", "v//vn".
                int ti = -1, ni = -1;
                const char* s1 = t;
                while (s1 < q && *s1 != '/') ++s1;        // 1st '/'
                if (s1 < q) {
                    const char* f2 = s1 + 1;
                    const char* s2 = f2;
                    while (s2 < q && *s2 != '/') ++s2;    // 2nd '/' (or token end)
                    if (loadUV && s2 > f2)                // non-empty vt field
                        ti = objResolveIndex(objParseInt(f2, s2), (int)texcoords.size());
                    if (s2 < q && s2 + 1 < q)             // non-empty vn field
                        ni = objResolveIndex(objParseInt(s2 + 1, q), (int)normals.size());
                }
                fIdx.push_back(vi);
                fTidx.push_back(ti);
                fNidx.push_back(ni);
            }
            // Fan-triangulate the polygon (0, k, k+1).
            auto uvAt = [&](int ti) -> Vec3 {
                return (ti >= 0 && ti < (int)texcoords.size()) ? texcoords[ti] : Vec3{0, 0, 0};
            };
            auto nAt = [&](int ni) -> Vec3 {
                return (ni >= 0 && ni < (int)normals.size()) ? normals[ni] : Vec3{0, 0, 0};
            };
            for (size_t k = 1; k + 1 < fIdx.size(); ++k) {
                Tri t{verts[fIdx[0]], verts[fIdx[k]], verts[fIdx[k + 1]], curMat, -1, {}};
                if (loadUV) {
                    t.uv0 = uvAt(fTidx[0]); t.uv1 = uvAt(fTidx[k]); t.uv2 = uvAt(fTidx[k + 1]);
                }
                // Per-vertex shading normals (zero => finalize() falls back to gn,
                // preserving exact flat-shading for meshes without `vn`).
                t.n0 = nAt(fNidx[0]); t.n1 = nAt(fNidx[k]); t.n2 = nAt(fNidx[k + 1]);
                s.tris.push_back(t);
                if (recordVI) triVI.push_back({fIdx[0], fIdx[k], fIdx[k + 1]});
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
