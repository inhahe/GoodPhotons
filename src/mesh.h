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
#include <unordered_map>
#include "assetbytes.h"
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

// ---------------------------------------------------------------------------
// Shared tail of the mesh loaders (extracted from loadObj, 0.147.0).
//
// Both remaining passes need the mesh AS A WHOLE — the procedural UV projection
// needs its AABB, crease smoothing needs its vertex adjacency — so neither can run
// while faces are still streaming in. Keeping them in one function is what lets the
// binary `.ftmesh` path produce bit-identical shading to the OBJ path instead of a
// second implementation that can drift.
//
// `verts` are WORLD-space positions; `triVI` gives, for each triangle appended at
// `triStart + i`, the three indices into `verts` it was built from (empty when the
// caller needed neither pass). `haveNormals` means the file supplied authored
// normals, which always win over synthesized ones. Returns whether it smoothed.
inline bool meshFinishTris(Scene& s, size_t triStart,
                           const std::vector<Vec3>& verts,
                           const std::vector<std::array<int, 3>>& triVI,
                           bool proceduralUV, UvProjection uvProj, int uvAxis,
                           bool wantSmooth, bool haveNormals, double creaseAngleDeg) {
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
    if (wantSmooth && !haveNormals && !triVI.empty() && !verts.empty()) {
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
        // Hashed, not ordered: nothing downstream depends on the welded ids being
        // sorted — they are only slots, and both containers assign them in
        // first-seen (ascending v) order, so the result is bit-identical.
        struct QKey {
            long long x, y, z;
            bool operator==(const QKey& o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct QHash {
            size_t operator()(const QKey& k) const {
                unsigned long long h = 1469598103934665603ull;
                auto mix = [&h](long long v) {
                    h ^= (unsigned long long)v; h *= 1099511628211ull;
                    h ^= h >> 29;
                };
                mix(k.x); mix(k.y); mix(k.z);
                return (size_t)h;
            }
        };
        std::unordered_map<QKey, int, QHash> weldMap;
        weldMap.reserve(verts.size() * 2);
        std::vector<int> weld(verts.size());
        int nWeld = 0;
        for (size_t v = 0; v < verts.size(); ++v) {
            QKey key{(long long)std::llround(verts[v].x * inv),
                     (long long)std::llround(verts[v].y * inv),
                     (long long)std::llround(verts[v].z * inv)};
            auto it = weldMap.find(key);
            if (it == weldMap.end()) { weldMap.emplace(key, nWeld); weld[v] = nWeld++; }
            else                     { weld[v] = it->second; }
        }
        // Welded-vertex -> incident CORNER list, as CSR (one flat array + offsets)
        // rather than nWeld separate std::vectors: same traversal order, no
        // per-vertex heap allocation. A corner id `j*3+c` identifies both the
        // triangle (`j`) and which of its three corners touches this vertex, so the
        // corner angle below is a table lookup instead of a search.
        std::vector<int> voff(nWeld + 1, 0);
        for (size_t i = 0; i < nt; ++i)
            for (int c = 0; c < 3; ++c) ++voff[weld[triVI[i][c]] + 1];
        for (int v = 0; v < nWeld; ++v) voff[v + 1] += voff[v];
        std::vector<int> vcorner((size_t)nt * 3);
        {   // fill in ascending triangle order, matching the old push_back order
            std::vector<int> cur(voff.begin(), voff.end() - 1);
            for (size_t i = 0; i < nt; ++i)
                for (int c = 0; c < 3; ++c)
                    vcorner[cur[weld[triVI[i][c]]]++] = (int)i * 3 + c;
        }
        // Per-corner interior angle (Thurmer & Wuthrich weight), computed ONCE per
        // corner instead of once per incident pair. The old code called this from the
        // innermost loop, so a vertex of degree d paid d^2 acos() calls per triangle
        // fan; now it is 3*nt total. Same formula, same inputs => same bits.
        const double cosThresh = std::cos(creaseAngleDeg * 3.14159265358979323846 / 180.0);
        std::vector<double> ang((size_t)nt * 3);
        for (size_t i = 0; i < nt; ++i) {
            const std::array<int, 3>& vi = triVI[i];
            for (int c = 0; c < 3; ++c) {
                const Vec3& P = verts[vi[c]];
                Vec3 e1 = verts[vi[(c + 1) % 3]] - P, e2 = verts[vi[(c + 2) % 3]] - P;
                double l1 = std::sqrt(dot(e1, e1)), l2 = std::sqrt(dot(e2, e2));
                double a = 0.0;
                if (l1 >= 1e-18 && l2 >= 1e-18) {
                    double ca = dot(e1, e2) / (l1 * l2);
                    ca = ca < -1.0 ? -1.0 : (ca > 1.0 ? 1.0 : ca);
                    a = std::acos(ca);
                }
                ang[i * 3 + c] = a;
            }
        }
        for (size_t i = 0; i < nt; ++i) {
            Tri& t = s.tris[triStart + i];
            const Vec3 fni = fn[i];
            for (int c = 0; c < 3; ++c) {
                int wv = weld[triVI[i][c]];
                Vec3 sum{0, 0, 0};
                for (int k = voff[wv], e = voff[wv + 1]; k < e; ++k) {
                    int cid = vcorner[k];
                    const Vec3& fnj = fn[cid / 3];
                    if (dot(fni, fnj) >= cosThresh) sum += fnj * ang[cid];
                }
                double l = std::sqrt(dot(sum, sum));
                Vec3 sn = (l > 1e-12) ? sum * (1.0 / l) : fni;
                if (c == 0) t.n0 = sn; else if (c == 1) t.n1 = sn; else t.n2 = sn;
            }
        }
        didSmooth = true;
    }
    return didSmooth;
}

// Load an OBJ into the scene as triangles of material `matId`, applying the full
// affine transform `xf` (translate + rotation + non-uniform scale). When `loadUV`
// is set, per-vertex texture coordinates are read from `vt` lines and assigned to
// the triangles (for textured materials); otherwise the Tri default UVs are kept.
// When `matResolver` is non-null, OBJ `usemtl <name>` records switch the active
// material for subsequent faces to `matResolver(name)` (falling back to `matId`
// when the name is unknown) — this is the per-face `usemtl use_names` path.
// Returns the number of triangles added (0 on failure). Call before Scene::build().
// `buf` is the file's bytes and `path` is only a label for diagnostics, so the same
// parse serves a file on disk and bytes that arrived over the loom pipe (see
// assetbytes.h). `loadObj` below is the file-reading wrapper.
inline int loadObjBytes(Scene& s, const std::string& buf, const char* path, int matId,
                        const Affine& xf,
                        bool loadUV = false, const MtlResolver* matResolver = nullptr,
                        UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                        double creaseAngleDeg = -1.0) {
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
    bool didSmooth = meshFinishTris(s, triStart, verts, triVI, proceduralUV, uvProj,
                                    uvAxis, wantSmooth, !normals.empty(), creaseAngleDeg);
    std::printf("loadObj: %s -> %d verts, %d tris (mat %d)%s%s\n",
                path, (int)verts.size(), added, matId,
                proceduralUV ? " [procedural UVs]" : "",
                didSmooth ? " [crease-smoothed]" : "");
    return added;
}

// Read the file (binary; CRLF is handled at the line split inside, so the parsed
// lines are exactly what text-mode getline used to produce) and parse it.
inline int loadObj(Scene& s, const char* path, int matId, const Affine& xf,
                   bool loadUV = false, const MtlResolver* matResolver = nullptr,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                   double creaseAngleDeg = -1.0) {
    std::string buf;
    if (!assetbytes::readFile(path, buf)) {
        std::fprintf(stderr, "loadObj: cannot open %s\n", path);
        return 0;
    }
    return loadObjBytes(s, buf, path, matId, xf, loadUV, matResolver, uvProj, uvAxis,
                        creaseAngleDeg);
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

// ===========================================================================
// .ftmesh — a binary indexed triangle mesh (0.147.0)
// ===========================================================================
//
// WHY. The loom live viewer (§F4) re-derives its scene every frame and hands the
// swept mesh to the renderer through a file. As OBJ text that costs, per frame,
// a float->decimal format on loom's side and a decimal->float parse on ftrace's,
// for a mesh that was f64 in memory on both ends. `.ftmesh` is the same geometry
// as raw little-endian f32, so both of those disappear. It is deliberately NOT an
// interchange format: no materials, no groups, no names, one mesh per file. OBJ,
// glTF and FBX remain the formats for authored assets — this one exists for
// machine-to-machine handoff within a single run.
//
// FIDELITY. f32 is a strict IMPROVEMENT over the text it replaces: `write_obj`
// formatted with `%.6g`, i.e. 6 significant decimal digits, where f32 carries
// ~7.2. Nothing gets coarser.
//
// LAYOUT (little-endian; writer and reader are the same machine, and the header
// magic catches the case where they somehow aren't):
//
//   0   char  magic[8]   "FTMESH\0\0"
//   8   u32   version    1
//   12  u32   flags      bit0 = has normals, bit1 = has UVs
//   16  u32   nverts
//   20  u32   ntris
//   24  f32   pos[nverts*3]
//       f32   nrm[nverts*3]     if flags & 1
//       f32   uv [nverts*2]     if flags & 2
//       u32   idx[ntris*3]
//
// Everything is 4-byte aligned and the size is fully determined by the header, so
// a truncated file (a reader racing a writer that is not using an atomic replace)
// is detected rather than silently producing garbage geometry.
//
// The writer is `tools/loom/loom/ftmesh.py`; keep the two in sync.
static const char FTMESH_MAGIC[8] = {'F', 'T', 'M', 'E', 'S', 'H', '\0', '\0'};
enum : unsigned { FTMESH_HAS_NORMALS = 1u, FTMESH_HAS_UVS = 2u };

// Load a `.ftmesh` into the scene as triangles of material `matId` under transform
// `xf`. Semantics match loadObj's exactly — same world-space transform, same
// authored-normals-win rule, same procedural UV and crease-smoothing passes (it
// calls the same `meshFinishTris`) — so swapping a mesh's file format cannot change
// how it shades. Returns triangles added (0 on failure, with `err` set).
// As with `loadObjBytes`, `buf` is the content and `path` is only a label for the
// diagnostics — so the identical decode serves a file and bytes handed over the loom
// pipe. Everything the format guarantees (magic, version, the header-implied size,
// index range) is checked here, where the bytes are, rather than at whatever supplied
// them: the live channel is exactly the caller that must not be trusted to have
// finished writing.
inline int loadFtmeshBytes(Scene& s, const std::string& buf, const char* path,
                           int matId, const Affine& xf,
                           bool loadUV, std::string& err,
                           UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                           double creaseAngleDeg = -1.0) {
    const size_t HDR = 24;
    if (buf.size() < HDR || std::memcmp(buf.data(), FTMESH_MAGIC, 8) != 0) {
        err = std::string("ftmesh: ") + path + " is not a .ftmesh (bad magic)";
        return 0;
    }
    auto u32at = [&](size_t off) {
        unsigned v; std::memcpy(&v, buf.data() + off, 4); return v;
    };
    unsigned version = u32at(8), flags = u32at(12);
    unsigned nverts = u32at(16), ntris = u32at(20);
    if (version != 1) {
        err = "ftmesh: " + std::string(path) + " is version "
            + std::to_string(version) + ", this build reads version 1";
        return 0;
    }
    const bool hasN  = (flags & FTMESH_HAS_NORMALS) != 0;
    const bool hasUV = (flags & FTMESH_HAS_UVS) != 0;
    // Size the file from the header before touching it, so truncation is an error
    // and not an out-of-bounds read.
    size_t need = HDR + (size_t)nverts * 3 * 4
                + (hasN ? (size_t)nverts * 3 * 4 : 0)
                + (hasUV ? (size_t)nverts * 2 * 4 : 0)
                + (size_t)ntris * 3 * 4;
    if (buf.size() < need) {
        err = "ftmesh: " + std::string(path) + " is truncated ("
            + std::to_string(buf.size()) + " bytes, header implies "
            + std::to_string(need) + ")";
        return 0;
    }
    auto f32at = [&](size_t off) {
        float v; std::memcpy(&v, buf.data() + off, 4); return (double)v;
    };

    size_t off = HDR;
    std::vector<Vec3> verts(nverts);
    for (unsigned i = 0; i < nverts; ++i, off += 12)
        verts[i] = xf.apply(Vec3{f32at(off), f32at(off + 4), f32at(off + 8)});

    std::vector<Vec3> normals;
    if (hasN) {
        normals.resize(nverts);
        for (unsigned i = 0; i < nverts; ++i, off += 12) {
            Vec3 wn = xf.applyNormal(Vec3{f32at(off), f32at(off + 4), f32at(off + 8)});
            double l = std::sqrt(dot(wn, wn));
            normals[i] = (l > 1e-18) ? wn * (1.0 / l) : Vec3{0, 0, 0};
        }
    }
    std::vector<Vec3> texcoords;
    if (hasUV) {
        texcoords.resize(nverts);
        for (unsigned i = 0; i < nverts; ++i, off += 8)
            texcoords[i] = Vec3{f32at(off), f32at(off + 4), 0};
    }

    const bool proceduralUV = (uvProj != UvProjection::None) && !loadUV;
    const bool wantSmooth = (creaseAngleDeg >= 0.0);
    const bool recordVI = proceduralUV || wantSmooth;
    const bool useUV = loadUV && hasUV;
    size_t triStart = s.tris.size();
    std::vector<std::array<int, 3>> triVI;
    if (recordVI) triVI.reserve(ntris);
    s.tris.reserve(s.tris.size() + ntris);
    int added = 0;
    for (unsigned i = 0; i < ntris; ++i, off += 12) {
        unsigned a = u32at(off), b = u32at(off + 4), c = u32at(off + 8);
        // A bad index is a corrupt file, not a recoverable per-face condition the
        // way a malformed OBJ line is — say so once rather than emitting a mesh
        // with silently missing triangles.
        if (a >= nverts || b >= nverts || c >= nverts) {
            err = "ftmesh: " + std::string(path) + " triangle " + std::to_string(i)
                + " references a vertex out of range (nverts=" + std::to_string(nverts) + ")";
            s.tris.resize(triStart);
            return 0;
        }
        Tri t{verts[a], verts[b], verts[c], matId, -1, {}};
        if (useUV) { t.uv0 = texcoords[a]; t.uv1 = texcoords[b]; t.uv2 = texcoords[c]; }
        if (hasN)  { t.n0 = normals[a];    t.n1 = normals[b];    t.n2 = normals[c]; }
        s.tris.push_back(t);
        if (recordVI) triVI.push_back({(int)a, (int)b, (int)c});
        ++added;
    }

    bool didSmooth = meshFinishTris(s, triStart, verts, triVI, proceduralUV, uvProj,
                                    uvAxis, wantSmooth, hasN, creaseAngleDeg);
    std::printf("loadFtmesh: %s -> %u verts, %d tris (mat %d)%s%s%s\n",
                path, nverts, added, matId,
                hasN ? " [normals]" : "",
                proceduralUV ? " [procedural UVs]" : "",
                didSmooth ? " [crease-smoothed]" : "");
    return added;
}

inline int loadFtmesh(Scene& s, const char* path, int matId, const Affine& xf,
                      bool loadUV, std::string& err,
                      UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                      double creaseAngleDeg = -1.0) {
    std::string buf;
    if (!assetbytes::readFile(path, buf)) {
        err = std::string("ftmesh: cannot open ") + path;
        return 0;
    }
    return loadFtmeshBytes(s, buf, path, matId, xf, loadUV, err, uvProj, uvAxis,
                           creaseAngleDeg);
}

inline int loadFtmesh(Scene& s, const char* path, int matId, const MeshXform& xf,
                      bool loadUV, std::string& err,
                      UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                      double creaseAngleDeg = -1.0) {
    return loadFtmesh(s, path, matId, xf.toAffine(), loadUV, err, uvProj, uvAxis,
                      creaseAngleDeg);
}
