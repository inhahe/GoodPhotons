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
#include "parallel.h"
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
// Wavefront .mtl — the companion library an OBJ names with `mtllib`
// ---------------------------------------------------------------------------
//
// An OBJ file carries no material data at all; everything but the geometry lives in a
// sibling .mtl. Skipping it does not lose a nuance, it loses the entire look: every face
// arrives as whatever single fallback the caller supplied, so a model of twelve
// differently coloured parts renders as twelve identical ones and nothing in it is ever
// transmissive.
//
// The slots that map onto ftrace's spectral BSDFs, and how:
//
//   Kd r g b   diffuse albedo        -> `reflect`, upsampled to a reflectance spectrum
//   d  <a>     dissolve, 1 = opaque  -> a < 0.5 makes the material a DIELECTRIC
//   Tr <t>     the same thing inverted (t = 1 - d); both spellings are in the wild
//   Ni <n>     optical density       -> `ior`
//   Tf r g b   transmission filter   -> `absorb`, via -ln(Tf) (see below)
//   Ks / Ns    specular + shininess  -> Glossy, with roughness from the Phong exponent
//   illum      lighting model        -> 4/6/7/9 all mean "transmissive" in practice
//
// The transmissive cut is at d < 0.5, mirroring the 0.5 cuts the glTF importer uses on
// `metallic` and `transmission`: these are single-BSDF materials, so a half-dissolved
// surface has to be called one thing or the other.
//
// `Tf` is a dimensionless per-channel transmittance with no distance attached, unlike
// glTF's attenuationColor/attenuationDistance pair. Recording it as `absorb = -ln(Tf)`
// with `absorbRefDist = 1` says exactly that: one scene unit of this glass transmits Tf.
// The preview then halves the distance across the two crossings a closed solid presents,
// so a closed object shows the authored Tf overall.
//
// Everything else (maps, Ka, Ke, roughness extensions) is ignored for now; a slot ftrace
// cannot represent is better left at its default than approximated into something else.
inline std::unordered_map<std::string, int> parseMtl(Scene& s, const std::string& text,
                                                     const char* path, int* countOut = nullptr) {
    std::unordered_map<std::string, int> out;
    std::string name;
    // Per-material accumulator, flushed by the next `newmtl` and at end of file.
    Vec3   Kd{0.8, 0.8, 0.8}, Ks{0.0, 0.0, 0.0}, Tf{1.0, 1.0, 1.0};
    double d = 1.0, Ni = 1.5, Ns = 0.0;
    bool   haveTf = false;
    int    illum = 2;
    auto flush = [&]() {
        if (name.empty()) return;
        Material m;
        m.reflect = rgbToReflectanceJH(Kd.x, Kd.y, Kd.z);
        const bool transmissive = (d < 0.5) || (illum == 4 || illum == 6 || illum == 7 || illum == 9);
        const double ksLum = 0.2126 * Ks.x + 0.7152 * Ks.y + 0.0722 * Ks.z;
        if (transmissive) {
            m.type = MatType::Dielectric;
            m.ior  = iorConstant(Ni > 1.0 ? Ni : 1.5);
            if (haveTf && (Tf.x < 1.0 || Tf.y < 1.0 || Tf.z < 1.0)) {
                auto sigma = [](double c) {
                    c = std::min(1.0, std::max(1e-6, c));
                    return -std::log(c);                    // over one scene unit
                };
                m.absorb = rgbToReflectanceJH(sigma(Tf.x), sigma(Tf.y), sigma(Tf.z));
                m.absorbRefDist = 1.0;
            } else if (Kd.x < 0.99 || Kd.y < 0.99 || Kd.z < 0.99) {
                // No Tf, but a tinted Kd on a transmissive material is how a lot of
                // exporters colour glass. Read it the same way rather than throwing the
                // only colour the material has.
                auto sigma = [](double c) {
                    c = std::min(1.0, std::max(1e-6, c));
                    return -std::log(c);
                };
                m.absorb = rgbToReflectanceJH(sigma(Kd.x), sigma(Kd.y), sigma(Kd.z));
                m.absorbRefDist = 1.0;
            }
        } else if ((illum == 3 || illum == 5 || illum == 8) && ksLum >= 0.5) {
            // Only the RAYTRACE-REFLECTION illumination models mean metal. A high Ns with
            // a strong Ks does not: Blender writes `Ks 0.5 / Ns 250` for an ordinary
            // diffuse Principled BSDF, so keying off those alone would silently turn every
            // Blender OBJ export into a mirror. `illum` is the only slot in the format that
            // states the intent rather than a coefficient.
            m.type = MatType::Glossy;
            // Phong exponent -> roughness. The usual approximation r = sqrt(2/(Ns+2)),
            // floored so a mirror-sharp Ns still has a lobe the preview can show.
            m.roughness = std::max(0.02, std::sqrt(2.0 / (Ns + 2.0)));
        } else {
            m.type = MatType::Diffuse;
        }
        out[name] = (int)s.mats.size();
        s.mats.push_back(std::move(m));
        // Reset to defaults for the next block.
        Kd = Vec3{0.8, 0.8, 0.8}; Ks = Vec3{0, 0, 0}; Tf = Vec3{1, 1, 1};
        d = 1.0; Ni = 1.5; Ns = 0.0; haveTf = false; illum = 2;
    };

    const char* p = text.data();
    const char* end = p + text.size();
    auto words = [](const char* a, const char* b, double* v, int n) {
        for (int i = 0; i < n; ++i) v[i] = 0.0;
        objParseDoubles(a, b, v, n);
    };
    while (p < end) {
        const char* ls = p;
        while (p < end && *p != '\n') ++p;
        const char* le = p;
        if (le > ls && le[-1] == '\r') --le;
        if (p < end) ++p;
        ls = objSkipWs(ls, le);
        if (ls >= le || *ls == '#') continue;
        auto is = [&](const char* kw, size_t n) {
            return (size_t)(le - ls) > n && std::memcmp(ls, kw, n) == 0 && objIsWs(ls[n]);
        };
        double v[3];
        if (is("newmtl", 6)) {
            flush();
            const char* q = objSkipWs(ls + 6, le);
            name.assign(q, le);
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
        } else if (is("Kd", 2)) { words(ls + 2, le, v, 3); Kd = Vec3{v[0], v[1], v[2]}; }
        else if (is("Ks", 2))   { words(ls + 2, le, v, 3); Ks = Vec3{v[0], v[1], v[2]}; }
        else if (is("Tf", 2))   { words(ls + 2, le, v, 3); Tf = Vec3{v[0], v[1], v[2]}; haveTf = true; }
        else if (is("Ni", 2))   { words(ls + 2, le, v, 1); Ni = v[0]; }
        else if (is("Ns", 2))   { words(ls + 2, le, v, 1); Ns = v[0]; }
        else if (is("d", 1))    { words(ls + 1, le, v, 1); d  = v[0]; }
        else if (is("Tr", 2))   { words(ls + 2, le, v, 1); d  = 1.0 - v[0]; }
        else if (is("illum", 5)){ words(ls + 5, le, v, 1); illum = (int)v[0]; }
    }
    flush();
    if (countOut) *countOut = (int)out.size();
    if (!out.empty())
        std::printf("loadMtl: %s -> %zu material(s)\n", path, out.size());
    return out;
}

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
        // Per triangle, writing only its own three slots: threads with no coordination.
        (void)ft::parallelFor(nt, 4096, [&](size_t i) {
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
        });
        // The fan gather. Every table it reads (fn, ang, weld, voff, vcorner, triVI) is
        // finished and read-only by now, and iteration i writes exactly one triangle's
        // three normals, so this parallelises with no locking and no change of result --
        // each corner still sums the same incident faces in the same order, so it is
        // bit-identical to the serial version, not merely equivalent.
        //
        // It is worth doing because this is the dominant cost of a warp: an -nd extrude
        // re-derives normals over the whole projected mesh on every slider event, and at
        // 4M triangles this single loop was ~2.0 s of a 2.5 s warp, on one core of twelve.
        (void)ft::parallelFor(nt, 2048, [&](size_t i) {
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
        });
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
                        double creaseAngleDeg = -1.0, bool importMaterials = true) {
    // `mtllib`: load the companion library and let its names resolve `usemtl`. An
    // explicit `matResolver` (the FTSL `use_names` path) WINS -- a scene that declares
    // its own materials is stating an intent the file cannot override -- so the imported
    // table is only consulted for names the caller does not know. The library is resolved
    // beside the OBJ, which is where the format says it lives; a missing or unreadable one
    // is silent, because plenty of OBJs name an .mtl that was never shipped with them.
    std::unordered_map<std::string, int> mtlTable;
    if (importMaterials) {
        std::string libName;
        {
            const char* p = buf.data();
            const char* end = p + buf.size();
            while (p < end) {
                const char* ls = p;
                while (p < end && *p != '\n') ++p;
                const char* le = p;
                if (le > ls && le[-1] == '\r') --le;
                if (p < end) ++p;
                ls = objSkipWs(ls, le);
                if ((size_t)(le - ls) > 6 && std::memcmp(ls, "mtllib", 6) == 0 && objIsWs(ls[6])) {
                    const char* q = objSkipWs(ls + 6, le);
                    libName.assign(q, le);
                    while (!libName.empty() && (libName.back() == ' ' || libName.back() == '\t'))
                        libName.pop_back();
                    break;
                }
                // Only the header carries mtllib in practice, and scanning a 100 MB body
                // for it would cost a whole extra pass: stop at the first face.
                if ((size_t)(le - ls) >= 2 && ls[0] == 'f' && objIsWs(ls[1])) break;
            }
        }
        if (!libName.empty()) {
            std::string dir(path);
            const size_t sl = dir.find_last_of("/\\");
            dir = (sl == std::string::npos) ? std::string() : dir.substr(0, sl + 1);
            const std::string libPath = dir + libName;
            std::string libText;
            if (assetbytes::readFile(libPath.c_str(), libText))
                mtlTable = parseMtl(s, libText, libPath.c_str());
        }
    }
    MtlResolver mtlLookup = [&mtlTable](const std::string& n) -> int {
        auto it = mtlTable.find(n);
        return (it == mtlTable.end()) ? -1 : it->second;
    };
    const bool haveMtl = !mtlTable.empty();

    std::vector<Vec3> verts;
    std::vector<Vec3> objColors;   // per-vertex colour from the extended `v x y z r g b`
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
            // The extended form `v x y z r g b` (MeshLab, most scanner exports) puts a
            // per-vertex colour after the position. Parse six and count how many were
            // really there — a plain `v` leaves the last three at the sentinel.
            double d[6] = {0, 0, 0, -1, -1, -1};
            objParseDoubles(ls + 2, le, d, 6);
            verts.push_back(xf.apply(Vec3{d[0], d[1], d[2]}));
            if (d[3] >= 0.0 && d[4] >= 0.0 && d[5] >= 0.0) {
                // Written 0..1 in practice; a 0..255 file is accepted by scaling when any
                // channel exceeds 1. Unlike PLY there is no declared type to consult.
                const double mx = std::max({d[3], d[4], d[5]});
                const double k = (mx > 1.0) ? (1.0 / 255.0) : 1.0;
                auto lin = [](double c) {
                    c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
                    return (c <= 0.04045) ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
                };
                objColors.resize(verts.size() - 1, Vec3{1, 1, 1});   // pad any uncoloured prefix
                objColors.push_back(Vec3{lin(d[3] * k), lin(d[4] * k), lin(d[5] * k)});
            }
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
        } else if ((matResolver || haveMtl) && le - ls >= 6 &&
                   std::memcmp(ls, "usemtl", 6) == 0) {
            // A material NAME can legally contain spaces, so take the rest of the line
            // rather than the first token — trimmed, since exporters pad it.
            const char* q = objSkipWs(ls + 6, le);
            std::string name(q, le);
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
            int resolved = -1;
            if (!name.empty()) {
                if (matResolver) resolved = (*matResolver)(name);       // FTSL names win
                if (resolved < 0 && haveMtl) resolved = mtlLookup(name);
            }
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
                if (objColors.size() == verts.size()) {
                    t.vcol = (int)(s.vertColors.size() / 3);
                    for (int vi : {fIdx[0], fIdx[k], fIdx[k + 1]}) {
                        const Vec3& c = objColors[(size_t)vi];
                        s.vertColors.push_back((float)c.x);
                        s.vertColors.push_back((float)c.y);
                        s.vertColors.push_back((float)c.z);
                    }
                }
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
                   double creaseAngleDeg = -1.0, bool importMaterials = true) {
    std::string buf;
    if (!assetbytes::readFile(path, buf)) {
        std::fprintf(stderr, "loadObj: %s: %s\n", path,
                     assetbytes::describeOpenFailure(path).c_str());
        return 0;
    }
    return loadObjBytes(s, buf, path, matId, xf, loadUV, matResolver, uvProj, uvAxis,
                        creaseAngleDeg, importMaterials);
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
// .ply — Stanford polygon format (0.183.0)
// ===========================================================================
//
// WHY IT EXISTS NOW AND DID NOT BEFORE. `.ply` and `.stl` were already advertised
// by `-help` and already accepted by the bare-mesh quick-viewer's extension list,
// but no loader ever backed either one: the format dispatch in ftsl.h sent
// everything that was not .gltf/.glb/.fbx/.ftmesh to the OBJ parser, which finds no
// `v `/`f ` lines in a PLY and returns an EMPTY mesh — and the call site discarded
// its return value, so nothing failed. The user-visible symptom was a render of
// nothing at all: `ftrace foo.ply` auto-framed an empty scene and produced a
// uniform grey field, with the only clue a `0 verts, 0 tris` line in the log.
// Both formats are real loaders now, and an empty load is a hard error at the call
// site rather than a silent grey image.
//
// COVERAGE. All three PLY encodings (`ascii`, `binary_little_endian`,
// `binary_big_endian`, the last byte-swapped on read); arbitrary element and
// property layouts, with unknown elements and properties consumed by their declared
// type rather than assumed away — that is what lets a 60-property Gaussian-splat
// PLY parse instead of derailing on the first `f_rest_*` field. Vertices come from
// `x`/`y`/`z`, optional shading normals from `nx`/`ny`/`nz`, optional texture
// coordinates from whichever of the four name pairs the exporter used. Faces come
// from `vertex_indices` (or the singular `vertex_index` some writers emit) and are
// fan-triangulated exactly like OBJ polygons, so a quad mesh loads identically
// through either format. Semantics match loadObj's throughout: same world-space
// transform, same authored-normals-win rule, same `meshFinishTris` finishing pass,
// so swapping a mesh between .obj and .ply cannot change how it shades.
namespace plydetail {

enum class PT { I8, U8, I16, U16, I32, U32, F32, F64, None };

inline PT ptOf(const std::string& t) {
    if (t == "char"   || t == "int8")    return PT::I8;
    if (t == "uchar"  || t == "uint8")   return PT::U8;
    if (t == "short"  || t == "int16")   return PT::I16;
    if (t == "ushort" || t == "uint16")  return PT::U16;
    if (t == "int"    || t == "int32")   return PT::I32;
    if (t == "uint"   || t == "uint32")  return PT::U32;
    if (t == "float"  || t == "float32") return PT::F32;
    if (t == "double" || t == "float64") return PT::F64;
    return PT::None;
}
inline int ptSize(PT t) {
    switch (t) {
        case PT::I8:  case PT::U8:               return 1;
        case PT::I16: case PT::U16:              return 2;
        case PT::I32: case PT::U32: case PT::F32: return 4;
        case PT::F64:                            return 8;
        default:                                 return 0;
    }
}

struct Prop { std::string name; PT type = PT::None; bool isList = false; PT countType = PT::None; };
struct Elem { std::string name; long long count = 0; std::vector<Prop> props; };

// Binary body cursor. Every value is read through `num`, including ones we intend to
// throw away — reading-and-discarding an unknown property costs one byte-swap more
// than seeking past it and removes a whole class of size-arithmetic bug, which is the
// better trade for a loader whose job is to survive layouts we have never seen.
struct Reader {
    const unsigned char* p;
    const unsigned char* end;
    bool swap;
    bool ok = true;
    double num(PT t) {
        const int n = ptSize(t);
        if (n == 0 || (size_t)(end - p) < (size_t)n) { ok = false; return 0.0; }
        unsigned char b[8];
        for (int i = 0; i < n; ++i) b[i] = swap ? p[n - 1 - i] : p[i];
        p += n;
        switch (t) {
            case PT::I8:  { signed char   v; std::memcpy(&v, b, 1); return (double)v; }
            case PT::U8:                                            return (double)b[0];
            case PT::I16: { short         v; std::memcpy(&v, b, 2); return (double)v; }
            case PT::U16: { unsigned short v; std::memcpy(&v, b, 2); return (double)v; }
            case PT::I32: { int           v; std::memcpy(&v, b, 4); return (double)v; }
            case PT::U32: { unsigned      v; std::memcpy(&v, b, 4); return (double)v; }
            case PT::F32: { float         v; std::memcpy(&v, b, 4); return (double)v; }
            case PT::F64: { double        v; std::memcpy(&v, b, 8); return v; }
            default:                                                return 0.0;
        }
    }
};

// ASCII body cursor. A PLY ascii body is nominally one element instance per line,
// but list properties make the per-line token count variable and some writers wrap
// long lists, so this reads a flat whitespace-delimited token stream instead of
// binding to line structure. Newlines are whitespace here (unlike objIsWs, which
// deliberately stops at one).
struct AsciiReader {
    const char* p;
    const char* end;
    bool ok = true;
    double num(PT) {
        while (p < end && (unsigned char)*p <= ' ') ++p;
        if (p >= end) { ok = false; return 0.0; }
        double v = 0.0;
        bool good = true;
        const char* q = objParseDouble(p, end, v, good);
        if (!good) { ok = false; return 0.0; }
        p = q;
        return v;
    }
};

// Header: ASCII lines up to and including `end_header`, whatever the body encoding.
inline bool parseHeader(const std::string& buf, std::vector<Elem>& elems, int& fmt,
                        size_t& bodyOff, std::string& err) {
    fmt = -1;
    size_t pos = 0;
    bool sawMagic = false, sawEnd = false;
    while (pos < buf.size()) {
        const size_t nl = buf.find('\n', pos);
        size_t lend = (nl == std::string::npos) ? buf.size() : nl;
        if (lend > pos && buf[lend - 1] == '\r') --lend;
        const std::string line = buf.substr(pos, lend - pos);
        pos = (nl == std::string::npos) ? buf.size() : nl + 1;
        std::vector<std::string> w;
        for (size_t i = 0; i < line.size();) {
            while (i < line.size() && (unsigned char)line[i] <= ' ') ++i;
            const size_t b = i;
            while (i < line.size() && (unsigned char)line[i] > ' ') ++i;
            if (i > b) w.push_back(line.substr(b, i - b));
        }
        if (w.empty()) continue;
        if (!sawMagic) {
            if (w[0] != "ply") { err = "not a PLY file (no `ply` magic on the first line)"; return false; }
            sawMagic = true;
            continue;
        }
        if (w[0] == "comment" || w[0] == "obj_info") continue;
        if (w[0] == "format") {
            if (w.size() < 2) { err = "malformed `format` line"; return false; }
            if      (w[1] == "ascii")                fmt = 0;
            else if (w[1] == "binary_little_endian") fmt = 1;
            else if (w[1] == "binary_big_endian")    fmt = 2;
            else { err = "unsupported PLY encoding `" + w[1] + "`"; return false; }
        } else if (w[0] == "element") {
            if (w.size() < 3) { err = "malformed `element` line"; return false; }
            Elem e;
            e.name  = w[1];
            e.count = std::strtoll(w[2].c_str(), nullptr, 10);
            if (e.count < 0) { err = "negative element count on `element " + w[1] + "`"; return false; }
            elems.push_back(std::move(e));
        } else if (w[0] == "property") {
            if (elems.empty()) { err = "`property` line before any `element`"; return false; }
            Prop pr;
            if (w.size() >= 5 && w[1] == "list") {
                pr.isList    = true;
                pr.countType = ptOf(w[2]);
                pr.type      = ptOf(w[3]);
                pr.name      = w[4];
                if (pr.countType == PT::None || pr.type == PT::None) {
                    err = "unsupported type on list property `" + pr.name + "`"; return false;
                }
            } else if (w.size() >= 3) {
                pr.type = ptOf(w[1]);
                pr.name = w[2];
                if (pr.type == PT::None) {
                    err = "unsupported property type `" + w[1] + "`"; return false;
                }
            } else { err = "malformed `property` line"; return false; }
            elems.back().props.push_back(std::move(pr));
        } else if (w[0] == "end_header") {
            sawEnd  = true;
            bodyOff = pos;
            break;
        }
    }
    if (!sawMagic) { err = "not a PLY file (empty)"; return false; }
    if (!sawEnd)   { err = "PLY header is truncated (no `end_header`)"; return false; }
    if (fmt < 0)   { err = "PLY header has no `format` line"; return false; }
    return true;
}

// Body: walk every element in declaration order, keeping the vertex and face data
// and consuming everything else. `R` is Reader or AsciiReader.
template <class R>
inline void readBody(R& rd, const std::vector<Elem>& elems, const Affine& xf, bool loadUV,
                     std::vector<Vec3>& verts, std::vector<Vec3>& normals,
                     std::vector<Vec3>& uvs, std::vector<std::array<int, 3>>& faces,
                     std::vector<Vec3>& colors) {
    for (const Elem& e : elems) {
        const bool isVert = (e.name == "vertex");
        const bool isFace = (e.name == "face");
        int ix = -1, iy = -1, iz = -1, inx = -1, iny = -1, inz = -1, iu = -1, iv = -1, iIdx = -1;
        int icr = -1, icg = -1, icb = -1;
        for (size_t i = 0; i < e.props.size(); ++i) {
            const std::string& n = e.props[i].name;
            if (isVert) {
                if      (n == "x")  ix  = (int)i;
                else if (n == "y")  iy  = (int)i;
                else if (n == "z")  iz  = (int)i;
                else if (n == "nx") inx = (int)i;
                else if (n == "ny") iny = (int)i;
                else if (n == "nz") inz = (int)i;
                else if (n == "u" || n == "s" || n == "texture_u" || n == "texture_s") iu = (int)i;
                else if (n == "v" || n == "t" || n == "texture_v" || n == "texture_t") iv = (int)i;
                // Vertex colour. PLY has no material block, so this is the only place a
                // .ply can state a colour at all, and it is how scan / photogrammetry
                // pipelines ship one. Both the plain and the `diffuse_` spellings are in
                // the wild; `alpha` is read only to be skipped.
                else if (n == "red"   || n == "diffuse_red")   icr = (int)i;
                else if (n == "green" || n == "diffuse_green") icg = (int)i;
                else if (n == "blue"  || n == "diffuse_blue")  icb = (int)i;
            } else if (isFace && e.props[i].isList &&
                       (n == "vertex_indices" || n == "vertex_index")) {
                iIdx = (int)i;
            }
        }
        const bool wantVert = isVert && ix >= 0 && iy >= 0 && iz >= 0;
        const bool wantFace = isFace && iIdx >= 0;
        std::vector<double> vals(e.props.size(), 0.0);
        std::vector<int> poly;
        for (long long k = 0; k < e.count; ++k) {
            if (!rd.ok) return;
            poly.clear();
            for (size_t i = 0; i < e.props.size(); ++i) {
                const Prop& pr = e.props[i];
                if (!pr.isList) { vals[i] = rd.num(pr.type); continue; }
                const long long n = (long long)rd.num(pr.countType);
                if (!rd.ok || n < 0) { rd.ok = false; return; }
                if ((int)i == iIdx) {
                    poly.resize((size_t)n);
                    for (long long j = 0; j < n; ++j) poly[(size_t)j] = (int)rd.num(pr.type);
                } else {
                    for (long long j = 0; j < n; ++j) (void)rd.num(pr.type);
                }
                if (!rd.ok) return;
            }
            if (wantVert) {
                verts.push_back(xf.apply(Vec3{vals[ix], vals[iy], vals[iz]}));
                if (inx >= 0 && iny >= 0 && inz >= 0) {
                    // Object->world by the inverse-transpose of the linear part, then
                    // normalize — identical treatment to an OBJ `vn` (finalize()
                    // renormalizes again).
                    const Vec3 wn = xf.applyNormal(Vec3{vals[inx], vals[iny], vals[inz]});
                    const double l = std::sqrt(dot(wn, wn));
                    normals.push_back(l > 1e-18 ? wn * (1.0 / l) : Vec3{0, 0, 0});
                }
                if (loadUV && iu >= 0 && iv >= 0) uvs.push_back(Vec3{vals[iu], vals[iv], 0});
                if (icr >= 0 && icg >= 0 && icb >= 0) {
                    // uchar 0..255 is overwhelmingly the common encoding; a float
                    // property is already 0..1. Decide by the declared TYPE rather than
                    // by sniffing the values, which would misread a legitimately dark
                    // float colour as an 8-bit one.
                    const bool byteScale = (e.props[icr].type == PT::U8 ||
                                            e.props[icr].type == PT::I8);
                    const double k = byteScale ? (1.0 / 255.0) : 1.0;
                    // sRGB -> LINEAR: vertex colours are authored/scanned in display
                    // space, and every other colour in ftrace is linear by the time it
                    // reaches a material. Skipping this is what makes an imported scan
                    // read washed out.
                    auto lin = [](double c) {
                        c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
                        return (c <= 0.04045) ? c / 12.92
                                              : std::pow((c + 0.055) / 1.055, 2.4);
                    };
                    colors.push_back(Vec3{lin(vals[icr] * k), lin(vals[icg] * k),
                                          lin(vals[icb] * k)});
                }
            } else if (wantFace) {
                for (size_t j = 1; j + 1 < poly.size(); ++j)
                    faces.push_back({poly[0], poly[(int)j], poly[(int)j + 1]});
            }
        }
    }
}

}  // namespace plydetail

// Load a PLY from memory. Returns triangles added (0 on failure, with `err` set).
inline int loadPlyBytes(Scene& s, const std::string& buf, const char* path, int matId,
                        const Affine& xf, bool loadUV, std::string& err,
                        UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                        double creaseAngleDeg = -1.0) {
    std::vector<plydetail::Elem> elems;
    int fmt = -1;
    size_t bodyOff = 0;
    if (!plydetail::parseHeader(buf, elems, fmt, bodyOff, err)) return 0;

    std::vector<Vec3> verts, normals, uvs, colors;
    std::vector<std::array<int, 3>> faces;
    bool truncated = false;
    if (fmt == 0) {
        plydetail::AsciiReader rd{buf.data() + bodyOff, buf.data() + buf.size()};
        plydetail::readBody(rd, elems, xf, loadUV, verts, normals, uvs, faces, colors);
        truncated = !rd.ok;
    } else {
        const unsigned short one = 1;
        const bool hostLE = (*(const unsigned char*)&one == 1);
        const bool fileLE = (fmt == 1);
        plydetail::Reader rd{(const unsigned char*)buf.data() + bodyOff,
                             (const unsigned char*)buf.data() + buf.size(),
                             hostLE != fileLE};
        plydetail::readBody(rd, elems, xf, loadUV, verts, normals, uvs, faces, colors);
        truncated = !rd.ok;
    }
    if (verts.empty()) {
        err = std::string("PLY has no vertex data (no `element vertex` with x/y/z)");
        return 0;
    }
    if (faces.empty()) {
        // The common and confusing case: a point cloud or a Gaussian-splat export.
        // Say what the file *is*, because "0 triangles" alone reads like a parse bug.
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "PLY holds %zu vertices but no faces — it is a point cloud, not a "
                      "surface mesh, so there is nothing to render. Meshify it first "
                      "(e.g. Poisson reconstruction) and load the result.", verts.size());
        err = msg;
        return 0;
    }

    const size_t triStart = s.tris.size();
    const bool proceduralUV = (uvProj != UvProjection::None) && !loadUV;
    const bool wantSmooth   = (creaseAngleDeg >= 0.0);
    const bool haveN  = (normals.size() == verts.size());
    const bool haveUV = loadUV && (uvs.size() == verts.size());
    const bool haveC  = (colors.size() == verts.size());
    std::vector<std::array<int, 3>> triVI;
    triVI.reserve(faces.size());
    long long dropped = 0;
    const int nv = (int)verts.size();
    for (const std::array<int, 3>& f : faces) {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 || f[0] >= nv || f[1] >= nv || f[2] >= nv) {
            ++dropped;
            continue;
        }
        Tri t{verts[f[0]], verts[f[1]], verts[f[2]], matId, -1, {}};
        if (haveUV) { t.uv0 = uvs[f[0]];     t.uv1 = uvs[f[1]];     t.uv2 = uvs[f[2]]; }
        if (haveN)  { t.n0  = normals[f[0]]; t.n1  = normals[f[1]]; t.n2  = normals[f[2]]; }
        if (haveC) {
            t.vcol = (int)(s.vertColors.size() / 3);
            for (int c = 0; c < 3; ++c) {
                const Vec3& col = colors[(size_t)f[c]];
                s.vertColors.push_back((float)col.x);
                s.vertColors.push_back((float)col.y);
                s.vertColors.push_back((float)col.z);
            }
        }
        s.tris.push_back(t);
        triVI.push_back(f);
    }
    const int added = (int)triVI.size();
    const bool didSmooth = meshFinishTris(s, triStart, verts, triVI, proceduralUV, uvProj,
                                          uvAxis, wantSmooth, haveN, creaseAngleDeg);
    std::printf("loadPly: %s -> %d verts, %d tris (mat %d) [%s]%s%s%s%s\n",
                path, nv, added, matId,
                fmt == 0 ? "ascii" : (fmt == 1 ? "binary LE" : "binary BE"),
                haveN ? " [normals]" : "", haveUV ? " [uv]" : "",
                proceduralUV ? " [procedural UVs]" : "",
                didSmooth ? " [crease-smoothed]" : "");
    if (dropped)
        std::fprintf(stderr, "loadPly: %s -> dropped %lld face(s) with out-of-range indices\n",
                     path, dropped);
    if (truncated)
        std::fprintf(stderr, "loadPly: %s -> body ended early; the file is truncated and "
                             "only %d triangle(s) were recovered\n", path, added);
    return added;
}

// Read a `.ply` from disk and parse it.
inline int loadPly(Scene& s, const char* path, int matId, const Affine& xf,
                   bool loadUV, std::string& err,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                   double creaseAngleDeg = -1.0) {
    std::string buf;
    if (!assetbytes::readFile(path, buf)) {
        err = std::string(path) + ": " + assetbytes::describeOpenFailure(path);
        return 0;
    }
    return loadPlyBytes(s, buf, path, matId, xf, loadUV, err, uvProj, uvAxis, creaseAngleDeg);
}

// ===========================================================================
// .stl — stereolithography, binary and ascii (0.183.0)
// ===========================================================================
//
// Same story as PLY above: advertised, accepted, never implemented. STL carries no
// indices, no UVs and no shared vertices — every triangle repeats its three corners
// — so the per-facet normal the format stores is exactly the geometric normal
// `Tri::finalize` computes anyway. It is therefore read and discarded rather than
// stored: keeping it would pin the mesh to flat shading, whereas leaving the shading
// normals zero lets `smooth` work through the ordinary crease pass (which welds by
// position first, and so recovers the vertex sharing the format threw away).
inline int loadStlBytes(Scene& s, const std::string& buf, const char* path, int matId,
                        const Affine& xf, std::string& err,
                        UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                        double creaseAngleDeg = -1.0) {
    std::vector<Vec3> verts;   // 3 per triangle, in order
    bool binary = false;
    // An ascii STL starts with "solid", but so do plenty of binary ones (the 80-byte
    // header is arbitrary text). The reliable discriminator is the size the binary
    // layout implies: 80-byte header + u32 count + 50 bytes per facet.
    if (buf.size() >= 84) {
        unsigned n = 0;
        std::memcpy(&n, buf.data() + 80, 4);
        if ((size_t)84 + (size_t)n * 50 == buf.size()) binary = true;
    }
    if (binary) {
        unsigned n = 0;
        std::memcpy(&n, buf.data() + 80, 4);
        verts.reserve((size_t)n * 3);
        const char* p = buf.data() + 84;
        for (unsigned i = 0; i < n; ++i, p += 50) {
            float f[12];
            std::memcpy(f, p, 48);           // normal (discarded), then 3 vertices
            for (int k = 0; k < 3; ++k)
                verts.push_back(xf.apply(Vec3{f[3 + k * 3], f[4 + k * 3], f[5 + k * 3]}));
        }
    } else {
        // ASCII: pull the three floats after each `vertex` keyword. Facet/loop
        // structure carries no information a triangle list needs.
        const char* p = buf.data();
        const char* const e = p + buf.size();
        while (p < e) {
            const char* v = (const char*)std::memchr(p, 'v', (size_t)(e - p));
            if (!v) break;
            if ((size_t)(e - v) >= 6 && std::memcmp(v, "vertex", 6) == 0) {
                double d[3] = {0, 0, 0};
                objParseDoubles(v + 6, e, d, 3);
                verts.push_back(xf.apply(Vec3{d[0], d[1], d[2]}));
                p = v + 6;
            } else {
                p = v + 1;
            }
        }
        if (verts.empty()) {
            err = "STL has no `vertex` records (not an ascii STL, and its size does not "
                  "match the binary layout either)";
            return 0;
        }
    }
    if (verts.size() < 3) { err = "STL has no complete triangles"; return 0; }
    const size_t nTri = verts.size() / 3;
    const size_t triStart = s.tris.size();
    const bool proceduralUV = (uvProj != UvProjection::None);
    const bool wantSmooth   = (creaseAngleDeg >= 0.0);
    std::vector<std::array<int, 3>> triVI;
    triVI.reserve(nTri);
    for (size_t i = 0; i < nTri; ++i) {
        s.tris.push_back(Tri{verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2], matId, -1, {}});
        triVI.push_back({(int)(i * 3), (int)(i * 3 + 1), (int)(i * 3 + 2)});
    }
    const bool didSmooth = meshFinishTris(s, triStart, verts, triVI, proceduralUV, uvProj,
                                          uvAxis, wantSmooth, false, creaseAngleDeg);
    std::printf("loadStl: %s -> %zu verts, %zu tris (mat %d) [%s]%s%s\n",
                path, verts.size(), nTri, matId, binary ? "binary" : "ascii",
                proceduralUV ? " [procedural UVs]" : "",
                didSmooth ? " [crease-smoothed]" : "");
    return (int)nTri;
}

// Read a `.stl` from disk and parse it.
inline int loadStl(Scene& s, const char* path, int matId, const Affine& xf, std::string& err,
                   UvProjection uvProj = UvProjection::None, int uvAxis = 1,
                   double creaseAngleDeg = -1.0) {
    std::string buf;
    if (!assetbytes::readFile(path, buf)) {
        err = std::string(path) + ": " + assetbytes::describeOpenFailure(path);
        return 0;
    }
    return loadStlBytes(s, buf, path, matId, xf, err, uvProj, uvAxis, creaseAngleDeg);
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
