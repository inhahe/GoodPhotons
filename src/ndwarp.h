// ndwarp.h — lift a model into N dimensions, rotate it there, project it back to 3-D.
//
// WHAT THIS IS. Load any mesh ftrace can read, treat its vertices as points of R^N
// (the extra coordinates supplied by the per-dimension FILL below), rotate with the
// n(n-1)/2 independent plane rotations an n-D space has, and project orthographically
// back to three dimensions. The result is a real triangle mesh that replaces the
// scene's own — so it previews in the rasterizer, path-traces in every render mode,
// and can be written back out as a model in its own right.
//
// THE ONE FACT THAT DECIDES THE DESIGN. A 3-D model lifted with ZEROS in dims 4..n
// and projected back orthographically is EXACTLY a 3x3 matrix multiply, no matter how
// many dimensions or sliders are involved:
//
//     P    = (x, y, z, 0, ..., 0)                 lift
//     Q    = R P                                  only R's first THREE columns are read
//     p'   = (Q0, Q1, Q2) = M p,  M = R[0:3,0:3]  orthographic projection
//
// M is the top-left 3x3 block of an orthogonal matrix, so its singular values are all
// <= 1: the model can only ever be rotated and SQUASHED, never stretched, and every
// vertex takes the same M. Rotating by t in the x-w plane is literally "scale x by
// cos t" — at 90 degrees the model is a flat sheet, at 180 it is its own MIRROR IMAGE
// (det M < 0, which is the one genuinely 4-D trick on offer, being unreachable by any
// 3-D rotation). loom's `mathnd.py` states the same result for fields; this is its
// mesh counterpart.
//
// That is why FILL exists. The projection is only affine because the extra coordinates
// are constant. Give them CONTENT and it stops being affine:
//
//   * `Fill::Zero`    — the classic lift. Affine, per the above. Exact and cheap: the
//                       linear fast path below skips the topology machinery entirely.
//   * `Fill::Emboss`  — x_k = amp * f(vertex) for a per-vertex scalar f (curvature,
//                       radius, height, noise, u, v). Now x_k VARIES over the surface,
//                       so p' = M p + sum_k col_k f_k(p) is genuinely non-linear in p
//                       and the projection reveals shape that was not in the original.
//                       Topology is untouched — every vertex and every edge connection
//                       survives exactly, which is the whole point.
//   * `Fill::Extrude` — sweep the mesh into a real prism along that axis. The boundary
//                       of a 4-D prism is a 3-manifold, which no triangle rasterizer
//                       can draw, so what is built is its 2-SKELETON: the original
//                       triangles at one end, their copy at the other, and every mesh
//                       EDGE swept into a quad. That is exactly how a tesseract is
//                       drawn as its square faces, and it is a genuine N-D solid whose
//                       3-D projection changes qualitatively as it turns.
//
// The three fills are INDEPENDENT and per-dimension: dim 4 can extrude while dim 5
// embosses curvature and dim 6 stays zero. They compose in the obvious order (lift,
// then each extrusion in turn).
//
// COSTS. Extrusion multiplies the 2-skeleton: V -> 2V, E -> 2E + V, F -> 2F + E, and
// each swept edge is a quad (two triangles). For a closed mesh E ~ 1.5 F, so one
// extrusion is about 5x the triangles and a second about 5x again — a budget check
// lives in `projectedTriCount` so a 500k-triangle dragon cannot silently become 12M.
#pragma once
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <unordered_map>
#include "linalg.h"
#include "geometry.h"
#include "scene.h"
#include "mesh.h"            // meshFinishTris: the loaders' own crease-smoothing pass
#include "pov_functions.h"   // povSolidNoise: the renderer's own solid noise

namespace ndwarp {

// ---------------------------------------------------------------------------
// Dimensions, planes, names
// ---------------------------------------------------------------------------

// Axis letters. x/y/z are the real ones; w, v, u, t, s are the traditional
// continuation, and past that we run out of letters and number them (a8, a9, ...).
inline std::string axisName(int i) {
    static const char* kNames[] = {"x", "y", "z", "w", "v", "u", "t", "s"};
    if (i >= 0 && i < 8) return kNames[i];
    return "a" + std::to_string(i);
}

// An n-D space has one independent rotation per PLANE, not per axis — n(n-1)/2 of
// them. (Three dimensions is the coincidence that makes "rotate about an axis" work:
// 3*2/2 == 3 == the number of axes. It stops being true at n = 4.)
inline int planeCount(int n) { return (n < 2) ? 0 : n * (n - 1) / 2; }

// Plane k in the canonical lexicographic order: (0,1), (0,2), ... (0,n-1), (1,2), ...
inline void planeAxes(int n, int k, int& ai, int& aj) {
    ai = aj = 0;
    int idx = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (idx == k) { ai = i; aj = j; return; }
            ++idx;
        }
}

inline int planeIndex(int n, int ai, int aj) {
    if (ai > aj) std::swap(ai, aj);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (i == ai && j == aj) return idx;
            ++idx;
        }
    return -1;
}

inline std::string planeLabel(int n, int k) {
    int i, j; planeAxes(n, k, i, j);
    return axisName(i) + axisName(j);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class Fill : int { Zero = 0, Emboss, Extrude };

// Per-vertex scalar fields available to `Fill::Emboss`. Every one is CENTERED to zero
// mean and normalized to unit peak before `amp` scales it, so the embossed model stays
// put in the extra dimension instead of drifting off-origin (a constant offset in x_k
// would translate the projection under rotation, which reads as the model sliding away
// rather than deforming).
enum class Emb : int { Curvature = 0, Radius, Height, Noise, U, V };

inline const char* embName(Emb e) {
    switch (e) {
        case Emb::Curvature: return "curvature";
        case Emb::Radius:    return "radius";
        case Emb::Height:    return "height";
        case Emb::Noise:     return "noise";
        case Emb::U:         return "u";
        case Emb::V:         return "v";
    }
    return "?";
}

struct DimSpec {
    Fill   fill = Fill::Zero;
    Emb    src  = Emb::Curvature;
    // Emboss amplitude / extrude depth, in units of the MODEL RADIUS — so the same
    // number means the same visual amount whether the model is a 2 cm ring or a 40 m
    // building, and a scene's units never have to be thought about.
    double amp  = 0.25;
    double freq = 4.0;      // Emb::Noise only: cycles across the model's diameter
};

struct Config {
    int n = 3;                       // total dimensions; 3 = the warp is a no-op
    std::vector<DimSpec> extra;      // n-3 entries, for dims 3 .. n-1
    std::vector<double>  angle;      // planeCount(n) angles, RADIANS
    double creaseDeg = 30.0;         // crease angle used when normals are re-derived
    std::string object;              // "" = every mesh group; else only this one

    void resize(int dims) {
        n = std::max(3, dims);
        extra.resize((size_t)(n - 3));
        angle.resize((size_t)planeCount(n), 0.0);
    }
    bool anyExtrude() const {
        for (const DimSpec& d : extra) if (d.fill == Fill::Extrude && d.amp != 0.0) return true;
        return false;
    }
    bool anyEmboss() const {
        for (const DimSpec& d : extra) if (d.fill == Fill::Emboss && d.amp != 0.0) return true;
        return false;
    }
    // True when the whole warp collapses to one 3x3 matrix (see the header note).
    bool isLinear() const { return !anyExtrude() && !anyEmboss(); }
};

// ---------------------------------------------------------------------------
// The rotation matrix: a product of Givens (plane) rotations
// ---------------------------------------------------------------------------
//
// R = G_{m-1} ... G_1 G_0, i.e. plane 0 is applied FIRST and each later plane composes
// on the outside. Rotations in n >= 4 dimensions do not commute, so this order is part
// of the contract: the same slider values always produce the same matrix.
inline std::vector<double> rotationMatrix(const Config& cfg) {
    const int n = std::max(3, cfg.n);
    std::vector<double> R((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) R[(size_t)i * n + i] = 1.0;
    const int m = planeCount(n);
    for (int k = 0; k < m && k < (int)cfg.angle.size(); ++k) {
        const double a = cfg.angle[(size_t)k];
        if (a == 0.0) continue;
        int p, q; planeAxes(n, k, p, q);
        const double c = std::cos(a), s = std::sin(a);
        // Left-multiply by the Givens rotation: only rows p and q change.
        for (int col = 0; col < n; ++col) {
            const double rp = R[(size_t)p * n + col], rq = R[(size_t)q * n + col];
            R[(size_t)p * n + col] = c * rp - s * rq;
            R[(size_t)q * n + col] = s * rp + c * rq;
        }
    }
    return R;
}

// The 3x3 block that an orthographic projection actually reads.
struct Mat3 {
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Vec3 operator*(const Vec3& p) const {
        return Vec3{m[0] * p.x + m[1] * p.y + m[2] * p.z,
                    m[3] * p.x + m[4] * p.y + m[5] * p.z,
                    m[6] * p.x + m[7] * p.y + m[8] * p.z};
    }
    double det() const {
        return m[0] * (m[4] * m[8] - m[5] * m[7])
             - m[1] * (m[3] * m[8] - m[5] * m[6])
             + m[2] * (m[3] * m[7] - m[4] * m[6]);
    }
};

inline Mat3 topLeft3(const std::vector<double>& R, int n) {
    Mat3 M;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M.m[i * 3 + j] = R[(size_t)i * n + j];
    return M;
}

// Inverse transpose, for carrying normals through a linear map. Returns false when the
// map is singular (a 90-degree plane rotation collapses the model to a sheet, and there
// the surface genuinely has no well-defined normal) — the caller then leaves the shading
// normals zero so Tri::finalize() falls them back to the geometric normal.
inline bool inverseTranspose(const Mat3& M, Mat3& out) {
    const double* m = M.m;
    const double c00 =  (m[4] * m[8] - m[5] * m[7]);
    const double c01 = -(m[3] * m[8] - m[5] * m[6]);
    const double c02 =  (m[3] * m[7] - m[4] * m[6]);
    const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if (!(std::fabs(det) > 1e-12)) return false;
    const double inv = 1.0 / det;
    // The inverse transpose is the cofactor matrix over the determinant.
    out.m[0] = c00 * inv;
    out.m[1] = c01 * inv;
    out.m[2] = c02 * inv;
    out.m[3] = -(m[1] * m[8] - m[2] * m[7]) * inv;
    out.m[4] =  (m[0] * m[8] - m[2] * m[6]) * inv;
    out.m[5] = -(m[0] * m[7] - m[1] * m[6]) * inv;
    out.m[6] =  (m[1] * m[5] - m[2] * m[4]) * inv;
    out.m[7] = -(m[0] * m[5] - m[2] * m[3]) * inv;
    out.m[8] =  (m[0] * m[4] - m[1] * m[3]) * inv;
    return true;
}

// ---------------------------------------------------------------------------
// The captured base model
// ---------------------------------------------------------------------------

// One authored mesh object taken as warp input, as a range into Model::base.
struct Source {
    std::string name;
    size_t start = 0, count = 0;   // range into Model::base
    size_t groupIdx = 0;           // index into Scene::meshGroups (for the range rewrite)
};

// A welded, indexed view of the base triangles. Welding is by QUANTIZED POSITION at
// 1e-6 of the mesh diagonal — the same rule (and the same epsilon) the mesh loaders'
// crease smoothing uses, so an exporter that split one position into several file
// vertices still yields one topological vertex here.
//
// Attributes (UV, material) are NOT welded: they stay per-corner, because a UV seam is
// precisely a place where one position carries two different UVs. Original faces keep
// their own corner attributes exactly; only swept side walls fall back to a
// representative corner per vertex (`vref`), where there is no better answer.
struct Topo {
    int nv = 0;
    std::vector<Vec3>     pos;      // nv welded positions
    std::vector<uint32_t> vtri;     // 3 * ntris: welded vertex id per corner
    std::vector<uint32_t> vref;     // nv: a representative corner id (tri*3 + corner)
    std::vector<uint32_t> edge;     // 2 * nedges: welded vertex ids, lower first
    std::vector<Vec3>     vnrm;     // nv: area-weighted vertex normals of the BASE mesh
};

struct Model {
    bool ok = false;
    std::vector<Tri>    base;       // pristine copy of every warped triangle
    std::vector<Source> groups;     // the authored objects `base` came from
    std::vector<Tri>    keep;       // pristine copy of every UNwarped triangle, in order
    std::vector<size_t> keepOrig;   // their original Scene::tris indices
    Topo   topo;
    Vec3   center{0, 0, 0};
    double radius = 1.0;
    // Per-vertex emboss signals, centered and unit-peak normalized. Built lazily on
    // first use of each source and cached, because curvature in particular costs a full
    // adjacency walk and a slider drag would otherwise pay for it sixty times a second.
    mutable std::vector<std::vector<double>> embCache;
    mutable std::vector<double>              embCacheFreq;
    mutable std::vector<char>                embCacheValid;
};

// ---------------------------------------------------------------------------
// Welding
// ---------------------------------------------------------------------------

namespace detail {

struct QKey {
    long long x, y, z;
    bool operator==(const QKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct QHash {
    size_t operator()(const QKey& k) const {
        unsigned long long h = 1469598103934665603ull;
        auto mix = [&h](long long v) { h ^= (unsigned long long)v; h *= 1099511628211ull; h ^= h >> 29; };
        mix(k.x); mix(k.y); mix(k.z);
        return (size_t)h;
    }
};

inline void weld(const std::vector<Tri>& tris, Topo& t) {
    t = Topo{};
    if (tris.empty()) return;
    Vec3 lo = tris[0].v0, hi = tris[0].v0;
    auto grow = [&](const Vec3& v) {
        lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    };
    for (const Tri& tr : tris) { grow(tr.v0); grow(tr.v1); grow(tr.v2); }
    const Vec3 ext = hi - lo;
    const double diag = std::sqrt(dot(ext, ext));
    const double eps  = (diag > 0.0 ? diag : 1.0) * 1e-6;
    const double invE = 1.0 / eps;

    std::unordered_map<QKey, int, QHash> map;
    map.reserve(tris.size() * 2);
    t.vtri.resize(tris.size() * 3);
    auto id = [&](const Vec3& v, uint32_t cornerRef) -> uint32_t {
        QKey k{(long long)std::llround(v.x * invE),
               (long long)std::llround(v.y * invE),
               (long long)std::llround(v.z * invE)};
        auto it = map.find(k);
        if (it != map.end()) return (uint32_t)it->second;
        const int nid = t.nv++;
        map.emplace(k, nid);
        t.pos.push_back(v);
        t.vref.push_back(cornerRef);
        return (uint32_t)nid;
    };
    for (size_t i = 0; i < tris.size(); ++i) {
        t.vtri[i * 3 + 0] = id(tris[i].v0, (uint32_t)(i * 3 + 0));
        t.vtri[i * 3 + 1] = id(tris[i].v1, (uint32_t)(i * 3 + 1));
        t.vtri[i * 3 + 2] = id(tris[i].v2, (uint32_t)(i * 3 + 2));
    }

    // Unique undirected edges. Hashed on the ordered pair packed into 64 bits, which
    // caps a warped model at 4 G vertices — far past anything a rasterizer will draw.
    std::unordered_map<unsigned long long, uint32_t> emap;
    emap.reserve(tris.size() * 3);
    auto addEdge = [&](uint32_t a, uint32_t b) {
        if (a == b) return;                       // a degenerate corner pair is not an edge
        if (a > b) std::swap(a, b);
        const unsigned long long key = ((unsigned long long)a << 32) | (unsigned long long)b;
        auto it = emap.find(key);
        if (it != emap.end()) return;
        emap.emplace(key, (uint32_t)(t.edge.size() / 2));
        t.edge.push_back(a);
        t.edge.push_back(b);
    };
    for (size_t i = 0; i < tris.size(); ++i) {
        const uint32_t a = t.vtri[i * 3 + 0], b = t.vtri[i * 3 + 1], c = t.vtri[i * 3 + 2];
        addEdge(a, b); addEdge(b, c); addEdge(c, a);
    }

    // Area-weighted vertex normals of the base mesh — needed to sign the discrete
    // curvature, and a reasonable seed anywhere else a vertex normal is wanted.
    t.vnrm.assign((size_t)t.nv, Vec3{0, 0, 0});
    for (size_t i = 0; i < tris.size(); ++i) {
        const Vec3 n = cross(tris[i].v1 - tris[i].v0, tris[i].v2 - tris[i].v0);  // length = 2*area
        for (int c = 0; c < 3; ++c) t.vnrm[t.vtri[i * 3 + c]] += n;
    }
    for (Vec3& n : t.vnrm) {
        const double l = std::sqrt(dot(n, n));
        n = (l > 1e-18) ? n * (1.0 / l) : Vec3{0, 0, 0};
    }
}

// Center to zero mean, then scale to unit peak. A signal that is entirely constant
// (a perfectly flat mesh's curvature, say) becomes exactly zero, which is the honest
// answer — there is nothing in it to emboss with.
inline void normalizeSignal(std::vector<double>& s) {
    if (s.empty()) return;
    double mean = 0.0;
    for (double v : s) mean += v;
    mean /= (double)s.size();
    double peak = 0.0;
    for (double& v : s) { v -= mean; peak = std::max(peak, std::fabs(v)); }
    if (peak > 1e-15) { const double inv = 1.0 / peak; for (double& v : s) v *= inv; }
    else              { for (double& v : s) v = 0.0; }
}

inline void appendDouble(std::string& b, double v) {
    char tmp[40];
    const int k = std::snprintf(tmp, sizeof tmp, "%.6g", v);
    if (k > 0) b.append(tmp, (size_t)k);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Emboss signals
// ---------------------------------------------------------------------------

// Build (and cache) the per-vertex scalar field for one emboss source.
inline const std::vector<double>& embossSignal(const Model& m, Emb src, double freq) {
    const size_t idx = (size_t)src;
    if (m.embCache.size() <= idx) {
        m.embCache.resize(idx + 1);
        m.embCacheFreq.resize(idx + 1, -1.0);
        m.embCacheValid.resize(idx + 1, 0);
    }
    const bool freqMatters = (src == Emb::Noise);
    if (m.embCacheValid[idx] && (!freqMatters || m.embCacheFreq[idx] == freq))
        return m.embCache[idx];

    const Topo& t = m.topo;
    std::vector<double> s((size_t)t.nv, 0.0);
    switch (src) {
        case Emb::Curvature: {
            // Discrete mean curvature by the umbrella (uniform Laplacian) operator:
            // the vertex's offset from the centroid of its neighbours, signed against
            // the vertex normal so a bulge reads positive and a pit negative. Cheap,
            // and unlike Tri::curvature it works on a FLAT-SHADED mesh too (that field
            // is derived from the shading-normal field, which is exactly zero when a
            // mesh carries no `vn`).
            std::vector<Vec3>   acc((size_t)t.nv, Vec3{0, 0, 0});
            std::vector<double> deg((size_t)t.nv, 0.0);
            for (size_t e = 0; e + 1 < t.edge.size(); e += 2) {
                const uint32_t a = t.edge[e], b = t.edge[e + 1];
                acc[a] += t.pos[b] - t.pos[a]; deg[a] += 1.0;
                acc[b] += t.pos[a] - t.pos[b]; deg[b] += 1.0;
            }
            const double invR = (m.radius > 0.0) ? 1.0 / m.radius : 1.0;
            for (int v = 0; v < t.nv; ++v) {
                if (deg[(size_t)v] <= 0.0) continue;
                const Vec3 L = acc[(size_t)v] * (1.0 / deg[(size_t)v]);
                s[(size_t)v] = dot(L, t.vnrm[(size_t)v]) * invR;
            }
            break;
        }
        case Emb::Radius: {
            for (int v = 0; v < t.nv; ++v) s[(size_t)v] = length(t.pos[(size_t)v] - m.center);
            break;
        }
        case Emb::Height: {
            for (int v = 0; v < t.nv; ++v) s[(size_t)v] = t.pos[(size_t)v].y;
            break;
        }
        case Emb::Noise: {
            // The renderer's own solid noise, sampled in model-radius units so `freq`
            // reads as "cycles across the model" whatever the scene's scale is.
            const double k = (m.radius > 0.0) ? (freq * 0.5 / m.radius) : freq;
            for (int v = 0; v < t.nv; ++v) {
                const Vec3 p = (t.pos[(size_t)v] - m.center) * k;
                s[(size_t)v] = povSolidNoise(p.x, p.y, p.z);
            }
            break;
        }
        case Emb::U:
        case Emb::V: {
            // Read the representative corner's texture coordinate. A mesh with no UVs
            // carries Tri's defaults ((0,0),(1,0),(1,1)), which is a real if arbitrary
            // signal — the normalization below still centers it.
            for (int v = 0; v < t.nv; ++v) {
                const uint32_t ref = t.vref[(size_t)v];
                const Tri& tr = m.base[ref / 3];
                const Vec3& uv = (ref % 3 == 0) ? tr.uv0 : (ref % 3 == 1) ? tr.uv1 : tr.uv2;
                s[(size_t)v] = (src == Emb::U) ? uv.x : uv.y;
            }
            break;
        }
    }
    detail::normalizeSignal(s);
    m.embCache[idx]      = std::move(s);
    m.embCacheFreq[idx]  = freq;
    m.embCacheValid[idx] = 1;
    return m.embCache[idx];
}

// ---------------------------------------------------------------------------
// Capture: take the scene's authored meshes as the warp source
// ---------------------------------------------------------------------------

// Which triangles are eligible. Native primitives (a floor quad, a box, a sphere's
// tessellation) are deliberately NOT warped: `-nd` is a tool for looking at a MODEL in
// N dimensions, and a scene's ground plane folding up with it would be noise. Emissive
// meshes are skipped too — Scene::addMeshLight COPIES their triangles into the emitter
// at load time, so warping the surface without rebuilding the light would leave the
// two disagreeing about where the light is.
inline bool capture(const Scene& s, const std::string& object, Model& m, std::string& note) {
    m = Model{};
    note.clear();
    std::vector<char> warped(s.tris.size(), 0);
    size_t skippedEmissive = 0, skippedBlas = 0;

    // A mesh is a LIGHT when some emitter names its material: Scene::addMeshLight
    // records the material it was registered with, and Material::emit is a
    // std::function with no integral to interrogate, so the emitter table is the
    // authoritative (and only) place that fact is written down.
    auto materialIsEmissive = [&](int matId) {
        if (matId < 0) return false;
        for (const Emitter& e : s.emitters) if (e.matId == matId) return true;
        return false;
    };

    for (size_t gi = 0; gi < s.meshGroups.size(); ++gi) {
        const MeshGroup& g = s.meshGroups[gi];
        if (g.blasId >= 0) { ++skippedBlas; continue; }          // shared instanced asset
        if (g.shapeOnly || g.triCount == 0) continue;
        if (!object.empty() && g.name != object) continue;
        if (materialIsEmissive(g.matId)) { ++skippedEmissive; continue; }
        Source src;
        src.name     = g.name;
        src.groupIdx = gi;
        src.start    = m.base.size();
        const size_t end = std::min(g.triStart + g.triCount, s.tris.size());
        for (size_t i = g.triStart; i < end; ++i) {
            if (s.tris[i].sensorId >= 0) continue;               // a sensor must not move
            m.base.push_back(s.tris[i]);
            warped[i] = 1;
        }
        src.count = m.base.size() - src.start;
        if (src.count) m.groups.push_back(std::move(src));
        else           m.base.resize(src.start);
    }

    if (m.base.empty()) {
        note = object.empty()
             ? "no authored mesh geometry to warp (native primitives and instanced "
               "assets are left alone)"
             : "no mesh object named '" + object + "'";
        return false;
    }

    for (size_t i = 0; i < s.tris.size(); ++i)
        if (!warped[i]) { m.keep.push_back(s.tris[i]); m.keepOrig.push_back(i); }

    // Bounds of the warped geometry only: the rotation is about the MODEL's centre, so
    // a scene's distant floor must not drag it off-centre.
    Vec3 lo = m.base[0].v0, hi = m.base[0].v0;
    auto grow = [&](const Vec3& v) {
        lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    };
    for (const Tri& t : m.base) { grow(t.v0); grow(t.v1); grow(t.v2); }
    m.center = (lo + hi) * 0.5;
    m.radius = std::max(1e-9, length(hi - lo) * 0.5);

    detail::weld(m.base, m.topo);
    m.ok = true;

    if (skippedEmissive || skippedBlas) {
        note = "skipped ";
        if (skippedEmissive) note += std::to_string(skippedEmissive) + " emissive mesh(es)";
        if (skippedEmissive && skippedBlas) note += " and ";
        if (skippedBlas)     note += std::to_string(skippedBlas) + " instanced asset(s)";
    }
    return true;
}

// ---------------------------------------------------------------------------
// The N-D complex
// ---------------------------------------------------------------------------

// Vertices in R^n plus the 2-skeleton over them. `ref` carries, per triangle corner,
// the id of the BASE corner (tri*3 + corner) whose material and UV that corner
// inherits — which is what lets a swept prism keep the skin of the model it came from.
struct Complex {
    int n = 4;
    int nv = 0;
    std::vector<double>   pos;    // nv * n
    std::vector<uint32_t> tri;    // 3 * nt vertex ids
    std::vector<uint32_t> ref;    // 3 * nt base-corner ids
    std::vector<uint32_t> edge;   // 2 * ne vertex ids (carried so a further sweep can use them)
    std::vector<uint32_t> vref;   // nv representative base-corner ids
};

// Triangle count the current config would produce, without building anything — so a
// budget can be enforced before the allocation rather than after it.
inline size_t projectedTriCount(const Model& m, const Config& cfg) {
    size_t V = (size_t)m.topo.nv;
    size_t E = m.topo.edge.size() / 2;
    size_t F = m.base.size();
    for (const DimSpec& d : cfg.extra) {
        if (d.fill != Fill::Extrude || d.amp == 0.0) continue;
        const size_t F2 = 2 * F + 2 * E;   // both copies, plus two triangles per swept edge
        const size_t E2 = 2 * E + V;
        const size_t V2 = 2 * V;
        F = F2; E = E2; V = V2;
    }
    return F;
}

// Lift the welded base mesh into R^n, filling each extra dimension per its DimSpec,
// then sweep along every extruded axis in turn.
inline Complex buildComplex(const Model& m, const Config& cfg) {
    const int n = std::max(3, cfg.n);
    const Topo& t = m.topo;
    Complex c;
    c.n  = n;
    c.nv = t.nv;
    c.pos.assign((size_t)t.nv * n, 0.0);

    // Dimensions 0..2: the model itself, about its own centre.
    for (int v = 0; v < t.nv; ++v) {
        const Vec3 p = t.pos[(size_t)v] - m.center;
        c.pos[(size_t)v * n + 0] = p.x;
        c.pos[(size_t)v * n + 1] = p.y;
        c.pos[(size_t)v * n + 2] = p.z;
    }
    // Dimensions 3..n-1: zero, embossed, or (below) swept.
    for (int k = 3; k < n; ++k) {
        const DimSpec& d = cfg.extra[(size_t)(k - 3)];
        if (d.fill != Fill::Emboss || d.amp == 0.0) continue;
        const std::vector<double>& sig = embossSignal(m, d.src, d.freq);
        const double a = d.amp * m.radius;
        for (int v = 0; v < t.nv; ++v) c.pos[(size_t)v * n + k] = sig[(size_t)v] * a;
    }

    c.tri = t.vtri;
    c.ref.resize(t.vtri.size());
    for (size_t i = 0; i < c.ref.size(); ++i) c.ref[i] = (uint32_t)i;   // corner i of triangle i/3
    c.edge = t.edge;
    c.vref = t.vref;

    // Sweep. Each pass turns the current 2-skeleton into that of the prism over it:
    // both end copies, plus a quad for every edge. The edge list has to be swept too
    // (2E + V) or a second extrusion would have nothing to raise its walls from.
    for (int k = 3; k < n; ++k) {
        const DimSpec& d = cfg.extra[(size_t)(k - 3)];
        if (d.fill != Fill::Extrude) continue;
        const double h = d.amp * m.radius;
        if (h == 0.0) continue;

        const int    nv0 = c.nv;
        const size_t nt0 = c.tri.size() / 3;
        const size_t ne0 = c.edge.size() / 2;

        // Vertices: the far copy, offset along axis k. Centring the sweep on the
        // original (-h/2 .. +h/2) keeps the solid's middle where the model was, so
        // turning on an extrusion does not shove the object sideways.
        std::vector<double> pos2((size_t)nv0 * 2 * n);
        std::memcpy(pos2.data(), c.pos.data(), c.pos.size() * sizeof(double));
        for (int v = 0; v < nv0; ++v) {
            std::memcpy(&pos2[(size_t)(nv0 + v) * n], &c.pos[(size_t)v * n],
                        (size_t)n * sizeof(double));
            pos2[(size_t)v * n + k]         -= h * 0.5;
            pos2[(size_t)(nv0 + v) * n + k] += h * 0.5;
        }

        std::vector<uint32_t> tri2, ref2;
        tri2.reserve((nt0 * 2 + ne0 * 2) * 3);
        ref2.reserve((nt0 * 2 + ne0 * 2) * 3);
        // Near copy, unchanged.
        tri2.insert(tri2.end(), c.tri.begin(), c.tri.end());
        ref2.insert(ref2.end(), c.ref.begin(), c.ref.end());
        // Far copy, wound the other way so the prism's two lids face opposite ways.
        for (size_t i = 0; i < nt0; ++i) {
            tri2.push_back(c.tri[i * 3 + 0] + (uint32_t)nv0);
            tri2.push_back(c.tri[i * 3 + 2] + (uint32_t)nv0);
            tri2.push_back(c.tri[i * 3 + 1] + (uint32_t)nv0);
            ref2.push_back(c.ref[i * 3 + 0]);
            ref2.push_back(c.ref[i * 3 + 2]);
            ref2.push_back(c.ref[i * 3 + 1]);
        }
        // Side walls: one quad per edge, as two triangles.
        for (size_t e = 0; e < ne0; ++e) {
            const uint32_t a = c.edge[e * 2 + 0], b = c.edge[e * 2 + 1];
            const uint32_t a2 = a + (uint32_t)nv0, b2 = b + (uint32_t)nv0;
            const uint32_t ra = c.vref[a], rb = c.vref[b];
            tri2.push_back(a);  tri2.push_back(b);  tri2.push_back(b2);
            ref2.push_back(ra); ref2.push_back(rb); ref2.push_back(rb);
            tri2.push_back(a);  tri2.push_back(b2); tri2.push_back(a2);
            ref2.push_back(ra); ref2.push_back(rb); ref2.push_back(ra);
        }

        std::vector<uint32_t> edge2;
        edge2.reserve((ne0 * 2 + (size_t)nv0) * 2);
        edge2.insert(edge2.end(), c.edge.begin(), c.edge.end());
        for (size_t e = 0; e < ne0; ++e) {
            edge2.push_back(c.edge[e * 2 + 0] + (uint32_t)nv0);
            edge2.push_back(c.edge[e * 2 + 1] + (uint32_t)nv0);
        }
        for (int v = 0; v < nv0; ++v) {
            edge2.push_back((uint32_t)v);
            edge2.push_back((uint32_t)(nv0 + v));
        }

        std::vector<uint32_t> vref2 = c.vref;
        vref2.insert(vref2.end(), c.vref.begin(), c.vref.end());

        c.nv   = nv0 * 2;
        c.pos  = std::move(pos2);
        c.tri  = std::move(tri2);
        c.ref  = std::move(ref2);
        c.edge = std::move(edge2);
        c.vref = std::move(vref2);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Apply: rotate, project, and rewrite the scene's triangles
// ---------------------------------------------------------------------------

struct Stats {
    size_t verts   = 0;      // vertices in the N-D complex
    size_t tris    = 0;      // triangles written back to the scene
    size_t dropped = 0;      // degenerate (zero-area) triangles culled
    bool   linear  = false;  // the whole warp collapsed to one 3x3 matrix
    double det     = 1.0;    // determinant of that matrix (negative == mirrored)
    bool   singular = false; // the model has been squashed flat
    // Extra dimensions that are FILLED but invisible, because no rotated plane mixes them
    // into the three coordinates a projection keeps. An extruded axis in that state has
    // every one of its side walls collapse to zero area (they are culled, so the triangle
    // count silently halves back to the two lids); an embossed one displaces nothing.
    // Not an error — it is the correct picture of a solid seen exactly edge-on — but it
    // looks like the fill did nothing, so callers report it.
    std::vector<int> edgeOn;
};

// Column k of the rotation matrix's first three rows is the entire contribution dimension
// k makes to an orthographic projection. Its length is therefore how much of that axis is
// visible: 1 = fully turned into view, 0 = exactly edge-on.
inline double axisVisibility(const std::vector<double>& R, int n, int k) {
    if (k < 0 || k >= n) return 0.0;
    double s = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double v = R[(size_t)i * n + k];
        s += v * v;
    }
    return std::sqrt(s);
}

// Rebuild `s.tris` as the warped model, and rewrite Scene::meshGroups so the ranges
// still name the right triangles. Does NOT rebuild the BVH or the emitter tables —
// the interactive previewer re-rasterizes many times a second and needs none of that,
// so the caller decides when to pay for `Scene::build()`.
inline Stats apply(const Model& m, const Config& cfg, Scene& s) {
    Stats st;
    if (!m.ok) return st;

    const int n = std::max(3, cfg.n);
    const std::vector<double> R = rotationMatrix(cfg);
    const Mat3 M = topLeft3(R, n);
    st.det    = M.det();
    st.linear = cfg.isLinear();
    for (int k = 3; k < n; ++k)
        if (cfg.extra[(size_t)(k - 3)].fill != Fill::Zero &&
            cfg.extra[(size_t)(k - 3)].amp != 0.0 &&
            axisVisibility(R, n, k) < 1e-9)
            st.edgeOn.push_back(k);

    std::vector<Tri> out;
    out.reserve(m.keep.size() + (st.linear ? m.base.size() : projectedTriCount(m, cfg)));
    out.insert(out.end(), m.keep.begin(), m.keep.end());
    const size_t warpStart = out.size();

    // The positions every warped vertex is built from, and which base corner each
    // output corner inherits attributes from — collected so meshFinishTris can
    // re-derive crease-correct normals over the whole warped mesh in one pass.
    std::vector<Vec3>               verts;
    std::vector<std::array<int, 3>> triVI;
    const bool reNormal = !st.linear;   // a non-linear warp has to re-derive normals

    // Which output triangles came from which source group, so the group ranges can be
    // rewritten. Warped groups stay contiguous because we emit them in order.
    std::vector<size_t> groupFirst(m.groups.size(), 0), groupCount(m.groups.size(), 0);

    auto emit = [&](const Tri& proto, const Vec3& p0, const Vec3& p1, const Vec3& p2,
                    const Vec3& n0, const Vec3& n1, const Vec3& n2,
                    const Vec3& uv0, const Vec3& uv1, const Vec3& uv2,
                    int vi0, int vi1, int vi2) {
        // Cull what the projection destroyed. A plane rotation through 90 degrees
        // squashes the model onto a sheet and every triangle's area goes to zero;
        // letting those through would hand finalize() a zero cross product and put
        // NaNs in the geometric normals, which the rasterizer then paints as holes.
        const Vec3 cr = cross(p1 - p0, p2 - p0);
        if (!(dot(cr, cr) > 1e-30)) { ++st.dropped; return; }
        Tri t = proto;
        t.v0 = p0; t.v1 = p1; t.v2 = p2;
        t.n0 = n0; t.n1 = n1; t.n2 = n2;
        t.uv0 = uv0; t.uv1 = uv1; t.uv2 = uv2;
        out.push_back(t);
        if (reNormal) triVI.push_back({vi0, vi1, vi2});
    };

    if (st.linear) {
        // ---- Fast path: one 3x3 matrix, applied about the model centre ------------
        // No welding, no complex, no re-derived normals: the map is linear, so the
        // authored shading normals carry through exactly under the inverse transpose
        // and every crease the model was exported with survives untouched.
        Mat3 NT;
        const bool haveNT = inverseTranspose(M, NT);
        st.singular = !haveNT;
        auto xf = [&](const Vec3& p) { return M * (p - m.center) + m.center; };
        auto xn = [&](const Vec3& nv) {
            if (!haveNT) return Vec3{0, 0, 0};          // finalize() falls back to gn
            const Vec3 r = NT * nv;
            const double l = std::sqrt(dot(r, r));
            return (l > 1e-18) ? r * (1.0 / l) : Vec3{0, 0, 0};
        };
        for (size_t gi = 0; gi < m.groups.size(); ++gi) {
            const Source& g = m.groups[gi];
            groupFirst[gi] = out.size();
            for (size_t i = g.start; i < g.start + g.count; ++i) {
                const Tri& b = m.base[i];
                emit(b, xf(b.v0), xf(b.v1), xf(b.v2), xn(b.n0), xn(b.n1), xn(b.n2),
                     b.uv0, b.uv1, b.uv2, 0, 0, 0);
            }
            groupCount[gi] = out.size() - groupFirst[gi];
        }
        st.verts = (size_t)m.topo.nv;
    } else {
        // ---- General path: build the N-D complex, rotate it, project it -----------
        const Complex c = buildComplex(m, cfg);
        st.verts = (size_t)c.nv;

        // Rotate every vertex and keep the first three coordinates. Only the first
        // three ROWS of R are ever read — the rest of the matrix decides nothing an
        // orthographic projection can see, which is the header's whole argument.
        std::vector<Vec3> proj((size_t)c.nv);
        for (int v = 0; v < c.nv; ++v) {
            const double* P = &c.pos[(size_t)v * n];
            double q[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                double acc = 0.0;
                const double* Ri = &R[(size_t)i * n];
                for (int j = 0; j < n; ++j) acc += Ri[j] * P[j];
                q[i] = acc;
            }
            proj[(size_t)v] = Vec3{q[0], q[1], q[2]} + m.center;
        }
        verts = proj;

        // Every warped triangle belongs to the group its base corner came from, so an
        // extruded prism stays part of the object it was swept out of.
        std::vector<size_t> baseTriGroup(m.base.size(), 0);
        for (size_t gi = 0; gi < m.groups.size(); ++gi)
            for (size_t i = m.groups[gi].start; i < m.groups[gi].start + m.groups[gi].count; ++i)
                baseTriGroup[i] = gi;

        const size_t nt = c.tri.size() / 3;
        std::vector<std::vector<size_t>> byGroup(m.groups.size());
        for (size_t i = 0; i < nt; ++i)
            byGroup[baseTriGroup[c.ref[i * 3] / 3]].push_back(i);

        auto uvOf = [&](uint32_t ref) -> const Vec3& {
            const Tri& tr = m.base[ref / 3];
            const unsigned k = ref % 3;
            return (k == 0) ? tr.uv0 : (k == 1) ? tr.uv1 : tr.uv2;
        };
        for (size_t gi = 0; gi < m.groups.size(); ++gi) {
            groupFirst[gi] = out.size();
            for (size_t i : byGroup[gi]) {
                const uint32_t a = c.tri[i * 3 + 0], b = c.tri[i * 3 + 1], d = c.tri[i * 3 + 2];
                const uint32_t ra = c.ref[i * 3 + 0], rb = c.ref[i * 3 + 1], rd = c.ref[i * 3 + 2];
                emit(m.base[ra / 3], proj[a], proj[b], proj[d],
                     Vec3{0, 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0},
                     uvOf(ra), uvOf(rb), uvOf(rd), (int)a, (int)b, (int)d);
            }
            groupCount[gi] = out.size() - groupFirst[gi];
        }
    }

    st.tris = out.size() - warpStart;
    s.tris.swap(out);

    // Re-derive shading normals over the warped surface. `wantSmooth=true,
    // haveNormals=false` is exactly the loaders' crease pass: angle-weighted averaging
    // of face normals, merged only across edges softer than `creaseDeg`, so an extruded
    // prism's lids stay sharp against its walls instead of smearing into them.
    if (reNormal && !triVI.empty())
        meshFinishTris(s, warpStart, verts, triVI, /*proceduralUV*/false,
                       UvProjection::None, 1, /*wantSmooth*/true, /*haveNormals*/false,
                       cfg.creaseDeg);

    for (Tri& t : s.tris) t.finalize();

    // Rewrite the mesh-group ranges. Untouched groups only SHIFT (their triangles kept
    // their relative order at the front of the array); warped ones move to the tail and
    // may have changed size entirely.
    if (!s.meshGroups.empty()) {
        std::vector<size_t> newIndex(m.keepOrig.empty() ? 0 : m.keepOrig.back() + 1, SIZE_MAX);
        for (size_t i = 0; i < m.keepOrig.size(); ++i) newIndex[m.keepOrig[i]] = i;
        std::vector<char> isWarped(s.meshGroups.size(), 0);
        for (size_t gi = 0; gi < m.groups.size(); ++gi) isWarped[m.groups[gi].groupIdx] = 1;
        for (size_t gi = 0; gi < s.meshGroups.size(); ++gi) {
            MeshGroup& g = s.meshGroups[gi];
            if (isWarped[gi] || g.blasId >= 0 || g.triCount == 0) continue;
            if (g.triStart < newIndex.size() && newIndex[g.triStart] != SIZE_MAX)
                g.triStart = newIndex[g.triStart];
        }
        for (size_t gi = 0; gi < m.groups.size(); ++gi) {
            MeshGroup& g = s.meshGroups[m.groups[gi].groupIdx];
            g.triStart = groupFirst[gi];
            g.triCount = groupCount[gi];
        }
    }
    return st;
}

// A one-line "you cannot see this yet" note for the dimensions in Stats::edgeOn, or an
// empty string when everything filled is visible.
inline std::string edgeOnNote(const Stats& st, int n) {
    if (st.edgeOn.empty()) return "";
    std::string names;
    for (size_t i = 0; i < st.edgeOn.size(); ++i) {
        if (i) names += ", ";
        names += axisName(st.edgeOn[i]);
    }
    // Name a plane that would actually reveal it, since "rotate something" is not
    // actionable advice when there are 45 sliders to choose from.
    const std::string hint = "z" + axisName(st.edgeOn[0]);
    return names + (st.edgeOn.size() > 1 ? " are" : " is") +
           " filled but edge-on to the projection (turn a plane containing " +
           (st.edgeOn.size() > 1 ? "them" : "it") + ", e.g. " + hint + ", to see it)";
}

// Put the scene back exactly as it was captured (the warp switched off).
inline void restore(const Model& m, Scene& s) {
    if (!m.ok) return;
    Config off;
    off.n = 3;
    apply(m, off, s);
}

// The contiguous range of `s.tris` holding the warped model (apply() puts it at the
// tail). Returns false when nothing has been warped.
inline bool warpedRange(const Scene& s, const Model& m, size_t& start, size_t& count) {
    start = SIZE_MAX; count = 0;
    for (const Source& g : m.groups) {
        if (g.groupIdx >= s.meshGroups.size()) continue;
        const MeshGroup& mg = s.meshGroups[g.groupIdx];
        if (mg.triCount == 0) continue;
        start = std::min(start, mg.triStart);
        count += mg.triCount;
    }
    return start != SIZE_MAX && count > 0;
}

// ---------------------------------------------------------------------------
// Export: write the projected model out as a model in its own right
// ---------------------------------------------------------------------------

// Write the warped triangles as a Wavefront OBJ: welded positions, per-corner normals
// and UVs, one `o <name>` object per source mesh group, `usemtl` per scene material so
// the assignment survives a round trip. Deliberately NOT via isomesh::writeObj — that
// writer is for isosurface meshes and carries no texture coordinates, which is exactly
// what a skinned model must not lose.
inline bool exportObj(const Scene& s, const Model& m, const std::string& path,
                      std::string& err, size_t* trisOut = nullptr) {
    size_t start = 0, count = 0;
    if (!warpedRange(s, m, start, count)) { err = "nothing to export"; return false; }

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot open " + path; return false; }

    std::string b;
    b.reserve(size_t(4) << 20);
    b += "# ftrace -nd export: an N-D rotation of the source model, projected to 3-D\n";

    // Weld positions across the whole export so the OBJ is indexed rather than a soup;
    // normals and UVs stay per corner (a crease or a UV seam is precisely a place where
    // one position needs two of them).
    std::unordered_map<detail::QKey, int, detail::QHash> pmap;
    std::vector<Vec3> pos;
    std::vector<int>  pidx(count * 3);
    {
        Vec3 lo = s.tris[start].v0, hi = lo;
        for (size_t i = 0; i < count; ++i) {
            const Tri& t = s.tris[start + i];
            const Vec3 vv[3] = {t.v0, t.v1, t.v2};
            for (const Vec3& v : vv) {
                lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
                hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
            }
        }
        const Vec3 ext = hi - lo;
        const double diag = std::sqrt(dot(ext, ext));
        const double invE = 1.0 / ((diag > 0.0 ? diag : 1.0) * 1e-6);
        pmap.reserve(count * 2);
        for (size_t i = 0; i < count; ++i) {
            const Tri& t = s.tris[start + i];
            const Vec3 vv[3] = {t.v0, t.v1, t.v2};
            for (int c = 0; c < 3; ++c) {
                detail::QKey k{(long long)std::llround(vv[c].x * invE),
                               (long long)std::llround(vv[c].y * invE),
                               (long long)std::llround(vv[c].z * invE)};
                auto it = pmap.find(k);
                if (it == pmap.end()) {
                    const int id = (int)pos.size();
                    pmap.emplace(k, id);
                    pos.push_back(vv[c]);
                    pidx[i * 3 + c] = id;
                } else {
                    pidx[i * 3 + c] = it->second;
                }
            }
        }
    }
    for (const Vec3& p : pos) {
        b += "v ";  detail::appendDouble(b, p.x);
        b += ' ';   detail::appendDouble(b, p.y);
        b += ' ';   detail::appendDouble(b, p.z);
        b += '\n';
    }
    for (size_t i = 0; i < count; ++i) {
        const Tri& t = s.tris[start + i];
        const Vec3 tt[3] = {t.uv0, t.uv1, t.uv2};
        for (const Vec3& uv : tt) {
            b += "vt "; detail::appendDouble(b, uv.x);
            b += ' ';   detail::appendDouble(b, uv.y);
            b += '\n';
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const Tri& t = s.tris[start + i];
        const Vec3 nn[3] = {t.n0, t.n1, t.n2};
        for (const Vec3& v : nn) {
            b += "vn "; detail::appendDouble(b, v.x);
            b += ' ';   detail::appendDouble(b, v.y);
            b += ' ';   detail::appendDouble(b, v.z);
            b += '\n';
        }
    }
    // Faces, grouped by source object and then by material run.
    int lastMat = INT32_MIN;
    for (const Source& g : m.groups) {
        if (g.groupIdx >= s.meshGroups.size()) continue;
        const MeshGroup& mg = s.meshGroups[g.groupIdx];
        if (mg.triCount == 0) continue;
        b += "o "; b += (g.name.empty() ? std::string("nd_object") : g.name); b += '\n';
        for (size_t i = mg.triStart; i < mg.triStart + mg.triCount; ++i) {
            const size_t li = i - start;
            const Tri& t = s.tris[i];
            if (t.matId != lastMat) {
                lastMat = t.matId;
                b += "usemtl mat"; b += std::to_string(t.matId); b += '\n';
            }
            b += 'f';
            for (int c = 0; c < 3; ++c) {
                const long vi = (long)pidx[li * 3 + c] + 1;
                const long ti = (long)(li * 3 + c) + 1;
                b += ' ';
                b += std::to_string(vi); b += '/';
                b += std::to_string(ti); b += '/';
                b += std::to_string(ti);
            }
            b += '\n';
        }
    }
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    std::fclose(f);
    if (!ok) { err = "write failed: " + path; return false; }
    if (trisOut) *trisOut = count;
    return true;
}

// Write the warped triangles as a binary `.ftmesh` (see mesh.h for the layout): the
// same geometry with no text round trip, which is what ftrace itself prefers to
// reload. One mesh per file by definition, so materials and object names are not
// carried — use the OBJ writer when those matter.
inline bool exportFtmesh(const Scene& s, const Model& m, const std::string& path,
                         std::string& err, size_t* trisOut = nullptr) {
    size_t start = 0, count = 0;
    if (!warpedRange(s, m, start, count)) { err = "nothing to export"; return false; }

    // Weld on (position, normal, uv) so the shading survives: two corners at the same
    // point across a crease must stay two vertices, since .ftmesh is a single indexed
    // array with one normal per vertex.
    struct VKey {
        float p[3], n[3], t[2];
        bool operator==(const VKey& o) const { return std::memcmp(this, &o, sizeof(VKey)) == 0; }
    };
    struct VHash {
        size_t operator()(const VKey& k) const {
            unsigned long long h = 1469598103934665603ull;
            const unsigned char* raw = (const unsigned char*)&k;
            for (size_t i = 0; i < sizeof(VKey); ++i) { h ^= raw[i]; h *= 1099511628211ull; }
            return (size_t)h;
        }
    };
    std::unordered_map<VKey, uint32_t, VHash> map;
    map.reserve(count * 2);
    std::vector<float>    pos, nrm, uv;
    std::vector<uint32_t> idx(count * 3);
    for (size_t i = 0; i < count; ++i) {
        const Tri& t = s.tris[start + i];
        const Vec3 vv[3] = {t.v0, t.v1, t.v2};
        const Vec3 nn[3] = {t.n0, t.n1, t.n2};
        const Vec3 tt[3] = {t.uv0, t.uv1, t.uv2};
        for (int c = 0; c < 3; ++c) {
            VKey k{};
            k.p[0] = (float)vv[c].x; k.p[1] = (float)vv[c].y; k.p[2] = (float)vv[c].z;
            k.n[0] = (float)nn[c].x; k.n[1] = (float)nn[c].y; k.n[2] = (float)nn[c].z;
            k.t[0] = (float)tt[c].x; k.t[1] = (float)tt[c].y;
            auto it = map.find(k);
            if (it == map.end()) {
                const uint32_t id = (uint32_t)(pos.size() / 3);
                map.emplace(k, id);
                pos.push_back(k.p[0]); pos.push_back(k.p[1]); pos.push_back(k.p[2]);
                nrm.push_back(k.n[0]); nrm.push_back(k.n[1]); nrm.push_back(k.n[2]);
                uv.push_back(k.t[0]);  uv.push_back(k.t[1]);
                idx[i * 3 + c] = id;
            } else {
                idx[i * 3 + c] = it->second;
            }
        }
    }

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot open " + path; return false; }
    const uint32_t ver = 1, flags = FTMESH_HAS_NORMALS | FTMESH_HAS_UVS;
    const uint32_t nv = (uint32_t)(pos.size() / 3), nt = (uint32_t)count;
    bool ok = std::fwrite(FTMESH_MAGIC, 1, 8, f) == 8;
    auto w32 = [&](uint32_t v) { ok = ok && std::fwrite(&v, 4, 1, f) == 1; };
    w32(ver); w32(flags); w32(nv); w32(nt);
    ok = ok && std::fwrite(pos.data(), 4, pos.size(), f) == pos.size();
    ok = ok && std::fwrite(nrm.data(), 4, nrm.size(), f) == nrm.size();
    ok = ok && std::fwrite(uv.data(),  4, uv.size(),  f) == uv.size();
    ok = ok && std::fwrite(idx.data(), 4, idx.size(), f) == idx.size();
    std::fclose(f);
    if (!ok) { err = "write failed: " + path; return false; }
    if (trisOut) *trisOut = count;
    return true;
}

// Dispatch on the extension. `.ftmesh` takes the binary writer, anything else the OBJ
// one — an unknown extension is a typo far more often than a request for a format we
// do not have, and OBJ is the one every other tool can read.
inline bool exportModel(const Scene& s, const Model& m, const std::string& path,
                        std::string& err, size_t* trisOut = nullptr) {
    std::string ext;
    const size_t dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot);
        for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    }
    if (ext == ".ftmesh") return exportFtmesh(s, m, path, err, trisOut);
    return exportObj(s, m, path, err, trisOut);
}

// ---------------------------------------------------------------------------
// Command-line spec parsing
// ---------------------------------------------------------------------------

// `zero` | `extrude[:<depth>]` | `emboss:<source>[:<amp>[:<freq>]]`
inline bool parseDimSpec(const std::string& spec, DimSpec& out, std::string& err) {
    std::vector<std::string> f;
    size_t i = 0;
    while (i <= spec.size()) {
        const size_t j = spec.find(':', i);
        f.push_back(spec.substr(i, (j == std::string::npos) ? std::string::npos : j - i));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    if (f.empty()) { err = "empty fill spec"; return false; }
    std::string head = f[0];
    for (char& c : head) c = (char)std::tolower((unsigned char)c);
    if (head == "zero" || head == "0" || head == "off") { out = DimSpec{}; return true; }
    if (head == "extrude" || head == "prism" || head == "sweep") {
        out.fill = Fill::Extrude;
        out.amp  = (f.size() > 1 && !f[1].empty()) ? std::atof(f[1].c_str()) : 0.5;
        return true;
    }
    if (head == "emboss") {
        out.fill = Fill::Emboss;
        std::string src = (f.size() > 1) ? f[1] : "curvature";
        for (char& c : src) c = (char)std::tolower((unsigned char)c);
        if      (src == "curvature" || src == "curv") out.src = Emb::Curvature;
        else if (src == "radius"    || src == "rad")  out.src = Emb::Radius;
        else if (src == "height"    || src == "y")    out.src = Emb::Height;
        else if (src == "noise")                      out.src = Emb::Noise;
        else if (src == "u")                          out.src = Emb::U;
        else if (src == "v")                          out.src = Emb::V;
        else { err = "unknown emboss source '" + src +
                     "' (curvature|radius|height|noise|u|v)"; return false; }
        out.amp  = (f.size() > 2 && !f[2].empty()) ? std::atof(f[2].c_str()) : 0.25;
        out.freq = (f.size() > 3 && !f[3].empty()) ? std::atof(f[3].c_str()) : 4.0;
        return true;
    }
    err = "unknown fill '" + f[0] + "' (zero|emboss:<src>[:amp]|extrude[:depth])";
    return false;
}

// ---------------------------------------------------------------------------
// The fill pick-list
// ---------------------------------------------------------------------------
//
// One flat list of every (fill, source) pair a dimension can take, so the viewer's combo
// and the CLI's spec grammar cannot drift apart: the combo reports an index into THIS
// list, and `applyFillChoice` is the only thing that turns an index into a DimSpec.

inline const std::vector<std::string>& fillChoices() {
    static const std::vector<std::string> kChoices = {
        "zero",
        "emboss curvature",
        "emboss radius",
        "emboss height",
        "emboss noise",
        "emboss u",
        "emboss v",
        "extrude",
    };
    return kChoices;
}

// Sensible starting amount for a fill the user has just switched ON. An emboss at 0 is
// indistinguishable from `zero`, so picking one and seeing nothing happen would be the
// same dead end the edge-on case produces.
inline double defaultAmountFor(int choice) {
    if (choice <= 0) return 0.0;
    return (choice == 7) ? 0.5 : 0.25;   // extrude reads better a little deeper
}

// Set `d`'s fill and source from a pick-list index, leaving `amp`/`freq` alone.
inline void applyFillChoice(int choice, DimSpec& d) {
    switch (choice) {
        case 1: d.fill = Fill::Emboss;  d.src = Emb::Curvature; break;
        case 2: d.fill = Fill::Emboss;  d.src = Emb::Radius;    break;
        case 3: d.fill = Fill::Emboss;  d.src = Emb::Height;    break;
        case 4: d.fill = Fill::Emboss;  d.src = Emb::Noise;     break;
        case 5: d.fill = Fill::Emboss;  d.src = Emb::U;         break;
        case 6: d.fill = Fill::Emboss;  d.src = Emb::V;         break;
        case 7: d.fill = Fill::Extrude;                         break;
        default: d.fill = Fill::Zero;                           break;
    }
}

// The inverse: which pick-list entry a DimSpec currently is.
inline int fillChoiceOf(const DimSpec& d) {
    if (d.fill == Fill::Extrude) return 7;
    if (d.fill != Fill::Emboss)  return 0;
    switch (d.src) {
        case Emb::Curvature: return 1;
        case Emb::Radius:    return 2;
        case Emb::Height:    return 3;
        case Emb::Noise:     return 4;
        case Emb::U:         return 5;
        case Emb::V:         return 6;
    }
    return 0;
}

inline std::string dimSpecText(const DimSpec& d) {
    switch (d.fill) {
        case Fill::Zero:    return "zero";
        case Fill::Extrude: return "extrude " + std::to_string(d.amp);
        case Fill::Emboss:  return std::string("emboss ") + embName(d.src) + " " +
                                   std::to_string(d.amp);
    }
    return "?";
}

// `<plane>=<degrees>`, where the plane is either a letter pair (`xw`, `zv`) or a pair
// of axis indices (`0-3`). Returns the plane index, or -1 with `err` set.
inline int parseAngleSpec(int n, const std::string& spec, double& deg, std::string& err) {
    const size_t eq = spec.find('=');
    if (eq == std::string::npos) { err = "expected <plane>=<degrees>, got '" + spec + "'"; return -1; }
    std::string pl = spec.substr(0, eq);
    deg = std::atof(spec.c_str() + eq + 1);
    for (char& c : pl) c = (char)std::tolower((unsigned char)c);

    auto axisOf = [&](const std::string& t) -> int {
        for (int a = 0; a < n; ++a) if (axisName(a) == t) return a;
        if (!t.empty() && std::isdigit((unsigned char)t[0])) {
            const int a = std::atoi(t.c_str());
            if (a >= 0 && a < n) return a;
        }
        return -1;
    };
    const size_t dash = pl.find_first_of("-,");
    int ai = -1, aj = -1;
    if (dash != std::string::npos) {
        ai = axisOf(pl.substr(0, dash));
        aj = axisOf(pl.substr(dash + 1));
    } else if (pl.size() == 2) {
        ai = axisOf(pl.substr(0, 1));
        aj = axisOf(pl.substr(1, 1));
    }
    if (ai < 0 || aj < 0 || ai == aj) {
        err = "'" + pl + "' is not a rotation plane of " + std::to_string(n) + "-D space";
        return -1;
    }
    const int k = planeIndex(n, ai, aj);
    if (k < 0) { err = "'" + pl + "' is out of range"; return -1; }
    return k;
}

}  // namespace ndwarp
