// Watertight / airtight surface check.
//
// A closed 2-manifold ("watertight") mesh is one where every edge is shared by
// exactly two triangles: no boundary edges (holes), no non-manifold edges (an edge
// shared by 3+ faces). For a DIELECTRIC (glass) surface this is not cosmetic — the
// renderer decides enter-vs-exit from the surface normal at each hit and carries the
// "which medium am I inside" state along the whole ray/photon path, so a hole lets a
// ray reach the interior without a refraction event (medium bookkeeping desyncs) and a
// non-manifold edge is geometrically ambiguous. Either way refraction/absorption go
// wrong for the rest of that path, and in forward tracing the mis-bent light lands in
// the wrong place — artifacts can appear far from the object.
//
// We also flag ORIENTATION inconsistency: a mesh can be topologically closed yet have
// some triangles wound the wrong way, so their geometric normals point INTO the solid.
// That inverts the enter/exit test for those facets — just as damaging for glass as a
// hole — so it is reported alongside true watertightness.
//
// Vertices are welded by position first (quantised to a small fraction of the model
// diagonal) because both OBJ meshes and marching-cubes output routinely store
// coincident-but-separate vertices along shared edges; without welding every internal
// edge would look like a boundary.
#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "geometry.h"   // Vec3, Tri

namespace watertight {

struct Report {
    size_t tris        = 0;   // triangles examined
    size_t verts       = 0;   // welded (unique-position) vertex count
    size_t edges       = 0;   // unique undirected edges
    size_t boundary    = 0;   // edges used by exactly 1 face  -> a hole / open border
    size_t nonManifold = 0;   // edges used by 3+ faces        -> ambiguous topology
    size_t flipped     = 0;   // closed edges whose two faces wind the same way -> bad normals
    bool closed() const { return boundary == 0 && nonManifold == 0; }
    bool ok()     const { return closed() && flipped == 0; }
};

namespace detail {
struct VKey { long long x, y, z; bool operator==(const VKey& o) const { return x==o.x && y==o.y && z==o.z; } };
struct VKeyHash {
    size_t operator()(const VKey& k) const {
        size_t h = 1469598103934665603ULL;
        h ^= (size_t)k.x; h *= 1099511628211ULL;
        h ^= (size_t)k.y; h *= 1099511628211ULL;
        h ^= (size_t)k.z; h *= 1099511628211ULL;
        return h;
    }
};
struct EKey { long long a, b; bool operator==(const EKey& o) const { return a==o.a && b==o.b; } };
struct EKeyHash {
    size_t operator()(const EKey& e) const { return (size_t)(e.a * 1000003LL) ^ (size_t)(e.b * 2654435761ULL); }
};
struct EdgeUse { int fwd = 0, rev = 0; };   // fwd: low->high half-edges, rev: high->low
} // namespace detail

// Core: weld `pos` by position, remap the triangle indices `idx` (3 per triangle) onto
// welded ids, then classify every mesh edge.
inline Report check(const std::vector<Vec3>& pos, const std::vector<int>& idx) {
    Report r;
    r.tris = idx.size() / 3;
    if (pos.empty() || idx.size() < 3) return r;

    Vec3 lo = pos[0], hi = pos[0];
    for (const Vec3& p : pos) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    double diag = std::sqrt((hi.x-lo.x)*(hi.x-lo.x) + (hi.y-lo.y)*(hi.y-lo.y) + (hi.z-lo.z)*(hi.z-lo.z));
    double eps  = std::max(1e-9, diag * 1e-7);
    double inv  = 1.0 / eps;

    std::unordered_map<detail::VKey, int, detail::VKeyHash> weld;
    weld.reserve(pos.size() * 2);
    std::vector<int> remap(pos.size());
    for (size_t i = 0; i < pos.size(); ++i) {
        detail::VKey k{ std::llround(pos[i].x*inv), std::llround(pos[i].y*inv), std::llround(pos[i].z*inv) };
        auto it = weld.find(k);
        if (it == weld.end()) { int id = (int)weld.size(); weld.emplace(k, id); remap[i] = id; }
        else remap[i] = it->second;
    }
    r.verts = weld.size();

    std::unordered_map<detail::EKey, detail::EdgeUse, detail::EKeyHash> edges;
    edges.reserve(idx.size());
    auto addEdge = [&](int u, int v) {
        if (u == v) return;                       // degenerate edge, ignore
        bool swap = u > v;
        detail::EKey e{ swap ? (long long)v : (long long)u, swap ? (long long)u : (long long)v };
        detail::EdgeUse& eu = edges[e];
        if (swap) eu.rev++; else eu.fwd++;
    };
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        int a = remap[idx[t]], b = remap[idx[t+1]], c = remap[idx[t+2]];
        addEdge(a, b); addEdge(b, c); addEdge(c, a);
    }
    r.edges = edges.size();
    for (const auto& kv : edges) {
        int total = kv.second.fwd + kv.second.rev;
        if      (total == 1) r.boundary++;
        else if (total  > 2) r.nonManifold++;
        else if (total == 2 && !(kv.second.fwd == 1 && kv.second.rev == 1)) r.flipped++;
    }
    return r;
}

// Convenience: check a contiguous run of Scene triangles (their explicit v0/v1/v2
// world positions; welding rebuilds shared vertices).
inline Report checkTris(const Tri* tris, size_t n) {
    std::vector<Vec3> pos; pos.reserve(n * 3);
    std::vector<int>  idx; idx.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
        pos.push_back(tris[i].v0); pos.push_back(tris[i].v1); pos.push_back(tris[i].v2);
        int base = (int)(i * 3);
        idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
    }
    return check(pos, idx);
}

// Marks the faces of `tris` (one mesh object) that bound a closed solid facing outward, so that a
// ray meeting one from behind is leaving the solid (Tri::solidFace, Hit::solid). "Facing" is the
// intersector's: it turns a face's `ng` to the side its shading normals are on (intersectTri), so
// a face counts by the side all three corner normals agree on, and one whose corners disagree
// cannot say. Taken per connected piece (triangles sharing a welded vertex), which must have:
//   - every edge used by exactly two of its faces, in opposite directions: closed and consistently
//     wound;
//   - every face's normals on one side of its winding, the same side over the whole piece, so
//     `back` keeps one meaning there (an OBJ wound against its normals is fine, if it is so
//     throughout);
//   - a positive enclosed volume, taken on that side: it faces outward.
// Every other face is cleared. Returns how many were marked.
inline size_t markSolidFaces(Tri* tris, size_t n) {
    if (n == 0) return 0;
    Vec3 lo = tris[0].v0, hi = tris[0].v0;
    for (size_t i = 0; i < n; ++i)
        for (const Vec3* p : {&tris[i].v0, &tris[i].v1, &tris[i].v2}) {
            lo.x = std::min(lo.x, p->x); lo.y = std::min(lo.y, p->y); lo.z = std::min(lo.z, p->z);
            hi.x = std::max(hi.x, p->x); hi.y = std::max(hi.y, p->y); hi.z = std::max(hi.z, p->z);
        }
    const double diag = std::sqrt((hi.x-lo.x)*(hi.x-lo.x) + (hi.y-lo.y)*(hi.y-lo.y) + (hi.z-lo.z)*(hi.z-lo.z));
    const double inv  = 1.0 / std::max(1e-9, diag * 1e-7);   // check()'s weld
    std::unordered_map<detail::VKey, int, detail::VKeyHash> weld;
    weld.reserve(n * 2);
    std::vector<int> vid(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const Vec3* c[3] = {&tris[i].v0, &tris[i].v1, &tris[i].v2};
        for (int k = 0; k < 3; ++k) {
            detail::VKey key{ std::llround(c[k]->x*inv), std::llround(c[k]->y*inv), std::llround(c[k]->z*inv) };
            auto it = weld.emplace(key, (int)weld.size()).first;
            vid[i * 3 + k] = it->second;
        }
    }
    // The pieces: union-find over welded vertices.
    std::vector<int> parent(weld.size());
    for (size_t v = 0; v < parent.size(); ++v) parent[v] = (int)v;
    const auto root = [&](int v) {
        while (parent[(size_t)v] != v) { parent[(size_t)v] = parent[(size_t)parent[(size_t)v]]; v = parent[(size_t)v]; }
        return v;
    };
    for (size_t i = 0; i < n; ++i) {
        const int a = root(vid[i*3]), b = root(vid[i*3+1]), c = root(vid[i*3+2]);
        parent[(size_t)b] = a; parent[(size_t)root(c)] = a;
    }
    std::vector<char> bad(parent.size(), 0);
    std::vector<char> side(parent.size(), 0);     // +1 / -1: the piece's faces' normals with / against winding
    std::vector<double> vol(parent.size(), 0.0);
    std::unordered_map<detail::EKey, detail::EdgeUse, detail::EKeyHash> edges;
    edges.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const Tri& t = tris[i];
        const int piece = root(vid[i*3]);
        const int c[3] = {vid[i*3], vid[i*3+1], vid[i*3+2]};
        for (int k = 0; k < 3; ++k) {
            const int u = c[k], v = c[(k + 1) % 3];
            if (u == v) continue;                    // a degenerate edge, as check() ignores it
            const bool swap = u > v;
            detail::EdgeUse& eu = edges[detail::EKey{ swap ? v : u, swap ? u : v }];
            if (swap) eu.rev++; else eu.fwd++;
        }
        // Volume relative to the box's corner keeps the sums' rounding to the object's own scale.
        const Vec3 a = t.v0 - lo, b = t.v1 - lo, d = t.v2 - lo;
        vol[(size_t)piece] += dot(a, cross(b, d));
        const double c0 = dot(t.gn, t.n0), c1 = dot(t.gn, t.n1), c2 = dot(t.gn, t.n2);
        const char s = (c0 > 0.0 && c1 > 0.0 && c2 > 0.0) ? 1 : (c0 < 0.0 && c1 < 0.0 && c2 < 0.0) ? -1 : 0;
        char& ps = side[(size_t)piece];
        if (s == 0 || (ps != 0 && ps != s)) bad[(size_t)piece] = 1;
        ps = s;
    }
    for (const auto& kv : edges)
        if (!(kv.second.fwd == 1 && kv.second.rev == 1)) bad[(size_t)root((int)kv.first.a)] = 1;
    size_t marked = 0;
    for (size_t i = 0; i < n; ++i) {
        const int piece = root(vid[i*3]);
        const bool s = !bad[(size_t)piece] && vol[(size_t)piece] * (double)side[(size_t)piece] > 0.0;
        tris[i].solidFace = s ? 1 : 0;
        marked += s ? 1 : 0;
    }
    return marked;
}

} // namespace watertight
