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
#include <cfloat>
#include <cmath>
#include <functional>
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

// Marks the faces of one mesh object (`tris`, pointers into wherever they live) that bound a closed
// solid, so a ray meeting one from behind its normal is leaving the solid or entering it
// (Tri::solidFace, Hit::solid). "Behind" is the intersector's: it turns a face's `ng` to the side its
// interpolated shading normal is on (intersectTri), so a face counts by the side its corner normals
// lie on. Corners exactly in its plane flip nothing; one corner on each side means it cannot say.
// Taken per connected piece (triangles sharing a welded vertex), which must have:
//   - every edge used by exactly two of its faces, in opposite directions: closed and consistently
//     wound;
//   - every face's normals on one side of its winding, the same side over the whole piece, so
//     `back` keeps one meaning there (an OBJ wound against its normals is fine, if it is so
//     throughout).
// Its enclosed volume, taken on that side, says which way the normals point: a positive one marks
// the piece's faces 1 (normals out: from behind is leaving), a negative one 2 (normals in, as a
// room's may be: from behind is entering). Every other face is 0. Returns how many were marked.
// `boxes`, when given, gets each closed piece's bounds (lo, hi) and the material of one of its faces,
// so a caller can tell where solids lie without looking at every face; `pieceOf`, each face's closed
// piece (a number shared by its faces), -1 for a face of none.
//
// `capOk`, where given (the footprint probes' pass alone: PROBE-PARITY-LIMITS), also closes a piece open
// along a planar rim that another face closes -- a statue's open base on the floor it stands on: a piece
// whose every edge is used twice in opposite directions or once, the once-used ones (its rim) all in one
// plane, which capOk(centre, normal, rim) says some other face covers where the rim encloses it. Its
// volume taken about a point in that plane is the capped solid's, the cap's own tetrahedra vanishing
// there, and its faces are marked 3 (normals out) or 4 (in): a side the probes read and nothing else does
// -- a camera or light inside one is not in a solid, a glowing one may still be two-sided -- as the cap
// is right only for the gather on the face that closes it (an upright cup's rim is planar too and closes
// on nothing, its floor lit through it). Its pieceOf is -1; its box is given.
struct PieceBox { Vec3 lo, hi; int matId = 0; };
using CapOk = std::function<bool(const Vec3& centre, const Vec3& normal, const std::vector<Vec3>& rim)>;
inline size_t markSolidFaces(Tri* const* tris, size_t n, std::vector<PieceBox>* boxes = nullptr,
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr) {
    if (pieceOf) pieceOf->assign(n, -1);
    if (n == 0) return 0;
    Vec3 lo = tris[0]->v0, hi = tris[0]->v0;
    for (size_t i = 0; i < n; ++i)
        for (const Vec3* p : {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2}) {
            lo.x = std::min(lo.x, p->x); lo.y = std::min(lo.y, p->y); lo.z = std::min(lo.z, p->z);
            hi.x = std::max(hi.x, p->x); hi.y = std::max(hi.y, p->y); hi.z = std::max(hi.z, p->z);
        }
    const double diag = std::sqrt((hi.x-lo.x)*(hi.x-lo.x) + (hi.y-lo.y)*(hi.y-lo.y) + (hi.z-lo.z)*(hi.z-lo.z));
    // The weld: check()'s, 1e-7 of the extent -- but no more than a thousandth of the short edges
    // (a pool of loose triangles spans a whole scene, and a ground quad kilometres wide must not
    // weld a millimetre box's corners), taken at their 1st percentile so one sliver does not shrink
    // it for every piece; and no less than the coordinates' own precision, a billionth of their
    // magnitude, which also keeps the quantised values far inside a long long.
    std::vector<double> el;
    el.reserve(n * 3);
    double maxAbs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const Vec3* c[3] = {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2};
        for (int k = 0; k < 3; ++k) {
            maxAbs = std::max({maxAbs, std::fabs(c[k]->x), std::fabs(c[k]->y), std::fabs(c[k]->z)});
            const Vec3 e = *c[(k + 1) % 3] - *c[k];
            const double l = std::sqrt(dot(e, e));
            if (l > 0.0) el.push_back(l);
        }
    }
    double weld = std::max(1e-9, diag * 1e-7);
    if (!el.empty()) {
        const size_t q = el.size() / 100;
        std::nth_element(el.begin(), el.begin() + (std::ptrdiff_t)q, el.end());
        weld = std::min(weld, 1e-3 * el[q]);
    }
    std::vector<double>().swap(el);
    weld = std::max(weld, std::max(1e-12, 1e-9 * maxAbs));
    const double inv  = 1.0 / weld;
    // Weld by sorting the corners' quantised positions (flat arrays: a scene's worth of triangles
    // costs a sort, where hash maps cost a node per corner).
    struct Corner { long long x, y, z; uint32_t at; };
    std::vector<Corner> cs(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const Vec3* c[3] = {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2};
        for (int k = 0; k < 3; ++k)
            cs[i * 3 + k] = Corner{ std::llround(c[k]->x*inv), std::llround(c[k]->y*inv),
                                    std::llround(c[k]->z*inv), (uint32_t)(i * 3 + k) };
    }
    std::sort(cs.begin(), cs.end(), [](const Corner& a, const Corner& b) {
        return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
    });
    std::vector<int> vid(n * 3);
    int nv = 0;
    for (size_t q = 0; q < cs.size(); ++q) {
        if (q > 0 && (cs[q].x != cs[q-1].x || cs[q].y != cs[q-1].y || cs[q].z != cs[q-1].z)) ++nv;
        vid[cs[q].at] = nv;
    }
    ++nv;
    std::vector<Corner>().swap(cs);
    // The pieces: union-find over welded vertices.
    std::vector<int> parent((size_t)nv);
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
    std::vector<char> rimmed(parent.size(), 0);   // bad only by edges used once (a rim): capOk may close it
    std::vector<char> side(parent.size(), 0);     // +1 / -1: the piece's faces' normals with / against winding
    std::vector<double> vol(parent.size(), 0.0);
    // Each piece's own extent, and a corner of it the volume is taken about, so the sums' rounding
    // and the no-volume test are at the piece's scale, not the call's.
    std::vector<Vec3> plo(parent.size()), phi(parent.size());
    std::vector<char> seenP(parent.size(), 0);
    for (size_t i = 0; i < n; ++i) {
        const size_t pc = (size_t)root(vid[i*3]);
        if (!seenP[pc]) { seenP[pc] = 1; plo[pc] = phi[pc] = tris[i]->v0; }
        for (const Vec3* p : {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2}) {
            plo[pc].x = std::min(plo[pc].x, p->x); plo[pc].y = std::min(plo[pc].y, p->y); plo[pc].z = std::min(plo[pc].z, p->z);
            phi[pc].x = std::max(phi[pc].x, p->x); phi[pc].y = std::max(phi[pc].y, p->y); phi[pc].z = std::max(phi[pc].z, p->z);
        }
    }
    // Each directed edge as (low, high) and which way it runs; sorted, every undirected edge's uses
    // are one run, which must be one each way.
    struct Edge { int a, b; bool rev; };
    std::vector<Edge> es;
    es.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const Tri& t = *tris[i];
        const int piece = root(vid[i*3]);
        const int c[3] = {vid[i*3], vid[i*3+1], vid[i*3+2]};
        for (int k = 0; k < 3; ++k) {
            const int u = c[k], v = c[(k + 1) % 3];
            if (u == v) continue;                    // a degenerate edge, as check() ignores it
            es.push_back(u < v ? Edge{u, v, false} : Edge{v, u, true});
        }
        const Vec3& o = plo[(size_t)piece];
        const Vec3 a = t.v0 - o, b = t.v1 - o, d = t.v2 - o;
        vol[(size_t)piece] += dot(a, cross(b, d));
        const double c0 = dot(t.gn, t.n0), c1 = dot(t.gn, t.n1), c2 = dot(t.gn, t.n2);
        const bool anyNeg = c0 < 0.0 || c1 < 0.0 || c2 < 0.0, anyPos = c0 > 0.0 || c1 > 0.0 || c2 > 0.0;
        const char s = (anyNeg && anyPos) ? 0 : anyNeg ? -1 : 1;   // all in-plane: nothing flips it
        char& ps = side[(size_t)piece];
        if (s == 0 || (ps != 0 && ps != s)) bad[(size_t)piece] = 1;
        ps = s;
    }
    std::sort(es.begin(), es.end(), [](const Edge& x, const Edge& y) {
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    // A rim's edges, by piece, as vertex pairs in their faces' direction (capOk's pass): each rim vertex
    // starts one, and the loops' area vector (Newell's) sums them.
    std::vector<std::vector<std::pair<int, int>>> rimOf(capOk ? parent.size() : 0);
    std::vector<char> worse(parent.size(), 0);    // an edge used past once each way: no rim closes it
    for (size_t q = 0; q < es.size();) {
        size_t e = q;
        int fwd = 0, rev = 0;
        for (; e < es.size() && es[e].a == es[q].a && es[e].b == es[q].b; ++e) (es[e].rev ? rev : fwd)++;
        const size_t pc = (size_t)root(es[q].a);
        if (!(fwd == 1 && rev == 1)) {
            bad[pc] = 1;
            // as its one face runs it (an edge is kept low vertex first, `rev` where the face ran it back)
            if (capOk && fwd + rev == 1) rimOf[pc].emplace_back(es[q].rev ? es[q].b : es[q].a, es[q].rev ? es[q].a : es[q].b);
            else worse[pc] = 1;
        }
        q = e;
    }
    // A face's side wrong for its piece is as bad as an edge used thrice: no cap mends it.
    std::vector<char> sideOk(parent.size(), 1);
    for (size_t i = 0; i < n; ++i) {
        const Tri& t = *tris[i];
        const double c0 = dot(t.gn, t.n0), c1 = dot(t.gn, t.n1), c2 = dot(t.gn, t.n2);
        const bool anyNeg = c0 < 0.0 || c1 < 0.0 || c2 < 0.0, anyPos = c0 > 0.0 || c1 > 0.0 || c2 > 0.0;
        const char s = (anyNeg && anyPos) ? 0 : anyNeg ? -1 : 1;
        const size_t pc = (size_t)root(vid[i*3]);
        if (s == 0 || s != side[pc]) sideOk[pc] = 0;
    }
    // Each rimmed piece's rim plane and the volume about its centre (capOk's pass).
    std::vector<Vec3> capC(capOk ? parent.size() : 0);
    std::vector<double> capVol(capOk ? parent.size() : 0, 0.0);
    if (capOk) {
        std::vector<Vec3> at((size_t)nv);   // a welded vertex's position
        for (size_t i = 0; i < n; ++i) {
            at[(size_t)vid[i*3]] = tris[i]->v0; at[(size_t)vid[i*3+1]] = tris[i]->v1; at[(size_t)vid[i*3+2]] = tris[i]->v2;
        }
        for (size_t pc = 0; pc < parent.size(); ++pc) {
            if (rimOf[pc].empty() || worse[pc] || !sideOk[pc]) continue;
            std::vector<Vec3> rim;
            Vec3 c{0, 0, 0};
            for (const auto& [a, b] : rimOf[pc]) { rim.push_back(at[(size_t)a]); c = c + at[(size_t)a]; }
            c = c * (1.0 / (double)rim.size());
            Vec3 nrm{0, 0, 0};   // Newell's: the rim loops' summed area vector
            for (const auto& [a, b] : rimOf[pc]) nrm = nrm + cross(at[(size_t)a] - c, at[(size_t)b] - c);
            const double nl = std::sqrt(dot(nrm, nrm));
            const Vec3 ext = phi[pc] - plo[pc];
            const double pd = std::sqrt(dot(ext, ext));
            if (!(nl > 1e-12 * pd * pd)) continue;   // no area: no plane to close it by
            nrm = nrm * (1.0 / nl);
            bool planar = true;
            for (const Vec3& p : rim) planar = planar && std::fabs(dot(p - c, nrm)) <= 1e-6 * pd;
            if (!planar || !(*capOk)(c, nrm, rim)) continue;
            rimmed[pc] = 1;
            capC[pc] = c;
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t pc = (size_t)root(vid[i*3]);
            if (!rimmed[pc]) continue;
            const Vec3 a = tris[i]->v0 - capC[pc], b = tris[i]->v1 - capC[pc], d = tris[i]->v2 - capC[pc];
            capVol[pc] += dot(a, cross(b, d));
        }
    }
    size_t marked = 0;
    std::vector<int> boxOf;                       // a marked piece's slot in *boxes, or -1
    if (boxes) boxOf.assign(parent.size(), -1);
    for (size_t i = 0; i < n; ++i) {
        const int piece = root(vid[i*3]);
        // A piece of no thickness -- a double-sided card, its two faces wound against each other
        // -- is closed and encloses nothing: its sign is rounding's, so it is no solid.
        const bool capped = !rimmed.empty() && rimmed[(size_t)piece];
        const double v = (capped ? capVol[(size_t)piece] : vol[(size_t)piece]) * (double)side[(size_t)piece];
        const Vec3 ext = phi[(size_t)piece] - plo[(size_t)piece];
        const double pd = std::sqrt(dot(ext, ext));
        const bool flat = std::fabs(v) <= 1e-9 * pd * pd * pd;
        const unsigned char s = capped ? (flat ? 0 : (v > 0.0 ? 3 : 4))
                                       : (bad[(size_t)piece] || flat) ? 0 : (v > 0.0 ? 1 : 2);
        tris[i]->solidFace = s;
        marked += s ? 1 : 0;
        if (pieceOf && (s == 1 || s == 2)) (*pieceOf)[i] = piece;
        if (boxes && s && boxOf[(size_t)piece] < 0) {
            boxOf[(size_t)piece] = (int)boxes->size();
            boxes->push_back(PieceBox{plo[(size_t)piece], phi[(size_t)piece], tris[i]->matId});
        }
    }
    return marked;
}
// The same over a contiguous run.
inline size_t markSolidFaces(Tri* tris, size_t n, std::vector<PieceBox>* boxes = nullptr,
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr) {
    std::vector<Tri*> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = tris + i;
    return markSolidFaces(p.data(), n, boxes, pieceOf, capOk);
}

} // namespace watertight
