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
#include "bvh.h"        // OwnFaces' tree

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
// along a rim that other faces close -- a statue's open base on the floor it stands on, planar or on
// uneven ground: a piece whose every edge is used twice in opposite directions or once, the once-used ones
// its rim, in one loop or several, each loop closed by the fan of triangles from its own centroid, which
// capOk(loops, own, motion) says rests on faces not the piece's own, every loop (CapLoop): a lampshade
// open at its top and bottom rests by its bottom alone, its top on nothing. `own(x, rad)`: whether x lies
// within `rad` of one of the piece's faces; `motion` the piece's motion set (Scene::motionSets; -1 for
// one that holds still or that moves part by part or deforms): a mover is asked where it is at each of
// its keys, so a piece lowered onto the floor by its motion rests there. Its volume taken about the
// first loop's centroid, with the other loops' fans, is the capped solid's (the first fan's tetrahedra
// vanish there), and its faces are marked 3 (normals out) or 4 (in): a side the probes read and nothing
// else does -- a camera or light inside one is not in a solid, a glowing one may still be two-sided -- as
// the cap is right only for the gather on the face that closes it (an upright cup's rim is planar too and
// closes on nothing, its floor lit through it). A piece that covers its own points (a near-flat sheet its
// rounding leaves a volume, a ground whose border closes on its own flat part round a pit) closes on
// nothing. The cap says only which side of the faces is in: whether that side walls a column off from
// the gather point the probes' low leg decides (gathercoverage.h inside()) -- a box with a lamp in it is a
// room the gather point lies in, a mover capped where it rests may be clear of the surface at a path's
// time. Its pieceOf is -1; its box is given.
//
// `capCands`, where given in capOk's stead (an asset's pass: its instances may each rest on something
// else or nothing), takes the pieces a cap would close, the first kCapCandsMax of them, as capOk would
// be asked of them (CapCand), and marks their faces 5 + 2k (normals out) or 6 + 2k (in), k the piece's
// place there: a cap each instance that places it resting decides for itself (Scene::markSolidFaces'
// MeshInstance::capMask, Hit::solid +-(4 + k) until the instance's hit resolves it). They are no solid by
// themselves, so not counted; their box is given, with the place.
//
// A marked piece's box, and (`faces`) its faces where it has kPieceNearFaces or fewer -- a room, a box,
// a crate -- empty for one with more (Scene::solidNear): pointers into the call's triangles, good while
// they stand; `cand`, an asset's candidate's place in `capCands`, or -1.
constexpr size_t kPieceNearFaces = 64;
struct PieceBox { Vec3 lo, hi; int matId = 0; std::vector<const Tri*> faces; size_t count = 0; int cand = -1; };
constexpr int kCapCandsMax = 64;   // MeshInstance::capMask's bits
// Whether `x` lies within `h` of triangle `t` (its box first).
inline bool nearFace(const Tri& t, const Vec3& x, double h) {
    if (x.x < std::min({t.v0.x, t.v1.x, t.v2.x}) - h || x.x > std::max({t.v0.x, t.v1.x, t.v2.x}) + h ||
        x.y < std::min({t.v0.y, t.v1.y, t.v2.y}) - h || x.y > std::max({t.v0.y, t.v1.y, t.v2.y}) + h ||
        x.z < std::min({t.v0.z, t.v1.z, t.v2.z}) - h || x.z > std::max({t.v0.z, t.v1.z, t.v2.z}) + h)
        return false;
    return pointTriDistSq(x, t.v0, t.v1, t.v2) <= h * h;
}
// One loop of a piece's rim as capOk asks it, in the piece's own space: its centroid `c`, the unit
// normal of its area vector (Newell's), `pts` the points a face other than the piece's must cover --
// the centroid and a tenth of the way in from a dozen of the loop's corners at most, on its fan -- and
// `rim` the points a hundredth of the way in from its corners (every one, up to 96), where the loop must
// touch one, neither any lying on the piece itself (a hat's brim on the table asks nothing); `reach` the farthest corner
// from the centroid, `slack` twice the loop's spread off its mean plane (0 for a planar one), how far a
// covering face may stray from a point along the normal beyond a planar loop's allowance.
struct CapLoop { Vec3 c{0, 0, 0}, nrm{0, 0, 0}; std::vector<Vec3> pts, rim; double reach = 0.0, slack = 0.0; };
// A piece's faces (indices into the call's triangles) and a tree over their boxes, leaf k for faces[k]:
// whether a point lies on the piece.
struct OwnFaces {
    std::vector<size_t> faces;
    Bvh tree;
    template <class TriAt>
    bool onPiece(const TriAt& triAt, const Vec3& x, double h) const {
        return tree.traverseSphereAny(x, h, [&](int k) { return nearFace(triAt(faces[(size_t)k]), x, h); });
    }
};
// An asset's piece a cap would close, as capOk would be asked of it in the asset's own space.
struct CapCand { std::vector<CapLoop> loops; OwnFaces own; };
using CapOk = std::function<bool(const std::vector<CapLoop>& loops, const std::function<bool(const Vec3&, double)>& own,
                                 int motion)>;
inline size_t markSolidFaces(Tri* const* tris, size_t n, std::vector<PieceBox>* boxes = nullptr,
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr,
                             std::vector<CapCand>* capCands = nullptr) {
    if (capCands) capCands->clear();
    const bool caps = capOk || capCands;
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
    std::vector<char> side(parent.size(), 0);     // +1 / -1: the piece's faces' normals with / against winding
    // capOk's pass: per piece, kCapSide a face's side is wrong for it, kCapEdge an edge is used past once
    // each way -- neither of which a cap mends -- and kCapRim it has a rim (edges used once).
    enum : char { kCapSide = 1, kCapEdge = 2, kCapRim = 4 };
    std::vector<char> capFlags(caps ? parent.size() : 0, 0);
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
        if (s == 0 || (ps != 0 && ps != s)) {
            bad[(size_t)piece] = 1;
            if (caps) capFlags[(size_t)piece] |= kCapSide;
        }
        ps = s;
    }
    std::sort(es.begin(), es.end(), [](const Edge& x, const Edge& y) {
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    // A rim's edges, as their one face runs them (an edge is kept low vertex first, `rev` where the face
    // ran it back): each rim vertex starts one, and the loops' area vector (Newell's) sums them.
    struct RimEdge { int piece, a, b; };
    std::vector<RimEdge> rimEdges;
    for (size_t q = 0; q < es.size();) {
        size_t e = q;
        int fwd = 0, rev = 0;
        for (; e < es.size() && es[e].a == es[q].a && es[e].b == es[q].b; ++e) (es[e].rev ? rev : fwd)++;
        if (!(fwd == 1 && rev == 1)) {
            const int pc = root(es[q].a);
            bad[(size_t)pc] = 1;
            if (caps) {
                if (fwd + rev == 1) {
                    capFlags[(size_t)pc] |= kCapRim;
                    rimEdges.push_back(RimEdge{pc, es[q].rev ? es[q].b : es[q].a, es[q].rev ? es[q].a : es[q].b});
                } else {
                    capFlags[(size_t)pc] |= kCapEdge;
                }
            }
        }
        q = e;
    }
    std::vector<Edge>().swap(es);
    // The pieces a cap may close (capOk's pass): open only along a rim, every face's side right.
    // Compact: a candidate's slot by its root (`candOf`), only where there is one.
    // `motion`: its faces' motion set, -1 where none moves, -2 where they differ or one deforms; -3 unset.
    // `place`: where it is in *capCands, or -1.
    struct Cand {
        int piece; size_t e0, e1; double vol = 0.0; bool capped = false; std::vector<size_t> faces;
        int motion = -3;
        int place = -1;
    };
    std::vector<Cand> cands;
    std::vector<int> candOf;
    if (caps && !rimEdges.empty()) {
        std::sort(rimEdges.begin(), rimEdges.end(), [](const RimEdge& x, const RimEdge& y) { return x.piece < y.piece; });
        for (size_t q = 0; q < rimEdges.size();) {
            size_t e = q;
            while (e < rimEdges.size() && rimEdges[e].piece == rimEdges[q].piece) ++e;
            if (capFlags[(size_t)rimEdges[q].piece] == kCapRim) cands.push_back(Cand{rimEdges[q].piece, q, e});
            q = e;
        }
    }
    if (!cands.empty()) {
        candOf.assign(parent.size(), -1);
        for (size_t k = 0; k < cands.size(); ++k) candOf[(size_t)cands[k].piece] = (int)k;
        std::unordered_map<int, Vec3> at;   // a rim vertex's position
        for (const Cand& cd : cands)
            for (size_t q = cd.e0; q < cd.e1; ++q) at.emplace(rimEdges[q].a, Vec3{0, 0, 0});
        for (size_t i = 0; i < n; ++i) {
            const int cn = candOf[(size_t)root(vid[i*3])];
            if (cn < 0) continue;
            Cand& cdn = cands[(size_t)cn];
            cdn.faces.push_back(i);
            const int fm = tris[i]->vkeys >= 0 ? -2 : tris[i]->motion;
            cdn.motion = cdn.motion == -3 ? fm : (cdn.motion == fm ? fm : -2);
            const Vec3* p[3] = {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2};
            for (int k = 0; k < 3; ++k) {
                const auto it = at.find(vid[i*3 + k]);
                if (it != at.end()) it->second = *p[k];
            }
        }
        const auto triAt = [&](size_t i) -> const Tri& { return *tris[i]; };
        for (Cand& cd : cands) {
            const size_t pc = (size_t)cd.piece;
            const Vec3 ext = phi[pc] - plo[pc];
            const double pd = std::sqrt(dot(ext, ext));
            // Its rim's loops: each edge runs a -> b as its one face runs it, so a loop follows b to the
            // edge that starts there, back to where it began. A vertex two loops share (a pinch) starts
            // two edges; either serves. An end that leads nowhere is no loop, and closes nothing.
            std::unordered_multimap<int, size_t> from;
            for (size_t q = cd.e0; q < cd.e1; ++q) from.emplace(rimEdges[q].a, q);
            std::vector<char> used(cd.e1 - cd.e0, 0);
            std::vector<std::vector<size_t>> loopEdges;
            bool broken = false;
            for (size_t q0 = cd.e0; q0 < cd.e1 && !broken; ++q0) {
                if (used[q0 - cd.e0]) continue;
                std::vector<size_t> le;
                size_t q = q0;
                while (true) {
                    used[q - cd.e0] = 1;
                    le.push_back(q);
                    if (rimEdges[q].b == rimEdges[q0].a) break;
                    size_t next = SIZE_MAX;
                    const auto range = from.equal_range(rimEdges[q].b);
                    for (auto it = range.first; it != range.second; ++it)
                        if (!used[it->second - cd.e0]) { next = it->second; break; }
                    if (next == SIZE_MAX) { broken = true; break; }
                    q = next;
                }
                loopEdges.push_back(std::move(le));
            }
            if (broken || loopEdges.empty()) continue;
            // Each loop's centroid, normal, spread and points (CapLoop).
            OwnFaces own;
            own.faces = cd.faces;
            {
                std::vector<Aabb> fb(cd.faces.size());
                for (size_t k = 0; k < cd.faces.size(); ++k) {
                    const Tri& t = *tris[cd.faces[k]];
                    fb[k].expand(t.v0); fb[k].expand(t.v1); fb[k].expand(t.v2);
                }
                own.tree.build(fb, 1);
            }
            std::vector<CapLoop> loops;
            bool degenerate = false;
            for (const std::vector<size_t>& le : loopEdges) {
                CapLoop L;
                std::vector<Vec3> corner;
                for (size_t q : le) { corner.push_back(at[rimEdges[q].a]); L.c = L.c + corner.back(); }
                L.c = L.c * (1.0 / (double)corner.size());
                Vec3 nrm{0, 0, 0};
                for (size_t q : le) nrm = nrm + cross(at[rimEdges[q].a] - L.c, at[rimEdges[q].b] - L.c);
                const double nl = std::sqrt(dot(nrm, nrm));
                if (!(nl > 1e-12 * pd * pd)) { degenerate = true; break; }   // no area: no fan to close it by
                L.nrm = nrm * (1.0 / nl);
                double spread = 0.0;                    // the loop's farthest corner off its mean plane
                for (const Vec3& p : corner) {
                    spread = std::max(spread, std::fabs(dot(p - L.c, L.nrm)));
                    L.reach = std::max(L.reach, length(p - L.c));
                }
                L.slack = spread <= 1e-6 * pd ? 0.0 : 2.0 * spread;   // planar, to within rounding: none
                // On the piece itself, to within rounding: a point it covers itself, which asks nothing.
                const double onR = 1e-6 * L.reach;
                if (!own.onPiece(triAt, L.c, onR)) L.pts.push_back(L.c);
                // Covering points at a dozen corners at most; touching points at every corner up to
                // kTouchAll, beyond them at an odd stride -- an even one aliases a base notched corner by
                // corner, every point then on a corner that stands on the floor.
                const size_t step = std::max<size_t>(1, corner.size() / 12);
                for (size_t i = 0; i < corner.size(); i += step) {
                    const Vec3 in10 = corner[i] + (L.c - corner[i]) * 0.1;
                    if (!own.onPiece(triAt, in10, onR)) L.pts.push_back(in10);
                }
                constexpr size_t kTouchAll = 96;
                const size_t tstep = corner.size() <= kTouchAll ? 1 : ((corner.size() / kTouchAll) | 1);
                for (size_t i = 0; i < corner.size(); i += tstep) {
                    const Vec3 in1 = corner[i] + (L.c - corner[i]) * 0.01;
                    if (!own.onPiece(triAt, in1, onR)) L.rim.push_back(in1);
                }
                loops.push_back(std::move(L));
            }
            if (degenerate) continue;
            // Its volume about the first loop's centroid, that loop's fan's tetrahedra nothing there, the
            // others' added: a fan triangle (c_j, b, a) runs its rim edge against the face that runs it a -> b.
            // A flat piece (a card, the floor, a light's quad) closes nothing, and asks capOk nothing.
            const Vec3 o = loops[0].c;
            for (size_t i : cd.faces) {
                const Vec3 a = tris[i]->v0 - o, b = tris[i]->v1 - o, d = tris[i]->v2 - o;
                cd.vol += dot(a, cross(b, d));
            }
            for (size_t j = 1; j < loops.size(); ++j)
                for (size_t q : loopEdges[j])
                    cd.vol += dot(loops[j].c - o, cross(at[rimEdges[q].b] - o, at[rimEdges[q].a] - o));
            if (std::fabs(cd.vol) <= 1e-9 * pd * pd * pd) continue;
            // An asset's: handed back for its instances to ask.
            if (capCands) {
                if ((int)capCands->size() < kCapCandsMax) {
                    cd.place = (int)capCands->size();
                    capCands->push_back(CapCand{std::move(loops), std::move(own)});
                }
                continue;
            }
            // The piece's own faces cover nothing: what it rests on is something else.
            if (!(*capOk)(loops, [&](const Vec3& x, double rad) { return own.onPiece(triAt, x, rad); },
                          std::max(cd.motion, -1)))
                continue;
            cd.capped = true;
        }
    }
    size_t marked = 0;
    std::vector<int> boxOf;                       // a marked piece's slot in *boxes, or -1
    if (boxes) boxOf.assign(parent.size(), -1);
    for (size_t i = 0; i < n; ++i) {
        const int piece = root(vid[i*3]);
        const int cn = candOf.empty() ? -1 : candOf[(size_t)piece];
        const bool capped = cn >= 0 && cands[(size_t)cn].capped;
        const int place = cn >= 0 ? cands[(size_t)cn].place : -1;
        // A piece of no thickness -- a double-sided card, its two faces wound against each other
        // -- is closed and encloses nothing: its sign is rounding's, so it is no solid.
        const double v = (capped || place >= 0 ? cands[(size_t)cn].vol : vol[(size_t)piece]) * (double)side[(size_t)piece];
        const Vec3 ext = phi[(size_t)piece] - plo[(size_t)piece];
        const double pd = std::sqrt(dot(ext, ext));
        const bool flat = std::fabs(v) <= 1e-9 * pd * pd * pd;
        const unsigned char s = place >= 0 ? (unsigned char)(5 + 2 * place + (v > 0.0 ? 0 : 1))
                              : capped ? (v > 0.0 ? 3 : 4)
                                       : (bad[(size_t)piece] || flat) ? 0 : (v > 0.0 ? 1 : 2);
        tris[i]->solidFace = s;
        marked += (s && place < 0) ? 1 : 0;
        if (pieceOf && (s == 1 || s == 2)) (*pieceOf)[i] = piece;
        if (boxes && s && boxOf[(size_t)piece] < 0) {
            boxOf[(size_t)piece] = (int)boxes->size();
            boxes->push_back(PieceBox{plo[(size_t)piece], phi[(size_t)piece], tris[i]->matId, {}, 0, place});
        }
        if (boxes && s) {
            PieceBox& pb = (*boxes)[(size_t)boxOf[(size_t)piece]];
            if (++pb.count <= kPieceNearFaces) pb.faces.push_back(tris[i]);
            else if (!pb.faces.empty()) std::vector<const Tri*>().swap(pb.faces);
        }
    }
    return marked;
}
// The same over a contiguous run.
inline size_t markSolidFaces(Tri* tris, size_t n, std::vector<PieceBox>* boxes = nullptr,
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr,
                             std::vector<CapCand>* capCands = nullptr) {
    std::vector<Tri*> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = tris + i;
    return markSolidFaces(p.data(), n, boxes, pieceOf, capOk, capCands);
}

} // namespace watertight
