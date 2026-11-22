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
// on nothing, its floor lit through it). Nor is a piece closed that moves or deforms (it rests where it
// does at the instant only), or one holding any of `litPoints` (the lights' positions), whose inside the
// light reaches: a box with a lamp in it is a room, not a solid. Its pieceOf is -1; its box is given.
struct PieceBox { Vec3 lo, hi; int matId = 0; };
using CapOk = std::function<bool(const Vec3& centre, const Vec3& normal, const std::vector<Vec3>& rim)>;
inline size_t markSolidFaces(Tri* const* tris, size_t n, std::vector<PieceBox>* boxes = nullptr,
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr,
                             const std::vector<Vec3>* litPoints = nullptr) {
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
    // each way, kCapMoves a face moves or deforms -- none of which a cap mends -- and kCapRim it has a rim
    // (edges used once).
    enum : char { kCapSide = 1, kCapEdge = 2, kCapMoves = 4, kCapRim = 8 };
    std::vector<char> capFlags(capOk ? parent.size() : 0, 0);
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
            if (capOk) capFlags[(size_t)piece] |= kCapSide;
        }
        if (capOk && (t.motion >= 0 || t.vkeys >= 0)) capFlags[(size_t)piece] |= kCapMoves;
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
            if (capOk) {
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
    // The pieces a cap may close (capOk's pass): open only along a rim, every face's side right, still.
    // Compact: a candidate's slot by its root (`candOf`), only where there is one.
    struct Cand { int piece; size_t e0, e1; Vec3 c, nrm; double vol = 0.0; bool capped = false; std::vector<size_t> faces; };
    std::vector<Cand> cands;
    std::vector<int> candOf;
    if (capOk && !rimEdges.empty()) {
        std::sort(rimEdges.begin(), rimEdges.end(), [](const RimEdge& x, const RimEdge& y) { return x.piece < y.piece; });
        for (size_t q = 0; q < rimEdges.size();) {
            size_t e = q;
            while (e < rimEdges.size() && rimEdges[e].piece == rimEdges[q].piece) ++e;
            if (capFlags[(size_t)rimEdges[q].piece] == kCapRim) cands.push_back(Cand{rimEdges[q].piece, q, e, {}, {}});
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
            cands[(size_t)cn].faces.push_back(i);
            const Vec3* p[3] = {&tris[i]->v0, &tris[i]->v1, &tris[i]->v2};
            for (int k = 0; k < 3; ++k) {
                const auto it = at.find(vid[i*3 + k]);
                if (it != at.end()) it->second = *p[k];
            }
        }
        // A ray from `x` along a fixed odd direction (no edge it can graze in any scene laid on axes):
        // whether it crosses the piece with its cap an odd count of times, so `x` lies inside.
        const Vec3 dirOdd = normalize(Vec3{0.5772, 0.5781, 0.5766});
        const auto inside = [&](const Cand& cd, const Vec3& x) {
            int cross_ = 0;
            for (size_t i : cd.faces) {
                const Tri& t = *tris[i];
                const Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0, p = cross(dirOdd, e2);
                const double det = dot(e1, p);
                if (std::fabs(det) < 1e-300) continue;
                const double inv = 1.0 / det;
                const Vec3 s = x - t.v0;
                const double u = dot(s, p) * inv;
                if (u < 0.0 || u > 1.0) continue;
                const Vec3 qv = cross(s, e1);
                const double v = dot(dirOdd, qv) * inv;
                if (v < 0.0 || u + v > 1.0) continue;
                if (dot(e2, qv) * inv > 0.0) ++cross_;
            }
            // The cap: the rim's plane where the ray meets it, inside the rim (even-odd, the plane's
            // dominant axis dropped).
            const double dn = dot(dirOdd, cd.nrm);
            if (std::fabs(dn) > 1e-12) {
                const double tc = dot(cd.c - x, cd.nrm) / dn;
                if (tc > 0.0) {
                    const Vec3 y = x + dirOdd * tc;
                    const Vec3 an{std::fabs(cd.nrm.x), std::fabs(cd.nrm.y), std::fabs(cd.nrm.z)};
                    const int drop = an.x >= an.y && an.x >= an.z ? 0 : an.y >= an.z ? 1 : 2;
                    const auto u2 = [&](const Vec3& p) { return drop == 0 ? p.y : p.x; };
                    const auto v2 = [&](const Vec3& p) { return drop == 2 ? p.y : p.z; };
                    bool in = false;
                    for (size_t q = cd.e0; q < cd.e1; ++q) {
                        const Vec3& pa = at[rimEdges[q].a];
                        const Vec3& pb = at[rimEdges[q].b];
                        if ((v2(pa) > v2(y)) != (v2(pb) > v2(y)) &&
                            u2(y) < u2(pa) + (u2(pb) - u2(pa)) * (v2(y) - v2(pa)) / (v2(pb) - v2(pa)))
                            in = !in;
                    }
                    if (in) ++cross_;
                }
            }
            return (cross_ & 1) != 0;
        };
        for (Cand& cd : cands) {
            const size_t pc = (size_t)cd.piece;
            std::vector<Vec3> rim;
            Vec3 c{0, 0, 0};
            for (size_t q = cd.e0; q < cd.e1; ++q) { rim.push_back(at[rimEdges[q].a]); c = c + rim.back(); }
            c = c * (1.0 / (double)rim.size());
            Vec3 nrm{0, 0, 0};
            for (size_t q = cd.e0; q < cd.e1; ++q) nrm = nrm + cross(at[rimEdges[q].a] - c, at[rimEdges[q].b] - c);
            const double nl = std::sqrt(dot(nrm, nrm));
            const Vec3 ext = phi[pc] - plo[pc];
            const double pd = std::sqrt(dot(ext, ext));
            if (!(nl > 1e-12 * pd * pd)) continue;   // no area: no plane to close it by
            nrm = nrm * (1.0 / nl);
            bool planar = true;
            for (const Vec3& p : rim) planar = planar && std::fabs(dot(p - c, nrm)) <= 1e-6 * pd;
            if (!planar) continue;
            cd.c = c; cd.nrm = nrm;
            // Its volume about the rim's centre, the cap's own tetrahedra nothing there; a flat piece (a
            // card, the floor, a light's quad) closes nothing, and asks capOk nothing.
            for (size_t i : cd.faces) {
                const Vec3 a = tris[i]->v0 - c, b = tris[i]->v1 - c, d = tris[i]->v2 - c;
                cd.vol += dot(a, cross(b, d));
            }
            if (std::fabs(cd.vol) <= 1e-9 * pd * pd * pd) continue;
            // A light inside it lights what it holds, which is then no solid: a box with a lamp in it.
            bool lit = false;
            if (litPoints)
                for (const Vec3& x : *litPoints)
                    if (x.x >= plo[pc].x && x.x <= phi[pc].x && x.y >= plo[pc].y && x.y <= phi[pc].y &&
                        x.z >= plo[pc].z && x.z <= phi[pc].z && inside(cd, x)) { lit = true; break; }
            if (lit || !(*capOk)(c, nrm, rim)) continue;
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
        // A piece of no thickness -- a double-sided card, its two faces wound against each other
        // -- is closed and encloses nothing: its sign is rounding's, so it is no solid.
        const double v = (capped ? cands[(size_t)cn].vol : vol[(size_t)piece]) * (double)side[(size_t)piece];
        const Vec3 ext = phi[(size_t)piece] - plo[(size_t)piece];
        const double pd = std::sqrt(dot(ext, ext));
        const bool flat = std::fabs(v) <= 1e-9 * pd * pd * pd;
        const unsigned char s = capped ? (v > 0.0 ? 3 : 4)
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
                             std::vector<int>* pieceOf = nullptr, const CapOk* capOk = nullptr,
                             const std::vector<Vec3>* litPoints = nullptr) {
    std::vector<Tri*> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = tris + i;
    return markSolidFaces(p.data(), n, boxes, pieceOf, capOk, litPoints);
}

} // namespace watertight
