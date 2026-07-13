// isomesh.h — isosurface -> triangle mesh converter (marching tetrahedra + optional
// quadric-error decimation for curvature-adaptive tessellation). Reuses the exact
// field the renderer sees: Implicit::eval (signed field, f<0 inside) and
// Implicit::gradient (unit outward normal). Output is a watertight, welded OBJ
// suitable for import into Unreal / Blender / etc.
//
// Entry points: marchImplicit(im, opts) -> Mesh; decimateAdaptive(mesh, ratio, im);
//               writeObj(path, groups, log).
//
// Design notes
//  * Marching *tetrahedra* (Kuhn/Freudenthal 6-tet split of each cell), NOT marching
//    cubes: tets have no face-ambiguous cases, so the surface is a guaranteed
//    watertight 2-manifold (marching cubes can leave holes / non-manifold edges).
//  * Vertices are welded by the *canonical unordered pair of lattice endpoints* the
//    crossing edge spans; the Kuhn split is consistent across shared cube faces, so
//    adjacent cells agree on every shared vertex -> no cracks.
//  * Edge crossings are placed by linear interpolation of the field, then refined
//    by a few bisection steps on the real field for curved surfaces.
//  * Triangle winding is reoriented per-face so the geometric normal agrees with
//    the field gradient (which points from inside->outside), i.e. outward.
//  * Adaptive tessellation: the uniform mesh is decimated with a quadric error
//    metric (QEM) edge-collapse pass that removes triangles in flat regions and
//    keeps them where curvature/detail is high, preserving the manifold.
#pragma once
#include "implicit.h"   // Implicit::eval / gradient / bounds, plus Vec3 (via linalg.h)
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace isomesh {

// Corner (di,dj,dk) offsets for the 8 cube corners (Bourke layout). The mesher
// uses marching *tetrahedra* (Kuhn/Freudenthal split, no ambiguous cases) rather
// than marching cubes, so no edge/triangle lookup tables are needed.
static const int kCorner[8][3] = {
    {0,0,0},{1,0,0},{1,0,1},{0,0,1},{0,1,0},{1,1,0},{1,1,1},{0,1,1}
};

struct Options {
    int    res = 128;        // cells along the LONGEST bounds axis (fineness)
    bool   adaptive = false; // run QEM decimation after marching
    double decimate = 0.5;   // adaptive target: keep this fraction of triangles
    int    refineIters = 4;  // bisection steps to place each edge vertex
    double pad = 0.02;       // fractional bounds pad so the surface isn't clipped
};

struct Mesh {
    std::vector<Vec3> pos;
    std::vector<Vec3> nrm;
    std::vector<int>  tri;   // 3 indices per triangle
};

// ---- Uniform marching tetrahedra over one implicit (box-capped) -------------
inline Mesh marchImplicit(const Implicit& im, const Options& opt) {
    Mesh m;
    Aabb capBox = im.bounds;              // solid is intersected with this box (caps)
    Vec3 ext0 = capBox.hi - capBox.lo;
    double maxe = std::max(ext0.x, std::max(ext0.y, ext0.z));
    if (maxe <= 0) return m;

    // Interior cell counts over the cap box (roughly cubic cells), then extend the
    // sampling lattice PAD cells beyond each face so the outermost lattice layer is
    // strictly outside the cap box (boxSDF > 0 there). Combined with the box-SDF
    // capping below, this guarantees every surface that reaches the domain seals
    // into a flat cap -> a closed, watertight solid (no open rim at the boundary).
    auto cells = [&](double e){ return std::max(1, (int)std::lround(opt.res * (e / maxe))); };
    int IX = cells(ext0.x), IY = cells(ext0.y), IZ = cells(ext0.z);
    double hx = ext0.x / IX, hy = ext0.y / IY, hz = ext0.z / IZ;
    const int PAD = 2;                    // sampling-lattice margin, in cells
    int NX = IX + 2*PAD, NY = IY + 2*PAD, NZ = IZ + 2*PAD;
    int SX = NX + 1, SY = NY + 1, SZ = NZ + 1;
    Vec3 org{ capBox.lo.x - PAD*hx, capBox.lo.y - PAD*hy, capBox.lo.z - PAD*hz };

    auto gridPos = [&](int i, int j, int k) {
        return Vec3{ org.x + hx*i, org.y + hy*j, org.z + hz*k };
    };

    // Signed distance to the cap box (negative inside, positive outside).
    auto boxSDF = [&](const Vec3& p)->double {
        double dx = std::max(capBox.lo.x - p.x, p.x - capBox.hi.x);
        double dy = std::max(capBox.lo.y - p.y, p.y - capBox.hi.y);
        double dz = std::max(capBox.lo.z - p.z, p.z - capBox.hi.z);
        double ox = std::max(dx,0.0), oy = std::max(dy,0.0), oz = std::max(dz,0.0);
        double outside = std::sqrt(ox*ox + oy*oy + oz*oz);
        double inside  = std::min(std::max(dx, std::max(dy,dz)), 0.0);
        return outside + inside;
    };
    // Augmented field = CSG intersection of the solid with the cap box. This is the
    // field the exported mesh actually represents; both sampling and vertex normals
    // use it, so box-face caps get correct outward normals.
    auto augEval = [&](const Vec3& p)->double {
        return std::max(im.eval(p), boxSDF(p));
    };
    double eps = maxe / std::max(NX, std::max(NY, NZ)) * 0.5;
    auto augGrad = [&](const Vec3& p)->Vec3 {
        double gx = augEval({p.x+eps,p.y,p.z}) - augEval({p.x-eps,p.y,p.z});
        double gy = augEval({p.x,p.y+eps,p.z}) - augEval({p.x,p.y-eps,p.z});
        double gz = augEval({p.x,p.y,p.z+eps}) - augEval({p.x,p.y,p.z-eps});
        Vec3 g{gx,gy,gz};
        double L = length(g);
        return (L > 1e-30) ? g*(1.0/L) : Vec3{0,1,0};
    };

    // Sample the augmented field on the full lattice.
    std::vector<double> val((size_t)SX * SY * SZ);
    auto vidx = [&](int i, int j, int k){ return ((size_t)i * SY + j) * SZ + k; };
    for (int i = 0; i < SX; ++i)
        for (int j = 0; j < SY; ++j)
            for (int k = 0; k < SZ; ++k)
                val[vidx(i,j,k)] = augEval(gridPos(i,j,k));

    // Vertices are welded by the canonical (unordered) pair of LATTICE ENDPOINTS the
    // crossing edge spans. The Kuhn/Freudenthal 6-tetrahedra split below tiles space
    // CONSISTENTLY across cube faces (every cube uses the same body diagonal 0->6),
    // so a face/edge shared by two cubes is split identically. Unlike marching cubes
    // (whose face-ambiguous cases can leave holes / non-manifold edges), marching
    // tetrahedra has NO ambiguous case, so the output is a watertight 2-manifold.
    auto latId = [&](int i,int j,int k)->uint64_t { return ((uint64_t)i * SY + j) * SZ + k; };
    std::unordered_map<uint64_t,int> edgeVert;
    edgeVert.reserve((size_t)SX * SY * 4);

    // Interpolate + bisection-refine the zero crossing on lattice edge (a)-(b); weld.
    auto edgeVertex = [&](int i0,int j0,int k0,double f0, int i1,int j1,int k1,double f1)->int {
        uint64_t a = latId(i0,j0,k0), b = latId(i1,j1,k1);
        uint64_t key = a < b ? (a << 32 | b) : (b << 32 | a);
        auto it = edgeVert.find(key);
        if (it != edgeVert.end()) return it->second;
        Vec3 p0 = gridPos(i0,j0,k0), p1 = gridPos(i1,j1,k1);
        double denom = f1 - f0;
        double t = (std::fabs(denom) > 1e-30) ? (-f0 / denom) : 0.5;
        t = std::min(1.0, std::max(0.0, t));
        double lo = 0.0, hi = 1.0, flo = f0, fhi = f1;
        if ((flo < 0) != (fhi < 0)) {
            for (int r = 0; r < opt.refineIters; ++r) {
                double tm = 0.5 * (lo + hi);
                double fm = augEval(p0 + (p1 - p0) * tm);
                if ((flo < 0) == (fm < 0)) { lo = tm; flo = fm; } else { hi = tm; fhi = fm; }
            }
            t = 0.5 * (lo + hi);
        }
        Vec3 p = p0 + (p1 - p0) * t;
        int idx = (int)m.pos.size();
        m.pos.push_back(p);
        m.nrm.push_back(augGrad(p));   // unit, outward (augmented field increases outward)
        edgeVert.emplace(key, idx);
        return idx;
    };

    // Kuhn decomposition: 6 tets, each = corner 0 (0,0,0) + a monotone lattice path
    // to corner 6 (1,1,1). Indices reference kCorner[] (Bourke corner layout).
    static const int kTet[6][4] = {
        {0,1,5,6}, {0,1,2,6}, {0,4,5,6}, {0,4,7,6}, {0,3,2,6}, {0,3,7,6}
    };
    auto emitTri = [&](int a,int b,int c){
        if (a<0||b<0||c<0||a==b||b==c||a==c) return;
        Vec3 fn = cross(m.pos[b]-m.pos[a], m.pos[c]-m.pos[a]);
        Vec3 gn = m.nrm[a] + m.nrm[b] + m.nrm[c];
        int bb=b, cc=c;
        if (dot(fn,gn) < 0.0) std::swap(bb,cc);   // wind so normal agrees with grad
        m.tri.push_back(a); m.tri.push_back(bb); m.tri.push_back(cc);
    };

    for (int ci = 0; ci < NX; ++ci)
    for (int cj = 0; cj < NY; ++cj)
    for (int ck = 0; ck < NZ; ++ck) {
        int cc3[8][3]; double cv[8];       // lattice coords + field value per corner
        for (int c = 0; c < 8; ++c) {
            int i = ci + kCorner[c][0], j = cj + kCorner[c][1], k = ck + kCorner[c][2];
            cc3[c][0]=i; cc3[c][1]=j; cc3[c][2]=k;
            cv[c] = val[vidx(i,j,k)];
        }
        for (int tt = 0; tt < 6; ++tt) {
            const int* T = kTet[tt];
            int in[4], nin=0, out[4], nout=0;
            for (int q = 0; q < 4; ++q) {
                if (cv[T[q]] < 0.0) in[nin++]=T[q]; else out[nout++]=T[q];
            }
            (void)nout;
            if (nin==0 || nin==4) continue;    // tet entirely inside or outside
            auto V = [&](int ca, int cb)->int {
                return edgeVertex(cc3[ca][0],cc3[ca][1],cc3[ca][2],cv[ca],
                                  cc3[cb][0],cc3[cb][1],cc3[cb][2],cv[cb]);
            };
            if (nin==1) {                      // 1 in : triangle on its 3 edges
                int o=in[0];
                emitTri(V(o,out[0]), V(o,out[1]), V(o,out[2]));
            } else if (nin==3) {               // 1 out: triangle on its 3 edges
                int o=out[0];
                emitTri(V(o,in[0]), V(o,in[1]), V(o,in[2]));
            } else {                           // 2 in / 2 out: quad -> two triangles
                int a=in[0], b=in[1], c=out[0], d=out[1];
                int ac=V(a,c), ad=V(a,d), bd=V(b,d), bc=V(b,c);
                emitTri(ac, ad, bd);
                emitTri(ac, bd, bc);
            }
        }
    }
    return m;
}

// ---- Curvature-adaptive decimation (quadric error metric edge collapse) -----
// Collapses cheap edges first. The QEM cost is tiny on flat regions (all incident
// planes coincide -> zero error to slide a vertex along them) and large where the
// surface curves, so triangles survive densely on detailed areas and thin out on
// flat ones — exactly the requested adaptive tessellation. Foldover-rejecting
// checks keep the mesh manifold/watertight. `ratio` is the target triangle
// fraction (0.3 => keep ~30%). `im` is used only to re-seat vertex normals.
namespace qem_detail {
struct Quad { double q[10] = {0,0,0,0,0,0,0,0,0,0}; };
inline void addPlane(Quad& Q, double a, double b, double c, double d) {
    double p[4] = {a,b,c,d};
    int t = 0;
    for (int i = 0; i < 4; ++i) for (int j = i; j < 4; ++j) Q.q[t++] += p[i]*p[j];
}
inline void add(Quad& A, const Quad& B){ for (int i=0;i<10;++i) A.q[i]+=B.q[i]; }
inline double error(const Quad& Q, const Vec3& v) {
    double x=v.x,y=v.y,z=v.z;
    return Q.q[0]*x*x + 2*Q.q[1]*x*y + 2*Q.q[2]*x*z + 2*Q.q[3]*x
         + Q.q[4]*y*y + 2*Q.q[5]*y*z + 2*Q.q[6]*y
         + Q.q[7]*z*z + 2*Q.q[8]*z + Q.q[9];
}
// Optimal collapse position: solve 3x3 [A]v = -b; fall back to the midpoint.
inline Vec3 optimum(const Quad& Q, const Vec3& fallback) {
    double a11=Q.q[0],a12=Q.q[1],a13=Q.q[2];
    double a22=Q.q[4],a23=Q.q[5],a33=Q.q[7];
    double b1=-Q.q[3],b2=-Q.q[6],b3=-Q.q[8];
    double det = a11*(a22*a33-a23*a23) - a12*(a12*a33-a23*a13) + a13*(a12*a23-a22*a13);
    if (std::fabs(det) < 1e-14) return fallback;
    double id = 1.0/det;
    double x = ( b1*(a22*a33-a23*a23) - a12*(b2*a33-a23*b3) + a13*(b2*a23-a22*b3)) * id;
    double y = ( a11*(b2*a33-a23*b3) - b1*(a12*a33-a23*a13) + a13*(a12*b3-b2*a13)) * id;
    double z = ( a11*(a22*b3-a23*b2) - a12*(a12*b3-a23*b1) + b1*(a12*a23-a22*a13)) * id;
    return Vec3{x,y,z};
}
} // namespace qem_detail

inline void decimateAdaptive(Mesh& m, double ratio, const Implicit& im) {
    using namespace qem_detail;
    int nv = (int)m.pos.size();
    int nt0 = (int)(m.tri.size() / 3);
    if (nt0 == 0) return;
    size_t targetTris = (size_t)std::max(4.0, ratio * nt0);

    struct Face { int v[3]; bool dead=false; };
    std::vector<Face> F(nt0);
    for (int t = 0; t < nt0; ++t) { F[t].v[0]=m.tri[3*t]; F[t].v[1]=m.tri[3*t+1]; F[t].v[2]=m.tri[3*t+2]; }

    std::vector<Vec3> P = m.pos;
    std::vector<Quad> Q(nv);
    std::vector<std::vector<int>> vf(nv);         // vertex -> incident faces
    std::vector<char> alive(nv, 1);
    std::vector<int>  ver(nv, 0);                 // version for lazy PQ invalidation

    auto faceNormalAreaW = [&](int a,int b,int c)->Vec3 {
        return cross(P[b]-P[a], P[c]-P[a]);       // length == 2*area, dir == normal
    };
    // Seed quadrics from face planes; build adjacency.
    for (int t = 0; t < nt0; ++t) {
        int a=F[t].v[0],b=F[t].v[1],c=F[t].v[2];
        Vec3 n = faceNormalAreaW(a,b,c);
        double len = length(n);
        if (len > 1e-20) {
            Vec3 un = n / len;
            double d = -dot(un, P[a]);
            Quad pq; addPlane(pq, un.x, un.y, un.z, d);
            add(Q[a], pq); add(Q[b], pq); add(Q[c], pq);
        }
        vf[a].push_back(t); vf[b].push_back(t); vf[c].push_back(t);
    }

    struct Cand { double cost; int i, j, vi, vj; Vec3 target; };
    struct Cmp { bool operator()(const Cand&x,const Cand&y)const{ return x.cost>y.cost; } };
    std::priority_queue<Cand,std::vector<Cand>,Cmp> pq;

    auto edgeKey = [](int a,int b){ if(a>b) std::swap(a,b); return ((uint64_t)a<<32)|(uint32_t)b; };
    auto pushEdge = [&](int a,int b){
        if (a==b || !alive[a] || !alive[b]) return;
        Quad qc = Q[a]; add(qc, Q[b]);
        Vec3 mid = (P[a]+P[b])*0.5;
        Vec3 x = optimum(qc, mid);
        pq.push(Cand{ error(qc,x), a, b, ver[a], ver[b], x });
    };

    { // seed unique edges
        std::unordered_map<uint64_t,char> seen;
        seen.reserve(nt0*3);
        for (int t=0;t<nt0;++t)
            for (int e=0;e<3;++e){
                int a=F[t].v[e], b=F[t].v[(e+1)%3];
                uint64_t k=edgeKey(a,b);
                if (seen.emplace(k,1).second) pushEdge(a,b);
            }
    }

    size_t liveTris = nt0;
    while (liveTris > targetTris && !pq.empty()) {
        Cand cd = pq.top(); pq.pop();
        int i=cd.i, j=cd.j;
        if (!alive[i] || !alive[j]) continue;
        if (ver[i]!=cd.vi || ver[j]!=cd.vj) continue;   // stale
        // Collect faces incident to either endpoint; classify shared (will die).
        // Foldover guard: with i moved to target, no surviving face may flip.
        Vec3 np = cd.target;
        bool reject = false;
        auto survivingFaces = [&](int vv, std::vector<int>& out){
            for (int t : vf[vv]) if (!F[t].dead) out.push_back(t);
        };
        std::vector<int> fi, fj; survivingFaces(i,fi); survivingFaces(j,fj);
        // Link condition (keeps the mesh 2-manifold): i and j may share ONLY the
        // vertices opposite the faces on edge (i,j). Any other common neighbour would
        // fuse two distinct edges into one -> a non-manifold (>2 faces) edge.
        {
            std::unordered_map<int,char> nbrI;      // edge-neighbours of i
            std::vector<int> opp;                   // 3rd vertex of each shared face
            for (int t : fi) {
                int a=F[t].v[0],b=F[t].v[1],c=F[t].v[2];
                for (int w : {a,b,c}) if (w!=i) nbrI.emplace(w,1);
                if (a==j||b==j||c==j) for (int w : {a,b,c}) if (w!=i && w!=j) opp.push_back(w);
            }
            for (int t : fj) {
                int a=F[t].v[0],b=F[t].v[1],c=F[t].v[2];
                for (int w : {a,b,c}) {
                    if (w==j || !nbrI.count(w)) continue; // only common neighbours
                    bool isOpp=false; for (int o : opp) if (o==w) { isOpp=true; break; }
                    if (!isOpp) { reject=true; break; }
                }
                if (reject) break;
            }
        }
        if (reject) continue;
        for (int t : fi) {
            int a=F[t].v[0],b=F[t].v[1],c=F[t].v[2];
            if ((a==j)||(b==j)||(c==j)) continue;        // shared face: collapses away
            Vec3 oldn = faceNormalAreaW(a,b,c);
            Vec3 pa=(a==i)?np:P[a], pb=(b==i)?np:P[b], pc=(c==i)?np:P[c];
            Vec3 newn = cross(pb-pa, pc-pa);
            if (dot(oldn,newn) <= 0.0 || length(newn) < 1e-18) { reject=true; break; }
        }
        if (!reject) for (int t : fj) {
            int a=F[t].v[0],b=F[t].v[1],c=F[t].v[2];
            if ((a==i)||(b==i)||(c==i)) continue;
            Vec3 oldn = faceNormalAreaW(a,b,c);
            Vec3 pa=(a==j)?np:P[a], pb=(b==j)?np:P[b], pc=(c==j)?np:P[c];
            Vec3 newn = cross(pb-pa, pc-pa);
            if (dot(oldn,newn) <= 0.0 || length(newn) < 1e-18) { reject=true; break; }
        }
        if (reject) continue;

        // Commit: move i to target, merge j into i.
        P[i] = np;
        add(Q[i], Q[j]);
        alive[j] = 0;
        for (int t : vf[j]) {
            if (F[t].dead) continue;
            int* v = F[t].v;
            bool touchesI = (v[0]==i||v[1]==i||v[2]==i);
            for (int e=0;e<3;++e) if (v[e]==j) v[e]=i;
            if (touchesI || v[0]==v[1] || v[1]==v[2] || v[0]==v[2]) {
                F[t].dead = true; --liveTris;      // degenerate shared face removed
            } else {
                vf[i].push_back(t);
            }
        }
        ++ver[i];
        // Re-push edges from i to its (still-live) neighbours.
        std::unordered_map<int,char> nb;
        for (int t : vf[i]) if (!F[t].dead)
            for (int e=0;e<3;++e){ int w=F[t].v[e]; if (w!=i && alive[w]) nb.emplace(w,1); }
        for (auto& kv : nb) pushEdge(i, kv.first);
    }

    // Compact surviving vertices/faces; re-seat normals as area-weighted geometric
    // normals. These are correct on box-face caps too (the field gradient is not),
    // and respect the outward winding already established by the mesher.
    (void)im;
    std::vector<int> remap(nv, -1);
    Mesh out;
    for (int t=0;t<nt0;++t){
        if (F[t].dead) continue;
        int idx[3];
        for (int e=0;e<3;++e){
            int v=F[t].v[e];
            if (remap[v]<0){ remap[v]=(int)out.pos.size(); out.pos.push_back(P[v]); out.nrm.push_back(Vec3{0,0,0}); }
            idx[e]=remap[v];
        }
        if (idx[0]==idx[1]||idx[1]==idx[2]||idx[0]==idx[2]) continue;
        out.tri.push_back(idx[0]); out.tri.push_back(idx[1]); out.tri.push_back(idx[2]);
    }
    for (size_t t=0; t+2 < out.tri.size(); t+=3) {
        int a=out.tri[t], b=out.tri[t+1], c=out.tri[t+2];
        Vec3 fn = cross(out.pos[b]-out.pos[a], out.pos[c]-out.pos[a]);  // area-weighted
        out.nrm[a]=out.nrm[a]+fn; out.nrm[b]=out.nrm[b]+fn; out.nrm[c]=out.nrm[c]+fn;
    }
    for (auto& n : out.nrm){ double L=length(n); n = (L>1e-30)? n*(1.0/L) : Vec3{0,1,0}; }
    m = std::move(out);
}

// ---- OBJ writer (v / vn / f v//vn), multiple groups ------------------------
inline bool writeObj(const std::string& path,
                     const std::vector<std::pair<std::string, Mesh>>& groups,
                     const std::function<void(const std::string&)>& log) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { if (log) log("[export-mesh] ERROR: cannot open " + path); return false; }
    std::fprintf(f, "# forward-raytracer isosurface export\n");
    size_t vbase = 0, totV = 0, totT = 0;
    for (const auto& g : groups) {
        const Mesh& m = g.second;
        std::fprintf(f, "o %s\n", g.first.c_str());
        for (const auto& p : m.pos) std::fprintf(f, "v %.6g %.6g %.6g\n", p.x, p.y, p.z);
        for (const auto& n : m.nrm) std::fprintf(f, "vn %.5f %.5f %.5f\n", n.x, n.y, n.z);
        for (size_t t = 0; t + 2 < m.tri.size(); t += 3) {
            long a = (long)(vbase + m.tri[t]   + 1);
            long b = (long)(vbase + m.tri[t+1] + 1);
            long c = (long)(vbase + m.tri[t+2] + 1);
            std::fprintf(f, "f %ld//%ld %ld//%ld %ld//%ld\n", a,a, b,b, c,c);
        }
        vbase += m.pos.size();
        totV  += m.pos.size();
        totT  += m.tri.size() / 3;
    }
    std::fclose(f);
    if (log) log("[export-mesh] wrote " + path + "  (" + std::to_string(totV) +
                 " verts, " + std::to_string(totT) + " tris, " +
                 std::to_string(groups.size()) + " objects)");
    return true;
}

} // namespace isomesh
