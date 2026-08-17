#pragma once
// ---------------------------------------------------------------------------
// Light BVH — Conty & Kulla, "Importance Sampling of Many Lights with Adaptive
// Tree Splitting" (Solid Angle / Arnold; the same structure Cycles and PBRT-v4
// use).  Replaces the "connect a shadow ray to EVERY emitter at EVERY vertex"
// splitting estimator with a spatially-aware SELECTION: descend a BVH over the
// emitters choosing children in proportion to a conservative bound on their
// contribution, and return the chosen emitter together with its selection pdf.
//
// Why: the old estimator is O(N_lights) per shading vertex and buys nothing once
// the lights are redundant.  Measured before this file existed (mode R, 256 spp,
// 256^2, same room, same TOTAL flux split across N ceiling panels — see
// scraps/gen_manylights.py): 1 light 0.4 s, 16 lights 3.3 s, 64 lights 15.1 s,
// 256 lights 75.9 s, all at an identical 6.25 % reported noise.  190x the cost
// for the same image.
//
// Unbiasedness.  The tree only ever changes HOW an emitter is chosen, never what
// its connection is worth: neeLight goes from `sum_e w_e` to `w_e / p(e)` for the
// drawn e, and E[w_e/p(e)] = sum_e w_e as long as p(e) > 0 whenever w_e != 0.
// That is why every bound here must be CONSERVATIVE (a cone that contains all the
// normals, an |cos| rather than a clamped cos at the receiver): a bound that is
// too tight would zero the probability of an emitter that can still contribute
// and silently darken the image.  Erring wide only costs a little variance.
//
// ADAPTIVE SPLITTING.  Near the top of the tree a node can subtend a large solid
// angle, and picking one of its children at random is then a bad bet — the two
// halves are genuinely different lights, not two samples of the same one.  The
// walk therefore VISITS BOTH children (contributing probability 1 to each) while
// the node is "too big" at the shading point, and only starts choosing once the
// remaining subtree is far/small enough to be treated as one light.  With the
// threshold wide open every leaf is visited with pdf 1 and the estimator degrades
// EXACTLY to the old sum-over-all-emitters; with it closed the walk costs one
// root-to-leaf descent.  So the old behaviour is the limit of the new one, which
// is what makes small-N scenes safe.
//
// This header is deliberately DEPENDENCY-FREE (only <cmath>/<cstdint>) and is
// written against plain scalar arrays rather than Vec3, so the CUDA translation
// unit — which does not include scene.h — can include and call exactly the same
// traversal code instead of carrying a hand-kept second copy.  The host-side
// BUILDER, which does need Emitter/Scene, lives in lighttree_build.h.
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LT_FN __host__ __device__ inline
#else
  #define LT_FN inline
#endif

// Render-side knobs for the tree, set once by the CLI (-no-lighttree / -lighttree /
// -light-split / -light-samples).  They live HERE, as inline globals, rather than as
// statics in main.cpp, for the same reason `hero::gSplit` does: the CUDA translation
// unit has to read the very same values when it fills DScene, and it sees neither
// main.cpp nor backward.h.  Two copies of a policy flag is two chances for the CPU and
// GPU renders of one scene to disagree.
namespace lt {
inline bool   gEnabled = true;   // false = the exact all-emitters splitting estimator
inline double gSplit   = 1.0;    // adaptive-splitting threshold, (node radius / distance)^2
inline int    gSamples = 8;      // cap on emitters connected per shading vertex
}

// One node of the light BVH. Interior nodes carry two child indices; leaves carry
// one emitter index. Kept as plain doubles/ints with no constructors so the whole
// array can be memcpy'd to the device as-is.
struct LightTreeNode {
    double bmin[3];      // spatial bounds of the subtree's emitters
    double bmax[3];
    double axis[3];      // bounding cone of the subtree's emission axes
    double cosThetaO;    // cos of the half-angle containing every emitter normal
    double cosThetaE;    // cos of the extra spread beyond the normal (0 = Lambertian hemisphere)
    double power;        // total emitted flux in the subtree (the selection weight)
    int left;            // interior: child node index; leaf: -1
    int right;           // interior: child node index; leaf: -1
    int emitter;         // leaf: index into Scene::emitters; interior: -1
};

LT_FN double ltSafeSqrt(double x) { return x <= 0.0 ? 0.0 : std::sqrt(x); }

// Conservative bound on how much this subtree can contribute at a shading point.
//
// Geometry, all bounded rather than evaluated (we have a whole subtree, not one
// light): `d2` is the squared distance to the node centre, floored by the node's
// own size so that standing inside a big node does not produce an infinite score;
// theta_u is the half-angle the node's bounding sphere subtends at p; theta is the
// angle between the emission-cone axis and the direction back to p. Widening theta
// by theta_o (the cone) and theta_u (the node's extent) gives the smallest angle
// off-axis any emitter in the subtree could actually radiate along, and if even
// that is outside the emission spread theta_e the subtree provably contributes
// nothing and may be given probability zero.
//
// `n` is the receiver's shading normal (pass hasN = false at a volume/phase-function
// vertex, which has no normal). The receiver cosine uses |cos| widened by theta_u —
// ABS, not max(.,0), because a one-sided clamp is not a conservative bound for a
// subtree that straddles the horizon.
LT_FN double ltImportance(const LightTreeNode& nd, const double p[3],
                          const double n[3], bool hasN) {
    if (nd.power <= 0.0) return 0.0;
    const double cx = 0.5 * (nd.bmin[0] + nd.bmax[0]);
    const double cy = 0.5 * (nd.bmin[1] + nd.bmax[1]);
    const double cz = 0.5 * (nd.bmin[2] + nd.bmax[2]);
    const double dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    double d2 = dx * dx + dy * dy + dz * dz;
    // Bounding-sphere radius of the node (half the box diagonal).
    const double ex = nd.bmax[0] - nd.bmin[0];
    const double ey = nd.bmax[1] - nd.bmin[1];
    const double ez = nd.bmax[2] - nd.bmin[2];
    const double r2 = 0.25 * (ex * ex + ey * ey + ez * ez);
    // Half the node radius as the distance floor: inside-the-node scores stay
    // large (it IS the nearest light) but finite.
    const double floor2 = 0.25 * r2;
    if (d2 < floor2) d2 = floor2;
    if (d2 <= 0.0) return nd.power * 1e12;      // degenerate point light at p

    // cos/sin of theta_u, the half-angle the node subtends at p.
    double cosThetaU, sinThetaU;
    if (d2 <= r2) { cosThetaU = -1.0; sinThetaU = 0.0; }   // p is inside the sphere
    else {
        const double s2 = r2 / d2;
        sinThetaU = ltSafeSqrt(s2);
        cosThetaU = ltSafeSqrt(1.0 - s2);
    }

    const double invD = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + 1e-300);
    const double wx = dx * invD, wy = dy * invD, wz = dz * invD;   // node -> p

    // theta = angle(axis, w). Widen by theta_o then by theta_u; both widenings are
    // angle subtractions done with the cos/sin addition formulae so no acos is needed.
    double cosTheta = nd.axis[0] * wx + nd.axis[1] * wy + nd.axis[2] * wz;
    if (cosTheta > 1.0) cosTheta = 1.0; else if (cosTheta < -1.0) cosTheta = -1.0;
    const double sinTheta = ltSafeSqrt(1.0 - cosTheta * cosTheta);
    const double sinThetaO = ltSafeSqrt(1.0 - nd.cosThetaO * nd.cosThetaO);

    double cosTP, sinTP;                                   // theta' = theta - theta_o
    if (cosTheta > nd.cosThetaO) { cosTP = 1.0; sinTP = 0.0; }   // already inside the cone
    else {
        cosTP = cosTheta * nd.cosThetaO + sinTheta * sinThetaO;
        sinTP = sinTheta * nd.cosThetaO - cosTheta * sinThetaO;
        if (sinTP < 0.0) sinTP = 0.0;
    }
    double cosTX;                                          // theta'' = theta' - theta_u
    if (cosTP > cosThetaU) cosTX = 1.0;
    else cosTX = cosTP * cosThetaU + sinTP * sinThetaU;
    if (cosTX <= nd.cosThetaE) return 0.0;                 // provably no emission this way
    if (cosTX > 1.0) cosTX = 1.0;

    double imp = nd.power * cosTX / d2;
    if (hasN) {
        double ci = n[0] * wx + n[1] * wy + n[2] * wz;
        ci = ci < 0.0 ? -ci : ci;                          // |cos|: conservative
        if (ci > 1.0) ci = 1.0;
        const double si = ltSafeSqrt(1.0 - ci * ci);
        double ciw;                                        // widen by theta_u
        if (ci > cosThetaU) ciw = 1.0;
        else ciw = ci * cosThetaU + si * sinThetaU;
        if (ciw <= 0.0) return 0.0;
        imp *= ciw;
    }
    return imp;
}

// One drawn emitter and the probability the walk had of drawing it.
struct LtSample { int emitter; double pdf; };

// Should the walk visit BOTH children of this node instead of choosing one?
// The test is purely geometric: a node whose bounding sphere still subtends a
// sizeable solid angle at the shading point is not "one light" yet, so averaging
// its halves under a single random choice would throw away real structure.
// `thresh` is (radius/distance)^2; 0 disables splitting entirely (pure selection),
// a huge value splits the whole tree (= the old sum-over-all-emitters estimator).
LT_FN bool ltShouldSplit(const LightTreeNode& nd, const double p[3], double thresh) {
    if (thresh <= 0.0) return false;
    const double ex = nd.bmax[0] - nd.bmin[0];
    const double ey = nd.bmax[1] - nd.bmin[1];
    const double ez = nd.bmax[2] - nd.bmin[2];
    const double r2 = 0.25 * (ex * ex + ey * ey + ez * ez);
    if (r2 <= 0.0) return false;
    const double cx = 0.5 * (nd.bmin[0] + nd.bmax[0]);
    const double cy = 0.5 * (nd.bmin[1] + nd.bmax[1]);
    const double cz = 0.5 * (nd.bmin[2] + nd.bmax[2]);
    const double dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= 0.0) return true;
    return r2 / d2 > thresh;
}

// Walk the tree and fill `out` with the emitters to connect to this vertex, each
// with the probability the walk had of reaching it. Returns how many were written.
//
// Iterative with an explicit stack (recursion is a poor idea in a CUDA kernel).
// A stack entry carries the node and the pdf accumulated on the way to it; a split
// pushes both children at the SAME pdf (probability 1 each), a stochastic step
// pushes one child at pdf * p(child). `maxOut` bounds the work per vertex — once
// the output is full the walk stops splitting and finishes the pending entries by
// pure selection, so the result stays unbiased rather than truncated.
//
// RNG is a template parameter so the host Pcg32 and the device DRng both work with
// no second copy of this function; both expose `uniform()` in [0,1).
//
// STACK bounds the explicit traversal stack. A split is only taken when the output
// still has room for BOTH halves, so `sp` can never exceed `maxOut` — STACK only has
// to be maxOut+2. It is a template parameter because the device wants it small: the
// array is a per-thread local-memory frame in the megakernel, and 64 entries of
// {int,double} is a kilobyte per thread that would be paid by every thread whether it
// touches a light tree or not.
template <class RNG, int STACK = 64>
LT_FN int ltSample(const LightTreeNode* nodes, int root, const double p[3],
                   const double n[3], bool hasN, double splitThresh,
                   int maxOut, LtSample* out, RNG& rng) {
    if (!nodes || root < 0 || maxOut <= 0) return 0;
    struct Entry { int node; double pdf; };
    Entry stack[STACK];
    int sp = 0, nOut = 0;
    stack[sp++] = Entry{root, 1.0};
    while (sp > 0) {
        Entry e = stack[--sp];
        const LightTreeNode& nd = nodes[e.node];
        if (nd.emitter >= 0) {                       // leaf
            if (nOut < maxOut) out[nOut++] = LtSample{nd.emitter, e.pdf};
            continue;
        }
        const int li = nd.left, ri = nd.right;
        if (li < 0) continue;
        if (ri < 0) { stack[sp++] = Entry{li, e.pdf}; continue; }
        // Room to split? Need a free stack slot AND a free output slot, since a
        // split can only pay off if both halves can still be reported.
        const bool canSplit = (sp + 2 <= STACK) && (nOut + sp + 2 <= maxOut) &&
                              ltShouldSplit(nd, p, splitThresh);
        if (canSplit) {
            stack[sp++] = Entry{li, e.pdf};
            stack[sp++] = Entry{ri, e.pdf};
            continue;
        }
        const double iL = ltImportance(nodes[li], p, n, hasN);
        const double iR = ltImportance(nodes[ri], p, n, hasN);
        const double sum = iL + iR;
        if (!(sum > 0.0)) continue;                  // neither half can contribute
        const double pL = iL / sum;
        const double xi = (double)rng.uniform();
        if (xi < pL) stack[sp++] = Entry{li, e.pdf * pL};
        else         stack[sp++] = Entry{ri, e.pdf * (1.0 - pL)};
    }
    return nOut;
}
