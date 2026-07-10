// Binned-SAH bounding volume hierarchy over an arbitrary primitive list.
// The BVH stores only bounding boxes and a primitive-index permutation; the
// caller supplies a leaf callback that intersects the actual primitive. This
// keeps the acceleration structure decoupled from primitive types (tris/spheres
// today, meshes tomorrow). It must not change the image vs the linear scan.
#pragma once
#include <vector>
#include <algorithm>
#include <cfloat>
#include "linalg.h"
#include "geometry.h"

inline double vget(const Vec3& v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

struct Aabb {
    Vec3 lo{ DBL_MAX,  DBL_MAX,  DBL_MAX};
    Vec3 hi{-DBL_MAX, -DBL_MAX, -DBL_MAX};

    void expand(const Vec3& p) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    void expand(const Aabb& b) { expand(b.lo); expand(b.hi); }
    Vec3 center() const { return (lo + hi) * 0.5; }
    double area() const {
        Vec3 d = hi - lo;
        if (d.x < 0 || d.y < 0 || d.z < 0) return 0.0;   // empty
        return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    int largestAxis() const {
        Vec3 d = hi - lo;
        if (d.x >= d.y && d.x >= d.z) return 0;
        return d.y >= d.z ? 1 : 2;
    }
    // Slab test against [tmin, tmax]. Returns true if the ray overlaps the box;
    // tEnter is the near intersection distance (>= tmin).
    bool hit(const Ray& r, const Vec3& invD, double tmin, double tmax, double& tEnter) const {
        double te = tmin, tx = tmax;
        for (int a = 0; a < 3; ++a) {
            double o = vget(r.o, a), id = vget(invD, a);
            double t0 = (vget(lo, a) - o) * id;
            double t1 = (vget(hi, a) - o) * id;
            if (t0 > t1) std::swap(t0, t1);
            te = t0 > te ? t0 : te;
            tx = t1 < tx ? t1 : tx;
            if (tx < te) return false;
        }
        tEnter = te;
        return true;
    }
};

struct BvhNode {
    Aabb box;
    int left = -1, right = -1;   // internal children
    int first = 0, count = 0;    // leaf: primIdx[first .. first+count)
    bool isLeaf() const { return count > 0; }
};

// Diagnostic counters for a single traversal (see -bvhstats). Optional; passing
// nullptr keeps the hot path branch-free in normal renders.
struct TraversalStats { long long nodeVisits = 0; long long leafTests = 0; };

struct Bvh {
    std::vector<BvhNode> nodes;
    std::vector<int> primIdx;    // permutation of [0, nPrims)
    static constexpr int LEAF_SIZE = 4;
    static constexpr int NUM_BINS = 16;

    struct BuildPrim { Aabb box; Vec3 centroid; int idx; };

    void build(const std::vector<Aabb>& boxes) {
        int n = (int)boxes.size();
        nodes.clear();
        primIdx.clear();
        if (n == 0) return;
        std::vector<BuildPrim> bp(n);
        for (int i = 0; i < n; ++i) { bp[i].box = boxes[i]; bp[i].centroid = boxes[i].center(); bp[i].idx = i; }
        nodes.reserve(2 * n);
        buildRecursive(bp, 0, n);
        primIdx.resize(n);
        for (int i = 0; i < n; ++i) primIdx[i] = bp[i].idx;
    }

    int buildRecursive(std::vector<BuildPrim>& bp, int start, int end) {
        int nodeIdx = (int)nodes.size();
        nodes.push_back(BvhNode{});
        Aabb bounds, cbounds;
        for (int i = start; i < end; ++i) { bounds.expand(bp[i].box); cbounds.expand(bp[i].centroid); }
        int count = end - start;

        auto makeLeaf = [&]() {
            BvhNode& node = nodes[nodeIdx];
            node.box = bounds; node.first = start; node.count = count; node.left = node.right = -1;
        };
        if (count <= LEAF_SIZE) { makeLeaf(); return nodeIdx; }

        int axis = cbounds.largestAxis();
        double cLo = vget(cbounds.lo, axis), cHi = vget(cbounds.hi, axis);
        if (cHi - cLo < 1e-12) { makeLeaf(); return nodeIdx; } // degenerate centroids

        // Bin primitives by centroid along the chosen axis.
        struct Bin { Aabb box; int count = 0; };
        Bin bins[NUM_BINS];
        double scale = NUM_BINS / (cHi - cLo);
        for (int i = start; i < end; ++i) {
            int b = (int)((vget(bp[i].centroid, axis) - cLo) * scale);
            if (b < 0) b = 0; if (b >= NUM_BINS) b = NUM_BINS - 1;
            bins[b].count++; bins[b].box.expand(bp[i].box);
        }
        // Suffix (right side) area/count for each split plane.
        double rightArea[NUM_BINS]; int rightCount[NUM_BINS];
        Aabb rAcc; int rC = 0;
        for (int b = NUM_BINS - 1; b >= 1; --b) {
            rAcc.expand(bins[b].box); rC += bins[b].count;
            rightArea[b] = rAcc.area(); rightCount[b] = rC;
        }
        // Prefix (left side) sweep -> SAH cost, pick best split plane.
        Aabb lAcc; int lC = 0; double bestCost = DBL_MAX; int bestSplit = -1;
        for (int b = 0; b < NUM_BINS - 1; ++b) {
            lAcc.expand(bins[b].box); lC += bins[b].count;
            if (lC == 0 || rightCount[b + 1] == 0) continue;
            double cost = lC * lAcc.area() + rightCount[b + 1] * rightArea[b + 1];
            if (cost < bestCost) { bestCost = cost; bestSplit = b; }
        }
        // Split down to LEAF_SIZE regardless of whether the SAH split "improves"
        // on the leaf cost. Object-SAH on ring-like shapes (e.g. a torus) hits a
        // top-level pathology: splitting the ring through its center yields two
        // C-shaped halves whose AABBs each nearly equal the whole box, so every
        // split's cost ~ the leaf cost. A "split only if it lowers SAH" test would
        // then give up immediately and leave enormous leaves (measured: one 9334-
        // primitive leaf). Instead use SAH only to CHOOSE the plane and fall back
        // to a median split so recursion always makes progress.
        int midIdx = start;
        if (bestSplit >= 0) {
            auto mid = std::partition(bp.begin() + start, bp.begin() + end, [&](const BuildPrim& p) {
                int b = (int)((vget(p.centroid, axis) - cLo) * scale);
                if (b < 0) b = 0; if (b >= NUM_BINS) b = NUM_BINS - 1;
                return b <= bestSplit;
            });
            midIdx = (int)(mid - bp.begin());
        }
        if (midIdx == start || midIdx == end) {
            // No usable SAH split (or a degenerate partition): median-split by
            // centroid along the chosen axis.
            midIdx = (start + end) / 2;
            std::nth_element(bp.begin() + start, bp.begin() + midIdx, bp.begin() + end,
                             [&](const BuildPrim& a, const BuildPrim& b) {
                                 return vget(a.centroid, axis) < vget(b.centroid, axis);
                             });
        }

        int l = buildRecursive(bp, start, midIdx);
        int r = buildRecursive(bp, midIdx, end);
        // nodes may have reallocated during recursion; index by nodeIdx.
        BvhNode& node = nodes[nodeIdx];
        node.box = bounds; node.left = l; node.right = r; node.count = 0;
        return nodeIdx;
    }

    // Nearest-hit traversal. leafTest(primIndex, tMax&) intersects the primitive
    // and, on a closer hit, updates tMax (used to prune farther nodes).
    template <class LeafFn>
    void traverseClosest(const Ray& r, double tmin, double& tMax, LeafFn&& leafTest,
                         TraversalStats* stats = nullptr) const {
        if (nodes.empty()) return;
        Vec3 invD{1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z};
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while (sp) {
            const BvhNode& n = nodes[stack[--sp]];
            double tEnter;
            if (stats) stats->nodeVisits++;
            if (!n.box.hit(r, invD, tmin, tMax, tEnter)) continue;
            if (n.isLeaf()) {
                if (stats) stats->leafTests += n.count;
                for (int i = 0; i < n.count; ++i) leafTest(primIdx[n.first + i], tMax);
            } else {
                // Front-to-back: cull children against the current tMax at push
                // time and descend the nearer child first (push far, then near),
                // so a near hit tightens tMax before the far subtree is visited.
                double tL, tR;
                bool hL = nodes[n.left].box.hit(r, invD, tmin, tMax, tL);
                bool hR = nodes[n.right].box.hit(r, invD, tmin, tMax, tR);
                if (hL && hR) {
                    if (tL <= tR) { stack[sp++] = n.right; stack[sp++] = n.left; }
                    else          { stack[sp++] = n.left;  stack[sp++] = n.right; }
                } else if (hL) {
                    stack[sp++] = n.left;
                } else if (hR) {
                    stack[sp++] = n.right;
                }
            }
        }
    }

    // Any-hit (occlusion) traversal. leafHit(primIndex) returns true if the
    // primitive blocks the segment; traversal stops at the first blocker.
    template <class LeafFn>
    bool traverseAny(const Ray& r, double tmin, double tMax, LeafFn&& leafHit) const {
        if (nodes.empty()) return false;
        Vec3 invD{1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z};
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while (sp) {
            const BvhNode& n = nodes[stack[--sp]];
            double tEnter;
            if (!n.box.hit(r, invD, tmin, tMax, tEnter)) continue;
            if (n.isLeaf()) {
                for (int i = 0; i < n.count; ++i) if (leafHit(primIdx[n.first + i])) return true;
            } else {
                double tc;
                if (nodes[n.left].box.hit(r, invD, tmin, tMax, tc))  stack[sp++] = n.left;
                if (nodes[n.right].box.hit(r, invD, tmin, tMax, tc)) stack[sp++] = n.right;
            }
        }
        return false;
    }
};
