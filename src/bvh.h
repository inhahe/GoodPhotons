// Binned-SAH bounding volume hierarchy over an arbitrary primitive list.
// The BVH stores only bounding boxes and a primitive-index permutation; the
// caller supplies a leaf callback that intersects the actual primitive. This
// keeps the acceleration structure decoupled from primitive types (tris/spheres
// today, meshes tomorrow). It must not change the image vs the linear scan.
#pragma once
#include <vector>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "linalg.h"
#include "geometry.h"
#include "parallel.h"    // ft::stopRequested — cooperative `-stop` during a long build

// Read once: a scene builds many trees and getenv is not free in a loop.
inline bool bvhTimeEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_BVH_TIME");
        return e && *e && *e != '0';
    }();
    return on;
}

// FTRACE_BVH_THREADS=<n> caps the build's thread count, for scaling measurements. 0/unset =
// hardware. Exists to tell apart the two things that can limit a parallel build -- memory
// bandwidth saturation, which flattens the curve early and is not worth fighting, from
// per-fork copying overhead, which can make high thread counts actively worse.
inline int bvhThreadOverride() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_BVH_THREADS");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return n;
}

// FTRACE_BVH_VERIFY=1 makes every build re-run itself serially and compare, so the
// parallel==serial invariant is checked on REAL scene geometry rather than on the synthetic
// primitives of -checkbvhparallel. Roughly triples build time, so it is a debugging switch,
// not something to leave on; but it is the only check that covers the trees an actual scene
// produces — degenerate centroids, coincident boxes, hair segments, split beams — none of
// which a generated point cloud reproduces faithfully.
inline bool bvhVerifyEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_BVH_VERIFY");
        return e && *e && *e != '0';
    }();
    return on;
}

// Branch-free component fetch (Vec3 is standard-layout with x,y,z contiguous —
// same trick as Vec3::operator[]); the old two-branch ternary showed up in
// profiles via the slab test's 6 calls per node.
inline double vget(const Vec3& v, int a) { return (&v.x)[a]; }

struct Aabb {
    Vec3 lo{ DBL_MAX,  DBL_MAX,  DBL_MAX};
    Vec3 hi{-DBL_MAX, -DBL_MAX, -DBL_MAX};

    void expand(const Vec3& p) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    void expand(const Aabb& b) { expand(b.lo); expand(b.hi); }
    Vec3 center() const { return (lo + hi) * 0.5; }
    // Squared distance from `p` to the nearest point of the box; 0 when inside. The
    // building block of the sphere-overlap traversal below, and exact rather than a
    // centre-plus-radius approximation — an approximate reject would silently drop
    // primitives that really do intersect the ball, which is the one failure a
    // footprint measurement cannot tolerate (it would under-count area and so
    // over-brighten, indistinguishably from the defect it is meant to fix).
    double dist2To(const Vec3& p) const {
        double d = 0.0, v;
        v = p.x < lo.x ? lo.x - p.x : (p.x > hi.x ? p.x - hi.x : 0.0); d += v * v;
        v = p.y < lo.y ? lo.y - p.y : (p.y > hi.y ? p.y - hi.y : 0.0); d += v * v;
        v = p.z < lo.z ? lo.z - p.z : (p.z > hi.z ? p.z - hi.z : 0.0); d += v * v;
        return d;
    }
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
    // tEnter is the near intersection distance (>= tmin). Fully unrolled over the
    // three axes (was a for-loop with vget component indirection — this is one of
    // the hottest functions in every CPU mode, and the straight-line form keeps
    // each axis's operands in registers). Per axis the arithmetic, the swap, and
    // the early-out compare are IDENTICAL to the old loop body in the same x,y,z
    // order, so results are bit-identical.
    bool hit(const Ray& r, const Vec3& invD, double tmin, double tmax, double& tEnter) const {
        double te = tmin, tx = tmax;
        double t0 = (lo.x - r.o.x) * invD.x;
        double t1 = (hi.x - r.o.x) * invD.x;
        if (t0 > t1) { double tt = t0; t0 = t1; t1 = tt; }
        te = t0 > te ? t0 : te;
        tx = t1 < tx ? t1 : tx;
        if (tx < te) return false;
        t0 = (lo.y - r.o.y) * invD.y;
        t1 = (hi.y - r.o.y) * invD.y;
        if (t0 > t1) { double tt = t0; t0 = t1; t1 = tt; }
        te = t0 > te ? t0 : te;
        tx = t1 < tx ? t1 : tx;
        if (tx < te) return false;
        t0 = (lo.z - r.o.z) * invD.z;
        t1 = (hi.z - r.o.z) * invD.z;
        if (t0 > t1) { double tt = t0; t0 = t1; t1 = tt; }
        te = t0 > te ? t0 : te;
        tx = t1 < tx ? t1 : tx;
        if (tx < te) return false;
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

    // Set when a `-stop` landed mid-build, so the caller can tell a finished tree from an
    // abandoned one without inspecting it. The tree is still STRUCTURALLY VALID either way
    // (see buildRecursive) — this says only that it is coarse, not that it is unusable.
    bool stopped = false;

    // `maxThreads` exists for the parallel-vs-serial self-test (-checkbvhparallel), which has
    // to build the SAME primitives both ways inside one process to compare the node arrays.
    // Default -1 = use the hardware count.
    void build(const std::vector<Aabb>& boxes, int maxThreads = -1) {
        int n = (int)boxes.size();
        nodes.clear();
        primIdx.clear();
        stopped = false;
        if (n == 0) return;
        // FTRACE_BVH_TIME=1 reports every build costing more than a tenth of a second.
        // Off by default because a scene builds many small trees and the noise would bury
        // the one that matters; on, it is the only way to see this cost at all, since the
        // build happens before the first pixel and no existing line reports it. Measure with
        // this rather than inferring from wall clock, which on this machine varied 4.5x across
        // one session and 2x even idle, because a concurrent render moves it. The thread count
        // it reports is the CAP, not an observed occupancy: a tree too small or too lopsided
        // to fork simply never spends the budget.
        const bool timeIt = bvhTimeEnabled();
        const auto t0 = timeIt ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        std::vector<BuildPrim> bp(n);
        for (int i = 0; i < n; ++i) { bp[i].box = boxes[i]; bp[i].centroid = boxes[i].center(); bp[i].idx = i; }
        nodes.reserve(2 * n);
        // One less than the hardware count: this thread is already one of the workers.
        unsigned hw = std::thread::hardware_concurrency();
        int threads = (hw < 2) ? 1 : (int)hw;
        if (maxThreads > 0) threads = maxThreads;
        else if (bvhThreadOverride() > 0) threads = bvhThreadOverride();
        BuildCtx ctx;
        ctx.budget.store(threads - 1, std::memory_order_relaxed);
        unsigned tick = 0;
        buildRange(bp, 0, n, nodes, tick, ctx);
        stopped = ctx.stop.load(std::memory_order_relaxed);
        primIdx.resize(n);
        for (int i = 0; i < n; ++i) primIdx[i] = bp[i].idx;
        // Shadow serial rebuild, compared node for node. Skipped when the build never forked
        // (threads == 1, or a `-stop` landed and made the tree coarse by a path that depends
        // on timing and so legitimately differs between runs).
        if (bvhVerifyEnabled() && threads > 1 && !stopped) {
            std::vector<BvhNode> parNodes = nodes;
            std::vector<int> parPrim = primIdx;
            std::vector<BuildPrim> bp2(n);
            for (int i = 0; i < n; ++i) {
                bp2[i].box = boxes[i]; bp2[i].centroid = boxes[i].center(); bp2[i].idx = i;
            }
            nodes.clear(); nodes.reserve(2 * n);
            BuildCtx sctx;                       // budget 0 == never forks == the serial build
            unsigned stick = 0;
            buildRange(bp2, 0, n, nodes, stick, sctx);
            primIdx.resize(n);
            for (int i = 0; i < n; ++i) primIdx[i] = bp2[i].idx;
            size_t bad = 0;
            if (nodes.size() != parNodes.size() || primIdx.size() != parPrim.size()) {
                bad = (size_t)-1;
            } else {
                for (size_t i = 0; i < nodes.size(); ++i) {
                    const BvhNode& x = nodes[i];
                    const BvhNode& y = parNodes[i];
                    if (x.left != y.left || x.right != y.right || x.first != y.first ||
                        x.count != y.count || std::memcmp(&x.box, &y.box, sizeof(Aabb)) != 0) ++bad;
                }
                for (size_t i = 0; i < primIdx.size(); ++i)
                    if (primIdx[i] != parPrim[i]) ++bad;
            }
            if (bad == (size_t)-1)
                std::printf("[bvh-verify] FAIL: %d prims -- serial %zu nodes vs parallel %zu\n",
                            n, nodes.size(), parNodes.size());
            else if (bad)
                std::printf("[bvh-verify] FAIL: %d prims -- %zu of %zu nodes/indices differ\n",
                            n, bad, nodes.size());
            else
                std::printf("[bvh-verify] ok: %d prims, %zu nodes identical\n", n, nodes.size());
            nodes.swap(parNodes); primIdx.swap(parPrim);   // keep the parallel result
        }
        if (timeIt) {
            const double el = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0).count();
            if (el >= 0.1)
                std::printf("[bvh] %d prims -> %zu nodes in %.2f s, up to %d threads "
                            "(%.1f MB of BuildPrim)\n", n, nodes.size(), el, threads,
                            (double)(n * sizeof(BuildPrim)) / (1024.0 * 1024.0));
        }
    }

    // PARALLEL BUILD. Left and right recursions own DISJOINT ranges of `bp` (std::partition
    // has already split them), so they can run at once without touching each other's
    // primitives. What they cannot share is the node array: a node's index is its position at
    // push time, so two threads appending to one vector would interleave and the layout would
    // depend on who won.
    //
    // BIT-IDENTITY comes from giving each side a PRIVATE buffer and splicing afterwards.
    // Sequentially the array reads: parent at k, the whole left subtree at k+1.., then the
    // whole right subtree. Splicing lbuf at k+1 and rbuf at k+1+lbuf.size() reproduces exactly
    // that, provided every internal node's child indices are shifted by its buffer's base.
    // Leaves carry `first`/`count` into `bp`, which are absolute and need no remap.
    //
    // The budget, not a fixed fork depth, is what bounds the threads. A fixed depth is easy to
    // reason about but balances badly: subtree sizes differ by orders of magnitude, so depth 4
    // hands one thread a tenth of the tree and eleven threads nothing. Spawning depth-first
    // while a counter allows it lets the split follow the tree's real shape, and the counter
    // caps total threads at hardware_concurrency no matter how lopsided it gets.
    // Deliberately NOT members. A std::atomic member deletes Bvh's implicit copy-assignment,
    // and scene.h:2508 assigns one Bvh to another -- so the build state lives in build()'s frame
    // and travels down the recursion by pointer instead. Caught at compile time, which is the
    // good case; a Bvh that silently stopped being copyable would have been a worse bug.
    struct BuildCtx { std::atomic<int> budget{0}; std::atomic<bool> stop{false}; };

    static void appendRemap(std::vector<BvhNode>& out, const std::vector<BvhNode>& buf) {
        const int base = (int)out.size();
        for (const BvhNode& src : buf) {
            BvhNode n = src;
            if (n.count == 0) { n.left += base; n.right += base; }   // internal: shift children
            out.push_back(n);
        }
    }

    // `tick` is per-call rather than a shared member so the stop poll costs no atomic in the
    // hot path. Each parallel task therefore polls on its own schedule, which changes only how
    // fast a `-stop` lands, never the tree: when nothing stops, the flag is never read true.
    int buildRange(std::vector<BuildPrim>& bp, int start, int end,
                   std::vector<BvhNode>& nodes, unsigned& m_pollTick, BuildCtx& ctx) {
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

        // Cooperative `-stop` seam. A beam-map BVH over millions of sub-beams is tens of
        // seconds of one thread doing nothing observable, and it used to be the last block in
        // the pre-gather region that a stop could not touch.
        //
        // Bailing turns the CURRENT range into a leaf instead of abandoning the tree half
        // written, which matters: the result is a perfectly VALID BVH — every primitive is
        // still reachable, every node still has correct bounds — merely a coarse one that
        // would traverse slowly. So there is no window in which a partially built tree can be
        // traversed and give a wrong answer, and callers that discard on stop (the beam upload
        // in render_cuda.cu) and callers that do not are both safe. Unwinding costs O(depth)
        // further calls, each of which takes this same exit.
        //
        // Polled every 256 nodes rather than every node because `stopRequested` is an indirect
        // call through a relaxed atomic slot, and a 5M-node build would make five million of
        // them. Each poll is amortised over at least 256 * LEAF_SIZE primitives of binning
        // work, so the interval is free and the latency is still sub-millisecond.
        if (!ctx.stop.load(std::memory_order_relaxed) && (++m_pollTick & 255) == 0 &&
            ft::stopRequested())
            ctx.stop.store(true, std::memory_order_relaxed);
        if (ctx.stop.load(std::memory_order_relaxed)) { makeLeaf(); return nodeIdx; }

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

        int l, r;
        // Only fork where the subtree is big enough to pay for a thread, and only while the
        // budget allows. fetch_sub returns the PREVIOUS value, so `> 0` is the test for having
        // actually claimed one; when it fails the decrement is undone immediately.
        constexpr int PAR_MIN = 50000;          // prims below which a thread costs more than it saves
        bool forked = false;
        if ((end - start) >= PAR_MIN) {
            if (ctx.budget.fetch_sub(1, std::memory_order_relaxed) > 0) forked = true;
            else ctx.budget.fetch_add(1, std::memory_order_relaxed);
        }
        if (forked) {
            std::vector<BvhNode> lbuf, rbuf;
            // Reserve, or each side's buffer reallocates its way up from nothing -- ~22
            // doublings for a multi-million-node subtree, every one of them copying what it
            // already holds. The node:prim ratio is 0.644 on both callers measured (scene
            // 1589917/2469624, beam 2389129/3709615), so one node per primitive is comfortably
            // above the true size without over-committing.
            lbuf.reserve((size_t)(midIdx - start));
            rbuf.reserve((size_t)(end - midIdx));
            unsigned lt = 0, rt = 0;
            // The spawned thread takes the LEFT half and this one continues with the right, so
            // the caller's stack keeps doing useful work instead of blocking on a join.
            std::thread th([&] { buildRange(bp, start, midIdx, lbuf, lt, ctx); });
            buildRange(bp, midIdx, end, rbuf, rt, ctx);
            th.join();
            ctx.budget.fetch_add(1, std::memory_order_relaxed);
            // `nodes` still holds exactly this parent as its last element, so the two bases
            // below are the same positions the sequential build would have used.
            l = (int)nodes.size(); appendRemap(nodes, lbuf);
            r = (int)nodes.size(); appendRemap(nodes, rbuf);
        } else {
            l = buildRange(bp, start, midIdx, nodes, m_pollTick, ctx);
            r = buildRange(bp, midIdx, end, nodes, m_pollTick, ctx);
        }
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
        // Children are slab-tested once at push time and their tEnter recorded with
        // the stack entry; the pop-time 6-plane retest is replaced by the exactly-
        // equivalent scalar prune tEnter > tMax. (hit() seeds te with tmin only, so
        // for a node that passed the push test, a retest against a since-shrunk
        // tMax passes iff tEnter <= tMax — one compare instead of a slab test.)
        int stack[64]; double tStack[64]; int sp = 0;
        double tRoot;
        if (!nodes[0].box.hit(r, invD, tmin, tMax, tRoot)) {
            if (stats) stats->nodeVisits++;   // the root "pop" the old loop counted
            return;
        }
        stack[0] = 0; tStack[0] = tRoot; sp = 1;
        while (sp) {
            --sp;
            if (stats) stats->nodeVisits++;
            if (tStack[sp] > tMax) continue;
            const BvhNode& n = nodes[stack[sp]];
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
                    if (tL <= tR) { stack[sp] = n.right; tStack[sp] = tR; ++sp;
                                    stack[sp] = n.left;  tStack[sp] = tL; ++sp; }
                    else          { stack[sp] = n.left;  tStack[sp] = tL; ++sp;
                                    stack[sp] = n.right; tStack[sp] = tR; ++sp; }
                } else if (hL) {
                    stack[sp] = n.left;  tStack[sp] = tL; ++sp;
                } else if (hR) {
                    stack[sp] = n.right; tStack[sp] = tR; ++sp;
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
        // tMax never shrinks in any-hit traversal, so a child that passed its
        // push-time slab test cannot fail the identical pop-time retest — test the
        // root once and drop the per-pop retest entirely.
        double tRoot;
        if (!nodes[0].box.hit(r, invD, tmin, tMax, tRoot)) return false;
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while (sp) {
            const BvhNode& n = nodes[stack[--sp]];
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

    // Sphere-overlap traversal: visit every primitive whose leaf belongs to a node box
    // intersecting the ball |x - c| <= r. Unlike the two traversals above this has no
    // ordering and no early exit — every candidate is reported, because the caller is
    // measuring an AREA and an area needs all of its pieces.
    //
    // WHY IT EXISTS (M-GATHERAREA). The gather divides by pi r^2, the area of a full
    // disc, while collecting only from the same-facing surface actually present inside
    // the ball. Where those differ the estimate is wrong, and on a tangle it is wrong by
    // tens of percent in a direction the shipped gates can only choose between: measured
    // on `fur_creature`'s belly, -33.6 % with the fiber gate and +46.8 % without, with
    // the truth bracketed in between and no setting reaching it. Every attempt to infer
    // the footprint from a PHOTON statistic has failed for the reason that entry gives —
    // the probe sees the nearest layer while the query gathers from the whole ball — so
    // the footprint has to be measured from the GEOMETRY, and that starts here.
    //
    // `leafFn(primIndex)` is called once per candidate primitive. Candidates are a
    // superset of the true overlap set: a box can intersect the ball when its primitive
    // does not, so the caller must still test the primitive itself. Erring that way round
    // is deliberate — a false candidate costs a test, a missed one costs correctness.
    template <class LeafFn>
    void traverseSphere(const Vec3& c, double r, LeafFn&& leafFn) const {
        // r == 0 is a legitimate degenerate query (is this point inside anything?) and
        // falls out of dist2To correctly, so only a negative radius is rejected.
        if (nodes.empty() || !(r >= 0.0)) return;
        const double r2 = r * r;
        if (nodes[0].box.dist2To(c) > r2) return;
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while (sp) {
            const BvhNode& n = nodes[stack[--sp]];
            if (n.isLeaf()) {
                for (int i = 0; i < n.count; ++i) leafFn(primIdx[n.first + i]);
            } else {
                // Guard the push: `stack[64]` is the same depth budget the ray
                // traversals use, but they shrink tMax and this cannot, so a pathological
                // tree could in principle keep both children live at every level. Drop
                // rather than overrun — a dropped subtree under-counts area, which the
                // caller can at least detect as a coverage above 1.
                if (sp + 2 <= 64) {
                    if (nodes[n.left].box.dist2To(c)  <= r2) stack[sp++] = n.left;
                    if (nodes[n.right].box.dist2To(c) <= r2) stack[sp++] = n.right;
                }
            }
        }
    }
};

// `-checkspherequery`: brute-force self-test for traverseSphere. Deterministic, needs no
// scene, and checks the ONE invariant the footprint measurement depends on — every
// primitive whose own box meets the ball is reported. A traversal that quietly skipped
// some would under-count area and so OVER-brighten the gather, which is indistinguishable
// from the defect it is being built to fix; a bug there would be attributed to the
// estimator for a long time before anyone suspected the query. So the check is written
// now, before a single caller exists, rather than after a measurement goes wrong.
// PARALLEL BUILD == SERIAL BUILD, asserted on the data structure rather than on an image.
// An image comparison cannot do this job on the GPU (accumulation order leaves a ~1e-7 floor),
// and on the CPU it would only prove the tree is *equivalent*, not that it is the SAME tree --
// a differently-shaped but still-correct BVH would pass while quietly changing traversal order
// and every downstream `-bvhstats` number. So compare the arrays directly.
//
// The primitive set is deliberately awkward: sizes spanning three orders of magnitude and a
// clustered distribution, because the fork path only triggers on large lopsided subtrees and a
// tidy uniform cloud would exercise the serial path almost everywhere and pass vacuously.
inline bool bvhParallelSelfTest() {
    uint64_t st = 0xD1B54A32D192ED03ull;
    auto rnd = [&]() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    // Large enough to cross PAR_MIN (50000) several times over, or the test proves nothing.
    const int N = 400000;
    std::vector<Aabb> boxes(N);
    for (int i = 0; i < N; ++i) {
        const double cluster = (i % 7 == 0) ? 40.0 : 1.0;      // lopsided, so splits are uneven
        const Vec3 c{(rnd() - 0.5) * cluster, (rnd() - 0.5) * cluster, (rnd() - 0.5) * cluster};
        const double r = 0.001 + rnd() * rnd() * 1.0;          // three decades of size
        boxes[i].expand(Vec3{c.x - r, c.y - r, c.z - r});
        boxes[i].expand(Vec3{c.x + r, c.y + r, c.z + r});
    }
    Bvh a, b;
    a.build(boxes, 1);                                          // forced serial
    b.build(boxes, (int)std::max(4u, std::thread::hardware_concurrency()));
    if (a.nodes.size() != b.nodes.size() || a.primIdx.size() != b.primIdx.size()) {
        std::printf("[checkbvhparallel] FAIL: size %zu/%zu nodes, %zu/%zu prims\n",
                    a.nodes.size(), b.nodes.size(), a.primIdx.size(), b.primIdx.size());
        return false;
    }
    for (size_t i = 0; i < a.nodes.size(); ++i) {
        const BvhNode& x = a.nodes[i];
        const BvhNode& y = b.nodes[i];
        if (x.left != y.left || x.right != y.right || x.first != y.first || x.count != y.count ||
            std::memcmp(&x.box, &y.box, sizeof(Aabb)) != 0) {
            std::printf("[checkbvhparallel] FAIL: node %zu differs "
                        "(l %d/%d r %d/%d first %d/%d count %d/%d)\n",
                        i, x.left, y.left, x.right, y.right, x.first, y.first, x.count, y.count);
            return false;
        }
    }
    for (size_t i = 0; i < a.primIdx.size(); ++i)
        if (a.primIdx[i] != b.primIdx[i]) {
            std::printf("[checkbvhparallel] FAIL: primIdx %zu differs (%d vs %d)\n",
                        i, a.primIdx[i], b.primIdx[i]);
            return false;
        }
    std::printf("[checkbvhparallel] PASS -- %zu nodes, %d prims bit-identical serial vs %u threads\n",
                a.nodes.size(), N, std::max(4u, std::thread::hardware_concurrency()));
    return true;
}

inline bool bvhSphereQuerySelfTest() {
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    std::vector<Aabb> boxes;
    boxes.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        Vec3 c{rnd() * 10.0, rnd() * 10.0, rnd() * 10.0};
        Vec3 h{rnd() * 0.3 + 0.01, rnd() * 0.3 + 0.01, rnd() * 0.3 + 0.01};
        Aabb b; b.expand(c - h); b.expand(c + h);
        boxes.push_back(b);
    }
    Bvh bvh; bvh.build(boxes);
    long long missed = 0, reported = 0, truth = 0;
    std::vector<char> seen(boxes.size(), 0);
    const int kQueries = 500;
    for (int q = 0; q < kQueries; ++q) {
        Vec3 c{rnd() * 10.0, rnd() * 10.0, rnd() * 10.0};
        const double r = (q == 0) ? 0.0 : rnd() * 2.0 + 0.05;   // q==0 exercises r == 0
        std::fill(seen.begin(), seen.end(), (char)0);
        bvh.traverseSphere(c, r, [&](int p) { seen[p] = 1; ++reported; });
        const double r2 = r * r;
        for (size_t p = 0; p < boxes.size(); ++p)
            if (boxes[p].dist2To(c) <= r2) { ++truth; if (!seen[p]) ++missed; }
    }
    std::printf("[checkspherequery] %d queries over %d boxes: %lld true overlaps, "
                "%lld candidates reported (%.2fx), %lld MISSED\n",
                kQueries, (int)boxes.size(), truth, reported,
                truth > 0 ? (double)reported / (double)truth : 0.0, missed);
    if (missed) {
        std::printf("[checkspherequery] FAIL — the traversal skipped %lld primitive(s) whose "
                    "own box meets the query ball.\n", missed);
        return false;
    }
    if (truth == 0) {
        std::printf("[checkspherequery] FAIL — no overlaps generated, so the test proved "
                    "nothing. Check the box/radius ranges.\n");
        return false;
    }
    std::printf("[checkspherequery] PASS\n");
    return true;
}
