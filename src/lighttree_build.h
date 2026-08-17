#pragma once
// ---------------------------------------------------------------------------
// Host-side builder for the light BVH declared in lighttree.h.
//
// Split cleanly from the traversal because the traversal must compile inside the
// CUDA translation unit (which knows nothing about Scene/Emitter/std::vector) while
// the build runs once, on the host, at Scene::finalizeEmitters() time.  The seam is
// LtEmitterBound: a caller describes each emitter as a box + an emission cone + a
// flux, and never has to hand this file a scene type.
//
// The split is the Conty & Kulla SAOH — the usual SAH, but with the surface area of
// each side multiplied by an ORIENTATION measure M_Omega(theta_o, theta_e), the
// solid-angle-weighted extent of that side's emission cone.  That extra factor is
// the whole reason a light tree beats a plain BVH here: two panels facing opposite
// walls are cheap to separate and expensive to keep together, which a purely
// spatial heuristic cannot see.  A regularisation factor Kr penalises splitting
// across a thin axis, exactly as in the paper.
// ---------------------------------------------------------------------------
#include "lighttree.h"
#include <vector>
#include <algorithm>

// One emitter, described for the builder. `emitter` is whatever index the caller
// wants handed back by the traversal (Scene::emitters index in ftrace's case).
struct LtEmitterBound {
    double bmin[3], bmax[3];
    double axis[3];
    double cosThetaO;    // cos of the half-angle containing this emitter's normals
    double cosThetaE;    // cos of the spread beyond the normal (0 = Lambertian)
    double power;
    int    emitter;
};

namespace ltdetail {

constexpr double kPi = 3.14159265358979323846;

inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Solid-angle-ish measure of an emission cone (Conty & Kulla eq. 1). Monotone in
// both angles, and equal to the cone's solid angle when theta_e = 0.
inline double orientationMeasure(double cosThetaO, double cosThetaE) {
    const double thetaO = std::acos(clampd(cosThetaO, -1.0, 1.0));
    const double thetaE = std::acos(clampd(cosThetaE, -1.0, 1.0));
    const double thetaW = std::min(thetaO + thetaE, kPi);
    const double sO = std::sin(thetaO), cO = std::cos(thetaO);
    return 2.0 * kPi * (1.0 - cO) +
           0.5 * kPi * (2.0 * thetaW * sO - std::cos(thetaO - 2.0 * thetaW) -
                        2.0 * thetaO * sO + cO);
}

struct Cone { double w[3]; double cosTheta; bool empty; };

inline Cone coneUnion(Cone a, Cone b) {
    if (a.empty) return b;
    if (b.empty) return a;
    const double thetaA = std::acos(clampd(a.cosTheta, -1.0, 1.0));
    const double thetaB = std::acos(clampd(b.cosTheta, -1.0, 1.0));
    double d = a.w[0] * b.w[0] + a.w[1] * b.w[1] + a.w[2] * b.w[2];
    const double thetaD = std::acos(clampd(d, -1.0, 1.0));
    if (std::min(thetaD + thetaB, kPi) <= thetaA) return a;   // b already inside a
    if (std::min(thetaD + thetaA, kPi) <= thetaB) return b;
    const double thetaO = 0.5 * (thetaA + thetaD + thetaB);
    if (thetaO >= kPi) { Cone c{{a.w[0], a.w[1], a.w[2]}, -1.0, false}; return c; }
    // Rotate a's axis toward b's by (thetaO - thetaA), in the plane they span.
    const double thetaR = thetaO - thetaA;
    double px = a.w[1] * b.w[2] - a.w[2] * b.w[1];
    double py = a.w[2] * b.w[0] - a.w[0] * b.w[2];
    double pz = a.w[0] * b.w[1] - a.w[1] * b.w[0];
    const double pl = std::sqrt(px * px + py * py + pz * pz);
    Cone c;
    c.empty = false;
    c.cosTheta = std::cos(thetaO);
    if (pl < 1e-12) {                       // (anti)parallel axes: nothing to rotate about
        c.w[0] = a.w[0]; c.w[1] = a.w[1]; c.w[2] = a.w[2];
        return c;
    }
    px /= pl; py /= pl; pz /= pl;
    const double cs = std::cos(thetaR), sn = std::sin(thetaR);
    // Rodrigues about the (unit) perpendicular p.
    const double dotpa = px * a.w[0] + py * a.w[1] + pz * a.w[2];   // == 0, kept for clarity
    c.w[0] = a.w[0] * cs + (py * a.w[2] - pz * a.w[1]) * sn + px * dotpa * (1.0 - cs);
    c.w[1] = a.w[1] * cs + (pz * a.w[0] - px * a.w[2]) * sn + py * dotpa * (1.0 - cs);
    c.w[2] = a.w[2] * cs + (px * a.w[1] - py * a.w[0]) * sn + pz * dotpa * (1.0 - cs);
    const double l = std::sqrt(c.w[0] * c.w[0] + c.w[1] * c.w[1] + c.w[2] * c.w[2]);
    if (l > 0.0) { c.w[0] /= l; c.w[1] /= l; c.w[2] /= l; }
    return c;
}

// Running union of a set of emitters: box, cone, flux.
struct Agg {
    double bmin[3] = { 1e300,  1e300,  1e300};
    double bmax[3] = {-1e300, -1e300, -1e300};
    Cone cone{{0, 0, 1}, 1.0, true};
    double cosThetaE = 1.0;    // widest (smallest cos) emission spread in the set
    double power = 0.0;
    int n = 0;

    void add(const LtEmitterBound& e) {
        for (int k = 0; k < 3; ++k) {
            bmin[k] = std::min(bmin[k], e.bmin[k]);
            bmax[k] = std::max(bmax[k], e.bmax[k]);
        }
        Cone c{{e.axis[0], e.axis[1], e.axis[2]}, e.cosThetaO, false};
        cone = coneUnion(cone, c);
        cosThetaE = std::min(cosThetaE, e.cosThetaE);
        power += e.power;
        ++n;
    }
    double surfaceArea() const {
        if (n == 0) return 0.0;
        const double dx = std::max(0.0, bmax[0] - bmin[0]);
        const double dy = std::max(0.0, bmax[1] - bmin[1]);
        const double dz = std::max(0.0, bmax[2] - bmin[2]);
        // A degenerate (planar / linear) set of lights still has to have a nonzero
        // measure or every SAOH cost collapses to 0 and the split becomes arbitrary.
        const double a = 2.0 * (dx * dy + dy * dz + dz * dx);
        return a > 0.0 ? a : 1e-12;
    }
    double measure() const { return orientationMeasure(cone.cosTheta, cosThetaE); }
};

}  // namespace ltdetail

// Build the tree. `items` is consumed by value order only (it is not reordered in
// place; an index permutation is). Returns the root node index, or -1 if there was
// nothing to build. `out` is cleared first.
inline int ltBuild(const std::vector<LtEmitterBound>& items,
                   std::vector<LightTreeNode>& out) {
    using namespace ltdetail;
    out.clear();
    const int n = (int)items.size();
    if (n == 0) return -1;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    out.reserve((size_t)std::max(1, 2 * n - 1));

    // Recursive build over idx[first, last). Returns the node's index in `out`.
    struct Builder {
        const std::vector<LtEmitterBound>& items;
        std::vector<int>& idx;
        std::vector<LightTreeNode>& out;

        int build(int first, int last) {
            Agg agg;
            for (int i = first; i < last; ++i) agg.add(items[idx[i]]);
            const int self = (int)out.size();
            out.push_back(LightTreeNode{});
            LightTreeNode nd{};
            // The node carries the DERIVED bounding-sphere form (centre, r^2) and the
            // baked sin(theta_o) — see the struct comment in lighttree.h. These MUST be
            // the exact expressions the traversal used to recompute them per call
            // (0.5*(bmin+bmax); 0.25*(ex^2+ey^2+ez^2) summed x,y,z; ltSafeSqrt(1-c^2))
            // so every importance value, selection pdf, and image stays bit-identical.
            for (int k = 0; k < 3; ++k) {
                nd.center[k] = 0.5 * (agg.bmin[k] + agg.bmax[k]);
                nd.axis[k]   = agg.cone.w[k];
            }
            {
                const double ex = agg.bmax[0] - agg.bmin[0];
                const double ey = agg.bmax[1] - agg.bmin[1];
                const double ez = agg.bmax[2] - agg.bmin[2];
                nd.r2 = 0.25 * (ex * ex + ey * ey + ez * ez);
            }
            nd.cosThetaO = agg.cone.cosTheta;
            nd.sinThetaO = ltSafeSqrt(1.0 - nd.cosThetaO * nd.cosThetaO);
            nd.cosThetaE = agg.cosThetaE;
            nd.power = agg.power;
            nd.left = nd.right = nd.emitter = -1;
            if (last - first == 1) {
                nd.emitter = items[idx[first]].emitter;
                out[self] = nd;
                return self;
            }
            const int mid = partition(first, last, agg);
            const int l = build(first, mid);
            const int r = build(mid, last);
            nd.left = l; nd.right = r;
            out[self] = nd;
            return self;
        }

        // SAOH over 12 buckets on each axis; falls back to an equal-count median
        // split when no bucket boundary beats it (identical centroids, or a set the
        // heuristic cannot separate) so the recursion always makes progress.
        int partition(int first, int last, const Agg& parent) {
            const int count = last - first;
            const double pArea = parent.surfaceArea();
            const double pMeas = parent.measure();
            const double denom = (pArea * pMeas > 0.0) ? (pArea * pMeas) : 1.0;
            double ext[3];
            for (int k = 0; k < 3; ++k) ext[k] = parent.bmax[k] - parent.bmin[k];
            const double maxExt = std::max(ext[0], std::max(ext[1], ext[2]));

            constexpr int kBuckets = 12;
            double bestCost = 1e300;
            int bestAxis = -1, bestSplit = -1;
            for (int axis = 0; axis < 3; ++axis) {
                if (ext[axis] <= 0.0) continue;
                const double Kr = maxExt / ext[axis];
                Agg buckets[kBuckets];
                const double scale = kBuckets / ext[axis];
                for (int i = first; i < last; ++i) {
                    const LtEmitterBound& e = items[idx[i]];
                    const double c = 0.5 * (e.bmin[axis] + e.bmax[axis]);
                    int b = (int)((c - parent.bmin[axis]) * scale);
                    b = b < 0 ? 0 : (b >= kBuckets ? kBuckets - 1 : b);
                    buckets[b].add(e);
                }
                // Prefix/suffix sweep.
                Agg left[kBuckets], right[kBuckets];
                Agg run;
                for (int b = 0; b < kBuckets; ++b) {
                    if (buckets[b].n) {
                        // Merge bucket b into the running aggregate.
                        for (int k = 0; k < 3; ++k) {
                            run.bmin[k] = std::min(run.bmin[k], buckets[b].bmin[k]);
                            run.bmax[k] = std::max(run.bmax[k], buckets[b].bmax[k]);
                        }
                        run.cone = coneUnion(run.cone, buckets[b].cone);
                        run.cosThetaE = std::min(run.cosThetaE, buckets[b].cosThetaE);
                        run.power += buckets[b].power;
                        run.n += buckets[b].n;
                    }
                    left[b] = run;
                }
                Agg run2;
                for (int b = kBuckets - 1; b >= 0; --b) {
                    if (buckets[b].n) {
                        for (int k = 0; k < 3; ++k) {
                            run2.bmin[k] = std::min(run2.bmin[k], buckets[b].bmin[k]);
                            run2.bmax[k] = std::max(run2.bmax[k], buckets[b].bmax[k]);
                        }
                        run2.cone = coneUnion(run2.cone, buckets[b].cone);
                        run2.cosThetaE = std::min(run2.cosThetaE, buckets[b].cosThetaE);
                        run2.power += buckets[b].power;
                        run2.n += buckets[b].n;
                    }
                    right[b] = run2;
                }
                for (int b = 0; b < kBuckets - 1; ++b) {
                    if (left[b].n == 0 || right[b + 1].n == 0) continue;
                    const double cost =
                        Kr * (left[b].power * left[b].surfaceArea() * left[b].measure() +
                              right[b + 1].power * right[b + 1].surfaceArea() * right[b + 1].measure()) /
                        denom;
                    if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = b; }
                }
            }
            if (bestAxis >= 0) {
                const double lo = parent.bmin[bestAxis];
                const double scale = kBuckets / (parent.bmax[bestAxis] - lo);
                const int split = bestSplit;
                auto* itemsP = &items;
                int* beg = idx.data() + first;
                int* end = idx.data() + last;
                int* midp = std::partition(beg, end, [&](int i) {
                    const LtEmitterBound& e = (*itemsP)[i];
                    const double c = 0.5 * (e.bmin[bestAxis] + e.bmax[bestAxis]);
                    int b = (int)((c - lo) * scale);
                    b = b < 0 ? 0 : (b >= kBuckets ? kBuckets - 1 : b);
                    return b <= split;
                });
                const int mid = (int)(midp - idx.data());
                if (mid > first && mid < last) return mid;
            }
            // Median fallback on the widest axis (or plain halving if that degenerates).
            int axis = 0;
            for (int k = 1; k < 3; ++k) if (ext[k] > ext[axis]) axis = k;
            const int mid = first + count / 2;
            auto* itemsP = &items;
            std::nth_element(idx.data() + first, idx.data() + mid, idx.data() + last,
                             [&](int a, int b) {
                                 const double ca = 0.5 * ((*itemsP)[a].bmin[axis] + (*itemsP)[a].bmax[axis]);
                                 const double cb = 0.5 * ((*itemsP)[b].bmin[axis] + (*itemsP)[b].bmax[axis]);
                                 return ca < cb;
                             });
            return mid;
        }
    } builder{items, idx, out};

    return builder.build(0, n);
}
