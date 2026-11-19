// cell_walk.h -- a ray's way into an axis-aligned box and through the regular lattice of cells in it,
// in the one form the host and the device both take. The box clip bounds media, fires, the fur coat
// and the volume caches (clipRayBox, dClipRayBox: Medium::clipToBounds and dMedClip, VolCache::raySpan
// and dVolCacheMarch); the walk steps through every lattice the renderer marches: a medium's majorant
// grid (Renderer::majorantWalk, dMajWalk), the aggregate fur coat's grid (FurGrid::walk: its march and
// FurVolume::sampleFlight) and a fire's emission grid (cellWalkT2Integral: firePointPdf and its
// device twin). Host and device take the same stepping; the spans they cut agree to the precision
// of the lattice and the ray each holds (the device's in float under FTRACE_GPU_FP32).
#pragma once
#include "unit_linkage.h"   // FT_UNIT_INLINE

#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define CW_HD __host__ __device__ FT_UNIT_INLINE
#else
#define CW_HD FT_UNIT_INLINE
#endif

// Slab clip of the ray o + t d, t in [t0, t1], to the box [lo, hi]: the overlap [ta, tb], or false
// when it is empty or a single point. An axis the ray runs parallel to (|d| < 1e-12) either contains
// the origin or misses; a slab that comes out NaN leaves the bounds as they were.
CW_HD bool rayBoxClip(const double lo[3], const double hi[3], const double o[3], const double d[3],
                      double t0, double t1, double& ta, double& tb) {
    for (int a = 0; a < 3; ++a) {
        if (fabs(d[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return false; continue; }
        const double inv = 1.0 / d[a];
        double s0 = (lo[a] - o[a]) * inv, s1 = (hi[a] - o[a]) * inv;
        if (s0 > s1) { const double s = s0; s0 = s1; s1 = s; }
        t0 = s0 > t0 ? s0 : t0;
        t1 = s1 < t1 ? s1 : t1;
        if (t0 > t1) return false;
    }
    ta = t0; tb = t1;
    return tb > ta;
}

// A walk in progress through the lattice of n[0] x n[1] x n[2] cells (Amanatides-Woo): cellWalkInit or
// cellWalkClip starts it, and each cellWalkNext hands back the next cell the ray crosses and the
// stretch of the ray inside it, in order along the ray. Cells are numbered x fastest,
// ((z * ny) + y) * nx + x.
struct CellWalk {
    int    c[3], step[3], n[3];   // the cell the walk is in, its step along each axis, the lattice
    double tNext[3], tDelta[3];   // where the ray next crosses a plane of each axis, and their spacing
    double t, tEnd;               // how far along the ray the walk is, and where it ends
};

// Start the walk of o + t d over t in [ta, tb], through the lattice whose low corner is `lo`, whose
// cells are cs[k] wide (ic[k] their reciprocal, as the caller rounds it) and n[k] a side. The segment
// must lie in the lattice's box up to rounding, which the first cell -- the one holding o + ta d --
// is clamped against: the ray's clip to it (cellWalkClip), or to a region it bounds (a medium's,
// Renderer::majorantWalk). An axis the clip takes the ray as parallel to (|d| < 1e-12) is never
// stepped, its next crossing being at infinity, so even a walk with no axis to step (a zero or NaN
// direction) hands back one span and ends.
CW_HD void cellWalkInit(CellWalk& w, const double lo[3], const double cs[3], const double ic[3],
                        const int n[3], const double o[3], const double d[3], double ta, double tb) {
    for (int k = 0; k < 3; ++k) {
        w.n[k] = n[k];
        const int i = (int)floor((o[k] + d[k] * ta - lo[k]) * ic[k]);
        w.c[k] = i < 0 ? 0 : (i >= n[k] ? n[k] - 1 : i);
        if (d[k] >= 1e-12) {
            w.step[k] = 1;
            w.tNext[k] = (lo[k] + (w.c[k] + 1) * cs[k] - o[k]) / d[k];
            w.tDelta[k] = cs[k] / d[k];
        } else if (d[k] <= -1e-12) {
            w.step[k] = -1;
            w.tNext[k] = (lo[k] + w.c[k] * cs[k] - o[k]) / d[k];
            w.tDelta[k] = -cs[k] / d[k];
        } else {
            w.step[k] = 0; w.tNext[k] = HUGE_VAL; w.tDelta[k] = HUGE_VAL;
        }
    }
    w.t = ta; w.tEnd = tb;
}

// The walk of the ray's stretch within [t0, t1] inside the lattice's box [lo, hi] (rayBoxClip), the
// lattice as cellWalkInit takes it: false when the ray misses the box.
CW_HD bool cellWalkClip(CellWalk& w, const double lo[3], const double hi[3], const double cs[3],
                        const double ic[3], const int n[3], const double o[3], const double d[3],
                        double t0, double t1) {
    double ta, tb;
    if (!rayBoxClip(lo, hi, o, d, t0, t1, ta, tb)) return false;
    cellWalkInit(w, lo, cs, ic, n, o, d, ta, tb);
    return true;
}

// The next cell of the walk: `ci` its index, [a, b] the stretch of the ray inside it, b > a (a cell
// the ray only touches is passed over), each stretch starting where the last ended. False once the
// walk has reached tEnd or left the lattice, or tEnd is NaN. A crossing the rounding of a nearly
// parallel axis puts behind the walk is passed over, not stepped back to, so a stretch never starts
// before the walk's segment or overlaps the last.
CW_HD bool cellWalkNext(CellWalk& w, size_t& ci, double& a, double& b) {
    for (;;) {
        if (!(w.t < w.tEnd)) return false;
        const int axis = (w.tNext[0] < w.tNext[1]) ? ((w.tNext[0] < w.tNext[2]) ? 0 : 2)
                                                   : ((w.tNext[1] < w.tNext[2]) ? 1 : 2);
        const double t1 = w.tNext[axis] < w.tEnd ? w.tNext[axis] : w.tEnd;
        const size_t cell = ((size_t)w.c[2] * w.n[1] + w.c[1]) * w.n[0] + w.c[0];
        const double s0 = w.t;
        if (t1 >= w.tEnd) {
            w.t = w.tEnd;
        } else {
            w.t = t1 > s0 ? t1 : s0;
            w.c[axis] += w.step[axis];
            if (w.c[axis] < 0 || w.c[axis] >= w.n[axis]) w.t = w.tEnd;   // left the lattice
            else w.tNext[axis] += w.tDelta[axis];
        }
        if (t1 > s0) { ci = cell; a = s0; b = t1; return true; }
    }
}

// The integral along o + t d, t >= 0, of w(x) t^2 dt, w the lattice's cells' `weight` (x fastest)
// times `scale` inside the box [lo, hi] (the lattice as cellWalkClip takes it) and 0 outside: the
// solid-angle density with which a point drawn by a density piecewise constant over the cells
// lies in the direction d from o -- a fire's point sampling's (firePointPdf, dFirePointPdf).
CW_HD double cellWalkT2Integral(const double lo[3], const double hi[3], const double cs[3],
                                const double ic[3], const int n[3], const double* weight, double scale,
                                const double o[3], const double d[3]) {
    CellWalk w;
    if (!cellWalkClip(w, lo, hi, cs, ic, n, o, d, 0.0, 1e300)) return 0.0;
    double sum = 0.0;
    size_t ci; double a, b;
    while (cellWalkNext(w, ci, a, b)) {
        const double cw = weight[ci];
        if (cw > 0.0) sum += cw * scale * (b * b * b - a * a * a) / 3.0;
    }
    return sum;
}
