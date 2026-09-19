"""How much do a groom's strands actually pass through each other?

    python tools/hair_penetration.py [dump.txt]

A settle solver is a big feature and the stated doubt was whether it could converge in acceptable
time. Before any of that: is there a problem, and how big? This reports the PENETRATION RATE -- the
share of segments lying closer to a segment of a DIFFERENT strand than the two radii allow -- on a
`-dumpcurves` polyline dump, plus the geometry a solver has to work at.

The headline result that motivated `settle { }`: Alice's 20 000-strand groom has 72.67 % of its
1.2 M segments overlapping, against 0.97 % for `fur` scattered at the SAME strand count, radius and
length. That 75x gap at equal density is what identifies the curve-of-curves blend, rather than
hair being dense, as the cause -- `count N` spaces instances by ARC LENGTH, and arc length does not
know what is already there.

PERFORMANCE NOTE, because it distorted several sweeps. Both halves of this file were originally
written the obvious way and were unusably slow on a 65 MB / 1.22 M-line dump: a per-line Python
`split()` cost ~90 s per file, and a per-sample neighbour loop filtering ~2253 candidates in Python
cost tens of minutes. Sweeps appeared to hang with no `ftrace` process running. Both are now batched
into numpy: `load()` hands one buffer to `np.fromstring`, and `measure()` flattens every sampled
segment's candidate list into one pair array and evaluates the segment-segment distances in chunks.
The rewrite is validated by re-deriving a KNOWN value -- the 72.67 % baseline -- rather than by
inspection.
"""
import io
import sys

import numpy as np
from scipy.spatial import cKDTree


def load(path):
    """(points, radii, strand_id) from a -dumpcurves file, parsed as a single buffer."""
    raw = io.open(path, encoding="utf-8").read().split(chr(10))
    vals, sid, cur = [], [], -1
    for ln in raw:
        if not ln:
            continue
        c = ln[0]
        if c == "s":
            cur += 1
            continue
        if c == "#":
            continue
        vals.append(ln)
        sid.append(cur)
    a = np.fromstring(" ".join(vals), sep=" ").reshape(-1, 4)
    return a[:, :3], a[:, 3], np.asarray(sid)


def _seg_seg(p1, q1, p2, q2):
    """Distance between segment (p1,q1) and segment (p2,q2), vectorised over the first axis.

    A clamped two-step solve: it returns a valid pair of parameters, so the result is an UPPER
    bound on the true minimum and can only ever UNDER-report penetration. That direction matters --
    the headline is a claim that overlap exists, so a conservative estimator is the safe one.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    s = np.where(den > 1e-30, np.clip((b * f - c * e) / np.where(den > 1e-30, den, 1.0), 0.0, 1.0), 0.0)
    t = np.clip((b * s + f) / np.where(e > 1e-30, e, 1.0), 0.0, 1.0)
    s = np.clip((b * t - c) / np.where(a > 1e-30, a, 1.0), 0.0, 1.0)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def measure(path, nsample=40000, verbose=True, seed=7, chunk=2000000):
    if verbose:
        print("== " + path)
    P, R, S = load(path)
    if verbose:
        print("strands %d | points %d | radius %.1f..%.1f um"
              % (S.max() + 1, len(P), 1e6 * R.min(), 1e6 * R.max()))

    same = S[1:] == S[:-1]                      # segments = consecutive points within one strand
    A, B = P[:-1][same], P[1:][same]
    SS, RS = S[:-1][same], 0.5 * (R[:-1][same] + R[1:][same])
    mid = 0.5 * (A + B)
    seglen = np.linalg.norm(B - A, axis=1)
    lo, hi = P.min(axis=0), P.max(axis=0)
    if verbose:
        print("segments %d | mean length %.3f mm | groom bbox %.3f x %.3f x %.3f m"
              % (len(A), 1e3 * seglen.mean(), *(hi - lo)))

    rng = np.random.default_rng(seed)
    n = min(nsample, len(A))
    sel = rng.choice(len(A), n, replace=False)
    # Reach must cover a full segment length: two segments crossing at an angle can touch with
    # their midpoints almost a segment apart, so a tighter radius would silently miss contacts.
    reach = float(seglen.mean() + 4.0 * RS.mean())
    if verbose:
        print("sampling %d segments, neighbour reach %.3f mm ..." % (n, 1e3 * reach))
        sys.stdout.flush()

    lists = cKDTree(mid).query_ball_point(mid[sel], reach)
    counts = np.fromiter((len(l) for l in lists), dtype=np.int64, count=n)
    total = int(counts.sum())
    flat = np.empty(total, dtype=np.int64)
    at = 0
    for l in lists:
        k = len(l)
        if k:
            flat[at:at + k] = l
            at += k
    samp = np.repeat(np.arange(n, dtype=np.int64), counts)

    keep = SS[flat] != SS[sel[samp]]            # different strands only
    flat, samp = flat[keep], samp[keep]

    nearest = np.full(n, np.inf)
    depth = np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    for s0 in range(0, len(flat), chunk):
        j = flat[s0:s0 + chunk]
        i = sel[samp[s0:s0 + chunk]]
        d = _seg_seg(A[i], B[i], A[j], B[j])
        lim = RS[i] + RS[j]
        si = samp[s0:s0 + chunk]
        np.minimum.at(nearest, si, d - lim)
        pen = lim - d
        m = pen > 0
        if m.any():
            np.maximum.at(depth, si[m], pen[m])
        seen[si] = True

    hit = depth > 0
    near = nearest[seen]
    if verbose:
        print(chr(10) + "  segments with a different-strand neighbour in reach : %d of %d (%.1f %%)"
              % (seen.sum(), n, 100.0 * seen.sum() / n))
        print("  PENETRATING (closer than r_i + r_j)                 : %d of %d = %.2f %%"
              % (hit.sum(), n, 100.0 * hit.sum() / n))
        if hit.any():
            dep = depth[hit]
            print("  penetration depth: median %.1f um, p95 %.1f um, max %.1f um"
                  % (1e6 * np.median(dep), 1e6 * np.percentile(dep, 95), 1e6 * dep.max()))
        if len(near):
            print("  clearance to the nearest other strand: median %.2f mm, p5 %.3f mm"
                  % (1e3 * np.median(near), 1e3 * np.percentile(near, 5)))

    return {"strands": int(S.max() + 1), "segments": int(len(A)),
            "pen_pct": 100.0 * hit.sum() / n,
            "median_pen_um": (1e6 * float(np.median(depth[hit]))) if hit.any() else 0.0,
            "median_clear_mm": 1e3 * float(np.median(near)) if len(near) else 0.0}


if __name__ == "__main__":
    measure(sys.argv[1] if len(sys.argv) > 1 else "scraps/_settle/alice_strands.txt")
