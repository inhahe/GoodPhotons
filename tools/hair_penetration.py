"""How much do Alice's strands actually pass through each other? Measure before designing.

A settle solver is a big feature and the stated doubt is whether it can converge in acceptable
time. Before any of that: is there a problem, and how big? This reports the PENETRATION RATE --
the share of segments that lie closer to a segment of a DIFFERENT strand than the two radii allow
-- on the real 20 000-strand groom, plus the geometry a solver would have to work at.
"""
import io
import sys, numpy as np
from scipy.spatial import cKDTree

def load(path):
    """(points, radii, strand_id) from a -dumpcurves file, parsed as one buffer.

    The per-line Python split this replaces cost ~90 s on a 1.22 M-line dump and was paid twice
    per comparison; four configurations was ~12 minutes of parsing, which is what made the
    sweeps look hung.
    """
    import numpy as np
    raw = io.open(path, encoding='utf-8').read().split(chr(10))
    vals, sid, cur = [], [], -1
    for ln in raw:
        if not ln:
            continue
        c = ln[0]
        if c == 's':
            cur += 1
            continue
        if c == '#':
            continue
        vals.append(ln)
        sid.append(cur)
    a = np.fromstring(' '.join(vals), sep=' ').reshape(-1, 4)
    return a[:, :3], a[:, 3], np.asarray(sid)


def measure(path, nsample=40000, verbose=True):
    if verbose: print("== " + path)
    pts, rad, sid = load(path)
    P, R, S = pts, rad, sid
    if verbose: print("strands %d | points %d | radius %.1f..%.1f um" % (S.max() + 1, len(P), 1e6 * R.min(), 1e6 * R.max()))

    # segments = consecutive points within one strand
    same = S[1:] == S[:-1]
    A, B = P[:-1][same], P[1:][same]
    SS, RS = S[:-1][same], 0.5 * (R[:-1][same] + R[1:][same])
    mid = 0.5 * (A + B)
    seglen = np.linalg.norm(B - A, axis=1)
    lo, hi = P.min(axis=0), P.max(axis=0)
    if verbose: print("segments %d | mean length %.3f mm | groom bbox %.3f x %.3f x %.3f m"
          % (len(A), 1e3 * seglen.mean(), *(hi - lo)))

    # Candidate pairs: midpoints closer than (contact + both half-lengths). Sampled, because the
    # exact all-pairs answer is 1.2M x neighbours and the RATE is what the design question needs.
    rng = np.random.default_rng(7)
    n = min(nsample, len(A))
    sel = rng.choice(len(A), n, replace=False)
    tree = cKDTree(mid)
    reach = float(seglen.mean() + 4.0 * RS.mean())
    if verbose: print("sampling %d segments, neighbour reach %.3f mm ..." % (n, 1e3 * reach)); sys.stdout.flush()

    def seg_seg(p1, q1, p2, q2):
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a = np.einsum('ij,ij->i', d1, d1); e = np.einsum('ij,ij->i', d2, d2)
        f = np.einsum('ij,ij->i', d2, r);  c = np.einsum('ij,ij->i', d1, r)
        b = np.einsum('ij,ij->i', d1, d2)
        den = a * e - b * b
        s = np.where(den > 1e-30, np.clip((b * f - c * e) / np.where(den > 1e-30, den, 1), 0, 1), 0.0)
        t = np.clip((b * s + f) / np.where(e > 1e-30, e, 1), 0, 1)
        s = np.clip((b * t - c) / np.where(a > 1e-30, a, 1), 0, 1)
        return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)

    hit = 0; depths = []; nearest = []
    for k, i in enumerate(sel):
        cand = tree.query_ball_point(mid[i], reach)
        cand = np.asarray([j for j in cand if SS[j] != SS[i]], dtype=np.int64)
        if cand.size == 0:
            continue
        d = seg_seg(np.repeat(A[i][None], cand.size, 0), np.repeat(B[i][None], cand.size, 0), A[cand], B[cand])
        lim = RS[i] + RS[cand]
        nearest.append((d - lim).min())
        pen = lim - d
        if (pen > 0).any():
            hit += 1
            depths.append(pen[pen > 0].max())

    near = np.asarray(nearest)
    if verbose: print("\n  segments with a different-strand neighbour in reach : %d of %d (%.1f %%)" % (len(near), n, 100.0 * len(near) / n))
    if verbose: print("  PENETRATING (closer than r_i + r_j)                 : %d of %d = %.2f %%" % (hit, n, 100.0 * hit / n))
    if depths:
        dep = np.asarray(depths)
        if verbose: print("  penetration depth: median %.1f um, p95 %.1f um, max %.1f um" % (1e6 * np.median(dep), 1e6 * np.percentile(dep, 95), 1e6 * dep.max()))
    if len(near):
        if verbose: print("  clearance to the nearest other strand: median %.2f mm, p5 %.3f mm" % (1e3 * np.median(near), 1e3 * np.percentile(near, 5)))

    return {"strands": int(S.max() + 1), "segments": int(len(A)),
            "pen_pct": 100.0 * hit / n,
            "median_pen_um": (1e6 * float(np.median(depths))) if depths else 0.0,
            "median_clear_mm": 1e3 * float(np.median(near)) if len(near) else 0.0}


if __name__ == "__main__":
    measure(sys.argv[1] if len(sys.argv) > 1 else "scraps/_settle/alice_strands.txt")
