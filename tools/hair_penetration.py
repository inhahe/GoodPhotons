"""How much do Alice's strands actually pass through each other? Measure before designing.

A settle solver is a big feature and the stated doubt is whether it can converge in acceptable
time. Before any of that: is there a problem, and how big? This reports the PENETRATION RATE --
the share of segments that lie closer to a segment of a DIFFERENT strand than the two radii allow
-- on the real 20 000-strand groom, plus the geometry a solver would have to work at.
"""
import sys, numpy as np
from scipy.spatial import cKDTree

path = sys.argv[1] if len(sys.argv) > 1 else "scraps/_settle/alice_strands.txt"
print("== " + path)
pts, sid, rad = [], [], []
cur = -1
with open(path) as f:
    for ln in f:
        if ln[0] == '#':
            continue
        if ln[0] == 's':
            cur += 1
            continue
        w = ln.split()
        if len(w) == 4:
            pts.append((float(w[0]), float(w[1]), float(w[2])))
            rad.append(float(w[3]))
            sid.append(cur)
P = np.asarray(pts); R = np.asarray(rad); S = np.asarray(sid)
print("strands %d | points %d | radius %.1f..%.1f um" % (S.max() + 1, len(P), 1e6 * R.min(), 1e6 * R.max()))

# segments = consecutive points within one strand
same = S[1:] == S[:-1]
A, B = P[:-1][same], P[1:][same]
SS, RS = S[:-1][same], 0.5 * (R[:-1][same] + R[1:][same])
mid = 0.5 * (A + B)
seglen = np.linalg.norm(B - A, axis=1)
lo, hi = P.min(axis=0), P.max(axis=0)
print("segments %d | mean length %.3f mm | groom bbox %.3f x %.3f x %.3f m"
      % (len(A), 1e3 * seglen.mean(), *(hi - lo)))

# Candidate pairs: midpoints closer than (contact + both half-lengths). Sampled, because the
# exact all-pairs answer is 1.2M x neighbours and the RATE is what the design question needs.
rng = np.random.default_rng(7)
n = min(40000, len(A))
sel = rng.choice(len(A), n, replace=False)
tree = cKDTree(mid)
reach = float(seglen.mean() + 4.0 * RS.mean())
print("sampling %d segments, neighbour reach %.3f mm ..." % (n, 1e3 * reach)); sys.stdout.flush()

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
print("\n  segments with a different-strand neighbour in reach : %d of %d (%.1f %%)" % (len(near), n, 100.0 * len(near) / n))
print("  PENETRATING (closer than r_i + r_j)                 : %d of %d = %.2f %%" % (hit, n, 100.0 * hit / n))
if depths:
    dep = np.asarray(depths)
    print("  penetration depth: median %.1f um, p95 %.1f um, max %.1f um" % (1e6 * np.median(dep), 1e6 * np.percentile(dep, 95), 1e6 * dep.max()))
if len(near):
    print("  clearance to the nearest other strand: median %.2f mm, p5 %.3f mm" % (1e3 * np.median(near), 1e3 * np.percentile(near, 5)))
