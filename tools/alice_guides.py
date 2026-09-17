"""Author Alice's hair guides by reading the flow off the sculpt.

    python tools/alice_guides.py [--scalp meshes/alice_scalp.obj] [--out scenes/alice_guides.ftsl]

The user's brief was to match "the ebbs and flows" of the sculpted hair by eye. The sculpt
already encodes its drape as a surface, so instead of guessing fifteen curves' coordinates, this
traces STREAMLINES down that surface: from each chosen root it steps along gravity projected onto
the local tangent plane, re-projects onto the nearest point of the scalp mesh, and repeats until
the hair bottom. Where the sculpt sweeps the fringe sideways, downhill-on-the-surface sweeps
sideways too; where the sides flare out at the tips, the surface flares and so do the guides.
What comes out is a set of `curve` definitions that hug the sculpt by construction, grouped into
curves of curves (FTSL 8.6) that a `fur ... guides` block (8.7) interpolates over the scalp.

Roots are placed on rings around the crown (the bow at (0.053, 0.894, -0.018); she faces +z and
+x is HER right), at several latitudes, so the crown radiates as the sculpt does. Everything here
is a starting point meant to be hand-adjusted in the emitted .ftsl -- that is the point of
authoring guides rather than strands.
"""
import argparse
import io
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Every constant below is authored in the RAW 1.9 m vertex frame of alice.glb. The scalp OBJ
# (tools/alice_scalp.py) carries the GLB's true size instead, so main() measures the frame
# scale from the OBJ's height and multiplies these through `FS` once. 1.0 in the raw frame.
FS = 1.0
CROWN_RAW = np.array([0.053, 0.894, -0.018])
HEAD_CENTRE_RAW = np.array([0.0, 0.80, 0.02])   # a rough skull centre, for placing the rings
HAIR_BOTTOM_RAW = 0.45                          # the tips, from the segmentation's y range
HAIR_SPAN_RAW = 0.4903                          # the mass's y extent in the raw frame (0.41..0.90)
CROWN = CROWN_RAW.copy(); HEAD_CENTRE = HEAD_CENTRE_RAW.copy(); HAIR_BOTTOM = HAIR_BOTTOM_RAW


def load_obj(path):
    V, N, F = [], [], []
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vn "):
            N.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(N), np.array(F)


class Surface:
    """Nearest point on a triangle soup, through a coarse uniform grid of triangle centroids."""

    def __init__(self, V, N, F, cell=0.02):
        self.V, self.N, self.F = V, N, F
        self.T = V[F]                                  # (n,3,3)
        self.C = self.T.mean(1)
        self.fn = np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0])
        self.fn /= np.maximum(np.linalg.norm(self.fn, axis=1, keepdims=True), 1e-12)
        self.cell = cell
        self.lo = self.C.min(0) - cell
        key = np.floor((self.C - self.lo) / cell).astype(np.int64)
        self.grid = {}
        for i, k in enumerate(map(tuple, key)):
            self.grid.setdefault(k, []).append(i)

    def candidates(self, p, r=2):
        k = np.floor((p - self.lo) / self.cell).astype(np.int64)
        out = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    out += self.grid.get((k[0] + dx, k[1] + dy, k[2] + dz), [])
        return np.array(out, dtype=np.int64)

    def closest(self, p):
        idx = self.candidates(p)
        if len(idx) == 0:
            idx = np.arange(len(self.F))
        T = self.T[idx]
        # closest point on each triangle (Ericson), vectorised
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        cp = p - c; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        q = np.empty_like(a)
        # vertex regions
        m = (d1 <= 0) & (d2 <= 0); q[m] = a[m]
        m2 = (d3 >= 0) & (d4 <= d3); q[m2] = b[m2]
        m3 = (d6 >= 0) & (d5 <= d6); q[m3] = c[m3]
        done = m | m2 | m3
        # edge ab
        v = np.where(np.abs(d1 - d3) > 1e-20, d1 / np.where(np.abs(d1 - d3) > 1e-20, d1 - d3, 1), 0)
        e1 = ~done & (vc <= 0) & (d1 >= 0) & (d3 <= 0); q[e1] = a[e1] + ab[e1] * v[e1][:, None]; done |= e1
        w = np.where(np.abs(d2 - d6) > 1e-20, d2 / np.where(np.abs(d2 - d6) > 1e-20, d2 - d6, 1), 0)
        e2 = ~done & (vb <= 0) & (d2 >= 0) & (d6 <= 0); q[e2] = a[e2] + ac[e2] * w[e2][:, None]; done |= e2
        den = (d4 - d3) + (d5 - d6)
        w2 = np.where(np.abs(den) > 1e-20, (d4 - d3) / np.where(np.abs(den) > 1e-20, den, 1), 0)
        e3 = ~done & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        q[e3] = b[e3] + (c[e3] - b[e3]) * w2[e3][:, None]; done |= e3
        # interior
        inn = ~done
        den2 = va + vb + vc
        den2 = np.where(np.abs(den2) > 1e-20, den2, 1)
        vv, ww = vb / den2, vc / den2
        q[inn] = a[inn] + ab[inn] * vv[inn][:, None] + ac[inn] * ww[inn][:, None]
        d = ((q - p) ** 2).sum(1)
        j = int(np.argmin(d))
        return q[j], self.fn[idx[j]]


def trace(surf, root, step=0.02, max_steps=60, bottom=None, outward_bias=0.15,
          sweep=0.0, lift0=0.004, lift1=0.006, part=0.03):
    """Downhill on the surface from `root`: gravity projected onto the tangent plane, with a small
    outward bias so a streamline never dives through the head where the sculpt overhangs."""
    # metric parameters arrive in the raw frame; scale them into the OBJ's frame
    step, lift0, lift1, part = step * FS, lift0 * FS, lift1 * FS, part * FS
    if bottom is None: bottom = HAIR_BOTTOM
    p, n = surf.closest(root)
    if np.dot(n, p - HEAD_CENTRE) < 0: n = -n          # face normals oriented OUTWARD
    pts = [p.copy()]; nrm = [n.copy()]
    g = np.array([0.0, -1.0, 0.0])
    for _ in range(max_steps):
        # outward = away from the head centre, projected on the tangent plane too
        out = p - HEAD_CENTRE; out[1] = 0.0
        out = out / max(np.linalg.norm(out), 1e-9)
        d = g - n * np.dot(g, n) + out * outward_bias
        # THE FRINGE: the sculpt sweeps its front hair across the forehead from her right (+x)
        # to her left, not straight down over the face. Downhill-on-the-surface cannot know
        # that, so front roots get a sideways bias that fades out past the temples.
        front = max(0.0, (p[2] - 0.04 * FS) / (0.14 * FS))   # 0 at the ears, 1 at the brow
        if front > 0.0:
            side = 1.0 if root[0] > part else -1.0            # her right of the part goes right
            d = d + np.array([side, 0.0, 0.0]) * (sweep * front)
        d = d - n * np.dot(d, n)
        if np.linalg.norm(d) < 1e-6:
            break
        d /= np.linalg.norm(d)
        q = p + d * step
        q, n2 = surf.closest(q)
        if np.linalg.norm(q - p) < 0.25 * step:       # stuck (a pocket): stop
            break
        p, n = q, n2
        if np.dot(n, p - HEAD_CENTRE) < 0: n = -n
        pts.append(p.copy()); nrm.append(n.copy())
        if p[1] < bottom:
            break
    pts, nrm = np.array(pts), np.array(nrm)
    # FLOAT the guide off the sculpt. A strand blended from surface-hugging guides on a CONVEX
    # head lies on the chord between them, i.e. INSIDE the surface, and a root pushed under the
    # skin never emerges -- the first render was bald on the crown for exactly this reason. So
    # each guide rides a little above the sculpt, more toward the tip where the hair mass is
    # thickest: h(t) = lift0 + lift1 * t along its own arc length.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]); t = t / max(t[-1], 1e-9)
    pts = pts + nrm * (lift0 + lift1 * t)[:, None]
    return pts


def decimate(pts, k=7):
    """Keep k points evenly spaced by arc length -- guides want few, well-placed control points."""
    if len(pts) <= k:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for t in np.linspace(0, s[-1], k):
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = min(max(i, 0), len(pts) - 2)
        u = (t - s[i]) / max(s[i + 1] - s[i], 1e-12)
        out.append(pts[i] * (1 - u) + pts[i + 1] * u)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalp", default="meshes/alice_scalp.obj")
    ap.add_argument("--out", default="scenes/alice_guides.ftsl")
    ap.add_argument("--rings", type=int, default=4)
    ap.add_argument("--around", type=int, default=10)
    ap.add_argument("--sweep", type=float, default=2.0, help="fringe sweep strength on the brow ring")
    ap.add_argument("--place", nargs=5, type=float, metavar=("TX", "TY", "TZ", "RY", "S"),
                    help="wrap the guides in a group { translate rotate scale } for a scene that shows her transformed")
    args = ap.parse_args()

    V, N, F = load_obj(args.scalp)
    global FS, CROWN, HEAD_CENTRE, HAIR_BOTTOM
    FS = (V[:, 1].max() - V[:, 1].min()) / HAIR_SPAN_RAW
    CROWN = CROWN_RAW * FS; HEAD_CENTRE = HEAD_CENTRE_RAW * FS; HAIR_BOTTOM = HAIR_BOTTOM_RAW * FS
    surf = Surface(V, N, F, cell=0.02 * FS)
    print("scalp: %d verts %d tris; frame scale %.5f of the raw 1.9 m frame (height %.4f m)"
          % (len(V), len(F), FS, V[:, 1].max() - V[:, 1].min()))

    # roots: HORIZONTAL rings (latitude circles about world +y through the head centre) at
    # increasing polar angle, the last one at the hairline. A root that lands on the face is
    # snapped by closest() onto the scalp's rim -- which IS the hairline, exactly where the
    # fringe guides must start. The crown's small offset toward the bow is left to the sculpt.
    up = np.array([0.0, 1.0, 0.0])
    t0 = np.array([1.0, 0.0, 0.0]); t1 = np.array([0.0, 0.0, 1.0])
    rows = []
    polar = np.linspace(0.35, 1.50, args.rings)            # radians from straight up; 1.5 ~ the hairline
    for ri, th in enumerate(polar):
        row = []
        around = args.around if ri == 0 else args.around + 4 * ri   # wider rings get more guides
        for ai in range(around):
            ph = 2 * math.pi * (ai + 0.5 * (ri % 2)) / around       # stagger alternate rings
            dirv = up * math.cos(th) + (t0 * math.cos(ph) + t1 * math.sin(ph)) * math.sin(th)
            root = HEAD_CENTRE + dirv * (0.17 * FS)
            # the sculpt's fringe is a SHORT swept bang over the brow, not a curtain over the
            # face: a hairline-ring root in front traces only ~7 steps (~0.14 m) before it stops.
            brow = (ri == args.rings - 1) and (root[2] > 0.06 * FS)
            pts = decimate(trace(surf, root, max_steps=5 if brow else 60,
                                 sweep=args.sweep if (ri == args.rings - 1) else 0.0),
                           k=5 if brow else 7)
            if len(pts) >= 3:
                row.append(pts)
        rows.append(row)
        print("ring %d (polar %.2f rad): %d guides, mean length %.3f m" %
              (ri, th, len(row), np.mean([np.linalg.norm(np.diff(g, axis=0), axis=1).sum() for g in row])))

    L = ["# Alice's hair guides -- GENERATED by tools/alice_guides.py from the sculpted hair surface",
         "# (streamlines down meshes/alice_scalp.obj). Hand-adjust freely; regenerate to start over.",
         "#",
         "# Each `ring_N` is a closed curve OF curves around the head (its children are the guides,",
         "# in order around the crown); `alice_hair` is the curve of those rings, crown to nape.",
         "# None of these carries a material: they are definitions for `fur ... guides`.",
         ""]
    for ri, row in enumerate(rows):
        for gi, pts in enumerate(row):
            L.append('curve "g_%d_%d" {' % (ri, gi))
            for p in pts:
                L.append('    point %.4f %.4f %.4f' % tuple(p))
            L.append('}')
        # `closed` is a bareword flag: it must stand ALONE on its line, or the next key becomes
        # its value (FTSL 1.1) -- and the curve parser refuses that, as it should.
        L.append('curve "ring_%d" {' % ri)
        L.append('    closed')
        L.append('    spline centripetal')
        for gi in range(len(row)):
            L.append('    curve "g_%d_%d"' % (ri, gi))
        L.append('}')
        L.append('')
    L.append('curve "alice_hair" {')
    for ri in range(len(rows)):
        L.append('    curve "ring_%d"' % ri)
    L.append('}')
    if args.place:
        tx, ty, tz, ry, sc = args.place
        # a curve of curves under a group is transformed as its control points (FTSL 8.6), so
        # the whole definition set rides inside one group and the scene places her once
        L = L[:6] + ['group "alice_hair_place" { translate %g %g %g  rotate 0 %g 0  scale %g' % (tx, ty, tz, ry, sc)] + \
            ["    " + l if l else l for l in L[6:]] + ["}"]
    io.open(args.out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote %s: %d guides in %d rings" % (args.out, sum(len(r) for r in rows), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
