# Where is frame 0, and what would rotating the loop's phase cost?
#
# Parses `camera_curve "fly"` straight out of the scene (no hardcoded copy of the points --
# that is how scraps/_flyplan.py goes stale), reproduces the loader's arc-length /
# density_at / frame placement exactly (src/ftsl.h ~7436-7770), and reports:
#
#   * the normalized ARC-LENGTH t of every control point, and the frame index that lands
#     nearest it -- so "point 23" and "frame 440" can be talked about in the same breath;
#   * for a candidate rotation, the shifted density_at stops.
#
# A `closed` centripetal Catmull-Rom loop is invariant under a cyclic rotation of its
# control points: the CURVE is identical, only the parameter origin moves. So rotating the
# point list is a pure phase change -- provided every density_at stop is shifted by the same
# arc-length delta (mod 1), which is what --rotate prints.
import math, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(os.path.dirname(HERE), "scenes", "gallery_rain.ftsl")
ALPHA = 0.5  # centripetal


def parse_curve(path, name="fly"):
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r'camera_curve\s+"%s"\s*\{' % re.escape(name), src)
    if not m:
        raise SystemExit("no camera_curve %r in %s" % (name, path))
    i, depth = m.end(), 1
    while depth:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    body = src[m.end():i - 1]
    body = re.sub(r"#[^\n]*", "", body)  # strip comments

    pts, dens = [], []
    for ln in body.splitlines():
        for mm in re.finditer(r"\bpoint\s+(\S+)\s+(\S+)\s+(\S+)", ln):
            pts.append(tuple(float(x) for x in mm.groups()))
        for mm in re.finditer(r"\bdensity_at\s+(\S+)\s+(\S+)", ln):
            dens.append((float(mm.group(1)), float(mm.group(2))))
    frames = int(re.search(r"\bframes\s+(\d+)", body).group(1))
    closed = bool(re.search(r"\bclosed\b", body))
    return pts, dens, frames, closed


def catmull(p, closed, g, alpha=ALPHA):
    """Transcription of catmullRomAt() in src/ftsl.h. g is the GLOBAL param in [0, nSeg]."""
    m = len(p)
    nseg = m if closed else m - 1
    g = min(max(g, 0.0), float(nseg))
    seg = min(int(math.floor(g)), nseg - 1)
    t = g - seg
    idx = lambda i: p[i % m] if closed else p[min(max(i, 0), m - 1)]
    P0, P1, P2, P3 = idx(seg - 1), idx(seg), idx(seg + 1), idx(seg + 2)
    k = lambda ti, a, b: ti + max(math.dist(a, b), 1e-9) ** alpha
    k0 = 0.0
    k1 = k(k0, P0, P1)
    k2 = k(k1, P1, P2)
    k3 = k(k2, P2, P3)
    tt = k1 + t * (k2 - k1)
    L = lambda a, b, u: tuple(a[c] * (1 - u) + b[c] * u for c in range(3))
    A1 = L(P0, P1, (tt - k0) / (k1 - k0))
    A2 = L(P1, P2, (tt - k1) / (k2 - k1))
    A3 = L(P2, P3, (tt - k2) / (k3 - k2))
    B1 = L(A1, A2, (tt - k0) / (k2 - k0))
    B2 = L(A2, A3, (tt - k1) / (k3 - k1))
    return L(B1, B2, (tt - k1) / (k2 - k1))


class Curve:
    """The loader's two-pass tables: pass 1 pure arc length, pass 2 rho integrated vs s/Smax."""

    def __init__(self, pts, dens, frames, closed):
        self.pts, self.dens, self.N, self.closed = pts, dens, frames, closed
        self.nseg = len(pts) if closed else len(pts) - 1
        self.M = max(64, 256 * self.nseg)          # finer than the loader's 64; same limit
        M, nseg = self.M, self.nseg
        self.sampG = [nseg * k / M for k in range(M + 1)]
        P = [catmull(pts, closed, g) for g in self.sampG]
        self.sampS = [0.0]
        for k in range(1, M + 1):
            self.sampS.append(self.sampS[-1] + math.dist(P[k - 1], P[k]))
        self.Smax = self.sampS[M]
        invS = 1.0 / self.Smax
        self.sampC = [0.0]
        for k in range(1, M + 1):
            self.sampC.append(self.sampC[-1]
                              + self.rho(self.sampS[k] * invS) * (self.sampS[k] - self.sampS[k - 1]))
        self.Cmax = self.sampC[M]

    def rho(self, u):
        d = self.dens
        if not d:
            return 1.0
        if u <= d[0][0]:
            return d[0][1]
        if u >= d[-1][0]:
            return d[-1][1]
        for j in range(len(d) - 1):
            if d[j][0] <= u <= d[j + 1][0]:
                sp = d[j + 1][0] - d[j][0]
                f = (u - d[j][0]) / sp if sp > 1e-12 else 0.0
                return d[j][1] + (d[j + 1][1] - d[j][1]) * f
        return d[-1][1]

    def _inv(self, tbl, target, hi_val):
        if target <= 0.0:
            return 0.0
        if target >= tbl[-1]:
            return hi_val
        lo, hi = 0, self.M
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if tbl[mid] <= target:
                lo = mid
            else:
                hi = mid
        c0, c1 = tbl[lo], tbl[lo + 1]
        f = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
        return self.sampG[lo] + (self.sampG[lo + 1] - self.sampG[lo]) * f

    def g_of_frame(self, i):
        fr = i / self.N if self.closed else (0.5 if self.N == 1 else i / (self.N - 1))
        return self._inv(self.sampC, fr * self.Cmax, float(self.nseg))

    def arc_at_g(self, g):
        kf = g / self.nseg * self.M
        lo = min(max(int(kf), 0), self.M - 1)
        f = kf - lo
        return self.sampS[lo] + (self.sampS[lo + 1] - self.sampS[lo]) * f

    def eye_dir(self, i):
        """Frame i's eye and `look tangent` direction (lookAheadFrac 0.045, per ftsl.h)."""
        g = self.g_of_frame(i)
        eye = catmull(self.pts, self.closed, g)
        s = self.arc_at_g(g) + 0.045 * self.Smax
        gT = self._inv(self.sampS, s % self.Smax if self.closed else min(s, self.Smax),
                       float(self.nseg))
        tgt = catmull(self.pts, self.closed, gT)
        d = [tgt[c] - eye[c] for c in range(3)]
        n = math.sqrt(sum(x * x for x in d)) or 1.0
        return eye, tuple(x / n for x in d)


def main():
    pts, dens, frames, closed = parse_curve(SCENE)
    cv = Curve(pts, dens, frames, closed)
    print("points %d   frames %d   closed %s   spline length %.3f m   Cmax %.1f"
          % (len(pts), frames, closed, cv.Smax, cv.Cmax))

    # arc-length t of each control point, and the frame nearest it
    fs = [cv.arc_at_g(cv.g_of_frame(i)) / cv.Smax for i in range(frames)]
    print("\n  idx      x      y      z     arc-t   ~frame   heading")
    for j, p in enumerate(pts):
        t = cv.arc_at_g(float(j)) / cv.Smax
        i = min(range(frames), key=lambda k: min(abs(fs[k] - t), 1 - abs(fs[k] - t)))
        _, d = cv.eye_dir(i)
        print("  %3d  %6.2f %6.2f %6.2f    %.4f    %3d    (%+.2f %+.2f %+.2f)"
              % (j, p[0], p[1], p[2], t, i, d[0], d[1], d[2]))

    if len(sys.argv) > 1 and sys.argv[1] == "--rotate":
        j0 = int(sys.argv[2])
        t0 = cv.arc_at_g(float(j0)) / cv.Smax
        print("\nrotate so control point %d (arc-t %.4f) becomes the new START:" % (j0, t0))
        print("  shifted density_at stops (old t - %.4f, mod 1, re-sorted):" % t0)
        sh = sorted(((t - t0) % 1.0, r) for t, r in dens)
        # a closed track needs an explicit wrap pair so 0 and 1 agree
        if sh and sh[0][0] > 1e-9:
            r0 = cv.rho(t0)
            sh = [(0.0, r0)] + sh + [(1.0, r0)]
        out = []
        for t, r in sh:
            out.append("density_at %.3f %s" % (t, ("%g" % r)))
        for k in range(0, len(out), 4):
            print("    " + "  ".join(out[k:k + 4]))
        print("\n  rotated point order: " + " ".join(
            str((j0 + k) % len(pts)) for k in range(len(pts))))


main()
