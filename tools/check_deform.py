"""Regression check: a mesh's deformers (ROADMAP 11 stage E).

    D:/python314/python.exe -u tools/check_deform.py [--out <dir>] [--exe ftrace.exe]

Each case loads a scene with `ftrace -export-tris` (no render) and reads the triangles back:

  * falloff moves a vertex by the handle's move times its profile's weight at the vertex's
    distance (smooth, linear and gaussian), and leaves the vertices past the radius alone.
  * rbf takes each handle's point exactly to its target, moves the mesh exactly so when every
    handle moves alike (a translation), or by one rotation (four handles off a plane: the affine
    part), and is a translation with a single handle.
  * elastic holds each handle's vertices at their targets; a bar clamped at its root, one tip corner
    pulled where a 60-degree arc would put it, bends keeping its length within 3 % and its worst
    edge strain under half that of the falloff making the same move; a bar whose two ends move
    alike moves rigidly.
  * Deformers stack in order, each on the last one's output.
  * A keyed handle gives each vertex keys: the instant's shape is the middle key's, the shape at a
    key's time (-export-tris-at) is the still deform to that key's target, and between two keys
    each corner is their blend.
  * A dynamic elastic (`dynamic { cache ... frame N }`): a frame not baked is refused, naming the
    bake; `-bake-elastic` bakes the shot; a bar whose root is lifted sharply and held lags behind
    it, overshoots and settles; a bake resumed part-way writes byte for byte the records one bake
    writes; a record baked for other settings is refused, by a render and by a bake continuing it.
  * `-checkbvh` passes on a mesh a keyed handle waves: the motion trees bound it at every time.
  * The guide pass carries a deforming surface: its prevposition is where it sat a frame earlier.
  * A blurred render of a mesh an rbf handle slides matches the same mesh slid by a group's keys,
    and differs from the still render.
  * Errors: an unknown deformer, keyed handles with different counts, a deforming mesh in a
    moving group, an elastic handle that holds nothing.
Exits 1 on any failure.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_motion import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
LIGHT = 'light area { origin -0.5 3 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:bb6500  power 40 }'
LIGHT_TRIS = 2       # the light's quad, written after the mesh: the export's last two triangles


def grid_obj(path, n=21, size=2.0):
    """An n x n vertex grid in the xz plane (y = 0), `size` across, centred on the origin."""
    lines = []
    for j in range(n):
        for i in range(n):
            lines.append("v %r 0 %r" % (-size / 2 + size * i / (n - 1), -size / 2 + size * j / (n - 1)))
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i + 1
            lines.append("f %d %d %d" % (a, a + n, a + 1))
            lines.append("f %d %d %d" % (a + 1, a + n, a + n + 1))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def bar_obj(path, segs=20, length=2.0, w=0.2):
    """A box bar along x, from 0 to `length`, w x w across, cut into `segs` rings of 4 vertices,
    capped at both ends."""
    lines, ring = [], []
    for k in range(segs + 1):
        x = length * k / segs
        for (y, z) in ((-w / 2, -w / 2), (w / 2, -w / 2), (w / 2, w / 2), (-w / 2, w / 2)):
            lines.append("v %r %r %r" % (x, y, z))
    for k in range(segs):
        for s in range(4):
            a = k * 4 + s + 1
            b = k * 4 + (s + 1) % 4 + 1
            lines.append("f %d %d %d" % (a, b, b + 4))
            lines.append("f %d %d %d" % (a, b + 4, a + 4))
    lines.append("f 1 3 2")
    lines.append("f 1 4 3")
    e = segs * 4
    lines.append("f %d %d %d" % (e + 1, e + 2, e + 3))
    lines.append("f %d %d %d" % (e + 1, e + 3, e + 4))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def read_obj_tris(path):
    """The corners of an OBJ's triangles, (T, 3, 3)."""
    vs, fs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                vs.append([float(x) for x in t[1:4]])
            elif t[0] == "f":
                fs.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    v = np.array(vs)
    return v[np.array(fs)]


class Ctx:
    def __init__(self, exe, out):
        self.exe, self.out, self.ok, self.n = exe, out, True, 0

    def verdict(self, name, good, detail=""):
        self.ok &= bool(good)
        print("%-66s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""), flush=True)

    def scene(self, body, interval=True):
        self.n += 1
        p = os.path.join(self.out, "s%02d.ftsl" % self.n)
        head = "scene { units meters%s }" % ("  motion_interval -0.5 0.5" if interval else "")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join([head, 'material "m" { type diffuse reflect 0.6 }', body, LIGHT,
                             'camera "c" { eye 0 3 4  look_at 0 0 0  up 0 1 0  fov_y 40  film { res 40 30 } }']) + NL)
        return p

    def export(self, body, at=None, interval=True):
        """The scene's triangles, or the error text if it fails to load."""
        p = self.scene(body, interval)
        o = p[:-5] + ".obj"
        if os.path.exists(o):
            os.remove(o)
        cmd = [self.exe, "-in", p, "-export-tris", o] + (["-export-tris-at", repr(at)] if at is not None else [])
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(o):
            return None, r.stdout + r.stderr
        return read_obj_tris(o)[:-LIGHT_TRIS], r.stdout + r.stderr


def smooth(s):
    return (1 - s * s) ** 3 if s < 1 else 0.0


def linear(s):
    return 1 - s if s < 1 else 0.0


def gaussian(s):
    if s >= 1:
        return 0.0
    e4 = math.exp(-4.0)
    return (math.exp(-4 * s * s) - e4) / (1 - e4) * (1 - s * s) ** 2


def max_diff(t, want):
    """The largest coordinate difference between triangles `t` and `want`; 1 when `t` failed."""
    return float(np.abs(t - want).max()) if t is not None else 1.0


def centreline(rest, now):
    """The length of the bar's centreline: the polyline through its rings' centres."""
    rp, tp = rest.reshape(-1, 3), now.reshape(-1, 3)
    xs = np.round(rp[:, 0], 6)
    cs = [tp[xs == x].mean(axis=0) for x in np.unique(xs)]
    return float(sum(np.linalg.norm(cs[i + 1] - cs[i]) for i in range(len(cs) - 1)))


def edge_strain(rest, now):
    """The worst relative change of any triangle edge's length."""
    worst = 0.0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        l0 = np.linalg.norm(rest[:, a] - rest[:, b], axis=1)
        l1 = np.linalg.norm(now[:, a] - now[:, b], axis=1)
        worst = max(worst, float(np.max(np.abs(l1 / l0 - 1))))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_deform"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    c = Ctx(os.path.abspath(a.exe), out)
    grid, bar = os.path.join(out, "grid.obj").replace(chr(92), "/"), os.path.join(out, "bar.obj").replace(chr(92), "/")
    grid_obj(grid)
    bar_obj(bar)

    def mesh(f, deform):
        return 'mesh "g" { file "%s"  material m  %s }' % (f, deform)

    rest, log = c.export(mesh(grid, ""))
    rbar, log2 = c.export(mesh(bar, ""))
    if rest is None or rbar is None:
        c.verdict("the rest meshes load", False, (log + log2).strip()[-300:])
        print("FAIL")
        return 1

    # --- falloff ---
    for prof, fn in (("smooth", smooth), ("linear", linear), ("gaussian", gaussian)):
        t, log = c.export(mesh(grid, "deform falloff { profile %s  handle { at 0 0 0  to 0 0.4 0  radius 0.6 } }" % prof))
        if t is None:
            c.verdict("falloff %s loads" % prof, False, log[-300:])
            continue
        r = np.linalg.norm(rest[..., [0, 2]], axis=-1)
        want = rest.copy()
        want[..., 1] += 0.4 * np.vectorize(fn)(r / 0.6)
        e = np.abs(t - want).max()
        c.verdict("falloff %s: each vertex moves by its weight" % prof, e < 1e-9, "worst %.2g m" % e)
    t, _ = c.export(mesh(grid, "deform falloff { handle { at 0 0 0  to 0 0.4 0  radius 0.6 } }"))
    far = np.linalg.norm(rest[..., [0, 2]], axis=-1) >= 0.6
    c.verdict("... and past the radius nothing moves", t is not None and max_diff(t[far], rest[far]) == 0.0)

    # --- rbf ---
    hs = [((-0.5, 0, -0.5), (-0.5, 0.2, -0.5)), ((0.5, 0, -0.5), (0.5, -0.1, -0.4)),
          ((0.5, 0, 0.5), (0.6, 0.3, 0.5)), ((-0.5, 0, 0.5), (-0.5, 0.0, 0.5)), ((0, 0, 0), (0, 0.25, 0))]
    body = " ".join("handle { at %r %r %r  to %r %r %r }" % (h[0] + h[1]) for h in hs)
    t, log = c.export(mesh(grid, "deform rbf { %s }" % body))
    if t is None:
        c.verdict("rbf loads", False, log[-300:])
    else:
        worst = 0.0
        for at, to in hs:
            m = np.all(np.abs(rest - np.array(at)) < 1e-9, axis=-1)
            worst = max(worst, np.abs(t[m] - np.array(to)).max())
        c.verdict("rbf takes each handle's point to its target", worst < 1e-9, "worst %.2g m" % worst)
    body = " ".join("handle { at %r %r %r  to %r %r %r }" % (h[0] + tuple(float(x) for x in np.add(h[0], (0.1, 0.2, -0.3))))
                    for h in hs)
    t, _ = c.export(mesh(grid, "deform rbf { %s }" % body))
    e = max_diff(t, rest + np.array([0.1, 0.2, -0.3]))
    c.verdict("... handles moved alike move the mesh exactly so", e < 1e-9, "worst %.2g m" % e)
    t, _ = c.export(mesh(grid, "deform rbf { handle { at 0.3 0 0.3  to 0.5 0.1 0.3 } }"))
    e = max_diff(t, rest + np.array([0.2, 0.1, 0.0]))
    c.verdict("... one handle is a translation", e < 1e-9, "worst %.2g m" % e)
    ang = math.radians(30)
    R = np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
    pts = [(0, 0, 0), (2, 0, 0), (0.6, 0.1, 0.1), (1.4, -0.1, 0.1), (1.0, 0.1, -0.1)]
    body = " ".join("handle { at %r %r %r  to %r %r %r }" % (p + tuple(float(x) for x in R @ np.array(p)))
                    for p in pts)
    t, log = c.export(mesh(bar, "deform rbf { kernel triharmonic  %s }" % body))
    e = max_diff(t, rbar @ R.T)
    c.verdict("... handles turned alike turn the mesh (the affine part)", e < 1e-7, "worst %.2g m" % e)

    # --- elastic ---
    # The root clamped over three rings, one tip corner pulled to where a 60-degree arc of the bar
    # would put it: the elastic bends the bar, keeping its length; a falloff making the same move
    # strains it far more.
    bend = ("handle { at 0 0 0  to 0 0 0  radius 0.25 }  handle { at 2 0.1 0.1  to 1.65 1.05 0.1  radius 0.01 }")
    t, log = c.export(mesh(bar, "deform elastic { %s }" % bend))
    if t is None:
        c.verdict("elastic loads", False, log[-300:])
    else:
        held = np.linalg.norm(rbar - np.array([0.0, 0, 0]), axis=-1) <= 0.25
        tip = np.linalg.norm(rbar - np.array([2.0, 0.1, 0.1]), axis=-1) <= 0.01
        e = max(np.abs(t[held] - rbar[held]).max(), np.abs(t[tip] - np.array([1.65, 1.05, 0.1])).max())
        c.verdict("elastic holds each handle's vertices at their targets", e < 1e-9, "worst %.2g m" % e)
        se, length = edge_strain(rbar, t), centreline(rbar, t)
        tf, _ = c.export(mesh(bar, "deform falloff { handle { at 2 0.1 0.1  to 1.65 1.05 0.1  radius 2.0 } }"))
        sf = edge_strain(rbar, tf) if tf is not None else 0.0
        c.verdict("... and bends a bar, keeping its length (and half falloff's strain)",
                  abs(length / 2.0 - 1) < 0.03 and se < sf / 2,
                  "centreline %.3f m of 2; worst strain %.3f, falloff's %.3f" % (length, se, sf))
    t, _ = c.export(mesh(bar, "deform elastic { handle { at 0 0 0  to 0.2 0.3 0  radius 0.15 }  "
                              "handle { at 2 0 0  to 2.2 0.3 0  radius 0.15 } }"))
    e = max_diff(t, rbar + np.array([0.2, 0.3, 0]))
    c.verdict("... both ends moved alike move it rigidly", e < 1e-6, "worst %.2g m" % e)

    # --- a stack ---
    t, _ = c.export(mesh(grid, "deform rbf { handle { at 0 0 0  to 0.1 0 0 } }  "
                               "deform falloff { handle { at 0.1 0 0  to 0.1 0.4 0  radius 0.6 } }"))
    shifted = rest + np.array([0.1, 0, 0])
    want = shifted.copy()
    want[..., 1] += 0.4 * np.vectorize(smooth)(np.linalg.norm(shifted[..., [0, 2]] - [0.1, 0], axis=-1) / 0.6)
    e = max_diff(t, want)
    c.verdict("deformers stack in order, each on the last one's output", e < 1e-9, "worst %.2g m" % e)

    # --- keyed handles: vertex keys ---
    keys = [(0, 0.0, 0), (0, 0.4, 0), (0, 0.2, 0.3)]
    keyed = ("deform falloff { handle { at 0 0 0  radius 0.6  motion { %s } } }"
             % " ".join("key { to %r %r %r }" % k for k in keys))
    t0, log = c.export(mesh(grid, keyed))
    if t0 is None:
        c.verdict("a keyed handle loads", False, log[-300:])
    else:
        still = [c.export(mesh(grid, "deform falloff { handle { at 0 0 0  to %r %r %r  radius 0.6 } }" % k))[0]
                 for k in keys]
        e = max_diff(t0, still[1])
        c.verdict("a keyed handle: the instant's shape is the middle key's", e < 1e-12, "worst %.2g m" % e)
        t1, _ = c.export(mesh(grid, keyed), at=0.5)
        e = max_diff(t1, still[2])
        c.verdict("... the shape at a key's time is the still deform to it", e < 1e-12, "worst %.2g m" % e)
        th, _ = c.export(mesh(grid, keyed), at=-0.25)
        e = max_diff(th, 0.5 * (still[0] + still[1])) if still[0] is not None and still[1] is not None else 1.0
        c.verdict("... and between two keys each corner is their blend", e < 1e-12, "worst %.2g m" % e)

    # --- the motion trees hold a deforming mesh at every time (-checkbvh: BVH against a linear walk,
    # every other ray at a random time) ---
    wave = ("deform falloff { handle { at 0 0 0  radius 0.8  motion { key { to 0 -0.5 0 } key { to 0.3 0 0.2 } "
            "key { to 0 0.6 0 } } } }")
    p = c.scene(mesh(grid, wave))
    r = subprocess.run([c.exe, "-in", p, "-checkbvh"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    line = [x for x in (r.stdout + r.stderr).splitlines() if "checkbvh" in x][-1:] or [(r.stdout + r.stderr)[-200:]]
    c.verdict("-checkbvh holds a deforming mesh at every time", r.returncode == 0 and "PASS" in line[0], line[0][:90])

    # --- the guide pass on a deforming surface; a blurred render ---
    slide = 'deform rbf { handle { at 0 0 0  motion { key { to -0.2 0 0 } key { to 0 0 0 } key { to 0.2 0 0 } } } }'
    floor = 'quad { origin -3 -0.5 -3  u 0 0 6  v 6 0 0  material m }'
    light = 'light area { origin -0.5 2.5 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:bb6500  power 40 }'
    def render(body, name, spp, extra=(), shutter=True):
        cam = ('camera "c" { eye 0 3 4  look_at 0 0 0  up 0 1 0  fov_y 40  %sfilm { res 80 60 } }'
               % ("shutter_open -0.5  shutter_close 0.5  " if shutter else ""))
        p = os.path.join(out, name + ".ftsl")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(["scene { units meters  motion_interval -0.5 0.5 }",
                             'material "m" { type diffuse reflect 0.6 }', floor, light, body, cam]) + NL)
        png = os.path.join(out, name + ".png")
        r = subprocess.run([c.exe, "-in", p, "-mode", "R", "-device", "cpu", "-spp", str(spp), "-seed", "3",
                            "-hdr", "-no-meter", "-o", png] + list(extra),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-1200:])
            return None
        return png[:-4]

    stem = render(mesh(grid, slide), "aov_deform", 1, ["-write-aovs"])
    if stem is None:
        c.verdict("the guide pass renders a deforming mesh", False)
    else:
        pos = read_pfm(stem + ".position.pfm")
        prev = read_pfm(stem + ".prevposition.pfm")
        hit = read_pfm(stem + ".depth.pfm")[..., 0] > 0           # the background's position is 0
        on = hit & (np.abs(pos[..., 1]) < 1e-6) & (np.abs(pos[..., 0]) < 0.9) & (np.abs(pos[..., 2]) < 0.9)
        e = np.abs(prev[on] - (pos[on] - np.array([0.4, 0, 0]))).max() if on.any() else 1.0
        c.verdict("the guide pass: a deforming surface sat where its keys put it", on.any() and e < 1e-6,
                  "worst %.2g m over %d pixels" % (e, on.sum()))
    sd = render(mesh(grid, slide), "blur_deform", 256)
    sg = render('group { motion { key { translate -0.2 0 0 } key { translate 0 0 0 } key { translate 0.2 0 0 } }  %s }'
                % mesh(grid, ""), "blur_group", 256)
    ss = render(mesh(grid, slide), "blur_still", 256, shutter=False)
    if sd is None or sg is None or ss is None:
        c.verdict("a blurred deforming mesh renders", False)
    else:
        d, g, st = read_pfm(sd + ".pfm"), read_pfm(sg + ".pfm"), read_pfm(ss + ".pfm")
        rel = np.abs(d - g).mean() / g.mean()
        off = np.abs(d - st).mean() / st.mean()
        c.verdict("a blurred slide by an rbf handle matches a group's", rel < 0.03 and off > 3 * rel,
                  "mean diff %.3f, from the still render %.3f" % (rel, off))

    # --- the dynamic elastic: baked through a shot ---
    dynamic_tests(c, bar, rbar)

    # --- errors ---
    for name, body, want in (
            ("an unknown deformer", mesh(grid, "deform twist { handle { at 0 0 0 } }"), "unknown deformer"),
            ("keyed handles with different counts",
             mesh(grid, "deform falloff { handle { at 0 0 0  radius 0.5  motion { key { to 0 0 0 } key { to 0 1 0 } key { to 0 0 0 } } }"
                        "  handle { at 0.5 0 0  radius 0.5  motion { key { to 0 0 0 } key { to 0 1 0 } } } }"),
             "same number of keys"),
            ("a deforming mesh in a moving group",
             "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  %s }"
             % mesh(grid, "deform falloff { handle { at 0 0 0  to 0 1 0  radius 0.5 } }"), "moving group"),
            ("a handle's target it cannot read",
             mesh(grid, "deform falloff { handle { at 0 0 0  to 0 x 0  radius 0.5 } }"), "three numbers"),
            ("an elastic handle that holds nothing",
             mesh(bar, "deform elastic { handle { at 5 5 5  to 5 6 5  radius 0.1 } }"), "holds no vertex")):
        t, log = c.export(body)
        c.verdict("refused: " + name, t is None and want in log, log.strip().splitlines()[-1][:90] if log.strip() else "")

    print("PASS" if c.ok else "FAIL")
    return 0 if c.ok else 1


DYN_FRAMES = 40
DYN_STIFF = 2.0e5


def lift(k):
    """The bar's held root's lift at frame k: up 0.3 m over frames 2-4, then still."""
    return 0.3 * min(1.0, max(0.0, (k - 2) / 2.0))


def dynamic_tests(c, bar, rbar):
    """A bar whose root is lifted sharply and held: the dynamic elastic lags, overshoots and
    settles where the quasi-static one is at once; a bake resumed part-way writes what one bake
    writes; a frame not baked, or baked for other settings, is refused."""
    shot = os.path.join(c.out, "dyn")
    os.makedirs(shot, exist_ok=True)
    for f in os.listdir(shot):
        os.remove(os.path.join(shot, f))
    cache = os.path.join(shot, "bar_%04d.ftec").replace(chr(92), "/")

    def frame_scene(k, stiff=DYN_STIFF, name="f%04d"):
        body = ('mesh "bar" { file "%s"  material m  deform elastic { dynamic { cache "%s"  frame %d  '
                'mass 1  stiffness %r  damping 0.9 }  handle { at 0 0 0  to 0 %r 0  radius 0.25 } } }'
                % (bar, cache, k, stiff, lift(k)))
        p = os.path.join(shot, (name % k) + ".ftsl")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(["scene { units meters }", 'material "m" { type diffuse reflect 0.6 }', body, LIGHT,
                             'camera "c" { eye 1 1 5  look_at 1 0 0  up 0 1 0  fov_y 40  film { res 40 30 } }']) + NL)
        return p

    def bake(first, last, name="f%04d"):
        r = subprocess.run([c.exe, "-bake-elastic", os.path.join(shot, name + ".ftsl"), str(first), str(last)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def tip(k):
        """The far end's height at frame k, read back by -export-tris."""
        o = os.path.join(shot, "x%04d.obj" % k)
        r = subprocess.run([c.exe, "-in", os.path.join(shot, "f%04d.ftsl" % k), "-export-tris", o],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            return None, r.stdout + r.stderr
        t = read_obj_tris(o)[:-LIGHT_TRIS]
        far = np.abs(rbar[..., 0] - 2.0) < 1e-9
        return float((t[far][:, 1] - rbar[far][:, 1]).mean()), ""

    for k in range(DYN_FRAMES):
        frame_scene(k)
    _, log = tip(5)
    c.verdict("a frame not yet baked is refused, naming the bake", "bake the shot first" in log,
              log.strip().splitlines()[-1][:90] if log.strip() else "")
    rc, log = bake(0, DYN_FRAMES - 1)
    c.verdict("-bake-elastic bakes the shot", rc == 0 and "baked frames" in log, log.strip()[-200:] if rc else "")
    ys = []
    for k in range(DYN_FRAMES):
        y, log = tip(k)
        if y is None:
            c.verdict("a baked frame loads", False, log.strip()[-200:])
            return
        ys.append(y)
    lag = ys[3] < 0.85 * lift(3)
    over = max(ys) > 0.3 * 1.03
    settled = abs(ys[-1] - 0.3) < 0.01
    c.verdict("the dynamic elastic lags, overshoots and settles", lag and over and settled,
              "tip at frame 3 %.3f (root %.3f), peak %.3f, last %.3f" % (ys[3], lift(3), max(ys), ys[-1]))
    # A bake resumed part-way writes the records one bake writes.
    one = {f: open(os.path.join(shot, f), "rb").read() for f in os.listdir(shot) if f.endswith(".ftec")}
    for f in one:
        if int(f[4:8]) >= 20:
            os.remove(os.path.join(shot, f))
    rc, log = bake(20, DYN_FRAMES - 1)
    same = rc == 0 and all(open(os.path.join(shot, f), "rb").read() == b for f, b in one.items())
    c.verdict("... a bake resumed part-way writes what one bake writes", same, log.strip()[-200:] if rc else "")
    # A record baked for other settings is refused, both by the render and by a bake continuing it.
    frame_scene(21, stiff=DYN_STIFF * 2, name="g%04d")
    r = subprocess.run([c.exe, "-in", os.path.join(shot, "g0021.ftsl"), "-export-tris", os.path.join(shot, "g.obj")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    c.verdict("... a record baked for other settings is refused", r.returncode != 0 and "another" in r.stdout + r.stderr,
              (r.stdout + r.stderr).strip()[-120:])
    rc, log = bake(21, 21, name="g%04d")
    c.verdict("... and a bake will not continue one", rc != 0 and "another" in log, log.strip()[-120:])


if __name__ == "__main__":
    sys.exit(main())
