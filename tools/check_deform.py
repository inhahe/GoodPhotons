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
    alike moves rigidly; a 2 mm bar bends as the 2 m one does, scaled (its rotations judged
    against its own size).
  * Deformers stack in order, each on the last one's output.
  * Handles are in the mesh's own coordinates: a mesh translated, turned and scaled unevenly by
    its own transform and a group's deforms as the bare mesh does, then is placed; a vertex the
    deformer leaves alone keeps its placed position exactly.
  * A deforming mesh rides a moving group: at a time its triangles are the group's transform then
    applied to the deformed shape (keyed handles and not); -checkbvh holds it at every time; a
    group sliding one way and a deformer sliding it back render as the still mesh.
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
  * At a time, a deforming surface's patterns read the instant's point and normal: a pattern
    stepping at a sheet's plane reads the side it faces, and a pattern of the normal on a sheet an
    rbf turns 90 degrees reads the instant's, each rendering exactly as the constant it should
    be. A fin a deformer folds exactly flat at a key renders as one folding a hair short of it
    (no normal is taken from a collapsed face).
  * Blurred mode B sees a mirror sheet a deformer lifts and lowers as mode R does, on the CPU and
    on the GPU.
  * On the GPU (where there is one), a waving mesh riding a moving group renders blurred as it
    does on the CPU.
  * Handle keys that all agree are no motion.
  * The dynamic elastic, further: under a motion interval it has keys though its handles have
    none; a bake of an unchanged shot keeps every record; one after a frame's handle target
    changes re-simulates from that frame on, and a render of it before then is refused;
    `ftrace -stop` stops a bake cleanly; a shot whose mesh a translate places differently every
    frame bakes, and writes byte for byte the unplaced shot's records (its cache key is the mesh
    in its own coordinates, exactly).
  * The guide pass of a deforming mesh riding a turning group: where its deformer moves nothing,
    its prevposition is its rigid twin's, carried by the group's motion rather than along a chord.
  * A face a deformer collapses at the instant: a pattern read on it at another time is finite (it
    renders as one collapsing a hair short of it).
  * Errors: an unknown deformer, keyed handles with different counts, an elastic handle that
    holds nothing, a falloff handle with no radius, a dynamic elastic's own `iterations`.
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
    # ARAP is unchanged by a similarity. A 2 mm bar's one-rings are ~1e-9 m^2: an absolute
    # threshold in the rotation's eigen-solve once left them unrotated (5.7 cm off, of 2 m).
    full, _ = c.export(mesh(bar, "deform elastic { %s }" % bend))
    sc = 1e-3
    tiny_bar = os.path.join(out, "bar_2mm.obj").replace(chr(92), "/")
    bar_obj(tiny_bar, length=2.0 * sc, w=0.2 * sc)
    tiny = ("handle { at 0 0 0  to 0 0 0  radius %r }  handle { at %r %r %r  to %r %r %r  radius %r }"
            % tuple(sc * x for x in (0.25, 2, 0.1, 0.1, 1.65, 1.05, 0.1, 0.01)))
    small, _ = c.export(mesh(tiny_bar, "deform elastic { %s }" % tiny))
    e = max_diff(small / sc, full) if small is not None and full is not None else 1.0
    c.verdict("... and a 2 mm bar bends as the 2 m one does, scaled", e < 1e-6, "worst %.2g m (of 2)" % e)

    # --- a stack ---
    t, _ = c.export(mesh(grid, "deform rbf { handle { at 0 0 0  to 0.1 0 0 } }  "
                               "deform falloff { handle { at 0.1 0 0  to 0.1 0.4 0  radius 0.6 } }"))
    shifted = rest + np.array([0.1, 0, 0])
    want = shifted.copy()
    want[..., 1] += 0.4 * np.vectorize(smooth)(np.linalg.norm(shifted[..., [0, 2]] - [0.1, 0], axis=-1) / 0.6)
    e = max_diff(t, want)
    c.verdict("deformers stack in order, each on the last one's output", e < 1e-9, "worst %.2g m" % e)

    wave = ("deform falloff { handle { at 0 0 0  radius 0.8  motion { key { to 0 -0.5 0 } key { to 0.3 0 0.2 } "
            "key { to 0 0.6 0 } } } }")

    # --- the mesh's own coordinates ---
    # Its own transform and a group's, uneven scale among them: the bar deforms as the bare one,
    # then is placed. The placement is read off the undeformed bar placed the same way.
    place = "translate 0.3 -0.2 0.5  rotate 10 35 -20  scale 1.5 0.7 1.2"
    def placed(deform):
        return 'group { translate -1 0.5 2  rotate 0 20 0  %s }' % (
            'mesh "g" { file "%s"  material m  %s  %s }' % (bar, place, deform))
    flat, _ = c.export(placed(""))
    bent0, _ = c.export(mesh(bar, "deform elastic { %s }" % bend))
    bent, log = c.export(placed("deform elastic { %s }" % bend))
    if flat is None or bent0 is None or bent is None:
        c.verdict("a placed mesh deforms in its own coordinates", False, log[-300:])
    else:
        X = np.c_[rbar.reshape(-1, 3), np.ones(rbar.size // 3)]
        A, *_ = np.linalg.lstsq(X, flat.reshape(-1, 3), rcond=None)     # rest -> placed: the affine
        want = (np.c_[bent0.reshape(-1, 3), np.ones(bent0.size // 3)] @ A).reshape(bent0.shape)
        e = max_diff(bent, want)
        c.verdict("a placed mesh deforms in its own coordinates, then is placed", e < 1e-9, "worst %.2g m" % e)
        held = np.linalg.norm(rbar - np.array([0.0, 0, 0]), axis=-1) <= 0.25
        c.verdict("... and a vertex it leaves alone keeps its placed position exactly",
                  max_diff(bent[held], flat[held]) == 0.0)

    # --- riding a moving group ---
    # The group turns and slides through the shutter: at each time the triangles are its transform
    # then (read off the undeformed bar in the same group) applied to the deformed shape.
    ride = ("group { motion { key { translate -0.3 0 0  rotate 0 -20 0 } key { translate 0 0 0 } "
            "key { translate 0.3 0.1 0  rotate 0 20 10 } }  %s }")
    waved = ("deform falloff { handle { at 2 0 0  radius 0.8  motion { key { to 2 -0.3 0 } key { to 2 0 0 } "
             "key { to 2 0.4 0.2 } } } }")
    for name, deform in (("a still deformer", "deform elastic { %s }" % bend), ("a keyed one", waved)):
        worst = 0.0
        for at in (-0.4, -0.1, 0.25, 0.5):
            moved, _ = c.export(ride % mesh(bar, ""), at=at)
            mine, _ = c.export(mesh(bar, deform), at=at)
            both, log = c.export(ride % mesh(bar, deform), at=at)
            if moved is None or mine is None or both is None:
                worst = 1.0
                break
            X = np.c_[rbar.reshape(-1, 3), np.ones(rbar.size // 3)]
            A, *_ = np.linalg.lstsq(X, moved.reshape(-1, 3), rcond=None)
            want = (np.c_[mine.reshape(-1, 3), np.ones(mine.size // 3)] @ A).reshape(mine.shape)
            worst = max(worst, max_diff(both, want))
        c.verdict("a deforming mesh rides a moving group (%s)" % name, worst < 1e-9, "worst %.2g m" % worst)
    p = c.scene(ride % mesh(grid, wave))
    r = subprocess.run([c.exe, "-in", p, "-checkbvh"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    line = [x for x in (r.stdout + r.stderr).splitlines() if "checkbvh" in x][-1:] or [(r.stdout + r.stderr)[-200:]]
    c.verdict("-checkbvh holds a deforming mesh in a moving group", r.returncode == 0 and "PASS" in line[0],
              line[0][:90])

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
    p = c.scene(mesh(grid, wave))
    r = subprocess.run([c.exe, "-in", p, "-checkbvh"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    line = [x for x in (r.stdout + r.stderr).splitlines() if "checkbvh" in x][-1:] or [(r.stdout + r.stderr)[-200:]]
    c.verdict("-checkbvh holds a deforming mesh at every time", r.returncode == 0 and "PASS" in line[0], line[0][:90])

    # --- the guide pass on a deforming surface; a blurred render ---
    slide = 'deform rbf { handle { at 0 0 0  motion { key { to -0.2 0 0 } key { to 0 0 0 } key { to 0.2 0 0 } } } }'
    floor = 'quad { origin -3 -0.5 -3  u 0 0 6  v 6 0 0  material m }'
    light = 'light area { origin -0.5 2.5 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:bb6500  power 40 }'
    def render(body, name, spp, extra=(), shutter=True, mode="R", eye="0 3 4", look="0 0 0",
               device="cpu", seed=3, log=None):
        """`body` rendered in `mode` (`spp` samples, or photons in mode B) on `device` to
        <out>/<name>; its HDR stem, or None. `log`, a list, gets the render's output."""
        cam = ('camera "c" { eye %s  look_at %s  up 0 1 0  fov_y 40  %sfilm { res 80 60 } }'
               % (eye, look, "shutter_open -0.5  shutter_close 0.5  " if shutter else ""))
        p = os.path.join(out, name + ".ftsl")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(["scene { units meters  motion_interval -0.5 0.5 }",
                             'material "m" { type diffuse reflect 0.6 }', floor, light, body, cam]) + NL)
        png = os.path.join(out, name + ".png")
        budget = ["-n", str(spp)] if mode == "B" else ["-spp", str(spp)]
        r = subprocess.run([c.exe, "-in", p, "-mode", mode, "-device", device, *budget, "-seed", str(seed),
                            "-hdr", "-no-meter", "-o", png] + list(extra),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if log is not None:
            log.append(r.stdout + r.stderr)
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
    # Riding a turning group, where its falloff moves nothing (beyond the handle's radius), a
    # deforming mesh's prevposition is its rigid twin's: carried by the group's motion, not along
    # a chord to the interval's end.
    turn = ("group { motion { key { rotate 0 -20 0 } key { rotate 0 0 0 } key { rotate 0 20 0 } }  %s }")
    local = ("deform falloff { handle { at 0 0 0  radius 0.3  motion { key { to 0 -0.1 0 } key { to 0 0 0 } "
             "key { to 0 0.1 0 } } } }")
    sdef = render(turn % mesh(grid, local), "aov_turn_def", 1, ["-write-aovs"])
    srig = render(turn % mesh(grid, ""), "aov_turn_rig", 1, ["-write-aovs"])
    if sdef is None or srig is None:
        c.verdict("the guide pass renders a deforming mesh in a turning group", False)
    else:
        pd, pr = read_pfm(sdef + ".prevposition.pfm"), read_pfm(srig + ".prevposition.pfm")
        pos = read_pfm(srig + ".position.pfm")
        hit = read_pfm(srig + ".depth.pfm")[..., 0] > 0
        far = hit & (np.abs(pos[..., 1]) < 1e-6) & (np.hypot(pos[..., 0], pos[..., 2]) > 0.45) & \
              (np.abs(pos[..., 0]) < 0.9) & (np.abs(pos[..., 2]) < 0.9)
        e = float(np.abs(pd[far] - pr[far]).max()) if far.any() else 1.0
        c.verdict("... and riding a turning group, it is carried as its rigid twin is", far.any() and e < 1e-5,
                  "worst %.2g m over %d pixels" % (e, far.sum()))
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
    back = 'deform rbf { handle { at 0 0 0  motion { key { to 0.2 0 0 } key { to 0 0 0 } key { to -0.2 0 0 } } } }'
    sb = render('group { motion { key { translate -0.2 0 0 } key { translate 0 0 0 } key { translate 0.2 0 0 } }  %s }'
                % mesh(grid, back), "blur_back", 256)
    if sb is None or ss is None:
        c.verdict("a group's slide undone by a deformer renders still", False)
    else:
        b, still = read_pfm(sb + ".pfm"), read_pfm(ss + ".pfm")
        rel = np.abs(b - still).mean() / still.mean()
        c.verdict("a group's slide undone by a deformer renders as the still mesh", rel < 0.01,
                  "mean diff %.4f of the still render" % rel)
    # On the GPU as on the CPU (where there is one): a waving mesh riding a moving group, blurred.
    ride_wave = ride % mesh(grid, wave)
    logs = []
    sg = [render(ride_wave, "gpu_ride_%d" % s, 256, device="gpu", seed=s, log=logs) for s in (1, 2)]
    if all(x is not None for x in sg) and all("-> GPU" in l for l in logs):
        sc_ = [render(ride_wave, "cpu_ride_%d" % s, 256, seed=s) for s in (1, 2)]
        from check_motion import agree, lum
        good, detail = agree([lum(read_pfm(x + ".pfm")) for x in sg], [lum(read_pfm(x + ".pfm")) for x in sc_])
        c.verdict("a waving mesh riding a moving group, blurred, renders on the GPU as on the CPU", good, detail)
    else:
        print("(no GPU here: the GPU's deforming-mesh case is skipped)")
    pattern_tests(c, grid, render)
    mirror_test(c, grid, render)

    # Handle keys that all agree move nothing: mode V under a shutter has no blur to say it drops.
    same = "deform falloff { handle { at 0 0 0  radius 0.5  motion { %s } } }" % " ".join(["key { to 0 0.3 0 }"] * 3)
    p = os.path.join(out, "nomotion.ftsl")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(["scene { units meters  motion_interval -0.5 0.5 }", 'material "m" { type diffuse reflect 0.6 }',
                         mesh(grid, same), light, 'camera "c" { eye 0 3 4  look_at 0 0 0  up 0 1 0  fov_y 40  '
                         'shutter_open -0.5  shutter_close 0.5  film { res 32 24 } }']) + NL)
    r = subprocess.run([c.exe, "-in", p, "-mode", "V", "-device", "cpu", "-spp", "1", "-n", "100000", "-no-meter",
                        "-o", os.path.join(out, "nomotion.png")], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    c.verdict("handle keys that all agree are no motion",
              r.returncode == 0 and "renders the frame's instant" not in r.stdout + r.stderr,
              "" if r.returncode == 0 else (r.stdout + r.stderr).strip()[-200:])

    # --- the dynamic elastic: baked through a shot ---
    dynamic_tests(c, bar, rbar)

    # --- errors ---
    for name, body, want in (
            ("an unknown deformer", mesh(grid, "deform twist { handle { at 0 0 0 } }"), "unknown deformer"),
            ("keyed handles with different counts",
             mesh(grid, "deform falloff { handle { at 0 0 0  radius 0.5  motion { key { to 0 0 0 } key { to 0 1 0 } key { to 0 0 0 } } }"
                        "  handle { at 0.5 0 0  radius 0.5  motion { key { to 0 0 0 } key { to 0 1 0 } } } }"),
             "same number of keys"),
            ("a handle's target it cannot read",
             mesh(grid, "deform falloff { handle { at 0 0 0  to 0 x 0  radius 0.5 } }"), "three numbers"),
            ("an elastic handle that holds nothing",
             mesh(bar, "deform elastic { handle { at 5 5 5  to 5 6 5  radius 0.1 } }"), "holds no vertex"),
            ("a falloff handle with no radius",
             mesh(grid, "deform falloff { handle { at 0 0 0  to 0 0.4 0 } }"), "needs a `radius`"),
            ("a dynamic elastic's own iterations",
             mesh(bar, 'deform elastic { iterations 5  dynamic { cache "x_%04d.ftec"  frame 0 }  '
                       'handle { at 0 0 0  radius 0.25 } }'), "rounds per step")):
        t, log = c.export(body)
        c.verdict("refused: " + name, t is None and want in log, log.strip().splitlines()[-1][:90] if log.strip() else "")

    print("PASS" if c.ok else "FAIL")
    return 0 if c.ok else 1


def same_render(c, name, a, b, tol):
    """Verdict: HDR renders `a` and `b` (stems; same seed) agree to `tol` of b's mean."""
    if a is None or b is None:
        c.verdict(name, False, "a render failed")
        return
    x, y = read_pfm(a + ".pfm"), read_pfm(b + ".pfm")
    rel = float(np.abs(x - y).mean() / max(float(y.mean()), 1e-12))
    c.verdict(name, rel < tol, "mean diff %.2g of the reference's" % rel)


def pattern_tests(c, grid, render):
    """At a time a deforming surface's patterns read the instant's point and normal, and a face a
    deformer folds flat at a key takes no normal from it: each case renders as a reference that
    must come out the same (the patterns draw no random numbers, so the same seed matches)."""
    # A sheet in the plane y = 0.3, bent at one corner by a keyed falloff, under a pattern that
    # steps exactly there: every hit reads the side its face faces (the instant's point nudged off
    # the instant's face), so it renders as that side's constant. (Read on the surface itself, the
    # point fell either side of 0.3 from hit to hit: 7.5 % of the image off.)
    def sheet(mat):
        return ('mesh "s" { file "%s"  material %s  translate 0 0.3 0  deform falloff { handle { at 0.9 0 -0.9  '
                'radius 0.6  motion { key { to 0.9 -0.3 -0.9 } key { to 0.9 0 -0.9 } key { to 0.9 0.3 -0.9 } } } } }'
                % (grid, mat))
    mats = NL.join(['pattern "step" { expr "0.1 + 0.8 * step(0.0, y - 0.3)" }',
                    'material "p" { type diffuse  reflect pattern:step }',
                    'material "q" { type diffuse  reflect 0.9 }'])
    eye = dict(eye="0 2.2 3.2", look="0 0.2 0")
    same_render(c, "a pattern stepping at a deforming sheet's plane reads one side at every time",
                render(mats + NL + sheet("p"), "pat_step", 64, **eye),
                render(mats + NL + sheet("q"), "pat_step_ref", 64, **eye), 1e-6)
    # A smooth-shaded plate, its normals tilted 45 degrees, that an rbf turns 90 degrees about x
    # each way through the shutter: a pattern of ny reads the instant's normal at every time.
    out = c.out
    tilted = os.path.join(out, "tilted.obj").replace(chr(92), "/")
    with open(tilted, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(["v -0.5 0 -0.5", "v 0.5 0 -0.5", "v 0.5 0 0.5", "v -0.5 0 0.5",
                         "vn 0 0.70710678 0.70710678", "f 1//1 3//1 2//1", "f 1//1 4//1 3//1"]) + NL)

    def turned(p, deg):        # about the x axis, in the plate's own coordinates
        a = math.radians(deg)
        return (p[0], p[1] * math.cos(a) - p[2] * math.sin(a), p[1] * math.sin(a) + p[2] * math.cos(a))
    hs = " ".join("handle { at %r %r %r  motion { %s } }"
                  % (turned(p, 0) + (" ".join("key { to %r %r %r }" % turned(p, d) for d in (-90, 0, 90)),))
                  for p in ((-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)))

    def plate(mat):
        return 'mesh "t" { file "%s"  material %s  translate 0 0.4 0  deform rbf { %s } }' % (tilted, mat, hs)
    mats = NL.join(['pattern "tilt" { expr "0.1 + 0.8 * step(0.9, ny)" }',
                    'material "p" { type diffuse  reflect pattern:tilt }',
                    'material "q" { type diffuse  reflect 0.1 }'])
    same_render(c, "a pattern of the normal on a turning sheet reads the instant's at every time",
                render(mats + NL + plate("p"), "pat_turn", 64, **eye),
                render(mats + NL + plate("q"), "pat_turn_ref", 64, **eye), 1e-6)
    # A fin a keyed falloff folds exactly flat at its first key (every coordinate a binary
    # fraction, so its corners are exactly collinear there): it renders as one folding a hair
    # short of flat. (A face normal stored for that key was NaN, and the hits between the first
    # two keys shaded black: 0.0021 of the image off, against 8e-5.)
    fin_obj = os.path.join(out, "fin.obj").replace(chr(92), "/")
    with open(fin_obj, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(["v -1 0 -0.25", "v 0 0.5 -0.5", "v 1 0 -0.25", "f 1 2 3"]) + NL)

    def fin(first):
        return ('mesh "s" { file "%s"  material m  deform falloff { handle { at 0 0.5 -0.5  radius 0.5  '
                'motion { key { to %s } key { to 0 0.5 -0.5 } key { to 0 1 -0.75 } } } } }' % (fin_obj, first))
    same_render(c, "a fin a deformer folds flat at a key renders as one folding a hair short",
                render(fin("0 0 -0.25"), "fin", 64, **eye), render(fin("0 0.001 -0.251"), "fin_ref", 64, **eye), 5e-4)
    # The same fin folded flat at the instant (the middle key), under a pattern of its position:
    # read at another time, the point is nudged along the hit's own normal, the instant's face
    # having none (it was NaN, and its samples were dropped).
    pat = NL.join(['pattern "px" { expr "0.2 + 0.6 * step(0.0, sin(9 * x + 3 * y))" }',
                   'material "pm" { type diffuse  reflect pattern:px }'])

    def fin_mid(mid):
        return (pat + NL + 'mesh "s" { file "%s"  material pm  deform falloff { handle { at 0 0.5 -0.5  radius 0.5  '
                'motion { key { to 0 0.5 -0.5 } key { to %s } key { to 0 1 -0.75 } } } } }' % (fin_obj, mid))
    same_render(c, "... and folded flat at the instant, its pattern at other times is finite",
                render(fin_mid("0 0 -0.25"), "fin_mid", 64, **eye),
                render(fin_mid("0 0.001 -0.251"), "fin_mid_ref", 64, **eye), 5e-3)


def mirror_test(c, grid, render):
    """A mirror sheet an rbf lifts and lowers through the shutter, under a lit sphere: blurred mode
    B, whose mirror connection must follow the sheet at each photon's time, agrees with blurred
    mode R over the sphere's reflection, on the CPU and (where there is one) on the GPU. (Taken as
    the still sheet, the connection found no mirror there: 0.16 of mode R.)"""
    body = NL.join(['material "mir" { type mirror }', 'sphere { center 0 0.8 -0.4  radius 0.3  material m }',
                    'mesh "mirror" { file "%s"  material mir  deform rbf { handle { at 0 0 0  motion { '
                    'key { to 0 -0.25 0 } key { to 0 0 0 } key { to 0 0.25 0 } } } } }' % grid])
    eye = dict(eye="0 2.2 3.2", look="0 0.2 0")
    sr, sb = render(body, "mirror_R", 512, **eye), render(body, "mirror_B", 40000000, mode="B", **eye)
    if sr is None or sb is None:
        c.verdict("blurred mode B sees a moving mirror as mode R does", False, "a render failed")
        return
    lr, lb = read_pfm(sr + ".pfm").mean(axis=-1), read_pfm(sb + ".pfm").mean(axis=-1)
    ratio = float(lb[30:50, 20:60].sum() / lr[30:50, 20:60].sum())   # the reflection, in the sheet
    c.verdict("blurred mode B sees a moving mirror as mode R does", abs(ratio - 1) < 0.1,
              "the reflection %.3f of mode R's" % ratio)
    logs = []
    sg = render(body, "mirror_B_gpu", 40000000, mode="B", device="gpu", log=logs, **eye)
    if sg is not None and all("-> GPU" in l for l in logs):
        lg = read_pfm(sg + ".pfm").mean(axis=-1)
        ratio = float(lg[30:50, 20:60].sum() / lr[30:50, 20:60].sum())
        c.verdict("... and on the GPU", abs(ratio - 1) < 0.1, "the reflection %.3f of mode R's" % ratio)
    else:
        print("(no GPU here: the GPU's moving-mirror case is skipped)")


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

    def frame_scene(k, stiff=DYN_STIFF, name="f%04d", up=None, cache=cache, interval=False, place=""):
        body = ('mesh "bar" { file "%s"  material m  %s  deform elastic { dynamic { cache "%s"  frame %d  '
                'mass 1  stiffness %r  damping 0.9 }  handle { at 0 0 0  to 0 %r 0  radius 0.25 } } }'
                % (bar, place, cache, k, stiff, lift(k) if up is None else up))
        p = os.path.join(shot, (name % k) + ".ftsl")
        head = "scene { units meters%s }" % ("  motion_interval -0.5 0.5" if interval else "")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join([head, 'material "m" { type diffuse reflect 0.6 }', body, LIGHT,
                             'camera "c" { eye 1 1 5  look_at 1 0 0  up 0 1 0  fov_y 40  film { res 40 30 } }']) + NL)
        return p

    def bake(first, last, name="f%04d"):
        r = subprocess.run([c.exe, "-bake-elastic", os.path.join(shot, name + ".ftsl"), str(first), str(last)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def records():
        return {f: open(os.path.join(shot, f), "rb").read() for f in os.listdir(shot)
                if f.startswith("bar_") and f.endswith(".ftec")}

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

    # A bake of the shot unchanged keeps every record, simulating nothing.
    before = records()
    rc, log = bake(0, DYN_FRAMES - 1)
    kept = log.count("is up to date")
    c.verdict("... a bake of an unchanged shot keeps every record", rc == 0 and kept == DYN_FRAMES
              and records() == before, "%d of %d frames up to date" % (kept, DYN_FRAMES))
    # Frame 20's handle target changed: a render of it is refused until the shot is baked again,
    # and the bake re-simulates from frame 20 on, keeping the frames before it.
    frame_scene(20, up=lift(20) + 0.05)
    _, log = tip(20)
    c.verdict("... a frame whose handle target changed is refused", "other handle targets" in log,
              log.strip().splitlines()[-1][:90] if log.strip() else "")
    rc, log = bake(0, DYN_FRAMES - 1)
    kept, baked = log.count("is up to date"), log.count("[elastic] baked frame ")
    after = records()
    same_before = all(after[f] == before[f] for f in before if int(f[4:8]) < 20)
    c.verdict("... and a bake after it re-simulates from that frame on", rc == 0 and kept == 20
              and baked == DYN_FRAMES - 20 and same_before and tip(20)[0] is not None,
              "%d kept, %d baked" % (kept, baked))
    frame_scene(20)

    # Placed by a translate that differs every frame, the bar bakes (its cache key is its mesh in
    # its own coordinates, exactly: carried back from the placed triangles, their rounding followed
    # the placement, and frame 1 found frame 0's record "baked for another mesh"), and its records
    # are the unplaced shot's byte for byte.
    pcache = os.path.join(shot, "pbar_%04d.ftec").replace(chr(92), "/")
    for k in range(8):
        frame_scene(k, name="p%04d", cache=pcache, place="translate %r 0.1 0" % (0.1 * k + 0.3))
    rc, log = bake(0, 7, name="p%04d")
    after = records()
    same = rc == 0 and all(open(os.path.join(shot, "pbar_%04d.ftec" % k), "rb").read() == after["bar_%04d.ftec" % k]
                           for k in range(8))
    c.verdict("... placed differently every frame, it bakes the unplaced shot's records", same,
              log.strip().splitlines()[-1][:100] if rc else "")

    # Under a motion interval the elastic has keys of its own, though its handles have none: the
    # bar, still ringing at frame 6, is somewhere else at each end of the frame.
    mcache = os.path.join(shot, "mbar_%04d.ftec").replace(chr(92), "/")
    for k in range(8):
        frame_scene(k, name="m%04d", cache=mcache, interval=True)
    rc, log = bake(0, 7, name="m%04d")
    ends = []
    for at in (-0.5, 0.5):
        o = os.path.join(shot, "m_at.obj")
        r = subprocess.run([c.exe, "-in", os.path.join(shot, "m0006.ftsl"), "-export-tris", o, "-export-tris-at",
                            repr(at)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        ends.append(read_obj_tris(o)[:-LIGHT_TRIS] if r.returncode == 0 else None)
    moved = max_diff(ends[0], ends[1]) if ends[0] is not None and ends[1] is not None else 0.0
    c.verdict("... under a motion interval it has keys though its handles have none", rc == 0 and moved > 1e-3,
              "moves %.3g m across frame 6" % moved if rc == 0 else log.strip()[-200:])

    # `ftrace -stop` reaches the frame being baked; the bake then exits with -stop-exit's code.
    for f in os.listdir(shot):
        if f.startswith("bar_") and f.endswith(".ftec"):
            os.remove(os.path.join(shot, f))
    for k in range(DYN_FRAMES, 4 * DYN_FRAMES):
        frame_scene(k)
    pr = subprocess.Popen([c.exe, "-bake-elastic", os.path.join(shot, "f%04d.ftsl"), "0", str(4 * DYN_FRAMES - 1),
                           "-stop-exit", "75"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace")
    seen = []
    for line in pr.stdout:
        seen.append(line)
        if "baked frame 2 " in line:
            subprocess.run([c.exe, "-stop", str(pr.pid)], capture_output=True)
            break
    seen.append(pr.stdout.read())
    rc = pr.wait()
    text = "".join(seen)
    left = len([f for f in os.listdir(shot) if f.startswith("bar_") and f.endswith(".ftec")])
    c.verdict("... and `ftrace -stop` stops a bake cleanly", rc == 75 and "stopped" in text and left < 4 * DYN_FRAMES,
              "exit %d, %d of %d frames baked" % (rc, left, 4 * DYN_FRAMES))


if __name__ == "__main__":
    sys.exit(main())
