"""Regression check: which open pieces the footprint probes' marking pass closes by a cap (PROBE-PARITY-LIMITS).

    python tools/check_cap_marks.py [--out png/check_cap_marks] [--exe ftrace.exe]

Modes M and S close a piece open along a planar rim that rests on another face -- a bottomless box of
quads on the floor -- for their footprint probes alone (watertight::markSolidFaces' capOk; design.md,
the probes), and the render's `[gather] N of M triangles ... bound closed solids` line counts the faces
so marked. Each case is a tiny mode-M render of a floor quad and a piece, and the line must read
exactly its count -- the marking pass's alone, with no noise:

  - `resting`: a bottomless box of five quads (ten triangles) on the floor -- capped, ten;
  - `lamp`: the same with an area light under its lid -- capped too: the cap says only which side of
    its faces is in, and the probes' low leg finds the room the gather point lies in (check_gather_base's
    `litbox`, which 0.451.16 left uncapped instead);
  - `moving`: the box in a group lifting it 5 cm over the shutter -- capped where its faces are stored,
    the low leg deciding at each path's time whether it rests there;
  - `lifted`: the box 0.25 mm up, its rim resting on nothing -- none;
  - `closed`: the box with its bottom face, a closed solid whatever is under it -- twelve;
  - `sheet`: two triangles hovering over the floor, one corner 1e-7 off the others' plane, which leaves
    them a volume, covering the points the cap asks about themselves -- none (0.451.16 capped it on
    itself, two);
  - `pit`: a ground grid of 72 triangles over the floor with a planar border and a pit off its middle --
    none (0.451.16 capped the pit on the ground's own flat part, 72);
  - `uneven`: a bottomless box of 16 triangles standing on a V-shaped ground, its walls following the
    valley down, so its rim is no plane -- capped by the fan from the rim's centroid, sixteen (0.451.18
    capped planar rims alone, none);
  - `hat`: a crown with a flat brim of 18 triangles lying on the floor, open along the brim's outer edge
    -- capped, eighteen: the points on the brim, the piece itself, ask nothing, and the floor covers the
    rest (0.451.17 refused it for lying on its own faces there, none).

Exits 1 on any mismatch.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_gather_base as g  # noqa: E402
import check_motion as cm  # noqa: E402
from check_meshvox import write_obj  # noqa: E402

B, H = 0.15, 0.04   # the box's half-width and height


def sheet_obj(path):
    """A 20 cm square tilted about two axes, 10 cm over the floor, its fourth corner 1e-7 off the plane
    the other three span."""
    p0, u, v = (-0.1, 0.1, -0.1), (0.2, 0.02, 0.0), (0.0, 0.01, 0.2)
    p1 = tuple(a + b for a, b in zip(p0, u))
    p3 = tuple(a + b for a, b in zip(p0, v))
    p2 = tuple(a + b + c for a, b, c in zip(p0, u, v))
    p2 = (p2[0], p2[1] + 1e-7, p2[2])
    write_obj(path, [p0, p1, p2, p3], [(0, 2, 1), (0, 3, 2)])


def pit_obj(path):
    """A 60 cm ground grid of 6 x 6 cells 5 cm over the floor, its border planar, one inner vertex (off
    the middle, two cells from the border, so the cells along it stay flat) 2 cm down."""
    verts = [(-0.3 + 0.1 * i, 0.05 - (0.02 if (i, j) == (2, 2) else 0.0), -0.3 + 0.1 * j)
             for j in range(7) for i in range(7)]
    faces = []
    for j in range(6):
        for i in range(6):
            a, b, c, d = j * 7 + i, j * 7 + i + 1, (j + 1) * 7 + i + 1, (j + 1) * 7 + i
            faces += [(a, c, b), (a, d, c)]
    write_obj(path, verts, faces)


K = 0.2   # the V-ground's slope: y = K |x|


def uneven_obj(path):
    """A bottomless box of half-width B whose walls stand on the V-ground y = K |x|, H over its rim's
    highest corners, wound outward: its rim six corners, the two at x = 0 K B below the rest."""
    t = K * B + H
    vs = {}
    order = []

    def v(x, y, z):
        if (x, y, z) not in vs:
            vs[(x, y, z)] = len(order)
            order.append((x, y, z))
        return vs[(x, y, z)]
    faces = []

    def quad(a, b, c, d):   # a b c d counter-clockwise seen from outside
        faces.extend([(a, b, c), (a, c, d)])
    # the walls at x = -B and x = +B, their feet on the ground at y = K B
    quad(v(-B, K * B, -B), v(-B, K * B, B), v(-B, t, B), v(-B, t, -B))
    quad(v(B, K * B, B), v(B, K * B, -B), v(B, t, -B), v(B, t, B))
    for z, s in ((-B, 1), (B, -1)):   # the walls at z = -B (outward -z) and z = +B, each in two halves
        for x0, x1 in ((-B, 0.0), (0.0, B)):
            a, b = (x0, x1) if s > 0 else (x1, x0)
            quad(v(a, K * abs(a), z), v(a, t, z), v(b, t, z), v(b, K * abs(b), z))
    quad(v(-B, t, -B), v(-B, t, B), v(0.0, t, B), v(0.0, t, -B))   # the top, in two halves
    quad(v(0.0, t, -B), v(0.0, t, B), v(B, t, B), v(B, t, -B))
    write_obj(path, order, faces)


def oriented(verts, loop, out):
    """The quad `loop` (four indices into `verts`, in order round it) as two triangles whose normal
    points along `out`."""
    a, b, c = (verts[i] for i in loop[:3])
    n = [(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
         (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
         (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])]
    if sum(x * y for x, y in zip(n, out)) < 0.0:
        loop = loop[::-1]
    return [(loop[0], loop[1], loop[2]), (loop[0], loop[2], loop[3])]


def hat_obj(path):
    """A crown of half-width 8 cm and height 6 cm with a flat brim out to B, on the floor, wound
    outward: open along the brim's outer edge alone."""
    a, h = 0.08, 0.06
    verts, faces = [], []

    def v(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1
    o = [v(-B, 0.0, -B), v(B, 0.0, -B), v(B, 0.0, B), v(-B, 0.0, B)]      # the brim's outer corners
    i = [v(-a, 0.0, -a), v(a, 0.0, -a), v(a, 0.0, a), v(-a, 0.0, a)]      # its inner corners, the crown's feet
    t = [v(-a, h, -a), v(a, h, -a), v(a, h, a), v(-a, h, a)]              # the crown's top
    sides = [(0, 1, (0, 0, -1)), (1, 2, (1, 0, 0)), (2, 3, (0, 0, 1)), (3, 0, (-1, 0, 0))]
    for k0, k1, out in sides:
        faces += oriented(verts, [o[k0], o[k1], i[k1], i[k0]], (0, 1, 0))   # the brim's trapezoid
        faces += oriented(verts, [i[k0], i[k1], t[k1], t[k0]], out)         # the crown's wall
    faces += oriented(verts, t, (0, 1, 0))                                     # the crown's top
    write_obj(path, verts, faces)


def scene(kind, out):
    geo = []
    if kind in ("resting", "lamp", "moving", "lifted", "closed"):
        lift = 0.00025 if kind == "lifted" else 0.0
        geo = (g.box if kind == "closed" else g.open_box)(-B, lift, -B, B, lift + H, B, "obj")
    if kind == "moving":
        geo = ["group { motion { key { translate 0 0 0 } key { translate 0 0.05 0 } }"] + geo + ["}"]
    if kind == "lamp":
        geo.append("light area { origin -0.03 0.035 -0.03  u 0.06 0 0  v 0 0 0.06  normal 0 -1 0  power 5  "
                   "spd preset:d65 }")
    if kind in ("sheet", "pit", "uneven", "hat"):
        obj = os.path.join(out, kind + ".obj")
        {"sheet": sheet_obj, "pit": pit_obj, "uneven": uneven_obj, "hat": hat_obj}[kind](obj)
        geo = ['mesh "%s" { file "%s"  material obj }' % (kind, kind + ".obj")]
    if kind == "uneven":   # the V-ground it stands on, two quads meeting along x = 0
        geo += [g.quad((-1, K, -1), (0, 0, 2), (1, -K, 0), "floor"), g.quad((0, 0, -1), (0, 0, 2), (1, K, 0), "floor")]
    shutter = "shutter_open 0  shutter_close 1  " if kind == "moving" else ""
    return NL.join([
        "scene { units meters%s }" % ("  motion_interval 0 1" if kind == "moving" else ""),
        'material "floor" { type diffuse reflect 0.5 }',
        'material "obj" { type diffuse reflect 0.5 }',
        g.quad((-2, 0, -2), (0, 0, 4), (4, 0, 0), "floor"),
        "light area { origin -0.4 1.5 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 300  spd preset:d65 }",
        ] + geo + [
        'camera "c" { eye 0 0.9 0.0001  look_at 0 0 0  up 0 0 1  fov_y 40  %sfilm { res 16 16 } }' % shutter,
    ]) + NL


CASES = {"resting": 10, "lamp": 10, "moving": 10, "lifted": 0, "closed": 12, "sheet": 0, "pit": 0, "uneven": 16,
         "hat": 18}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_cap_marks"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for kind, want in CASES.items():
        path = os.path.join(out, kind + ".ftsl")
        cm.write(path, scene(kind, out))
        rc, log = cm.run([exe, "-in", path, "-mode", "M", "-device", "cpu", "-n", "20000", "-spp", "1",
                          "-pmradius", "0.1", "-no-meter", "-window-min", "-o", os.path.join(out, kind + ".png")])
        m = g.GATHER_LINE.search(log)
        if rc != 0 or not m:
            print(log[-1500:])
            print("FAIL: %s: %s" % (kind, "exit %d" % rc if rc else "no [gather] line"))
            ok = False
            continue
        got = int(m.group(1))
        good = got == want
        ok &= good
        print("%-8s %2d of %s triangles marked (want %d)  %s" % (kind, got, m.group(2), want,
                                                              "ok" if good else "FAIL"), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
