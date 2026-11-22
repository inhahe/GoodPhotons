"""Regression check: which open pieces the footprint probes' marking pass closes by a cap (PROBE-PARITY-LIMITS).

    python tools/check_cap_marks.py [--out png/check_cap_marks] [--exe ftrace.exe]

Modes M and S close a piece open along a planar rim that rests on another face -- a bottomless box of
quads on the floor -- for their footprint probes alone (watertight::markSolidFaces' capOk; design.md,
the probes), and the render's `[gather] N of M triangles ... bound closed solids` line counts the faces
so marked. A cap is right only where the piece is a solid resting there, so four bottomless boxes of
five quads (ten triangles) on a floor quad are rendered, a tiny mode-M image each, and the line read:

  - `resting`: the box on the floor -- capped, its ten faces counted;
  - `lamp`: the same with an area light under its lid -- a room, not a solid: none counted (REVIEW-0458;
    capped, check_gather_base's `litbox` read its floor too bright);
  - `lifted`: the box 1 mm up, its rim resting on nothing -- none;
  - `moving`: the box in a group that lifts it 5 cm over the shutter -- it rests on the floor at the
    instant only, so none (REVIEW-0458);
  - `closed`: the box with its bottom face, a closed solid whatever is under it -- twelve.

The count is the marking pass's alone, with no noise: each case must read exactly its number. Exits 1
on any mismatch.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
B = 0.15            # the box's half-width
LINE = re.compile(r"\[gather\] (\d+) of (\d+) triangles")


def quad(o, u, v, mat):
    return "quad { origin %g %g %g  u %g %g %g  v %g %g %g  material %s }" % (o + u + v + (mat,))


def box(y0, h, bottom):
    """Six quads, normals out (u x v), or five without the bottom one."""
    x0, z0, d = -B, -B, 2 * B
    y1 = y0 + h
    faces = [quad((x0, y1, z0), (0, 0, d), (d, 0, 0), "obj"),
             quad((x0, y0, z0), (0, h, 0), (d, 0, 0), "obj"), quad((x0, y0, z0 + d), (d, 0, 0), (0, h, 0), "obj"),
             quad((x0, y0, z0), (0, 0, d), (0, h, 0), "obj"), quad((x0 + d, y0, z0), (0, h, 0), (0, 0, d), "obj")]
    if bottom:
        faces.append(quad((x0, y0, z0), (d, 0, 0), (0, 0, d), "obj"))
    return faces


def scene(kind):
    lift = 0.001 if kind == "lifted" else 0.0
    geo = box(lift, 0.04, kind == "closed")
    if kind == "moving":
        geo = ["group { motion { key { translate 0 0 0 } key { translate 0 0.05 0 } }"] + geo + ["}"]
    if kind == "lamp":
        geo.append("light area { origin -0.03 0.035 -0.03  u 0.06 0 0  v 0 0 0.06  normal 0 -1 0  power 5  "
                   "spd preset:d65 }")
    shutter = "shutter_open 0  shutter_close 1  " if kind == "moving" else ""
    return NL.join([
        "scene { units meters%s }" % ("  motion_interval 0 1" if kind == "moving" else ""),
        'material "floor" { type diffuse reflect 0.5 }',
        'material "obj" { type diffuse reflect 0.5 }',
        quad((-2, 0, -2), (0, 0, 4), (4, 0, 0), "floor"),
        "light area { origin -0.4 1.5 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 300  spd preset:d65 }",
        ] + geo + [
        'camera "c" { eye 0 0.9 0.0001  look_at 0 0 0  up 0 0 1  fov_y 40  %sfilm { res 16 16 } }' % shutter,
    ]) + NL


CASES = {"resting": 10, "lamp": 0, "lifted": 0, "moving": 0, "closed": 12}


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
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(scene(kind))
        png = os.path.join(out, kind + ".png")
        cmd = [exe, "-in", path, "-mode", "M", "-device", "cpu", "-n", "20000", "-spp", "1", "-pmradius", "0.1",
               "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = p.stdout + p.stderr
        m = LINE.search(log)
        if p.returncode != 0 or not m:
            print(log[-1500:])
            print("FAIL: %s: %s" % (kind, "exit %d" % p.returncode if p.returncode else "no [gather] line"))
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
