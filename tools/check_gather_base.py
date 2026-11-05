"""Regression check: modes M and S read the floor at the base of an object standing on it as mode R does.

    python tools/check_gather_base.py [--out png/check_gather_base] [--exe ftrace.exe]

A closed 3 cm box (six quads) standing on a lit floor, seen from straight above, modes M and S at a
pinned gather radius of 0.1 on the CPU and the GPU, against mode R on the GPU. A mode-M or mode-S
gather keeps only its own surface's photons (gatherplane.h, known-issues MERGE-PROXIMITY), so a
floor point beside the box drops the box top's, and its footprint probe must then not count the
column over the box as floor: the probe passes the box's top face and meets its bottom face (or the
floor inside a bottomless box) with an odd count of crossings, a miss. Counted as covered, the ring
of floor within 5 cm of the base read 0.75 of mode R in both modes.

The floor 0-2 cm and 2-5 cm out from the box's edge must each read within 7 % of mode R (two seeds
apiece). Exits 0 when every case holds, 1 otherwise.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_motion as cm  # noqa: E402

NL = chr(10)
W = H = 160
HALF = 0.5          # the frame covers x, z in [-HALF, HALF]
BOX = 0.15          # the box's half-width


def quad(o, u, v, mat):
    return "quad { origin %g %g %g  u %g %g %g  v %g %g %g  material %s }" % (o + u + v + (mat,))


def box(x0, y0, z0, x1, y1, z1, mat):
    """Six quads, normals out (u x v)."""
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    return [quad((x0, y1, z0), (0, 0, dz), (dx, 0, 0), mat), quad((x0, y0, z0), (dx, 0, 0), (0, 0, dz), mat),
            quad((x0, y0, z0), (0, dy, 0), (dx, 0, 0), mat), quad((x0, y0, z1), (dx, 0, 0), (0, dy, 0), mat),
            quad((x0, y0, z0), (0, 0, dz), (0, dy, 0), mat), quad((x1, y0, z0), (0, dy, 0), (0, 0, dz), mat)]


SCENE = NL.join([
    "scene { units meters }",
    'camera "c" { eye 0 0.9 0.0001  look_at 0 0 0  up 0 0 1  fov_y %g  film { res %d %d } }' % (
        2 * np.degrees(np.arctan(HALF / 0.9)), W, H),
    'material "floor" { type diffuse reflect 0.5 }',
    'material "obj" { type diffuse reflect 0.5 }',
    quad((-2, 0, -2), (0, 0, 4), (4, 0, 0), "floor"),
    "light area { origin -0.4 1.5 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 300  spd preset:d65 }",
] + box(-BOX, 0.0, -BOX, BOX, 0.03, BOX, "obj")) + NL
REF = ["-mode", "R", "-device", "gpu", "-spp", "512"]
CASES = [("M_gpu", ["-mode", "M", "-device", "gpu", "-n", "8000000", "-spp", "16", "-pmradius", "0.1"]),
         ("M_cpu", ["-mode", "M", "-device", "cpu", "-n", "2000000", "-spp", "4", "-pmradius", "0.1"]),
         ("S_gpu", ["-mode", "S", "-device", "gpu", "-spp", "32", "-pmradius", "0.1"]),
         ("S_cpu", ["-mode", "S", "-device", "cpu", "-spp", "8", "-pmradius", "0.1"])]
RINGS = ((0.0, 0.02), (0.02, 0.05))
TOL = 0.07


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gather_base"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "box.ftsl")
    cm.write(path, SCENE)
    xs = (np.arange(W) + 0.5) / W * 2 * HALF - HALF
    X, Z = np.meshgrid(xs, xs)
    dist = np.maximum(np.abs(X), np.abs(Z)) - BOX     # out from the box's edge (a square)

    def image(tag, args, gpu):
        return np.mean([cm.lum(cm.render(exe, out, path, "%s_%d" % (tag, s), args, seed=s, want_gpu=gpu)[0])
                        for s in (1, 2)], axis=0)

    R = image("R_gpu", REF, True)
    ok = True
    for name, args in CASES:
        M = image(name, args, "gpu" in name)
        for lo, hi in RINGS:
            m = (dist >= lo) & (dist < hi)
            q = M[m].mean() / R[m].mean()
            good = abs(q - 1.0) <= TOL
            ok &= good
            print("%-6s floor %2.0f-%2.0f cm from the base: %.3f of mode R  %s" % (
                name, lo * 100, hi * 100, q, "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
