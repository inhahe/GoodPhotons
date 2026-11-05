"""Regression check: modes M and S read the floor at the base of an object standing on it as mode R does.

    python tools/check_gather_base.py [--out png/check_gather_base] [--exe ftrace.exe]
                                      [--objects diffuse,mirror,coated,coaster,tower,ball,meshball,
                                                 diffuseball,smallball,mirrorsheet,lampsheet]

A closed 3 cm box (six quads) standing on a lit floor, seen from straight above, modes M and S at a
pinned gather radius of 0.1 on the CPU and the GPU, against mode R on the GPU. A mode-M or mode-S
gather keeps only its own surface's photons (gatherplane.h, known-issues MERGE-PROXIMITY), so a
floor point beside the box drops the box top's, and its footprint probe must then not count the
column over the box as floor: the probe passes the box's top face and meets its bottom face (or the
floor inside a bottomless box) with an odd count of crossings, a miss. Counted as covered, the ring
of floor within 5 cm of the base read 0.75 of mode R in both modes. `-gageom 1` measures the same
coverage from the geometry (gafootprint.h) and keeps the same rule per column; it looks for the
crossings as high as the probe does, r above the plane, or near the disc's edge a box top lies
beyond the column's chord and the floor under the box counts again: it must agree with the probe
within 2 % in each ring, where until 0.433.7 mode S's read 0.969 and 0.976 of the probe's (M 0.999
and 0.987; now M 1.012 and 1.000, S 0.998 and 1.000).

The box is diffuse, a mirror, or `layered` over a glossy body (`coated`). The last two hold no photon,
so the probe passes their faces, but they are opaque: their faces are crossings all the same
(passesLight), or the floor under them counts as covered again. A diffuse box 5 mm tall (`coaster`)
has its top in the floor's band from 5 cm away, so the floor's gather takes the top's photons there,
and the top must count as covered: met from outside, a closed solid's face in the band is the
surface, not a crossing. A diffuse box 30 cm tall (`tower`), off to one side so the foot of its near
face is seen past its top: every probe over it begins inside it, r above the floor, and crosses none
of its faces going down, so it must know from the path up from the gather point and across that it
began inside (PROBE-PATH). Its shadow at its foot is too steep for mode R to be a reference within
the radius, so it is held to its exact coverage (below). Then a glossy ball of the box's
half-width resting on the floor, analytic and as a closed OBJ, seen from low beside it with a low
light from the side reaching under its overhang. There, 4-12 cm from the point it rests on, the gap
to the ball is under the gather radius, so the probe begins inside the ball and leaves it before it
meets the floor, which a parity reads as inside. A closed solid's face says it is out
(PROBE-SOLID-SIDE). A diffuse ball's tilted underside holds photons and lies in the band there, so
the probe, met from inside, must pass it to the floor (PROBE-UNDERSIDE), as the march does. A glossy
ball of 4 cm (`smallball`) has its top within the probe's window over the floor around it and its
underside in the band: the probe enters it and must leave it by that underside, and the march,
climbing from below, must take the underside it enters to say the floor under it lies outside. Its
floor is held from the contact out to 10 cm.
Then a mirror quad and a light's own quad hovering 8 cm over the floor, seen
from the same low view: open sheets that hold no photon, which enclose nothing, so the floor under
them is the whole disc's coverage too (held as the balls' ring is).

The floor 0-2 cm and 2-5 cm out from a box's base must each read within 7 % of mode R (two seeds
apiece). The floor 4-12 cm from a ball's contact must read within 3 % of the same render with the
footprint correction off (-gatherarea 0), whose coverage there is the whole disc: the kernel's blur
from the lit floor beyond makes mode R no reference there. The tower's rings must read within 3 % of
that render divided by each pixel's share of the disc outside the tower, a circular segment's
complement, with mode S's radius held (-sppmalpha 1). Exits 0 when every case holds, 1 otherwise.
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
RINGS = ((0.0, 0.02), (0.02, 0.05))   # the floor out from an object's base, metres


def quad(o, u, v, mat):
    return "quad { origin %g %g %g  u %g %g %g  v %g %g %g  material %s }" % (o + u + v + (mat,))


def box(x0, y0, z0, x1, y1, z1, mat):
    """Six quads, normals out (u x v)."""
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    return [quad((x0, y1, z0), (0, 0, dz), (dx, 0, 0), mat), quad((x0, y0, z0), (dx, 0, 0), (0, 0, dz), mat),
            quad((x0, y0, z0), (0, dy, 0), (dx, 0, 0), mat), quad((x0, y0, z1), (dx, 0, 0), (0, dy, 0), mat),
            quad((x0, y0, z0), (0, 0, dz), (0, dy, 0), mat), quad((x1, y0, z0), (0, dy, 0), (0, 0, dz), mat)]


def write_ball_obj(path, r, n=48):
    """A closed UV sphere of radius `r` resting on y = 0, wound outward, its normals radial."""
    vs, ns, fs = [(0.0, 0.0, 0.0)], [(0.0, -1.0, 0.0)], []            # bottom pole
    for i in range(1, n):
        th = np.pi * i / n                                              # from the bottom
        for j in range(2 * n):
            ph = np.pi * j / n
            d = (np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph))
            vs.append((r * d[0], r + r * d[1], r * d[2]))
            ns.append(d)
    vs.append((0.0, 2 * r, 0.0))                                       # top pole
    ns.append((0.0, 1.0, 0.0))

    def ring(i, j):                                                     # 1-based OBJ index
        return 2 + (i - 1) * 2 * n + j % (2 * n)
    for j in range(2 * n):
        fs.append((1, ring(1, j), ring(1, j + 1)))
    for i in range(1, n - 1):
        for j in range(2 * n):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)
            fs.append((a, c, b))
            fs.append((a, d, c))
    top = len(vs)
    for j in range(2 * n):
        fs.append((ring(n - 1, j + 1), ring(n - 1, j), top))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines("v %.9g %.9g %.9g%s" % (v + (NL,)) for v in vs)
        f.writelines("vn %.9g %.9g %.9g%s" % (tuple(v) + (NL,)) for v in ns)
        f.writelines("f %d//%d %d//%d %d//%d%s" % (a, a, b, b, c, c, NL) for a, b, c in fs)


# The two views. A box is seen from straight above, its floor measured out from the box's edge. A
# ball is seen from low beside it, looking under its overhang, where a low light from the side
# reaches, and its floor is measured out from the point it rests on. That floor is what the probe
# begins inside the ball over: from above, the ball hides it.
TOP_CAM = 'camera "c" { eye 0 0.9 0.0001  look_at 0 0 0  up 0 0 1  fov_y %g  film { res %d %d } }' % (
    2 * np.degrees(np.arctan(HALF / 0.9)), W, H)
SW, SH, SFOV = 160, 100, 30.0
SEYE, SLOOK = np.array([0.0, 0.06, 0.55]), np.array([0.0, 0.03, 0.0])
SIDE_CAM = 'camera "c" { eye %g %g %g  look_at %g %g %g  up 0 1 0  fov_y %g  film { res %d %d } }' % (
    tuple(SEYE) + tuple(SLOOK) + (SFOV, SW, SH))
SIDE_LIGHT = ("light area { origin 0.5 0.005 -0.25  u 0 0 0.5  v 0 0.1 0  normal -1 0 0  power 40  "
              "spd preset:d65 }")


def top_dist(base):
    """Each pixel of the view from above as the floor's distance out from the box's edge. `base`
    takes the floor's coordinates under each pixel as the view lays them out, X rightward and Z
    downward, which are world -x and -z (the camera looks down with +z up the frame)."""
    xs = (np.arange(W) + 0.5) / W * 2 * HALF - HALF
    X, Z = np.meshgrid(xs, xs)
    return base(X, Z)


def side_dist(occluded=True, radius=BOX):
    """Each pixel of the view from beside the ball (of `radius`, resting on the floor at the origin)
    as the floor's distance from the point below the ball's centre, or NaN where the pixel sees no
    floor, or (`occluded`) sees it within 4 mm of the ball's silhouette (which the mesh ball's
    facets move). The camera's pinhole (fov_y, square pixels, row 0 at the top), checked against
    marker renders to a pixel."""
    w = SLOOK - SEYE
    w = w / np.linalg.norm(w)
    u = np.cross(w, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(u, w)
    t = np.tan(np.radians(SFOV) / 2)
    sx = ((np.arange(SW) + 0.5) / SW * 2 - 1) * t * SW / SH
    sy = (1 - (np.arange(SH) + 0.5) / SH * 2) * t
    SX, SY = np.meshgrid(sx, sy)
    d = w[None, None, :] + SX[..., None] * u + SY[..., None] * v
    with np.errstate(divide="ignore"):
        tf = -SEYE[1] / d[..., 1]                              # the floor, y = 0 (inf: parallel)
    oc = SEYE - np.array([0.0, radius, 0.0])
    b = d @ oc
    a = np.einsum("ijk,ijk->ij", d, d)
    disc = b * b - a * (oc @ oc - (radius + 0.004) ** 2)
    ts = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    seen = (d[..., 1] < 0) & (tf > 0) & ((tf < ts) | (not occluded))
    P = SEYE + d * tf[..., None]
    return np.where(seen, np.hypot(P[..., 0], P[..., 2]), np.nan)


# Each object: its scene lines, its view, its floor distance per pixel, the rings it is held to,
# and what each ring is held to. A box's floor is held to mode R. A ball's ring lies under its
# overhang, 4-12 cm from the point it rests on, where the gap to the ball is 0.5 to 6 cm: within the
# gather radius, so every probe there begins inside the ball. There the gather's disc reaches out to
# the floor beyond the ball, lit from above, and at radius 0.1 that kernel blur puts mode M some 40 %
# over mode R, and mode S 15-25 %, however the coverage is taken. So that ring holds each estimate to
# the same render with the footprint correction off (`-gatherarea 0`): the same photons, so the
# ratio is the coverage alone, which is the whole disc there since the floor is all present.
def _box(mat, height=0.03):
    return ([mat] + box(-BOX, 0.0, -BOX, BOX, height, BOX, "obj"), TOP_CAM,
            lambda: top_dist(lambda X, Z: np.maximum(np.abs(X), np.abs(Z)) - BOX), RINGS, "the base",
            "R")


def _ball(mat, mesh=False, radius=BOX, rings=((0.04, 0.12),)):
    geo = ('mesh "ball" { file "ball.obj"  material obj }' if mesh else
           'sphere { center 0 %g 0  radius %g  material obj }' % (radius, radius))
    return ([mat, geo, SIDE_LIGHT], SIDE_CAM, lambda: side_dist(radius=radius), rings, "the contact",
            "uncovered")


# A box taller than the radius, off to one side of the view from above (world x -0.45 to -0.15, the
# frame's right) so that its near face's foot is seen past its top, and the floor measured out from
# that face only (world x > -0.15, so X < 0.15 in the frame), within 5 cm of its middle, where the
# gather's disc meets the face along a straight edge. The tower shades the floor at its foot from the
# light overhead, so steeply that the kernel's blur puts the floor 0-2 cm out 15 % over mode R
# however the coverage is taken; the tower's rings are held instead to the same render with the
# footprint correction off divided by the share of the disc outside the tower (`seg_cover`): the
# same photons, so the ratio is the coverage alone. Mode S keeps its radius for it (-sppmalpha 1).
TOWER_X0 = -0.45
GATHER_R = 0.1      # the cases' -pmradius


def seg_cover(d, r=GATHER_R):
    """The share of a disc of radius `r` whose centre lies `d` (>= 0) from a straight edge, on the
    centre's side of it."""
    d = np.clip(d, 0.0, r)
    return 1.0 - (r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)) / (np.pi * r * r)


def _tower(mat):
    return ([mat] + box(TOWER_X0, 0.0, -BOX, -BOX, 0.3, BOX, "obj"), TOP_CAM,
            lambda: top_dist(lambda X, Z: np.where((np.abs(Z) < BOX - GATHER_R) & (X <= BOX), BOX - X, np.nan)),
            RINGS, "the base", "coverage")


# An open sheet hovering over the floor, above the side view's eye so the camera sees the floor
# under it: a mirror quad, and a light's own quad facing down. Neither holds a photon, and an open
# sheet encloses nothing, so the floor under it is the whole disc's coverage, as a ball's ring is.
def _sheet(lines):
    return lines, SIDE_CAM, lambda: side_dist(occluded=False), ((0.0, 0.1),), "the centre", "uncovered"


OBJECTS = {
    "diffuse": _box('material "obj" { type diffuse reflect 0.5 }'),
    "mirror": _box('material "obj" { type mirror reflect 0.9 }'),
    "coated": _box('material "obj" { type layered  ior 1.5  coat { }  layer "gl" 1.0 }'),
    # A box 5 mm tall: its top lies in the floor's band 5 cm away and more, where the floor's gather
    # takes its photons, so there the probe must end on it as covered (a closed solid's face met
    # from outside), and the march count it while dropping the floor inside the box below it.
    "coaster": _box('material "obj" { type diffuse reflect 0.5 }', height=0.005),
    # A box 30 cm tall holds every probe over it from its start r above the floor, so the probe
    # crosses none of its faces going down: it must know it begins inside (PROBE-PATH).
    "tower": _tower('material "obj" { type diffuse reflect 0.5 }'),
    # A ball resting on the floor holds the probe's origin over the floor under its overhang, so
    # the probe leaves it before it meets the floor: a closed solid's face says it is out
    # (PROBE-SOLID-SIDE), where a parity reads the floor as inside.
    "ball": _ball('material "obj" { type glossy reflect 0.6 roughness 0.3 }'),
    "meshball": _ball('material "obj" { type glossy reflect 0.6 roughness 0.3 }', mesh=True),
    # A diffuse ball's tilted underside holds photons, so the probe stops on it in the band; met
    # from inside, it must pass on to the floor below (PROBE-UNDERSIDE).
    "diffuseball": _ball('material "obj" { type diffuse reflect 0.5 }'),
    # A ball smaller than the radius: its top lies within the window over the floor around it, and
    # its underside in the band, which both the probe and the march must leave it by.
    "smallball": _ball('material "obj" { type glossy reflect 0.6 roughness 0.3 }', radius=0.04,
                       rings=((0.0, 0.1),)),
    "mirrorsheet": _sheet(['material "obj" { type mirror reflect 0.9 }',
                           quad((-BOX, 0.08, -BOX), (0, 0, 2 * BOX), (2 * BOX, 0, 0), "obj"), SIDE_LIGHT]),
    "lampsheet": _sheet(["light area { origin %g 0.08 %g  u %g 0 0  v 0 0 %g  normal 0 -1 0  power 5  "
                         "spd preset:d65 }" % (-BOX, -BOX, 2 * BOX, 2 * BOX), SIDE_LIGHT]),
}


def scene(obj_lines, cam):
    return NL.join([
    "scene { units meters }",
    cam,
    'material "floor" { type diffuse reflect 0.5 }',
    'material "gl" { type glossy reflect 0.6 roughness 0.3 }',
    quad((-2, 0, -2), (0, 0, 4), (4, 0, 0), "floor"),
    "light area { origin -0.4 1.5 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 300  spd preset:d65 }",
    ] + obj_lines) + NL
REF = ["-mode", "R", "-device", "gpu", "-spp", "512"]
CASES = [("M_gpu", ["-mode", "M", "-device", "gpu", "-n", "8000000", "-spp", "16", "-pmradius", "0.1"]),
         ("M_cpu", ["-mode", "M", "-device", "cpu", "-n", "2000000", "-spp", "4", "-pmradius", "0.1"]),
         ("S_gpu", ["-mode", "S", "-device", "gpu", "-spp", "32", "-pmradius", "0.1"]),
         ("S_cpu", ["-mode", "S", "-device", "cpu", "-spp", "8", "-pmradius", "0.1"]),
         # the geometric footprint in place of the probe (host only): the same crossing rule
         ("M_cpu_gageom", ["-mode", "M", "-device", "cpu", "-n", "2000000", "-spp", "4", "-pmradius", "0.1",
                           "-gageom", "1"]),
         ("S_cpu_gageom", ["-mode", "S", "-device", "cpu", "-spp", "8", "-pmradius", "0.1", "-gageom", "1"])]
TOL = 0.07
COVER_TOL = 0.03   # a ball's estimate against the same render with no footprint correction
# `-gageom` against the probe it stands in for, same mode and seeds, so the same photons: the two
# coverages must agree to 2 % in each ring.
PAIRS = (("M_cpu_gageom", "M_cpu"), ("S_cpu_gageom", "S_cpu"))
PAIR_TOL = 0.02


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gather_base"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--objects", default=",".join(OBJECTS))
    a = ap.parse_args()
    unknown = [b for b in a.objects.split(",") if b not in OBJECTS]
    if unknown:
        ap.error("unknown object: " + ", ".join(unknown))
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    write_ball_obj(os.path.join(out, "ball.obj"), BOX)
    ok = True
    for bname in a.objects.split(","):
        lines, cam, distf, rings, base, against = OBJECTS[bname]
        dist = distf()                                 # each pixel's floor distance from `base`
        path = os.path.join(out, "obj_%s.ftsl" % bname)
        cm.write(path, scene(lines, cam))

        def image(tag, args, gpu):
            return np.mean([cm.lum(cm.render(exe, out, path, "%s_%s_%d" % (bname, tag, s), args, seed=s,
                                             want_gpu=gpu)[0]) for s in (1, 2)], axis=0)

        R = image("R_gpu", REF, True) if against == "R" else None   # a ball's ring is held to U
        # The share of each pixel's disc on the floor, for a ring held to its exact coverage.
        cover = seg_cover(dist) if against == "coverage" else 1.0
        got = {}
        for name, args in CASES:
            if against == "coverage" and args[args.index("-mode") + 1] == "S":
                args = args + ["-sppmalpha", "1"]   # the radius the coverage is taken at, every pass
            M = image(name, args, "gpu" in name)
            # The same render with the footprint correction off, for a ring held to its coverage
            # (`against`); not for -gageom, which the pairs below hold to the probe.
            U = (image(name + "_uncovered", args + ["-gatherarea", "0"], "gpu" in name)
                 if against in ("uncovered", "coverage") and "gageom" not in name else None)
            for lo, hi in rings:
                m = (dist >= lo) & (dist < hi)
                q = got[name, lo] = M[m].mean() / (R[m].mean() if R is not None else 1.0)
                if U is None and against == "R":
                    good = abs(q - 1.0) <= TOL
                    ok &= good
                    print("%-9s %-12s floor %2.0f-%2.0f cm from %s: %.3f of mode R  %s" % (
                        bname, name, lo * 100, hi * 100, base, q, "ok" if good else "FAIL"), flush=True)
                elif U is not None:
                    c = M[m].mean() / (U / cover)[m].mean()
                    good = abs(c - 1.0) <= COVER_TOL
                    ok &= good
                    print("%-9s %-12s floor %2.0f-%2.0f cm from %s: %.3f of the uncorrected / %s  %s" % (
                        bname, name, lo * 100, hi * 100, base, c,
                        "its coverage" if against == "coverage" else "1", "ok" if good else "FAIL"), flush=True)
        for a_, b_ in PAIRS:
            for lo, hi in rings:
                q = got[a_, lo] / got[b_, lo]
                good = abs(q - 1.0) <= PAIR_TOL
                ok &= good
                print("%-9s %-12s floor %2.0f-%2.0f cm from %s: %.3f of %s's  %s" % (
                    bname, a_, lo * 100, hi * 100, base, q, b_, "ok" if good else "FAIL"), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
