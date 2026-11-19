"""Regression check: a GPU connection reads a vertex's patterns where the walk read them.

    python -u tools/check_vertex_hit_state.py [--out <dir>] [--exe ftrace.exe] [--modes DJU]

The device's bidirectional modes keep a path's vertices as DVertex and rebuild a hit from one
(dVertHit) for every BSDF, pdf and coat a connection or a merge evaluates there. The rebuilt hit
must carry what the walk's hit read its patterns by (DVertHitState, GPU-DVERTEX-HIT-STATE), among
it where a moving primitive's point sat at the instant. Two movers striped by a pattern in their
own space slide across the shutter interval and are held sharp at a time when they have travelled
half a stripe (`shutter_open 0.105`): a ball in a moving group (a rigid mover: its motion set), and
a panel a deformer slides (`deform rbf`, its eight handles keyed alike: a deforming triangle, read
back through the hit's barycentrics). A rebuilt hit that did not carry its point back to the
instant would read the stripes inverted, and the light a connection brings would stripe the mover
out of phase with the rest. It renders on the GPU in modes D, J and U (two seeds each; a render that fell back to the
CPU fails the check), which must agree with mode R on the CPU, whose walk reads every pattern at its
own hit, over 8x8 blocks of luminance (each within 5 standard errors or 3 %).

The state's other half, the hit's cavity, has no case: the one consumer whose reading differs, a
layered body's coat factor at a connection, is a ratio of the coat's transmissions at one film
thickness, which a cavity-driven titania film moves by under the noise.

Exits 1 on any failure.
"""
import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_motion import agree, lum, stripe_scene  # noqa: E402
import check_motion  # noqa: E402

W, H = 96, 72
MODES = {"D": ["-spp", "256"], "J": ["-spp", "256"], "U": ["-spp", "256"]}


def panel_obj(path, n=24):
    """A flat n x n grid, a metre square about the origin in y = 0, facing up."""
    lines = []
    for j in range(n + 1):
        for i in range(n + 1):
            lines.append("v %r 0 %r" % (-0.5 + i / n, -0.5 + j / n))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            b, c, d = a + 1, a + n + 2, a + n + 1
            lines += ["f %d %d %d" % (a, d, c), "f %d %d %d" % (a, c, b)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def deform_scene(obj, shutter):
    """The striped panel slid by a deformer from x = -0.6 to 0.6 over the interval, at rest at the
    instant: its handles, a box's corners about it, keyed alike."""
    def to(p, u):
        return "key { to %r %r %r }" % (p[0] - 0.6 + 1.2 * u, p[1], p[2])
    corners = [(x, y, z) for x in (-0.6, 0.6) for y in (-0.3, 0.3) for z in (-0.6, 0.6)]
    handles = "  ".join("handle { at %r %r %r  motion { %s } }" % (p + (" ".join(to(p, k / 2.0) for k in range(3)),))
                        for p in corners)
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'pattern "stripes" { expr "0.5 + 0.4 * sin(25 * x)" }',
        'material "striped" { type diffuse reflect pattern:stripes }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
        'group { translate 0 0.45 0.2  mesh "panel" { file "%s"  material striped  deform rbf { kernel triharmonic'
        '  %s } } }' % (obj.replace(chr(92), "/"), handles),
        'camera "c" { eye 0 1.6 2.2  look_at 0 0.45 0.2  up 0 1 0  fov_y 40  shutter_open %r  shutter_close %r'
        '  film { res %d %d } }' % (shutter + (W, H)),
    ]) + NL


def render(exe, out, path, tag, args, seed, want_gpu=False):
    """check_motion.render's image, as luminance."""
    return lum(check_motion.render(exe, out, path, tag, args, seed, want_gpu)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vertex_hit_state"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--modes", default="DJU")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    check_motion.W, check_motion.H = W, H
    obj = os.path.join(out, "panel.obj")
    panel_obj(obj)
    scenes = {"motion": stripe_scene(shutter=(0.105, 0.105)), "deform": deform_scene(obj, (0.105, 0.105))}
    ok = True
    for name, text in scenes.items():
        path = os.path.join(out, name + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        ref = [render(exe, out, path, "%s_R_cpu_s%d" % (name, s), ["-mode", "R", "-device", "cpu", "-spp", "256"], s)
               for s in (1, 2)]
        for m in a.modes:
            g = [render(exe, out, path, "%s_%s_gpu_s%d" % (name, m, s), ["-mode", m, "-device", "gpu"] + MODES[m], s,
                        want_gpu=True)
                 for s in (1, 2)]
            good, detail = agree(g, ref)
            ok &= good
            print("%-4s %s: GPU mode %s agrees with mode R on the CPU   %s" % ("ok" if good else "FAIL", name, m, detail),
                  flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
