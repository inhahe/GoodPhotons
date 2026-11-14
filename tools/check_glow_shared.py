"""Regression check: a glowing material shared by surfaces with different emitters stays one material.

    python tools/check_glow_shared.py [--out png/check_glow_shared] [--exe ftrace.exe] [--no-gpu]

Every emissive surface is a light (0.442.0), and a material lights the scene through as many
emitters as it has surfaces needing their own -- each mesh block, each group of other triangles,
each sphere and isosurface. Until 0.443.0 a hit found its emitter by its material, so each such
surface after the first took a copy of the material (Fable's review of 0.442.0). Two things that
broke, and must not:

SHARED. A glowing panel as a `mesh` block, and the same panel placed as a static `mesh_instance`,
both of one material: 0.442.0-0.442.4 refused the scene (the instance's triangles could not take
the copy the block's light forced). It must load, and the two panels, mirror images in the room,
must light their halves of the floor alike, in modes R, D and B on the CPU and the GPU (within
2.5 % of each other and of mode R).

SPLIT SOLID. An emissive, absorbing glass box made of a `mesh` block (five faces) and a bare `quad`
(the sixth, toward the wall behind it), seen through: the quad's emitter differs from the block's,
and with a copy of the material it was a different solid to the medium stack, so a ray leaving
through it stayed "inside" the glass and the wall behind read darker. It must render as the same
box written as one `mesh` block of six faces (mode R, CPU and GPU, the box's tiles within 2 %).

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

W, H = 96, 64
TOL = 0.025


def write(path, txt):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(txt)


def obj(path, verts, faces):
    write(path, NL.join(["v %g %g %g" % v for v in verts] + ["f " + " ".join(str(i + 1) for i in f)
                                                               for f in faces]) + NL)


def room(cam):
    return [
        "scene { units meters }",
        cam,
        'material "wall"  { type diffuse reflect 0.6 }',
        'quad { origin -2 0 -2  u 0 0 5  v 4 0 0  material wall }',             # floor
        'quad { origin -2 0 -2  u 4 0 0  v 0 2.6 0  material wall }',           # back
        'quad { origin -2 0 -2  u 0 2.6 0  v 0 0 5  material wall }',           # left
        'quad { origin 2 0 -2  u 0 0 5  v 0 2.6 0  material wall }',            # right
        'quad { origin -2 2.6 -2  u 0 0 5  v 4 0 0  material wall }',           # ceiling
    ]


def shared_scene(d):
    """Two glowing panels facing down, one a mesh block, one a static instance of the same panel."""
    # a 0.5 x 0.5 panel in its own space, wound to face -y
    obj(os.path.join(d, "panel.obj"), [(-0.25, 0, -0.25), (0.25, 0, -0.25), (0.25, 0, 0.25), (-0.25, 0, 0.25)],
        [(0, 1, 2), (0, 2, 3)])
    p = os.path.join(d, "panel.obj").replace(chr(92), "/")
    return NL.join(room('camera "c" { eye 0 1.6 2.8  look_at 0 0.3 0  up 0 1 0  fov_y 55  film { res %d %d } }'
                        % (W, H)) + [
        'material "lamp" { type diffuse reflect 0.0  emit 0.5 }',
        'group { translate -1 1.2 0  mesh { file "%s"  material lamp } }' % p,
        'mesh_asset "panel" { file "%s"  material lamp }' % p,
        'mesh_instance { of "panel"  translate 1 1.2 0 }',
        "",
    ])


def box_scene(d, split):
    """An emissive glass box, seen through toward the lit back wall: one mesh of six faces, or five
    faces in a mesh and the sixth (the back, toward the wall) a bare quad."""
    v = [(-0.3, 0.3, -0.3), (0.3, 0.3, -0.3), (0.3, 0.9, -0.3), (-0.3, 0.9, -0.3),
         (-0.3, 0.3, 0.3), (0.3, 0.3, 0.3), (0.3, 0.9, 0.3), (-0.3, 0.9, 0.3)]
    # outward-wound faces: back (z = -0.3), front, left, right, bottom, top
    back = (0, 3, 2, 1)
    rest = [(4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2)]
    tri = lambda q: [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    faces = [t for q in rest for t in tri(q)] + ([] if split else tri(back))
    name = "box_split.obj" if split else "box_whole.obj"
    obj(os.path.join(d, name), v, faces)
    p = os.path.join(d, name).replace(chr(92), "/")
    lines = room('camera "c" { eye 0 0.6 2.4  look_at 0 0.6 0  up 0 1 0  fov_y 30  film { res %d %d } }' % (W, H)) + [
        'light area { origin -0.4 2.59 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }',
        'material "glowglass" { type dielectric ior 1.5  absorb 3.0  emit 0.002 }',
        'mesh { file "%s"  material glowglass }' % p,
    ]
    if split:   # the back face as a bare quad, wound outward (-z), its emitter not the mesh's
        lines.append("quad { origin -0.3 0.3 -0.3  u 0 0.6 0  v 0.6 0 0  material glowglass }")
    return NL.join(lines + [""])


def render(exe, scn, mode, gpu, out, budget):
    pfm = out[:-4] + ".pfm"
    for f in (out, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scn, "-mode", mode, "-device", "gpu" if gpu else "cpu", "-hdr", "-no-meter",
           "-seed", "1", "-window-min", "-o", out] + budget
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        return None
    return read_pfm(pfm, top_down=True)[..., 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glow_shared"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    devs = [False] if a.no_gpu else [False, True]
    ok = True

    # SHARED: the floor's left and right halves' lower thirds, under the two panels
    scn = os.path.join(out, "shared.ftsl")
    write(scn, shared_scene(out))
    halves = {}
    for mode, budget in (("R", ["-spp", "1024"]), ("D", ["-spp", "512"]), ("B", ["-n", "40000000"])):
        for gpu in devs:
            tag = "%s on the %s" % (mode, "GPU" if gpu else "CPU")
            if gpu and budget[0] == "-spp":   # the GPU takes 4x the samples
                budget_dev = ["-spp", str(4 * int(budget[1]))]
            else:
                budget_dev = budget
            img = render(exe, scn, mode, gpu, os.path.join(out, "shared_%s_%s.png" % (mode, "gpu" if gpu else "cpu")),
                         budget_dev)
            if img is None:
                print("FAIL: the shared panels, mode %s: the scene did not render" % tag)
                ok = False
                continue
            lo = img[2 * H // 3:, :]
            l, r = float(lo[:, :W // 2].mean()), float(lo[:, W // 2:].mean())
            halves[(mode, gpu)] = (l, r)
            good = l > 0 and abs(l / r - 1.0) <= TOL
            ok = ok and good
            print("%s: the shared panels, mode %s: the floor under the instance reads %.4f of the mesh block's"
                  % ("ok" if good else "FAIL", tag, r / l if l > 0 else float("nan")), flush=True)
    ref = halves.get(("R", False))
    if ref:
        for key, (l, r) in halves.items():
            if key == ("R", False):
                continue
            w = max(abs(l / ref[0] - 1.0), abs(r / ref[1] - 1.0))
            good = w <= TOL
            ok = ok and good
            print("%s: the shared panels, mode %s on the %s against host mode R: off by %.4f"
                  % ("ok" if good else "FAIL", key[0], "GPU" if key[1] else "CPU", w), flush=True)

    # SPLIT SOLID: the box's own pixels, through which the wall shows
    imgs = {}
    for split in (False, True):
        s = os.path.join(out, "box_%s.ftsl" % ("split" if split else "whole"))
        write(s, box_scene(out, split))
        for gpu in devs:
            imgs[(split, gpu)] = render(exe, s, "R", gpu, os.path.join(out, "box_%s_%s.png" % (
                "split" if split else "whole", "gpu" if gpu else "cpu")), ["-spp", "1024" if not gpu else "4096"])
    for gpu in devs:
        a0, a1 = imgs[(False, gpu)], imgs[(True, gpu)]
        if a0 is None or a1 is None:
            print("FAIL: the split glass box on the %s did not render" % ("GPU" if gpu else "CPU"))
            ok = False
            continue
        c0 = float(a0[H // 4:3 * H // 4, 3 * W // 8:5 * W // 8].mean())
        c1 = float(a1[H // 4:3 * H // 4, 3 * W // 8:5 * W // 8].mean())
        r = c1 / c0 if c0 > 0 else float("nan")
        good = abs(r - 1.0) <= 0.02
        ok = ok and good
        print("%s: the glass box with its back a separate quad, mode R on the %s: %.4f of the one-mesh box"
              % ("ok" if good else "FAIL", "GPU" if gpu else "CPU", r), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
