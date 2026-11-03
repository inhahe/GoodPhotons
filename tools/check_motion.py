"""Regression check: motion keys and the camera shutter (ROADMAP 11 stage A).

    D:/python314/python.exe -u tools/check_motion.py [--out <dir>] [--exe ftrace.exe] [--no-gpu]

A box lit by a ceiling panel holds three moving groups: a sphere sliding left to right, a cube
turning a quarter turn about y, and a nested pair -- a group rising, holding a group that turns
-- over the scene's `motion_interval -0.5 0.5`. It requires:

  * Without a shutter, the moving scene renders BYTE FOR BYTE what a static scene holding each
    group at the frame's instant renders (mode R on the CPU, and on the GPU unless --no-gpu): the
    keys change nothing but the bounds the BVH is built from.
  * With `shutter_open -0.5  shutter_close 0.5`, mode R renders the average over the shutter of
    the static scenes: held against the mean of 32 static renders at evenly spaced times (the
    groups' transforms there written out as plain translate/rotate, interpolated as the loader
    interpolates keys), over 8x8-pixel blocks of luminance, each within 5 standard errors or
    3 %. The blur must be real: the blurred and instant frames differ in many blocks.
  * Mode W blurs too, deterministically (two runs agree byte for byte, and differ from the
    instant).
  * `-checkbvh` passes on a scene of moving triangles and spheres -- every other ray at a random
    time, so the swept bounds are held to the moved primitives.
  * Under a shutter, `-device gpu` says mode R stays on the CPU, and mode B says it renders the
    instant.
  * The loader refuses, with a message naming why: a `motion` block with no `motion_interval`; one
    with a single key; `translate` beside `motion`; an isosurface or an emissive mesh in a moving
    group; fur on, or a medium bounded by, a moving object; an interval that excludes the
    instant; `shutter_close` before `shutter_open`. And it builds a sphere its keys stretch as a
    mesh, as it does one an uneven static scale stretches.
Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
W, H = 96, 72
SPP = 1024                   # blurred mode R, per seed
TIMES = 32                   # static instants averaged for the reference
SPP_T = 64                   # per instant, per seed

# The three moving groups' keys at u = 0 and u = 1 of the motion interval [-0.5, 0.5]. Each key is
# (translate, rotate); the loader interpolates these linearly in between.
SLIDE = (((-0.8, 0.35, 0.3), (0, 0, 0)), ((0.8, 0.35, 0.3), (0, 0, 0)))
TURN = (((0.9, 0.3, -0.5), (0, 0, 0)), ((0.9, 0.3, -0.5), (0, 90, 0)))
RISE = (((-0.9, 0.0, -0.4), (0, 0, 0)), ((-0.9, 0.5, -0.4), (0, 0, 0)))
SPIN = (((0, 0.3, 0), (0, 0, 0)), ((0, 0.3, 0), (0, 0, 60)))


def lerp3(a, b, f):
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def at(keys, u):
    (t0, r0), (t1, r1) = keys
    return lerp3(t0, t1, u), lerp3(r0, r1, u)


def v3(v):
    return " ".join(repr(float(x)) for x in v)


def xf(keys, u):
    """The group transform at u, or its motion block when u is None."""
    if u is None:
        return "motion { %s }" % "  ".join("key { translate %s  rotate %s }" % (v3(t), v3(r))
                                          for t, r in keys)
    t, r = at(keys, u)
    return "translate %s  rotate %s" % (v3(t), v3(r))


CUBE = NL.join([
    "quad { origin -0.25 -0.25  0.25  u 0.5 0 0  v 0 0.5 0  material %(m)s }",
    "quad { origin  0.25 -0.25 -0.25  u -0.5 0 0  v 0 0.5 0  material %(m)s }",
    "quad { origin -0.25 -0.25 -0.25  u 0 0 0.5  v 0 0.5 0  material %(m)s }",
    "quad { origin  0.25 -0.25  0.25  u 0 0 -0.5  v 0 0.5 0  material %(m)s }",
    "quad { origin -0.25  0.25  0.25  u 0.5 0 0  v 0 0 -0.5  material %(m)s }",
    "quad { origin -0.25 -0.25 -0.25  u 0.5 0 0  v 0 0 0.5  material %(m)s }",
])


def scene(u=None, shutter=False, interval=True):
    """The box scene: its groups moving when u is None, else held at u."""
    cam = "shutter_open -0.5  shutter_close 0.5  " if shutter else ""
    return NL.join([
        "scene { units meters%s }" % ("  motion_interval -0.5 0.5" if interval else ""),
        'material "white" { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.15 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
        "group { %s  sphere { center 0 0 0  radius 0.3  material gold } }" % xf(SLIDE, u),
        "group { %s  %s }" % (xf(TURN, u), CUBE % {"m": "red"}),
        "group { %s  group { %s  %s } }" % (xf(RISE, u), xf(SPIN, u),
                                            CUBE.replace("0.25", "0.18") % {"m": "blue"}),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


def bvh_scene():
    """Many small moving primitives inside the unit cube -checkbvh fires its rays through."""
    rng = np.random.default_rng(7)
    out = ["scene { units meters  motion_interval -0.5 0.5 }",
           'material "w" { type diffuse reflect 0.5 }']
    for k in range(40):
        a, b = rng.uniform(0.1, 0.9, 3), rng.uniform(0.1, 0.9, 3)
        r0, r1 = rng.uniform(-90, 90, 3), rng.uniform(-90, 90, 3)
        body = ("sphere { center 0 0 0  radius 0.03  material w }" if k % 2 else
                "quad { origin -0.05 -0.05 0  u 0.1 0 0  v 0 0.1 0  material w }")
        out.append("group { motion { key { translate %s  rotate %s }  key { translate %s  "
                   "rotate %s } }  %s }" % (v3(a), v3(r0 if k % 2 == 0 else (0, 0, 0)), v3(b),
                                            v3(r1 if k % 2 == 0 else (0, 0, 0)), body))
    out.append("light area { origin 0.4 2 0.4  u 0.2 0 0  v 0 0 0.2  normal 0 -1 0  power 10 }")
    out.append('camera "c" { eye 0.5 0.5 3  look_at 0.5 0.5 0.5  up 0 1 0  fov_y 40 }')
    return NL.join(out) + NL


# (what is refused, the scene, a phrase its message must hold)
REFUSED = [
    ("motion without motion_interval",
     "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere { center 0 0 0 "
     "radius 0.1 material w } }", "needs the scene's `motion_interval", False),
    ("a single key",
     "group { motion { key { translate 0 0 0 } }  sphere { center 0 0 0  radius 0.1 material w } }",
     "at least two `key`", True),
    ("translate beside motion",
     "group { translate 1 0 0  motion { key { translate 0 0 0 } key { translate 1 0 0 } }  "
     "sphere { center 0 0 0  radius 0.1 material w } }", "beside a `motion` block", True),
    ("an isosurface in a moving group",
     "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  isosurface { material w  "
     "sphere { radius 0.2 } } }", "not supported yet", True),
    ("fur on a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL + 'fur "coat" { on "ball"  material w  count 10  length 0.02 }',
     "cannot follow a moving object", True),
    ("a medium bounded by a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL +
     'medium { sigma_t 1  albedo 0.9  bounds { object "ball" } }',
     "cannot follow a moving object", True),
    ("an emissive surface in a moving group",
     'material "glow" { type diffuse reflect 0.0  emit blackbody 5200 }' + NL +
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  mesh "m" { file "%(obj)s"'
     "  material glow } }", "emissive surface in a moving group", True),
]
LIGHT = ("light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
         "  power 60 }")
# A sphere uniform at the instant (the middle key) that its last key stretches: it must load, as a
# tessellated mesh -- an analytic sphere cannot become an ellipsoid.
UNEVEN = ("group { motion { key { scale 1 } key { scale 1 } key { scale 1 2 1 } }  sphere { center "
          "0 0 0  radius 0.1  material w } }")


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def read_pfm(path):
    """A PFM as h x w x 3, top row first."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data[::-1].astype(np.float64)


def render(exe, out, path, tag, args, seed=1, want_gpu=False):
    """Render `path` to <out>/<tag>.png; returns (HDR image, raw PFM bytes, log)."""
    png = os.path.join(out, tag + ".png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", path, "-seed", str(seed), "-hdr", "-no-meter", "-o", png] + args
    rc, log = run(cmd)
    if rc != 0 or not os.path.exists(pfm):
        print(log[-2000:])
        raise SystemExit("render failed (exit %d): %s" % (rc, " ".join(cmd)))
    if want_gpu and "-> GPU" not in log:
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    with open(pfm, "rb") as f:
        raw = f.read()
    return read_pfm(pfm), raw, log


def lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


def blocks(img, b=8):
    h, w = img.shape[0] // b * b, img.shape[1] // b * b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def block_err(a, b, n=8):
    """Standard error of the blocks of the mean of two seeds' images a, b (from their difference)."""
    d = a - b
    h, w = d.shape[0] // n * n, d.shape[1] // n * n
    s = (d[:h, :w] ** 2).reshape(h // n, n, w // n, n).sum(axis=(1, 3))
    return np.sqrt(s) / (2.0 * n * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_motion"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-58s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""))

    # --- the loader's refusals ---
    base = 'material "w" { type diffuse reflect 0.5 }' + NL + LIGHT
    obj = os.path.join(out, "tri.obj")
    write(obj, "v 0 0 0" + NL + "v 0.2 0 0" + NL + "v 0 0.2 0" + NL + "f 1 2 3" + NL)
    for name, body, phrase, interval in REFUSED:
        p = os.path.join(out, "refuse.ftsl")
        write(p, NL.join(["scene { units meters%s }" % ("  motion_interval -0.5 0.5" if interval
                                                        else ""), base,
                          body % {"obj": obj.replace(chr(92), "/")}]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        verdict("refuses " + name, rc != 0 and phrase in log,
                "" if rc != 0 and phrase in log else log.strip()[-300:])
    p = os.path.join(out, "uneven.ftsl")
    write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", base, UNEVEN]) + NL)
    rc, log = run([exe, "-in", p, "-parseonly"])
    m = re.search(r"(\d+) spheres, (\d+) tris", log)
    good = rc == 0 and m is not None and int(m.group(1)) == 0 and int(m.group(2)) > 100
    verdict("a sphere its keys stretch loads as a mesh", good, "" if good else log.strip()[-300:])
    for name, sc, cam, phrase in (
            ("refuses an interval that excludes the instant", "motion_interval 0.2 1", "",
             "motion_interval"),
            ("refuses shutter_close before shutter_open", "motion_interval -0.5 0.5",
             "shutter_open 0.5  shutter_close -0.5  ", "comes before `shutter_open`")):
        p = os.path.join(out, "refuse.ftsl")
        write(p, NL.join(["scene { units meters  %s }" % sc, base,
                          "sphere { center 0 0 0  radius 0.1  material w }",
                          'camera "c" { eye 0 0 3  look_at 0 0 0  up 0 1 0  fov_y 40  %s}' % cam])
              + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        verdict(name, rc != 0 and phrase in log, "" if rc != 0 and phrase in log else log.strip()[-300:])

    # --- -checkbvh across the motion interval ---
    p = os.path.join(out, "bvh.ftsl")
    write(p, bvh_scene())
    rc, log = run([exe, "-in", p, "-checkbvh"])
    verdict("-checkbvh on moving primitives", rc == 0 and "PASS" in log,
            [l for l in log.splitlines() if "checkbvh" in l][-1:][0] if "checkbvh" in log else log[-300:])

    moving, static0 = os.path.join(out, "moving.ftsl"), os.path.join(out, "static_instant.ftsl")
    blurred = os.path.join(out, "blurred.ftsl")
    write(moving, scene())
    write(static0, scene(u=0.5))
    write(blurred, scene(shutter=True))

    # --- no shutter: byte for byte the static scene at the instant ---
    devices = [("cpu", False)] + ([] if a.no_gpu else [("gpu", True)])
    for dev, gpu in devices:
        args = ["-mode", "R", "-device", dev, "-spp", "16"]
        _, raw_m, _ = render(exe, out, moving, "instant_moving_" + dev, args, want_gpu=gpu)
        _, raw_s, _ = render(exe, out, static0, "instant_static_" + dev, args, want_gpu=gpu)
        verdict("no shutter = static scene at the instant (mode R %s)" % dev.upper(), raw_m == raw_s)

    # --- blurred mode R = the average of static instants over the shutter ---
    args = ["-mode", "R", "-device", "cpu", "-spp", str(SPP)]
    bl = [render(exe, out, blurred, "blur_R_s%d" % s, args, seed=s)[0] for s in (1, 2)]
    inst = [np.zeros((H, W, 3)), np.zeros((H, W, 3))]
    for j in range(TIMES):
        u = (j + 0.5) / TIMES
        p = os.path.join(out, "static_t.ftsl")
        write(p, scene(u=u))
        for s in (1, 2):
            img = render(exe, out, p, "static_t_s%d" % s,
                         ["-mode", "R", "-device", "cpu", "-spp", str(SPP_T)], seed=100 * j + s)[0]
            inst[s - 1] += img / TIMES
    lb, li = [lum(x) for x in bl], [lum(x) for x in inst]
    mb, mi = blocks((lb[0] + lb[1]) / 2), blocks((li[0] + li[1]) / 2)
    err = np.hypot(block_err(lb[0], lb[1]), block_err(li[0], li[1]))
    z = np.abs(mb - mi) / np.maximum(err, 1e-30)
    rel = np.abs(mb - mi) / np.maximum(np.abs(mi), 1e-3 * np.abs(mi).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("blurred mode R = mean of %d instants (8x8 blocks)" % TIMES, not bad.any(),
            "worst block z %.1f rel %.3f; %d of %d blocks out" % (z.max(), rel.max(), bad.sum(), bad.size))
    total = abs(mb.mean() / mi.mean() - 1.0)
    verdict("blurred mode R, whole-frame mean", total < 0.01, "%.4f off" % total)
    still = blocks(lum(render(exe, out, static0, "instant_ref", args)[0]))
    moved = (np.abs(mb - still) / np.maximum(still, 1e-3 * still.max()) > 0.10).sum()
    verdict("the blur is there (blocks off the instant by > 10%)", moved >= 8, "%d blocks" % moved)

    # --- mode W: blurred, deterministically ---
    wargs = ["-mode", "W", "-device", "cpu", "-spp", "64"]
    _, w1, _ = render(exe, out, blurred, "blur_W_a", wargs)
    _, w2, _ = render(exe, out, blurred, "blur_W_b", wargs)
    _, w0, _ = render(exe, out, moving, "instant_W", wargs)
    verdict("mode W blurs, the same on every run", w1 == w2 and w1 != w0)

    # --- the notes a shutter brings ---
    if not a.no_gpu:
        rc, log = run([exe, "-in", blurred, "-mode", "R", "-device", "gpu", "-spp", "1", "-no-meter",
                       "-o", os.path.join(out, "gate.png")])
        verdict("-device gpu under a shutter says it stays on the CPU",
                rc == 0 and "renders on the CPU so far" in log, log.strip()[-300:] if rc else "")
    rc, log = run([exe, "-in", blurred, "-mode", "B", "-device", "cpu", "-n", "100000", "-no-meter",
                   "-o", os.path.join(out, "modeb.png")])
    verdict("mode B under a shutter says it renders the instant",
            rc == 0 and "renders the frame's instant" in log, log.strip()[-300:] if rc else "")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
