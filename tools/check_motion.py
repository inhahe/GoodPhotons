"""Regression check: motion keys and the camera shutter (ROADMAP 11 stage A).

    D:/python314/python.exe -u tools/check_motion.py [--out <dir>] [--exe ftrace.exe] [--no-gpu]

A box lit by a ceiling panel holds four moving groups over the scene's `motion_interval -0.5 0.5`:
a sphere sliding left to right, a cube turning a quarter turn about y, a nested pair -- a group
that rises and then, from the instant on, moves sideways, holding a group that turns -- and a
cube whose keys scale and shear it together. It requires:

  * Without a shutter, the moving scene renders BYTE FOR BYTE what a static scene holding each
    group at the frame's instant renders (mode R on the CPU, and on the GPU unless --no-gpu), even
    with a ball whose motion reaches 60 m off the stage, on a floor whose albedo follows `cavity`
    (whose default probe radius is a share of the scene's size): the keys change nothing but the
    bounds the BVH is built from, and the scene's own bounds stay its instant's.
  * With `shutter_open -0.5  shutter_close 0.5`, mode R renders the average over the shutter of
    the static scenes: held against the mean of 32 static renders at evenly spaced times (the
    groups' transforms there written out as plain translate/rotate/scale/shear, interpolated as
    the loader interpolates keys), over 8x8-pixel blocks of luminance, each within 5 standard
    errors or 3 %. The blur must be real: the blurred and instant frames differ in many blocks.
    The same scene authored in centimetres blurs the same.
  * An instantaneous shutter away from the instant (`shutter_open t  shutter_close t`) renders the
    static scene at t, for three t, the same block test -- with a ball added that orbits two whole
    turns on each side of the instant, which the keys must not lose.
  * A procedural pattern travels with its moving object: a striped ball held sharp late in the
    shutter equals the static ball there with its world-space stripes shifted by its travel
    (4x4 blocks, the same test).
  * Mode W blurs too, deterministically (two runs agree byte for byte, and differ from the
    instant).
  * `-checkbvh` passes on a scene of moving triangles and spheres -- every other ray at a random
    time, so the swept bounds are held to the moved primitives.
  * Blurred modes D (BDPT) and B (forward light tracing) on the CPU agree with blurred mode R.
    So do two mode B cameras with the shutter rendered beside a third without it (one flight per
    shutter, on any device), the third staying sharp; and the sky mode B sees directly behind a
    dark ball sweeping across it.
  * Under a shutter, `-device gpu` says mode R stays on the CPU, and mode M says it renders the
    instant.
  * The loader refuses, with a message naming why: a `motion` block with no `motion_interval`; one
    with a single key; `translate` beside `motion`; fur on, or a medium bounded by, a moving
    object; an interval that excludes the instant; `shutter_close` before `shutter_open`. It keeps
    an isosurface and an emissive mesh in a moving group at the instant, saying so -- and a render
    held sharp late in the shutter shows them there while their sibling ball has moved -- and builds
    a sphere its keys stretch as a mesh, as it does one an uneven static scale stretches.
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

# The moving groups, each a list of keys spread evenly over the motion interval [-0.5, 0.5]; a key
# sets translate / rotate / scale / shear, and the loader interpolates each linearly between keys.
SLIDE = [dict(translate=(-0.8, 0.35, 0.3)), dict(translate=(0.8, 0.35, 0.3))]
TURN = [dict(translate=(0.9, 0.3, -0.5)), dict(translate=(0.9, 0.3, -0.5), rotate=(0, 90, 0))]
# A parent that kinks at the instant -- up, then sideways -- holding a child that turns at its own
# pace: the child's keys must still include the parent's kink and the instant.
RISE = [dict(translate=(-0.9, 0.0, -0.4)), dict(translate=(-0.9, 0.4, -0.4)),
        dict(translate=(-0.5, 0.45, -0.4))]
SPIN = [dict(translate=(0, 0.3, 0)), dict(translate=(0, 0.3, 0), rotate=(0, 0, 15))]
# Scale and shear changing together: their product is not linear, so blending two keys is not
# the transform as written in between.
STRETCH = [dict(translate=(0.15, 0.12, -0.9)),
           dict(translate=(0.15, 0.12, -0.9), scale=(1.8, 1.8, 1.8), shear=(0.8, 0, 0))]
# Extras for the instant test: a ball that waits, then flies far off the stage -- the moving
# scene's sweep reaching well past the static one's bounds -- over a floor read through `cavity`,
# whose default probe radius is sized by those bounds.
FLY = [dict(translate=(-0.5, 1.6, -1.2)), dict(translate=(-0.5, 1.6, -1.2)),
       dict(translate=(60.0, 1.6, -1.2))]
# A ball orbiting its group's axis, two whole turns each side of the instant: a stretch between
# breaks whose midpoint equals the blend of its ends, so only a split by turn can follow it (used
# by the sharp-at-a-time tests).
TWIRL = [dict(translate=(0.55, 1.25, -1.0)), dict(translate=(0.55, 1.25, -1.0), rotate=(0, 0, 1440))]
FIELDS = (("translate", (0, 0, 0)), ("rotate", (0, 0, 0)), ("scale", (1, 1, 1)), ("shear", (0, 0, 0)))
# A dark ball sweeping across an open sky: the forward modes add the sky they see directly by a
# camera-side pass of their own, which must see the ball where each time in the shutter has it.
SWEEP = [dict(translate=(-0.9, 0.9, 0)), dict(translate=(0.9, 0.9, 0))]


def at(keys, u):
    """The group's fields at u in [0, 1], interpolated linearly between its evenly spread keys."""
    s = min(max(u, 0.0), 1.0) * (len(keys) - 1)
    i = min(int(s), len(keys) - 2)
    f = s - i
    out = {}
    for name, dflt in FIELDS:
        a, b = keys[i].get(name, dflt), keys[i + 1].get(name, dflt)
        out[name] = tuple(x + (y - x) * f for x, y in zip(a, b))
    return out


def v3(v, k=1.0):
    return " ".join(repr(float(x) * k) for x in v)


def fields(d, k):
    """A key's statements; translate is a length (times the unit factor k), the rest are not."""
    return "  ".join("%s %s" % (n, v3(d.get(n, dflt), k if n == "translate" else 1.0))
                     for n, dflt in FIELDS if n in d or n == "translate")


def xf(keys, u, k):
    """The group transform at u, or its motion block when u is None."""
    if u is None:
        return "motion { %s }" % "  ".join("key { %s }" % fields(key, k) for key in keys)
    return fields(at(keys, u), k)


def cube(h, m, k):
    """Six quads: a cube of half-size h about the origin, in material m, lengths times k."""
    f = lambda *xs: " ".join(repr(x * h * k) for x in xs)
    return NL.join([
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, 1), f(2, 0, 0), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(1, -1, -1), f(-2, 0, 0), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, -1), f(0, 0, 2), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(1, -1, 1), f(0, 0, -2), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, 1, 1), f(2, 0, 0), f(0, 0, -2), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, -1), f(2, 0, 0), f(0, 0, 2), m),
    ])


def scene(u=None, shutter=None, units="meters", extras=False, twirl=False):
    """The box scene: its groups moving when u is None, else held at u. `shutter` is
    (open, close) in frames; `units` "centimeters" writes every length 100 times larger."""
    k = {"meters": 1.0, "centimeters": 100.0}[units]
    L = lambda *xs: " ".join(repr(x * k) for x in xs)
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    lines = [
        "scene { units %s  motion_interval -0.5 0.5 }" % units,
        'material "white" { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "green" { type diffuse reflect rgb 0.15 0.6 0.2 }',
        'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.15 }',
        "quad { origin %s  u %s  v %s  material %s }" % (L(-2, 0, -2), L(0, 0, 4), L(4, 0, 0),
                                                          "cavfloor" if extras else "white"),
        "quad { origin %s  u %s  v %s  material white }" % (L(-2, 0, -2), L(4, 0, 0), L(0, 3, 0)),
        "light area { origin %s  u %s  v %s  normal 0 -1 0  spd preset:bb6500  power 60 }"
        % (L(-0.4, 2.9, -0.4), L(0.8, 0, 0), L(0, 0, 0.8)),
        "group { %s  sphere { center 0 0 0  radius %r  material gold } }" % (xf(SLIDE, u, k), 0.3 * k),
        "group { %s  %s }" % (xf(TURN, u, k), cube(0.25, "red", k)),
        "group { %s  group { %s  %s } }" % (xf(RISE, u, k), xf(SPIN, u, k), cube(0.18, "blue", k)),
        "group { %s  %s }" % (xf(STRETCH, u, k), cube(0.1, "green", k)),
        'camera "c" { eye %s  look_at %s  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
        % (L(0, 1.0, 4.2), L(0, 0.5, 0), cam, W, H),
    ]
    if twirl:
        lines.append("group { %s  sphere { center %s  radius %r  material red } }"
                     % (xf(TWIRL, u, k), L(0.45, 0, 0), 0.15 * k))
    if extras:
        # The floor's albedo follows `cavity`, whose default probe radius is a share of the
        # scene's size -- which a mover's sweep must not change.
        lines += ['pattern "cav" { expr "0.25 + 0.6 * cavity" }',
                  'material "cavfloor" { type diffuse reflect pattern:cav }',
                  "light env { spd preset:d65 }",
                  "group { %s  sphere { center 0 0 0  radius %r  material gold } }"
                  % (xf(FLY, u, k), 0.15 * k)]
    return NL.join(lines) + NL


def sky_scene(shutter):
    """The SWEEP ball over a floor under an environment light, nothing behind it but sky."""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.6 }',
        'material "dark" { type diffuse reflect 0.1 }',
        "light env { spd preset:d65 }",
        "quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material white }",
        "group { %s  sphere { center 0 0 0  radius 0.3  material dark } }" % xf(SWEEP, None, 1.0),
        'camera "c" { eye 0 0.8 3.5  look_at 0 0.8 0  up 0 1 0  fov_y 40  '
        'shutter_open %r  shutter_close %r  film { res %d %d } }' % (shutter + (W, H)),
    ]) + NL


def still_scene(obj, u=None, shutter=None):
    """A moving group holding a ball, an emissive triangle and an isosurface ball: the ball moves,
    the other two stay at the instant. With `u`, the static equivalent: the ball held at u and the
    other two at the instant (u = 0.5)."""
    keys = [dict(translate=(-0.6, 0.45, 0.2)), dict(translate=(0.6, 0.45, 0.2))]
    ball = "sphere { center 0 0 0  radius 0.25  material gold }"
    still = ('mesh "glow_tri" { file "%s"  material glow  translate -0.2 0.45 0  scale 2 }  '
             "isosurface { material blue  sphere { center 0.3 0.6 0  radius 0.15 } }" % obj)
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    lines = [
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.15 }',
        'material "glow"  { type diffuse reflect 0.0  emit blackbody 5200 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
    ]
    if u is None:
        lines.append("group { %s  %s  %s }" % (xf(keys, None, 1.0), ball, still))
    else:
        lines.append("group { %s  %s }" % (xf(keys, u, 1.0), ball))
        lines.append("group { %s  %s }" % (xf(keys, 0.5, 1.0), still))
    lines.append('camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
                 % (cam, W, H))
    return NL.join(lines) + NL


def stripe_scene(u=None, shutter=None, shift=0.0):
    """A ball striped by a world-space pattern, sliding left to right. Moving (u None), its
    stripes must travel with it; held at u with the pattern shifted by `shift` in x, the static
    equivalent of that."""
    keys = [dict(translate=(-0.6, 0.45, 0.2)), dict(translate=(0.6, 0.45, 0.2))]
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'pattern "stripes" { expr "0.5 + 0.4 * sin(25 * (x - %r))" }' % shift,
        'material "striped" { type diffuse reflect pattern:stripes }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
        "group { %s  sphere { center 0 0 0  radius 0.35  material striped } }" % xf(keys, u, 1.0),
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
    ("fur on a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL + 'fur "coat" { on "ball"  material w  count 10  length 0.02 }',
     "cannot follow a moving object", True),
    ("a medium bounded by a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL +
     'medium { sigma_t 1  albedo 0.9  bounds { object "ball" } }',
     "cannot follow a moving object", True),
]
# (what stays at the instant, the scene, the warning it must print) -- loaded, not refused
STILL = [
    ("an isosurface in a moving group",
     "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  isosurface { material w  "
     "sphere { radius 0.2 } } }", "an isosurface in a moving group stays where it is"),
    ("an emissive mesh in a moving group",
     'material "glow" { type diffuse reflect 0.0  emit blackbody 5200 }' + NL +
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  mesh "m" { file "%(obj)s"'
     "  material glow } }", "an emissive surface in a moving group stays where it is"),
]
LIGHT = ("light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
         "  power 60 }")
# A sphere uniform at the instant (the middle key) that its last key stretches: it must load, as a
# tessellated mesh -- an analytic sphere cannot become an ellipsoid.
UNEVEN = ("group { motion { key { scale 1 } key { scale 1 } key { scale 1 2 1 } }  sphere { center "
          "0 0 0  radius 0.1  material w } }")
# The same, the unevenness from a static ancestor: turning inside `scale 1 2 1` (undone below the
# turn, so the sphere is round at the instant) is no similarity.
UNEVEN_ANCESTOR = ("group { scale 1 2 1  group { motion { key { rotate 0 0 -45 } key { rotate 0 0 45 } }  "
                   "group { scale 1 0.5 1  sphere { center 0 0 0  radius 0.1  material w } } } }")


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
    for name, body, phrase in STILL:
        p = os.path.join(out, "still.ftsl")
        write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", base,
                          body % {"obj": obj.replace(chr(92), "/")}]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        good = rc == 0 and phrase in log
        verdict("keeps %s at the instant, saying so" % name, good, "" if good else log.strip()[-300:])
    # ...and render so: held sharp late in the shutter, the ball has moved and the triangle and
    # the isosurface have not.
    objf = obj.replace(chr(92), "/")
    sharp, ref = os.path.join(out, "still_sharp.ftsl"), os.path.join(out, "still_ref.ftsl")
    write(sharp, still_scene(objf, shutter=(0.4, 0.4)))
    write(ref, still_scene(objf, u=0.9))
    ra = [lum(render(exe, out, sharp, "still_sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                              "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    rb = [lum(render(exe, out, ref, "still_ref_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                          "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    ma, mr = blocks((ra[0] + ra[1]) / 2), blocks((rb[0] + rb[1]) / 2)
    err = np.hypot(block_err(ra[0], ra[1]), block_err(rb[0], rb[1]))
    z = np.abs(ma - mr) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("the kinds that cannot move stay; their siblings move", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))
    for name, body in (("a sphere its keys stretch loads as a mesh", UNEVEN),
                       ("a sphere turning inside an uneven scale loads as a mesh", UNEVEN_ANCESTOR)):
        p = os.path.join(out, "uneven.ftsl")
        write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", base, body]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        m = re.search(r"(\d+) spheres, (\d+) tris", log)
        good = rc == 0 and m is not None and int(m.group(1)) == 0 and int(m.group(2)) > 100
        verdict(name, good, "" if good else log.strip()[-300:])
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
    write(blurred, scene(shutter=(-0.5, 0.5)))

    # --- no shutter: byte for byte the static scene at the instant, even with a mover whose
    # sweep reaches far past the stage, under an environment light sized by the scene's bounds ---
    devices = [("cpu", False)] + ([] if a.no_gpu else [("gpu", True)])
    moving_x, static_x = os.path.join(out, "moving_x.ftsl"), os.path.join(out, "static_x.ftsl")
    write(moving_x, scene(extras=True))
    write(static_x, scene(u=0.5, extras=True))
    for dev, gpu in devices:
        args = ["-mode", "R", "-device", dev, "-spp", "16"]
        _, raw_m, _ = render(exe, out, moving_x, "instant_moving_" + dev, args, want_gpu=gpu)
        _, raw_s, _ = render(exe, out, static_x, "instant_static_" + dev, args, want_gpu=gpu)
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

    # --- the other modes that blur agree with mode R: BDPT (a time per path pair) and the
    # forward light tracer (a time per photon, its camera splat at the same time) ---
    for mode, margs in (("D", ["-spp", "256"]), ("B", ["-n", "40000000"])):
        bm = [lum(render(exe, out, blurred, "blur_%s_s%d" % (mode, s),
                         ["-mode", mode, "-device", "cpu"] + margs, seed=s)[0]) for s in (1, 2)]
        mm = blocks((bm[0] + bm[1]) / 2)
        err = np.hypot(block_err(lb[0], lb[1]), block_err(bm[0], bm[1]))
        z = np.abs(mm - mb) / np.maximum(err, 1e-30)
        rel = np.abs(mm - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
        bad = (z > 5.0) & (rel > 0.05)
        verdict("blurred mode %s = blurred mode R (8x8 blocks)" % mode, not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- several mode B cameras in one run: two with the shutter share a flight and both blur,
    # one without it (declared first) renders apart and stays sharp. On any device. ---
    text = open(blurred, encoding="utf-8").read()
    cam_c = [l for l in text.splitlines() if l.startswith('camera "c"')][0]
    cam_d = cam_c.replace('camera "c"', 'camera "d"')
    cam_a = re.sub(r"shutter_open \S+  shutter_close \S+  ", "", cam_c.replace('camera "c"', 'camera "a"'))
    multi = os.path.join(out, "blurred_multi.ftsl")
    write(multi, text.replace(cam_c, NL.join([cam_a, cam_c, cam_d])))
    imgs = {}
    for s in (1, 2):
        png = os.path.join(out, "multi_s%d.png" % s)
        rc, log = run([exe, "-in", multi, "-seed", str(s), "-hdr", "-no-meter", "-o", png,
                       "-mode", "B", "-n", "40000000"] + (["-device", "cpu"] if a.no_gpu else []))
        if rc != 0:
            print(log[-2000:])
            raise SystemExit("render failed: " + multi)
        for c in "acd":
            imgs.setdefault(c, []).append(lum(read_pfm(png[:-4] + "_%s.pfm" % c)))
    for c in "cd":
        mm = blocks((imgs[c][0] + imgs[c][1]) / 2)
        err = np.hypot(block_err(lb[0], lb[1]), block_err(imgs[c][0], imgs[c][1]))
        z = np.abs(mm - mb) / np.maximum(err, 1e-30)
        rel = np.abs(mm - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
        bad = (z > 5.0) & (rel > 0.05)
        verdict("... shared mode B flight, camera %s = blurred mode R" % c, not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))
    sharp_a = blocks((imgs["a"][0] + imgs["a"][1]) / 2)
    off = (np.abs(sharp_a - still) / np.maximum(still, 1e-3 * still.max()) > 0.10).sum()
    verdict("... and the camera without a shutter stays sharp", off < 4, "%d blocks off the instant" % off)

    # --- the sky a forward mode sees directly blurs behind a mover as mode R's does ---
    sky = os.path.join(out, "sky.ftsl")
    write(sky, sky_scene((-0.5, 0.5)))
    sr = [lum(render(exe, out, sky, "sky_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "512"],
                     seed=s)[0]) for s in (1, 2)]
    sbm = [lum(render(exe, out, sky, "sky_B_s%d" % s, ["-mode", "B", "-device", "cpu", "-n", "40000000"],
                      seed=s)[0]) for s in (1, 2)]
    mr, mbs = blocks((sr[0] + sr[1]) / 2), blocks((sbm[0] + sbm[1]) / 2)
    err = np.hypot(block_err(sr[0], sr[1]), block_err(sbm[0], sbm[1]))
    z = np.abs(mbs - mr) / np.maximum(err, 1e-30)
    rel = np.abs(mbs - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.05)
    verdict("blurred mode B's sky behind a mover = mode R's", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- the same blur authored in centimetres: the keys move metres, as the vertices are ---
    blurred_cm = os.path.join(out, "blurred_cm.ftsl")
    write(blurred_cm, scene(shutter=(-0.5, 0.5), units="centimeters"))
    bc = [lum(render(exe, out, blurred_cm, "blur_cm_s%d" % s, args, seed=s)[0]) for s in (1, 2)]
    mc = blocks((bc[0] + bc[1]) / 2)
    err = np.hypot(block_err(lb[0], lb[1]), block_err(bc[0], bc[1]))
    z = np.abs(mc - mb) / np.maximum(err, 1e-30)
    rel = np.abs(mc - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("the blur authored in centimetres = in metres", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- an instantaneous shutter away from the instant: the scene exactly as written there ---
    for t in (-0.35, 0.1, 0.4):
        sharp, still_t = os.path.join(out, "sharp.ftsl"), os.path.join(out, "sharp_static.ftsl")
        write(sharp, scene(shutter=(t, t), twirl=True))
        write(still_t, scene(u=t + 0.5, twirl=True))
        sa = [lum(render(exe, out, sharp, "sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                             "-spp", "256"], seed=s)[0]) for s in (1, 2)]
        sb = [lum(render(exe, out, still_t, "sharp_static_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                                     "-spp", "256"], seed=s)[0])
              for s in (1, 2)]
        ma, ms = blocks((sa[0] + sa[1]) / 2), blocks((sb[0] + sb[1]) / 2)
        err = np.hypot(block_err(sa[0], sa[1]), block_err(sb[0], sb[1]))
        z = np.abs(ma - ms) / np.maximum(err, 1e-30)
        rel = np.abs(ma - ms) / np.maximum(np.abs(ms), 1e-3 * np.abs(ms).max())
        bad = (z > 5.0) & (rel > 0.03)
        verdict("shutter %g..%g = the scene as written at %g" % (t, t, t), not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- a pattern travels with its moving object: held sharp at u = 0.9 the ball has slid 0.48
    # m, and its stripes with it ---
    sharp, ref = os.path.join(out, "stripe_sharp.ftsl"), os.path.join(out, "stripe_ref.ftsl")
    write(sharp, stripe_scene(shutter=(0.4, 0.4)))
    write(ref, stripe_scene(u=0.9, shift=-0.6 + 1.2 * 0.9 - 0.0))
    pa = [lum(render(exe, out, sharp, "stripe_sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                               "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    pb = [lum(render(exe, out, ref, "stripe_ref_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                           "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    ma, mr = blocks((pa[0] + pa[1]) / 2, 4), blocks((pb[0] + pb[1]) / 2, 4)
    err = np.hypot(block_err(pa[0], pa[1], 4), block_err(pb[0], pb[1], 4))
    z = np.abs(ma - mr) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("a pattern travels with its moving object", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

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
    rc, log = run([exe, "-in", blurred, "-mode", "M", "-device", "cpu", "-n", "100000", "-no-meter",
                   "-o", os.path.join(out, "modem.png")])
    verdict("mode M under a shutter says it renders the instant",
            rc == 0 and "renders the frame's instant" in log, log.strip()[-300:] if rc else "")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
