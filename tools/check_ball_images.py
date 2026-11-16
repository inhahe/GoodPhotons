"""Regression check: mode B draws a glass ball's reflections, seen from outside it and from inside it.

    python tools/check_ball_images.py [--out png/check_ball_images] [--exe ftrace.exe] [--no-gpu]

BALL-EYE-HALO. Mode B images a smooth dielectric sphere by solving each chain of surface events a ray
from the eye can take through it (Renderer::connectSpecularSphere). With only the refraction through
it, mode B drew a glass ball some 10 % dim and its rim as little as a fifth of mode R's, and an eye
inside a glass ball missed the halo the ball's inner surface shows round a lamp: the wall behind the
eye, where the ball focuses the lamp into a bright spot, seen in the inner surface. Each case is mode B
on the CPU and the GPU against mode R, the region's mean within the tolerance and 5 standard errors
over the seeds:
  * a glass ball in a lit room, seen from outside: the ball's body and its rim (the surface reflection
    is all of the light at grazing, and the light reflected many times inside carries much of the
    rest there); and a mirror ball in its place, its rim the reflection at grazing;
  * the eye at the centre of a small glass ball in the room, looking at the lamp: the rings round the
    lamp where the inner reflection shows the spot behind the eye, and the walls beyond;
  * a glass ball 5 cm from the back wall of a room whose walls all glow alike and reflect nothing:
    the ball's image reads what the walls read (its chains' products sum to one), within 2 % and 5
    standard errors -- with no reference to trust. The ball nearly focuses on a wall that close, and
    by the fold a vertex's two images lie within one step of the connector's scan, where they change
    no sign: a dark ring there had read 0.82 of the walls.
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

ROOM = NL.join([
    "scene { units meters }",
    'material "white" { type diffuse reflect 0.7 }',
    'material "glass" { type dielectric  ior 1.5 }',
    "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
    "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material white }",
    "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
    "quad { origin -2 0 2  u 0 3 0  v 4 0 0  material white }",
    "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material white }",
    "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material white }",
    "light area { origin -0.15 1.35 -1.0  u 0.3 0 0  v 0 0.3 0  normal 0 0 1  spd preset:bb6500  power 20 }",
])
BALL = "sphere { center 0 1.5 1.8  radius 0.15  material glass }"
# The ball seen from outside, past the bright spot it focuses on the wall behind it.
OUTSIDE = "camera \"c\" { eye 0.8 1.5 1.7  look_at 0 1.5 2.0  up 0 1 0  fov_y 12  film { res 96 72 } }"
# The eye at the ball's centre, looking at the lamp.
INSIDE = "camera \"c\" { eye 0 1.5 1.8  look_at 0 1.5 -1.0  up 0 1 0  fov_y 50  film { res 96 72 } }"
# The furnace: every wall glows alike and reflects nothing, the glass ball (no absorption) 5 cm from the
# back wall, the eye in front of it. Its chains' Fresnel products sum to one, so its image reads exactly
# what the walls read.
FURNACE = NL.join([
    "scene { units meters }",
    'material "glow" { type diffuse  reflect 0  emit preset:bb6500 }',
    'material "glass" { type dielectric  ior 1.5 }',
] + [q.replace("material white", "material glow") for q in ROOM.split(NL) if q.startswith("quad")] + [
    "sphere { center 0 1.5 1.65  radius 0.3  material glass }",
    'camera "c" { eye 0 1.5 0.75  look_at 0 1.5 1.65  up 0 1 0  fov_y 50  film { res 96 72 } }',
])
DEPTH = ["-max-bounce", "12"]
TOL = 0.03
SEEDS = (1, 2)


def scene(out, name, *parts):
    path = os.path.join(out, name + ".ftsl")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(parts) + NL)
    return path


def render(exe, out, scn, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scn, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-o", png] + DEPTH + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[..., 1].astype(np.float64)


def erode(m, k):
    # The mask's pixels whose k-neighbourhood (a square) lies wholly inside it.
    h, w = m.shape
    e = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            s = np.zeros_like(m)
            s[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h + min(-dy, 0),
                                                                          max(-dx, 0):w + min(-dx, 0)]
            e &= s
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_ball_images"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    refdev = "gpu" if not a.no_gpu else "cpu"

    outside = scene(out, "outside", ROOM, BALL, OUTSIDE)
    mirror = scene(out, "mirror", ROOM, 'material "chrome" { type mirror  reflect 0.9 }',
                   BALL.replace("material glass", "material chrome"), OUTSIDE)
    inside = scene(out, "inside", ROOM, BALL, INSIDE)
    # The ball's image, from the same view with the ball glowing and the lamp off.
    glow = scene(out, "outside_mask", ROOM.replace("power 20", "power 0.0001"),
                 'material "ember" { type diffuse  reflect 0  emit preset:bb6500 }',
                 BALL.replace("material glass", "material ember"), OUTSIDE)
    m = render(exe, out, glow, "mask", ["-mode", "R", "-device", "cpu", "-spp", "16"], 1)
    ball = m > 0.5 * m.max()
    body, rim = erode(ball, 3), ball & ~erode(ball, 2)
    yy, xx = np.mgrid[0:72, 0:96]
    r2 = (xx - 47.5) ** 2 + (yy - 35.5) ** 2
    cases = [("from outside", outside, [("the ball's body", body), ("its rim", rim)]),
             ("of mirror, from outside", mirror, [("the ball's body", body), ("its rim", rim)]),
             ("from inside", inside, [("rings 4-15 px round the lamp", (r2 >= 4 ** 2) & (r2 < 15 ** 2)),
                                      ("rings 15-25 px", (r2 >= 15 ** 2) & (r2 < 25 ** 2)),
                                      ("the walls beyond", r2 >= 25 ** 2)])]
    if not body.any() or not rim.any():
        raise SystemExit("the ball's image is empty: the frame is not the one this check reads")

    def stats(scn, tag, args):
        v = np.array([[render(exe, out, scn, tag, args, s)[reg].mean() for _, reg in regions] for s in SEEDS])
        return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(SEEDS))

    ok = True
    # The furnace, against itself: the ball's rings (its image some 28 px across the frame's centre) over
    # the walls.
    furnace = scene(out, "furnace", FURNACE)
    rings = [("rings 0-8 px", r2 < 8 ** 2), ("8-16 px", (r2 >= 8 ** 2) & (r2 < 16 ** 2)),
             ("16-22 px", (r2 >= 16 ** 2) & (r2 < 22 ** 2)), ("22-26 px", (r2 >= 22 ** 2) & (r2 < 26 ** 2))]
    for dev in devs:
        n = "100000000" if dev == "gpu" else "60000000"
        v = []
        for s in SEEDS:
            im = render(exe, out, furnace, "B_furnace_%s" % dev, ["-mode", "B", "-device", dev, "-n", n], s)
            wall = im[r2 > 40 ** 2].mean()
            v.append([im[reg].mean() / wall for _, reg in rings])
        v = np.array(v)
        q, err = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(SEEDS))
        for (rname, _), qi, ei in zip(rings, q, err):
            bad = not (np.isfinite(qi) and np.isfinite(ei)) or (abs(qi - 1.0) > 0.02 and abs(qi - 1.0) > 5.0 * ei)
            ok = ok and not bad
            print("%s: mode B on the %s, a glass ball in a glowing room, %s: %.4f (+- %.4f) of the walls"
                  % ("FAIL" if bad else "ok", dev.upper(), rname, qi, ei), flush=True)
    for what, scn, regions in cases:
        tag = os.path.splitext(os.path.basename(scn))[0]
        rm, rse = stats(scn, "R_%s" % tag, ["-mode", "R", "-device", refdev, "-spp", "8192"])
        for dev in devs:
            n = "100000000" if dev == "gpu" else "40000000"
            bm, bse = stats(scn, "B_%s_%s" % (tag, dev), ["-mode", "B", "-device", dev, "-n", n])
            q = bm / rm
            err = np.hypot(bse / rm, rse * bm / (rm * rm))
            for (rname, _), qi, ei in zip(regions, q, err):
                bad = not (np.isfinite(qi) and np.isfinite(ei)) or (abs(qi - 1.0) > TOL and abs(qi - 1.0) > 5.0 * ei)
                ok = ok and not bad
                print("%s: mode B on the %s, the ball %s, %s: %.4f (+- %.4f) of mode R"
                      % ("FAIL" if bad else "ok", dev.upper(), what, rname, qi, ei), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
