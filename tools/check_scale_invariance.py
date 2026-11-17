"""Regression check: a scene drawn 100 times smaller renders the same image.

    D:/python314/python.exe -u tools/check_scale_invariance.py [--out <dir>] [--exe ftrace.exe]

check_modew_fiber's fur ball under a stubble coat (6 mm strands, 1.2 mm thick), lit by its area lamp
alone, rendered in mode W at 16 spp (deterministic) in `units meters` and in `units centimeters` --
the same numbers, so the same scene a hundredth the size, its strands 12 um thick. The lamp keeps its
power, so the small scene's radiance is 1e4 times the large one's; scaled back, the two must agree:
the frame within 1 % and each live tile of a 4 x 4 grid within 3 %.

A renderer's self-intersection offsets are what break this. The device skipped an absolute 1e-4 at
the start of every trace and pushed origins by Wachter-Binder's constants for a scene of unit size,
and in centimetres read x2.5 over its own metre render (GPU-TMIN-SCALE, GPU-MINIATURE-SCALE); the host
offset everything by an absolute 1e-6, a twelfth of a strand there, and read x1.13, tiles to x1.45
(CPU-MINIATURE-SCALE). Both now take their distances from the scene's length scale (Scene::lenScale,
min(1, S); the device's c_lenScale). Exits 1 on failure.
"""
import argparse
import os
import sys

from check_modew_fiber import STUBBLE, compare, render_w  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCALE = 0.01        # centimetres against metres
SCENE = [l for l in STUBBLE if not l.startswith("light sun")]
# The same ball on a ground 400 m across, so that even drawn 100 times smaller the scene reaches past
# unit size and its scale (c_lenScale) is 1: a small object in a big scene, whose offsets are sized by
# its own primitives' reach (DHit::pscale) or not at all. The device only: the host's floor is an
# absolute 1e-6 in a scene of unit size or more (known-issues CPU-SMALL-IN-BIG).
BIG = [l.replace("quad { origin -2 0 -2   u 0 0 4   v 4 0 0", "quad { origin -200 0 -200   u 0 0 400   v 400 0 0")
       for l in SCENE]
CASES = [("ball", SCENE, ("cpu", "gpu")), ("ball_big_ground", BIG, ("gpu",))]


def render(exe, out, name, lines, units, dev):
    scene = os.path.join(out, "%s_%s.ftsl" % (name, units))
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(l.replace("units meters", "units " + units) for l in lines) + NL)
    return render_w(exe, out, scene, "%s_%s_%s" % (name, units, dev), dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_scale_invariance"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for name, lines, devices in CASES:
        for dev in devices:
            big = render(exe, out, name, lines, "meters", dev)
            small = render(exe, out, name, lines, "centimeters", dev) * SCALE ** 2
            frame, worst = compare(small, big)
            good = abs(frame - 1.0) <= 0.01 and worst <= 0.03
            ok &= good
            print("%s: %s, %s drawn 100x smaller against itself: frame x%.4f, worst tile off by %.2f %%"
                  % ("ok" if good else "FAIL", dev, name, frame, 100.0 * worst))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
