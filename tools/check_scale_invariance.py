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
import subprocess
import sys

import numpy as np
from pfm import read_pfm  # noqa: E402
from check_modew_fiber import STUBBLE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
DEVICES = ("cpu", "gpu")
SCALE = 0.01        # centimetres against metres
SCENE = [l for l in STUBBLE if not l.startswith("light sun")]


def render(exe, out, units, dev):
    scene = os.path.join(out, "ball_%s.ftsl" % units)
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(l.replace("units meters", "units " + units) for l in SCENE) + NL)
    png = os.path.join(out, "%s_%s.png" % (units, dev))
    cmd = [exe, "-in", scene, "-mode", "W", "-device", dev, "-spp", "16", "-hdr", "-no-meter", "-o", png]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    pfm = png[:-4] + ".pfm"
    if r.returncode != 0 or not os.path.exists(pfm):
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=False).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_scale_invariance"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for dev in DEVICES:
        big = render(exe, out, "meters", dev)
        small = render(exe, out, "centimeters", dev) * SCALE ** 2
        h, w = big.shape
        tile = lambda v: np.array([[v[h * i // 4:h * (i + 1) // 4, w * j // 4:w * (j + 1) // 4].mean()
                                    for j in range(4)] for i in range(4)])
        tb, ts = tile(big), tile(small)
        live = tb > 0.02 * tb.max()
        worst = float(np.max(np.where(live, np.abs(ts / np.maximum(tb, 1e-30) - 1.0), 0.0)))
        frame = float(small.mean() / max(big.mean(), 1e-30))
        good = abs(frame - 1.0) <= 0.01 and worst <= 0.03
        ok &= good
        print("%s: %s, the ball drawn 100x smaller against itself: frame x%.4f, worst tile off by %.2f %%"
              % ("ok" if good else "FAIL", dev, frame, 100.0 * worst))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
