"""Regression check: the sky a forward mode sees directly does not depend on when the render writes.

    python tools/check_sky_repeat.py [--out png/check_sky_repeat] [--exe ftrace.exe]

The forward modes draw the sky the camera sees directly in a pass of their own (main.cpp
addEnvBackground), and through a physical lens that pass is a Monte Carlo average whose sample count
grows with the photons per pixel, computed in batches as the writes ask for more. Until 0.440.11 a
batch's random streams were seeded per batch, so the same render written at another cadence -- a
live window's refreshes fall where the clock puts them -- drew other samples and showed another sky.
Each sample is now a function of its pixel and its index alone, added in index order.

The room open to a sky (check_bump_modes' fifth room, flat), through a 35 mm f/4 achromat, mode B,
written at the default interval and every 0.05 s: the two images must be byte-identical. 0.440.9
differed in 363 of the 3072 pixels. Exits 1 on a difference.
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
import check_bump_modes as cbm  # noqa: E402

LENS_CAM = ('camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  focus 4  '
            'film { res 64 48  format full-frame }  lens { preset achromat  focal 35  fstop 4 } }')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_sky_repeat"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    txt = cbm.room(out, 0.0, "sky")
    cam = [l for l in txt.split(NL) if l.startswith('camera "c"')]
    assert len(cam) == 1
    scene = os.path.join(out, "room.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(txt.replace(cam[0], LENS_CAM))
    imgs = []
    for tag, extra in (("slow", []), ("fast", ["-interval", "0.05"])):
        png = os.path.join(out, "B_%s.png" % tag)
        for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", scene, "-mode", "B", "-n", "20000000", "-seed", "1", "-hdr", "-no-meter",
               "-window-min", "-o", png] + extra
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        imgs.append(read_pfm(png[:-4] + ".pfm", top_down=True))
    diff = imgs[0] != imgs[1]
    good = not diff.any()
    print("%s: lens camera, sky seen directly, written at two cadences: %d of %d pixels differ"
          % ("ok" if good else "FAIL", int(diff.any(-1).sum()), diff.shape[0] * diff.shape[1]))
    print("PASS" if good else "FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
