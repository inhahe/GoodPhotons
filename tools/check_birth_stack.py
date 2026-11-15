"""Regression check: a light walk born inside a dielectric starts inside it (BIRTH-EMPTY-STACK).

    python tools/check_birth_stack.py [--out png/check_birth_stack] [--exe ftrace.exe] [--no-gpu]

A walk keeps the dielectric solids it is inside on a stack, pushing one as it enters and popping it as it
leaves. A light walk born inside one -- a lamp in a coloured glass globe -- must start with it there
(Scene::birthStack), or it carries none of the glass's absorption from its birth to the wall, while every
camera walk that goes in after it does. `scenes/_birth_in_glass.ftsl`: two small lamps inside an amber
glass ball (absorbing 0.5 / 3 / 9 per metre in R / G / B) in a white room. Modes B, D, J, U, M and S,
CPU and GPU, against mode R on the GPU: the walls' mean per channel within 3 % and 5 standard errors --
of the channel, or of a tenth of the frame's total where the channel is smaller (amber light leaves
linear sRGB's blue near zero, even below it). A light walk born outside the glass reads the room
several times too bright in green and blue. The walls, because the ball's own pixels show the lamp
through the glass, which mode B's light walks cannot reach a pinhole by (a path from a light through
specular vertices only to the eye).
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

SCENE = os.path.join(ROOT, "scenes", "_birth_in_glass.ftsl")
REF = ["-mode", "R", "-device", "gpu", "-spp", "8192"]
CASES = [("B", ["-n", "100000000"]), ("D", ["-spp", "512"]), ("J", ["-spp", "256"]), ("U", ["-spp", "512"]),
         ("M", ["-spp", "64", "-n", "2000000"]), ("S", ["-spp", "256", "-n", "200000"])]
SEEDS = (1, 2, 3)
TOL = 0.03
# The ball's silhouette, centred in the 96 x 72 frame (the camera looks at its centre), 8.5 pixels in
# radius, and a margin.
_Y, _X = np.mgrid[0:72, 0:96]
WALLS = (_X - 47.5) ** 2 + (_Y - 35.5) ** 2 >= 12.0 ** 2


def render(exe, out, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", SCENE, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-max-bounce", "12",
           "-o", png] + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[WALLS].mean(0)


def stats(exe, out, tag, args):
    v = np.array([render(exe, out, tag, args, s) for s in SEEDS])
    return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(SEEDS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_birth_stack"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ref_args = REF if not a.no_gpu else ["-mode", "R", "-device", "cpu", "-spp", "4096"]
    rm, rse = stats(exe, out, "R_ref", ref_args)
    print("mode R: %s" % " ".join("%.5f" % v for v in rm), flush=True)
    ok = True
    for mode, args in CASES:
        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            m, se = stats(exe, out, "%s_%s" % (mode, dev), ["-mode", mode, "-device", dev] + args)
            scale = np.maximum(np.abs(rm), 0.1 * np.abs(rm).sum())   # see the docstring
            ratio = 1.0 + (m - rm) / scale
            err = np.hypot(se, rse) / scale
            bad = (np.abs(ratio - 1.0) > TOL) & (np.abs(ratio - 1.0) > 5.0 * err)
            ok = ok and not bad.any()
            print("%s: mode %s on the %s: %s of mode R (+- %s)"
                  % ("FAIL" if bad.any() else "ok", mode, "GPU" if dev == "gpu" else "CPU",
                     " ".join("%.4f" % v for v in ratio), " ".join("%.4f" % v for v in err)), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
