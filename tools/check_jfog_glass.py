#!/usr/bin/env python3
"""Regression gate: modes J's and M's beam merges in a fog beside, behind and inside coloured glass.

Six scenes (scenes/_jfog_floor, _jfog_behind, _jfog_inside, _jfog_gel, _jfog_gelline, _jfog_gelkink; their headers
say what each guards), each rendered in mode R, the reference, and in modes J and M (`-beams`) on the GPU and on
the CPU. The fog's region of the frame is compared channel by channel. Every arm runs at
-max-bounce 32: modes D, J and U default to 8 edges and mode R to 32, and a comparison across two
caps measures the cap.

What failed before 0.391.4 (fog region, J / R):
  floor   GPU 1.043 (the device's beam-merge weight lacked the point merge at the vertex a beam
          left, J-GPU-ETAPREVS), CPU 0.998
  behind  G +7-8 % on both (JSPEC's carrier read by straight lines between its bins)
  inside  R 0.73, G 1.24 on both (the merge's partial camera segment took the glass's absorption
          at the camera's wavelength)
and what 0.391.4 reads: floor 0.999 / 0.998, behind R 0.99-1.00 G 1.025-1.036, inside R 0.98-0.99
G 1.00.

Mode M before 0.433.5, whose beam gather read its camera walk's tints and the glass around the
segment at the camera's wavelength (MODEM-BEAM-CARRIER), GPU / CPU: floor 0.997 / 1.012; behind
R 0.54 / 0.55, G 2.63 / 2.69; inside R 0.66 / 0.68, G 1.28 / 1.31. And 0.433.5: behind R 0.985 /
0.998, G 1.017 / 1.037; inside R 1.000 / 1.018, G 0.975 / 0.995.

The gel scene sees the fog through a filter that passes exactly nothing below 497 nm. Until 0.433.11
mode J read it at R 0.640, G 0.650 of mode R on both backends: a camera walk drawn where the gel is
opaque dies there, and the merges from the walks that pass did not stand in for it (known-issues
JSPEC-GAPS, the support). 0.433.11 reads R 0.987, G 1.003-1.004 in mode J, and mode M, which has
had the same rule since 0.381.4, 0.988-1.001 and 1.001-1.025.

The gelline scene sees the fog through a gel stepping on a boundary of the carrier's bins, lit by a
line at 610 nm in the bin above it. Mode M's beam gather carried that bin's read to its centre along
the line to the bin below, across the step (REVIEW-0458): 0.451.15 read the fog R 0.744 on the GPU and
0.755 on the CPU. Only red is checked there: a 610 nm line's linear-sRGB green and blue are small
negative shares of its red, which carry nothing red does not. The gelkink scene sees it through a gel
ramping up from that boundary instead, a bend between two bins' centres, which the uniform bins read
15 % low in modes J and M alike (JSPEC-BIN-RESOLUTION: 0.451.26 read the fog R 0.843 / 0.841 in mode
J, GPU / CPU, and 0.852 / 0.865 in mode M); the carriers on the scene's cells read 1.000 / 0.998 and
1.000 / 1.015 (0.451.27). Red alone again.

Blue is left out of the tinted scenes: the glass passes about 1 % of it, so its ratio is
noise. The behind scene's green keeps a wider tolerance for the bins' resolution (known-issues
J-GPU-ETAPREVS, the note on the carrier).

Usage::

    python tools/check_jfog_glass.py [--out DIR] [--exe EXE] [--scenes floor,behind,inside,gel,gelline,gelkink]
                                     [--arms J-gpu,J-cpu,M-gpu,M-cpu]

Exit status is non-zero if any channel of any arm misses mode R by more than its tolerance.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pfm import Linear  # noqa: E402  (scene-linear RGB, image order)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTRACE = os.path.join(_REPO, "ftrace.exe")
_BOX = (90, 60, 150, 140)          # the fog, x0 y0 x1 y1 in image order, at the scenes' 240x200
_DEPTH = ["-max-bounce", "32"]

# scene -> per-channel tolerances in % (None: channel not checked)
_SCENES = {
    "floor":  (2.0, 2.0, 2.0),
    "behind": (3.0, 5.0, None),
    "inside": (3.0, 3.0, None),
    "gel":    (3.0, 3.0, None),
    "gelline": (3.0, None, None),
    "gelkink": (3.0, None, None),
}
# Mode J at -jbeams auto: the knee-sized map these tolerances were set against; the sparse fixed
# default (0.435.0) gives the merges under test a far smaller share.
_ARMS = {
    "J-gpu": ["-mode", "J", "-device", "gpu", "-spp", "512", "-beamepochs", "32", "-jbeams", "auto"],
    "J-cpu": ["-mode", "J", "-device", "cpu", "-spp", "128", "-beamepochs", "16", "-jbeams", "auto"],
    "M-gpu": ["-mode", "M", "-beams", "-device", "gpu", "-n", "20000000", "-spp", "64"],
    "M-cpu": ["-mode", "M", "-beams", "-device", "cpu", "-n", "4000000", "-spp", "8"],
}


def box_means(img):
    x0, y0, x1, y1 = _BOX
    acc = [0.0, 0.0, 0.0]
    for y in range(y0, y1):
        for x in range(x0, x1):
            for c, v in enumerate(img.at(x, y)):
                acc[c] += v
    n = (x1 - x0) * (y1 - y0)
    return [a / n for a in acc]


def render(exe, scene, flags, out_png, log):
    cmd = [exe, "-in", scene, "-hdr", "-o", out_png] + _DEPTH + flags
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return box_means(Linear(os.path.splitext(out_png)[0] + ".pfm"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_jfog_glass"))
    ap.add_argument("--exe", default=_FTRACE)
    ap.add_argument("--scenes", default=",".join(_SCENES))
    ap.add_argument("--arms", default=",".join(_ARMS))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    os.makedirs(a.out, exist_ok=True)

    bad = 0
    for name in a.scenes.split(","):
        tol = _SCENES[name]
        scene = os.path.join(_REPO, "scenes", "_jfog_%s.ftsl" % name)
        stem = os.path.join(a.out, name)
        ref = render(a.exe, scene, ["-mode", "R", "-spp", "16384"], stem + "_R.png", stem + "_R.log")
        for arm in a.arms.split(","):
            flags = _ARMS[arm]
            m = render(a.exe, scene, flags, "%s_%s.png" % (stem, arm), "%s_%s.log" % (stem, arm))
            cells = []
            for c, (v, r, t) in enumerate(zip(m, ref, tol)):
                ratio = v / r if r else float("nan")
                miss = t is not None and not (abs(ratio - 1.0) * 100.0 <= t)
                bad += miss
                cells.append("%s %.3f%s" % ("RGB"[c], ratio, " FAIL" if miss else ("" if t else " (unchecked)")))
            print("%-7s %-6s %s" % (name, arm, "  ".join(cells)), flush=True)
    print("check_jfog_glass: %s" % ("FAIL" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
