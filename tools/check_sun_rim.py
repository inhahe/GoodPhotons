#!/usr/bin/env python3
"""Regression gate: the sun's light subpaths cover the whole bounding sphere (scenes/_sun_rim.ftsl).

Modes D, J and U start a sun light subpath on a disc perpendicular to the beam, one scene radius
upstream of the scene centre. A ray tilted by the sun cone's half-angle reaches a point on the
rim of the bounding sphere from outside a disc of radius R, so the disc has to be wider
(Emitter::sunDiscR, scene.h). With a radius-R disc, light tracing could not produce paths to the
rim while every MIS weight still counted it, and with the sun aim on (bdpt.h SunAim) the origin
density read 0 there: the floor's corners, which lie on the bounding sphere, rendered 22 % dark in
mode D (0.391.2, found by the Opus review of 0.385-0.388).

Each arm's floor is compared patch by patch against mode R, which lights the floor by NEE alone
and never draws a sun origin. The four corner patches sit in the rim band; the fifth, mid-floor,
is the control: if it fails too, something other than the rim is wrong.

The second scene, scenes/_sun_haze.ftsl, is the other side of the same disc: a sun-lit unbounded
haze seen from outside the bounding sphere, where no sun subpath can start toward. Light tracing
cannot reach it, so the sun's NEE must not share its weight with it there (bdpt.h misWeight,
`sunMiss`): mode D read that haze at 0.215 of mode R before 0.391.5. Whole frame, mode D on both
backends against mode R.

Usage::

    python tools/check_sun_rim.py [--spp N] [--tol PCT] [--out DIR] [--arms D-gpu,U-cpu,...] [--exe EXE]

Exit status is non-zero if any patch of any arm misses mode R by more than the tolerance.
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
_SCENE = os.path.join(_REPO, "scenes", "_sun_rim.ftsl")
_HAZE = os.path.join(_REPO, "scenes", "_sun_haze.ftsl")

# name -> (mode, extra flags, spp scale). The CPU arms take fewer samples: they are slower, and
# the tolerance is far wider than their noise.
_ARMS = {
    "D-gpu":       ("D", ["-device", "gpu"], 1.0),
    "D-gpu-noaim": ("D", ["-device", "gpu", "-sunaim", "0"], 1.0),
    "D-cpu":       ("D", ["-device", "cpu"], 0.25),
    "U-gpu":       ("U", ["-device", "gpu"], 1.0),
    "U-cpu":       ("U", ["-device", "cpu"], 0.25),
}

_PATCH = 28   # pixels a side, at the scene's 320x320


def patch_mean(img, x0, y0, s):
    acc = 0.0
    for y in range(y0, y0 + s):
        for x in range(x0, x0 + s):
            acc += sum(img.at(x, y))
    return acc / (s * s)


def frame_mean(img):
    return sum(sum(img.at(x, y)) for y in range(img.h) for x in range(img.w)) / (img.w * img.h)


def patches(w, h):
    """Patch corners in image coordinates (top-down); the scene's frame is square."""
    s, m = _PATCH, 14   # inset from the frame edge: the floor starts ~8 px in
    return {
        "corner 1": (m, h - m - s), "corner 2": (w - m - s, h - m - s),
        "corner 3": (m, m), "corner 4": (w - m - s, m),
        "mid floor": (w // 2 - s // 2 + 80, h // 2 - s // 2 + 80),
    }


def render(exe, mode, flags, spp, out_png, log, scene=_SCENE):
    # One depth for every arm: modes D, J and U default to 8 edges and mode R to 32, and a
    # comparison across two caps measures the cap, not the light.
    cmd = [exe, "-in", scene, "-mode", mode, "-spp", str(spp), "-hdr",
           "-max-bounce", "16", "-o", out_png] + flags
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return Linear(os.path.splitext(out_png)[0] + ".pfm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--tol", type=float, default=3.0, help="per-patch tolerance against mode R, %%")
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_sun_rim"))
    ap.add_argument("--arms", default=",".join(_ARMS), help="comma-separated: " + ", ".join(_ARMS))
    ap.add_argument("--exe", default=_FTRACE, help="the binary under test (the reference too)")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)   # CreateProcess resolves a relative path with / poorly
    os.makedirs(a.out, exist_ok=True)

    ref = render(a.exe, "R", [], a.spp, os.path.join(a.out, "R.png"), os.path.join(a.out, "R.log"))
    where = patches(ref.w, ref.h)
    refm = {k: patch_mean(ref, x, y, _PATCH) for k, (x, y) in where.items()}

    bad = 0
    for name in a.arms.split(","):
        mode, flags, scale = _ARMS[name]
        spp = max(16, int(a.spp * scale))
        img = render(a.exe, mode, flags, spp, os.path.join(a.out, name + ".png"),
                     os.path.join(a.out, name + ".log"))
        cells = []
        for k, (x, y) in where.items():
            ratio = patch_mean(img, x, y, _PATCH) / refm[k]
            miss = abs(ratio - 1.0) * 100.0 > a.tol
            bad += miss
            cells.append("%s %.3f%s" % (k, ratio, " FAIL" if miss else ""))
        print("%-12s %s" % (name, "  ".join(cells)))
    # The haze: the frame's mean, every channel.
    href = render(a.exe, "R", [], a.spp * 4, os.path.join(a.out, "haze_R.png"),
                  os.path.join(a.out, "haze_R.log"), _HAZE)
    hm = frame_mean(href)
    for name, flags, spp in (("haze D-gpu", ["-device", "gpu"], a.spp),
                             ("haze D-cpu", ["-device", "cpu"], max(16, a.spp // 4))):
        img = render(a.exe, "D", flags, spp, os.path.join(a.out, name.replace(" ", "_") + ".png"),
                     os.path.join(a.out, name.replace(" ", "_") + ".log"), _HAZE)
        ratio = frame_mean(img) / hm
        miss = abs(ratio - 1.0) * 100.0 > a.tol
        bad += miss
        print("%-12s frame %.3f%s" % (name, ratio, " FAIL" if miss else ""))
    print("check_sun_rim: %s (tolerance %.1f %% against mode R)" % ("FAIL" if bad else "ok", a.tol))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
