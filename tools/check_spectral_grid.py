#!/usr/bin/env python3
"""Regression gate: the GPU reads a tabulated spectrum as the host does (DEVICE-SPECTRAL-GRID).

The device kept every material, medium and emitter spectrum as a 96-entry grid 4.9 nm apart, so a
feature narrower than that came out differently on the two backends: a glass whose absorption has
3-5 nm windows (scenes/_glass_claim.ftsl) read red 1.25e10 on the GPU against 4.40e10 on the CPU
before 0.393.0, its red-channel being a small difference of large X, Y, Z terms. A `table` (or
measured-file) spectrum now reaches the device as its own knots (spectrum.h spectrumKnots, render_cuda
DSpec). This renders each scene in mode R on both backends and compares the frame means channel by
channel:

* scenes/_glass_claim.ftsl -- the windows in a material's absorption (the fog's box);
* scenes/_spectral_bands.ftsl -- fifteen 2 nm bands in a surface's reflectance, each between two
  points of the old grid (the whole frame).

The glass's red is noisy on both backends (it comes from the few paths that reach the light
through the glass: seed-to-seed spread +-4 % on the CPU at 1024 spp), hence the CPU's 2048 spp
and the 10 % tolerance -- far inside the error they catch (red 0.28 before the fix).

Usage::

    python tools/check_spectral_grid.py [--out DIR] [--exe EXE] [--spp-gpu N] [--spp-cpu N] [--tol PCT]

Exit status is non-zero if a channel differs by more than the tolerance.
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

# scene -> box (x0, y0, x1, y1) in image order, or None for the whole frame
_SCENES = {
    "_glass_claim.ftsl": (70, 50, 130, 110),
    "_spectral_bands.ftsl": None,
}


def box_means(path, box):
    img = Linear(path)
    x0, y0, x1, y1 = box if box else (0, 0, img.w, img.h)
    acc = [0.0, 0.0, 0.0]
    for y in range(y0, y1):
        for x in range(x0, x1):
            for c, v in enumerate(img.at(x, y)):
                acc[c] += v
    n = (x1 - x0) * (y1 - y0)
    return [a / n for a in acc]


def render(exe, scene, device, spp, out_png, log):
    cmd = [exe, "-in", os.path.join(_REPO, "scenes", scene), "-mode", "R", "-device", device,
           "-spp", str(spp), "-max-bounce", "32", "-hdr", "-o", out_png]
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=_REPO)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    # -device gpu still runs on the CPU when there is no device or the scene is outside the
    # GPU's scope, and then the gate would compare the CPU with itself.
    if device == "gpu":
        with open(log, encoding="utf-8", errors="replace") as lf:
            if "-> GPU" not in lf.read():
                raise RuntimeError("the GPU arm did not run on the GPU (log: %s)" % log)
    return os.path.splitext(out_png)[0] + ".pfm"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_spectral_grid"))
    ap.add_argument("--exe", default=_FTRACE)
    ap.add_argument("--spp-gpu", type=int, default=8192)
    ap.add_argument("--spp-cpu", type=int, default=2048)
    ap.add_argument("--tol", type=float, default=10.0, help="per-channel tolerance, %%")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for scene, box in _SCENES.items():
        tag = os.path.splitext(scene)[0].lstrip("_")
        out = lambda dev: os.path.join(a.out, "%s_%s" % (tag, dev))
        gpu = box_means(render(a.exe, scene, "gpu", a.spp_gpu, out("gpu") + ".png", out("gpu") + ".log"), box)
        cpu = box_means(render(a.exe, scene, "cpu", a.spp_cpu, out("cpu") + ".png", out("cpu") + ".log"), box)
        cells = []
        big = max(abs(v) for v in cpu)
        for c, (g, h) in enumerate(zip(gpu, cpu)):
            # A channel near zero (the bands' blue: amber lies outside sRGB) makes a ratio of noise.
            if abs(h) < 0.05 * big:
                cells.append("%s n/a" % "RGB"[c])
                continue
            ratio = g / h
            miss = abs(ratio - 1.0) * 100.0 > a.tol
            bad += miss
            cells.append("%s %.3f%s" % ("RGB"[c], ratio, " FAIL" if miss else ""))
        print("%-16s GPU / CPU: %s" % (tag, "  ".join(cells)), flush=True)
    print("check_spectral_grid: %s" % ("FAIL" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
