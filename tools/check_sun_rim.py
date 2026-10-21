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

Usage::

    python tools/check_sun_rim.py [--spp N] [--tol PCT] [--out DIR] [--arms D-gpu,U-cpu,...] [--exe EXE]

Exit status is non-zero if any patch of any arm misses mode R by more than the tolerance.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTRACE = os.path.join(_REPO, "ftrace.exe")
_SCENE = os.path.join(_REPO, "scenes", "_sun_rim.ftsl")

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


def read_pfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        if hdr not in (b"PF", b"Pf"):
            raise ValueError("not a PFM: %r" % hdr)
        nc = 3 if hdr == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        scale = float(f.readline().strip())
        n = w * h * nc
        data = struct.unpack("%s%df" % ("<" if scale < 0 else ">", n), f.read(4 * n))
    return w, h, nc, data


def patch_mean(img, x0, y0, s):
    w, _h, nc, d = img
    acc = 0.0
    for y in range(y0, y0 + s):
        for x in range(x0, x0 + s):
            i = (y * w + x) * nc
            acc += sum(d[i:i + nc])
    return acc / (s * s)


def patches(w):
    s, m = _PATCH, 14   # inset from the frame edge: the floor starts ~8 px in
    return {
        "corner 1": (m, m), "corner 2": (w - m - s, m),
        "corner 3": (m, w - m - s), "corner 4": (w - m - s, w - m - s),
        "mid floor": (w // 2 - s // 2 + 80, w // 2 - s // 2 - 80),
    }


def render(exe, mode, flags, spp, out_png, log):
    # One depth for every arm: modes D, J and U default to 8 edges and mode R to 32, and a
    # comparison across two caps measures the cap, not the light.
    cmd = [exe, "-in", _SCENE, "-camera", "top", "-mode", mode, "-spp", str(spp), "-hdr",
           "-max-bounce", "16", "-o", out_png] + flags
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return os.path.splitext(out_png)[0] + ".pfm"


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

    ref_pfm = render(a.exe, "R", [], a.spp, os.path.join(a.out, "R.png"), os.path.join(a.out, "R.log"))
    ref = read_pfm(ref_pfm)
    where = patches(ref[0])
    refm = {k: patch_mean(ref, x, y, _PATCH) for k, (x, y) in where.items()}

    bad = 0
    for name in a.arms.split(","):
        mode, flags, scale = _ARMS[name]
        spp = max(16, int(a.spp * scale))
        img = read_pfm(render(a.exe, mode, flags, spp, os.path.join(a.out, name + ".png"),
                              os.path.join(a.out, name + ".log")))
        cells = []
        for k, (x, y) in where.items():
            ratio = patch_mean(img, x, y, _PATCH) / refm[k]
            miss = abs(ratio - 1.0) * 100.0 > a.tol
            bad += miss
            cells.append("%s %.3f%s" % (k, ratio, " FAIL" if miss else ""))
        print("%-12s %s" % (name, "  ".join(cells)))
    print("check_sun_rim: %s (tolerance %.1f %% against mode R)" % ("FAIL" if bad else "ok", a.tol))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
