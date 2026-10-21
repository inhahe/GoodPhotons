#!/usr/bin/env python3
"""Regression gate: a dielectric solid whose parts carry different `mode` tags is still one solid.

scenes/_route_solid.ftsl is a tinted glass cube of six quads, two faces tagged `mode J` (like their
material's own tag) and one `mode D`; scenes/_route_solid_ref.ftsl is the same cube untagged. Each
tag gives its object a routed copy of the material, and the medium stack must know a copy by its
original (Material::solidId), or a ray entering through a tagged face never leaves the glass and
everything behind the cube darkens. Mode R ignores the tags, so the two must render alike: the
check wants the frame means equal to 0.1 %. Before 0.391.5 the tagged cube read 0.919 / 0.972 /
0.999 of the untagged one.

Usage::

    python tools/check_route_solid.py [--out DIR] [--exe EXE] [--device gpu|cpu]

Exit status is non-zero if any channel differs by more than the tolerance.
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


def frame_means(path):
    img = Linear(path)
    acc = [0.0, 0.0, 0.0]
    for y in range(img.h):
        for x in range(img.w):
            for c, v in enumerate(img.at(x, y)):
                acc[c] += v
    return [a / (img.w * img.h) for a in acc]


def render(exe, scene, device, out_png, log):
    cmd = [exe, "-in", os.path.join(_REPO, "scenes", scene), "-mode", "R", "-spp", "512",
           "-device", device, "-hdr", "-o", out_png]
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return frame_means(os.path.splitext(out_png)[0] + ".pfm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_route_solid"))
    ap.add_argument("--exe", default=_FTRACE)
    ap.add_argument("--tol", type=float, default=0.1, help="per-channel tolerance, %%")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for device in ("gpu", "cpu"):
        ref = render(a.exe, "_route_solid_ref.ftsl", device, os.path.join(a.out, "ref_%s.png" % device),
                     os.path.join(a.out, "ref_%s.log" % device))
        tag = render(a.exe, "_route_solid.ftsl", device, os.path.join(a.out, "tagged_%s.png" % device),
                     os.path.join(a.out, "tagged_%s.log" % device))
        cells = []
        for c, (t, r) in enumerate(zip(tag, ref)):
            ratio = t / r
            miss = abs(ratio - 1.0) * 100.0 > a.tol
            bad += miss
            cells.append("%s %.4f%s" % ("RGB"[c], ratio, " FAIL" if miss else ""))
        print("%-4s tagged / untagged: %s" % (device, "  ".join(cells)), flush=True)
    print("check_route_solid: %s" % ("FAIL" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
