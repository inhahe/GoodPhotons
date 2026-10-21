#!/usr/bin/env python3
"""Regression gate: a light walk's spectral claims do not cross glass that absorbs only between
the carriers' probe wavelengths (scenes/_glass_claim.ftsl).

The scene's glass absorbs the red half of the band except at the exact wavelengths the 24-bin
carriers probe, so a probe reads it as clear. A mode-M photon carries two claims that the
wavelength does not matter: a bundle of secondary wavelengths riding the hero's chord
(`-beamspec`, on by default) and the achromatic-path fold (`-beamachro`, on by default). A photon
whose hero wavelength falls in a clear part crosses this glass unabsorbed, and with its claims
alive it carries the red wavelengths the glass removes on to the fog. Material::absorbChroma (a
1 nm scan, 0.391.6) retires both in such glass. So mode M with its claims must render the fog the
colour it has with neither (`-beamachro off -beamspec 1`, one wavelength per photon); the check
compares the two over the fog, channel by channel. Before 0.391.6 the claims painted the fog white:
R/G/B 1.21 / 1.13 / 1.18 (x1e11) against 0.42 / 1.73 / 1.79 without them.

CPU only: the device's mode-M beam map carries no claim (its log says "achromatic fold n/a"), so
a GPU arm would compare an estimator with itself.

Usage::

    python tools/check_glass_claim.py [--out DIR] [--exe EXE] [--photons N] [--spp N]

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
_SCENE = os.path.join(_REPO, "scenes", "_glass_claim.ftsl")
_BOX = (70, 50, 130, 110)          # the fog, x0 y0 x1 y1 in image order, at the scene's 200x160


def box_means(path):
    img = Linear(path)
    x0, y0, x1, y1 = _BOX
    acc = [0.0, 0.0, 0.0]
    for y in range(y0, y1):
        for x in range(x0, x1):
            for c, v in enumerate(img.at(x, y)):
                acc[c] += v
    n = (x1 - x0) * (y1 - y0)
    return [a / n for a in acc]


def render(exe, flags, photons, spp, out_png, log):
    cmd = [exe, "-in", _SCENE, "-mode", "M", "-beams", "-n", str(photons), "-spp", str(spp),
           "-max-bounce", "32", "-device", "cpu", "-hdr", "-o", out_png] + flags
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=_REPO)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return box_means(os.path.splitext(out_png)[0] + ".pfm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_glass_claim"))
    ap.add_argument("--exe", default=_FTRACE)
    ap.add_argument("--photons", type=int, default=30000000)
    ap.add_argument("--spp", type=int, default=64, help="camera samples (the gather is the cost)")
    ap.add_argument("--tol", type=float, default=3.0, help="per-channel tolerance, %%")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    os.makedirs(a.out, exist_ok=True)
    out = lambda name: os.path.join(a.out, name)
    plain = render(a.exe, ["-beamachro", "off", "-beamspec", "1"], a.photons, a.spp,
                   out("plain.png"), out("plain.log"))
    fold = render(a.exe, [], a.photons, a.spp, out("fold.png"), out("fold.log"))
    bad = 0
    cells = []
    for c, (f, p) in enumerate(zip(fold, plain)):
        ratio = f / p
        miss = abs(ratio - 1.0) * 100.0 > a.tol
        bad += miss
        cells.append("%s %.3f%s" % ("RGB"[c], ratio, " FAIL" if miss else ""))
    print("mode M with its claims / without: %s" % "  ".join(cells), flush=True)
    print("check_glass_claim: %s" % ("FAIL" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
