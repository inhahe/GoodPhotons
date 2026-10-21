#!/usr/bin/env python3
"""Regression gate: mode J's merges in a fog beside, behind and inside coloured glass, GPU and CPU.

Three scenes (scenes/_jfog_floor, _jfog_behind, _jfog_inside; their headers say what each guards),
each rendered in mode R, the reference, and in mode J on the GPU and on the CPU. The fog's region
of the frame is compared channel by channel. Every arm runs at -max-bounce 32: modes D, J and U
default to 8 edges and mode R to 32, and a comparison across two caps measures the cap.

What failed before 0.391.3 (fog region, J / R):
  floor   GPU 1.043 (the device's beam-merge weight lacked the point merge at the vertex a beam
          left, J-GPU-ETAPREVS), CPU 0.998
  behind  G +7-8 % on both (JSPEC's carrier read by straight lines between its bins)
  inside  R 0.73, G 1.24 on both (the merge's partial camera segment took the glass's absorption
          at the camera's wavelength)
and what 0.391.3 reads: floor 0.999 / 0.998, behind R 0.99-1.00 G 1.025-1.036, inside R 0.98-0.99
G 1.00. Blue is left out of the tinted scenes: the glass passes about 1 % of it, so its ratio is
noise. The behind scene's green keeps a wider tolerance for the bins' resolution (known-issues
J-GPU-ETAPREVS, the note on the carrier).

Usage::

    python tools/check_jfog_glass.py [--out DIR] [--exe EXE] [--scenes floor,behind,inside]

Exit status is non-zero if any channel of any arm misses mode R by more than its tolerance.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTRACE = os.path.join(_REPO, "ftrace.exe")
_BOX = (90, 60, 150, 140)          # the fog, in PFM rows (bottom to top), at the scenes' 240x200
_DEPTH = ["-max-bounce", "32"]

# scene -> per-channel tolerances in % (None: channel not checked)
_SCENES = {
    "floor":  (2.0, 2.0, 2.0),
    "behind": (3.0, 5.0, None),
    "inside": (3.0, 3.0, None),
}
_ARMS = {
    "J-gpu": ["-mode", "J", "-device", "gpu", "-spp", "512", "-beamepochs", "32"],
    "J-cpu": ["-mode", "J", "-device", "cpu", "-spp", "128", "-beamepochs", "16"],
}


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


def box_means(img):
    w, _h, nc, d = img
    x0, y0, x1, y1 = _BOX
    acc = [0.0] * nc
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * w + x) * nc
            for c in range(nc):
                acc[c] += d[i + c]
    n = (x1 - x0) * (y1 - y0)
    return [a / n for a in acc]


def render(exe, scene, flags, out_png, log):
    cmd = [exe, "-in", scene, "-hdr", "-o", out_png] + _DEPTH + flags
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (" ".join(cmd), r.returncode, log))
    return box_means(read_pfm(os.path.splitext(out_png)[0] + ".pfm"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(_REPO, "png", "check_jfog_glass"))
    ap.add_argument("--exe", default=_FTRACE)
    ap.add_argument("--scenes", default=",".join(_SCENES))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    os.makedirs(a.out, exist_ok=True)

    bad = 0
    for name in a.scenes.split(","):
        tol = _SCENES[name]
        scene = os.path.join(_REPO, "scenes", "_jfog_%s.ftsl" % name)
        stem = os.path.join(a.out, name)
        ref = render(a.exe, scene, ["-mode", "R", "-spp", "16384"], stem + "_R.png", stem + "_R.log")
        for arm, flags in _ARMS.items():
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
