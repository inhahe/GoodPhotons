"""Regression check: `ftrace -topng` applies the denoiser's flags to a checkpoint, and refuses others.

    D:/python314/python.exe -u tools/check_topng_flags.py [--out <dir>] [--exe ftrace.exe]

Renders a small noisy frame with `-checkpoint`, then develops its `.ftbuf` with `-topng`:
  1. plain, and with `-denoise -denoise-luma 0.45` at the END of the line (a flag in the last
     position is the case a loop that wants a value after every flag skips). The luma filter is
     used because it changes any noisy image, where the default chroma-only filter may not; the
     two images must differ. Before 0.374.4 `-topng` ignored every denoise flag, so they matched.
  2. with a flag `-topng` does not take: it must exit 2 and write nothing. Before 0.374.4 it
     exited 0, having ignored the flag.
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_topng"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)          # CreateProcess wants a full path for a relative one
    d = os.path.abspath(a.out).replace(chr(92), "/")
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(d, "noisy.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join([
            'camera "c" { eye 0 0.6 4  look_at 0 0.4 0  up 0 1 0  fov_y 35  film { res 96 64 } }',
            'material "w" { type diffuse reflect 0.6 }',
            'material "red" { type diffuse reflect rgb 0.7 0.2 0.1 }',
            'light sun { dir 0.3 0.8 0.5  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
            'sphere { center 0 0.5 0  radius 0.5  material red }',
            'quad { corner -3 0 -3  u 6 0 0  v 0 0 6  material w }']) + NL)
    png = os.path.join(d, "noisy.png")
    ftbuf = png + ".ftbuf"
    for p in (png, ftbuf):
        if os.path.exists(p):
            os.remove(p)
    r = subprocess.run([a.exe, "-in", scene, "-mode", "D", "-device", "cpu", "-spp", "2", "-checkpoint",
                        "-window-min", "-o", png], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(ftbuf):
        print(r.stdout[-2000:], r.stderr[-2000:])
        print("FAIL: the render left no checkpoint")
        return 1

    def develop(name, extra):
        out = os.path.join(d, name)
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run([a.exe, "-topng", ftbuf, out] + extra, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        img = np.asarray(Image.open(out).convert("RGB")).astype(np.float64) if os.path.exists(out) else None
        return r.returncode, r.stdout + r.stderr, img

    ok = True
    rc0, _, plain = develop("plain.png", [])
    rc1, log1, filt = develop("denoised.png", ["-denoise", "-denoise-luma", "0.45"])
    if rc0 != 0 or rc1 != 0 or plain is None or filt is None:
        print("FAIL: -topng exited %d / %d" % (rc0, rc1)); print(log1); ok = False
    else:
        diff = float(np.abs(plain - filt).mean())
        good = diff > 0.05
        ok &= good
        print("%s: -denoise -denoise-luma 0.45 (last on the line) changes the developed image by %.3f levels"
              % ("ok" if good else "FAIL", diff))
    rc2, log2, img2 = develop("bogus.png", ["-not-a-flag"])
    good = rc2 == 2 and img2 is None
    ok &= good
    print("%s: an unknown flag exits %d%s" % ("ok" if good else "FAIL", rc2,
                                              "" if img2 is None else " and still writes the image"))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
