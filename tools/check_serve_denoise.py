"""Regression check: `-serve` renders every frame with the denoise settings it was started with.

    D:/python314/python.exe -u tools/check_serve_denoise.py [--out <dir>] [--exe ftrace.exe]

`-serve` re-runs the whole argument parse for every scene path it reads, and `-denoise <amount>`
MULTIPLIES the chroma tolerance, so before 0.374.6 each frame filtered harder than the last
(x4, x16, ...). Renders the same small scene twice in one `-serve` process with `-denoise 4`
(CPU, deterministic sampling) and requires the two frames to be identical. As a control, the same
pair without `-denoise` must be identical too, or the comparison would prove nothing. Exits 1 on
any failure.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def two_frames(exe, scene, out, extra):
    """Render `scene` twice in one -serve process; return both images as arrays."""
    if os.path.exists(out):
        os.remove(out)
    p = subprocess.Popen([exe, "-serve", "-in", scene, "-mode", "D", "-device", "cpu", "-spp", "4",
                          "-o", out] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    frames = []
    try:
        for k in range(2):
            for line in p.stdout:                    # wait for this frame's "[serve] done"
                if line.startswith("[serve] done"):
                    break
            else:
                raise RuntimeError("ftrace -serve ended before frame %d" % k)
            first = out + ".frame%d.png" % k
            shutil.copyfile(out, first)
            frames.append(np.asarray(Image.open(first).convert("RGB"), np.int16))
            if k == 0:
                p.stdin.write(scene + NL); p.stdin.flush()
        p.stdin.write("quit" + NL); p.stdin.flush()
        p.wait(timeout=120)
    finally:
        if p.poll() is None:
            subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, timeout=200)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_serve_denoise"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(d, "s.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join([
            'camera "c" { eye 0 0.6 4  look_at 0 0.4 0  up 0 1 0  fov_y 35  film { res 96 64 } }',
            'material "w" { type diffuse reflect 0.6 }',
            'material "red" { type diffuse reflect rgb 0.7 0.2 0.1 }',
            'light sun { dir 0.3 0.8 0.5  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
            'sphere { center 0 0.5 0  radius 0.5  material red }',
            'quad { corner -3 0 -3  u 6 0 0  v 0 0 6  material w }']) + NL)
    ok = True
    for label, extra in (("control, no -denoise", []), ("-denoise 4", ["-denoise", "4"])):
        f0, f1 = two_frames(a.exe, scene, os.path.join(d, "serve.png"), extra)
        diff = float(np.abs(f0 - f1).mean())
        good = diff == 0.0
        ok &= good
        print("%s: %s -> frame 2 differs from frame 1 by %.3f levels" % ("ok" if good else "FAIL", label, diff))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
