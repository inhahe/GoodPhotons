"""Regression check: a lens preset works for an ftrace.exe that cannot find the data directory.

    D:/python314/python.exe -u tools/check_lens_nodata.py [--exe ftrace.exe]

The lens presets are designed from their glasses' dispersion (an achromat's two powers come from
the Abbe numbers of BK7 and SF10). Those glasses are read from data/glass, beside the binary or
under the working directory -- and a copy of ftrace.exe run anywhere else used to fall back to
constant indices, which made the achromat's radii infinite and EVERY camera ray NaN: mode R
rendered such a camera black without a word (and, with a fire, sampled a grid at NaN points and
crashed). The presets now fall back to the built-in Sellmeier fits of the same glasses.

Copies the binary to a temporary directory with no data/ in reach, renders a lit sphere through
`lens { preset achromat }` in mode R, and requires a finite image that is not black. Exits 1 on
failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.5 -3  look_at 0 0.5 0  up 0 1 0  fov_y 35  film { res 48 32 }',
    '    lens { preset achromat  focal 50  fstop 2.8 } }',
    'material "m" { type diffuse  reflect 0.7 }',
    'sphere { center 0 0.5 0  radius 0.5  material m }',
    'light area { origin -1 2 -1  u 2 0 0  v 0 0 2  normal 0 -1 0  power 200 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix="ftrace_lens_nodata_")
    try:
        exe = os.path.join(d, "ftrace.exe")
        shutil.copy2(os.path.abspath(a.exe), exe)
        with open(os.path.join(d, "s.ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(SCENE) + NL)
        p = subprocess.run([exe, "-in", "s.ftsl", "-mode", "R", "-device", "cpu", "-spp", "16", "-hdr",
                            "-no-meter", "-o", "o.png"], cwd=d, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        pfm = os.path.join(d, "o.pfm")
        if p.returncode != 0 or not os.path.exists(pfm):
            print(p.stdout[-1500:], p.stderr[-1500:])
            print("FAILED: render failed (exit %d)" % p.returncode)
            return 1
        with open(pfm, "rb") as f:
            f.readline()
            f.readline()
            scale = float(f.readline())
            img = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
        finite = bool(np.isfinite(img).all())
        mean = float(img.mean()) if finite else float("nan")
        ok = finite and mean > 0.0
        print("%s: lens preset with no data/ in reach: image %s, mean %g" % (
            "ok" if ok else "FAIL", "finite" if finite else "NOT finite", mean))
        print("PASS" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
