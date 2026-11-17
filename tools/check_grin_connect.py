"""Regression check: no straight connection crosses a GRIN region.

    D:/python314/python.exe -u tools/check_grin_connect.py [--out <dir>] [--exe ftrace.exe] [--secs 60]

scenes/_grin_lin.ftsl (a checkerboard behind a sphere whose index falls linearly with height, lit by two
sphere lamps above) in mode R and mode B on the GPU, `--secs` each. Outside the lens's own image -- the
16 border tiles of a 6 x 6 grid it does not reach -- both must agree within 3 %: there the checker is lit, partly through
the lens, and seen directly. Mode B's photons bend through the field; mode R's light samples are straight
and must count blocked where they would cross it, the bent light arriving along its walks at full weight
(GRIN-STRAIGHT-CONNECT). Before 0.449.0 mode R lit the corners the lens shadows straight through it:
0.81-0.87 of B there. Also requires the lens's middle four tiles black in mode B, which cannot see
through a bending region (a straight camera leg across one counts blocked). Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from pfm import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def render(exe, scene, png, mode, secs):
    for f in (png, png[:-4] + ".pfm"):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scene, "-mode", mode, "-device", "gpu", "-time", str(secs), "-hdr", "-no-meter", "-o", png]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    pfm = png[:-4] + ".pfm"
    if r.returncode != 0 or not os.path.exists(pfm):
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    if "-> GPU" not in r.stdout + r.stderr:
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_grin_connect"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--secs", type=int, default=60)
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = os.path.join(ROOT, "scenes", "_grin_lin.ftsl")
    img = {m: render(exe, scene, os.path.join(out, "lin_%s.png" % m), m, a.secs) for m in ("R", "B")}
    h, w = img["R"].shape
    t = {k: np.array([[v[h * i // 6:h * (i + 1) // 6, w * j // 6:w * (j + 1) // 6].mean() for j in range(6)]
                      for i in range(6)]) for k, v in img.items()}
    q = t["B"] / np.maximum(t["R"], 1e-30)
    border = np.ones((6, 6), bool)   # the tiles clear of the lens's image (its disc reaches the top
    border[1:5, 1:5] = False         # and bottom rows' middle two)
    border[0, 2:4] = border[5, 2:4] = False
    worst = float(np.max(np.abs(q[border] - 1.0)))
    middle = float(q[2:4, 2:4].max())
    ok = worst <= 0.03 and middle <= 0.01
    print(np.round(q, 3))
    print("%s: _grin_lin, mode B / mode R outside the lens: worst border tile off by %.2f %%; through the "
          "lens, B at most %.3f of R" % ("ok" if ok else "FAIL", 100.0 * worst, middle))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
