"""Regression check: mode U renders at a merge radius whose dense cell lattice would not fit.

    python -u tools/check_vcm_grid_cap.py [--out png/check_vcm_grid_cap] [--exe ftrace.exe]

Mode U bins its light vertices in a dense lattice of radius-wide cells over their box (vcm.h
VcmGrid, the device's DVcmGrid). At a fine radius over a large box that lattice asks for more cells
than memory holds -- 10 m at a micrometre is 10^21 -- and the host's cellStart.assign threw, so the
render failed. Both grids cap the cells (64 M) and grow them past it instead: a merge tests distance
against the radius, so a bigger cell only adds candidates. A box of 10 m (two walls 10 m apart,
lit), rendered in mode U at `-pmradius 1e-6`, must finish with a finite image on the CPU and the
GPU, its mean within 5 % of the same frame at the scene's own radius (the merges at a micrometre
add almost nothing, and the rest is the same estimator's).

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

SCENE = """scene { units meters }
material "w" { type diffuse reflect 0.7 }
quad { origin -5 0 -5  u 0 0 10  v 10 0 0  material w }
quad { origin -5 0 -5  u 10 0 0  v 0 10 0  material w }
quad { origin -5 0 5  u 0 10 0  v 10 0 0  material w }
sphere { center 0 1 0  radius 1  material w }
light area { origin -0.5 9.5 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  power 400  spd preset:d65 }
camera "c" { eye 0 3 4.5  look_at 0 1 0  up 0 1 0  fov_y 60  film { res 64 48 } }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vcm_grid_cap"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "box.ftsl")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    ok = True

    def render(tag, device, extra):
        png = os.path.join(d, tag + ".png")
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        p = subprocess.run([exe, "-in", path, "-mode", "U", "-device", device, "-spp", "8", "-seed", "1", "-hdr",
                            "-no-meter", "-window-min", "-o", png] + extra, cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        log = p.stdout + p.stderr
        on_gpu = "-> GPU" in log
        img = read_pfm(pfm, top_down=True) if p.returncode == 0 and os.path.exists(pfm) else None
        return p.returncode, img, on_gpu, log

    for device in ("cpu", "gpu"):
        rc0, ref, g0, _ = render("own_" + device, device, [])
        rc1, fine, g1, log = render("fine_" + device, device, ["-pmradius", "1e-6"])
        ran = rc1 == 0 and fine is not None and (device == "cpu" or g1)
        finite = ran and bool(np.isfinite(fine).all())
        m0 = float(ref.mean()) if ref is not None else float("nan")
        m1 = float(fine.mean()) if finite else float("nan")
        good = finite and rc0 == 0 and ref is not None and abs(m1 / m0 - 1.0) <= 0.05
        ok &= good
        print("%-4s mode U on the %s at -pmradius 1e-6 over a 10 m box: %s, mean %.4g of the own radius's"
              % ("ok" if good else "FAIL", device.upper(),
                 ("exit %d" % rc1) if not ran else ("finite" if finite else "not finite"),
                 (m1 / m0) if (finite and m0 > 0) else float("nan")), flush=True)
        if not ran:
            print(log[-800:])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
