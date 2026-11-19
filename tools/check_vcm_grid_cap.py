"""Regression check: modes U and J render at a merge radius whose dense cell lattice would not fit.

    python -u tools/check_vcm_grid_cap.py [--out png/check_vcm_grid_cap] [--exe ftrace.exe]

Mode U bins its light vertices, and mode J its surface photons, in a lattice of radius-wide cells.
Dense over the points' box, at a fine radius over a large box that lattice asks for more cells than
memory holds -- 10 m at a micrometre is 10^21 -- and the host's cellStart.assign threw, so the
render failed; later both capped the cells (64 M) and grew them past it, every merge then weighing
more candidates. The lattice is hashed now (merge_grid.h), its table sized from the point count. A
box of 10 m (two walls 10 m apart, lit), rendered in mode U at `-pmradius 1e-6` and in mode J at
`-jsurf-radius 1e-6`, must finish with a finite image on the CPU and the GPU, its mean within 5 %
of the same frame at the scene's own radius (the merges at a micrometre add almost nothing, and
the rest is the same estimator's).

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

    def render(tag, device, mode, extra):
        png = os.path.join(d, tag + ".png")
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        p = subprocess.run([exe, "-in", path, "-mode", mode, "-device", device, "-spp", "8", "-seed", "1", "-hdr",
                            "-no-meter", "-window-min", "-o", png] + extra, cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        log = p.stdout + p.stderr
        on_gpu = "-> GPU" in log
        img = read_pfm(pfm, top_down=True) if p.returncode == 0 and os.path.exists(pfm) else None
        return p.returncode, img, on_gpu, log

    # mode J's light side held to one realization: its refresh epochs are paced by the work it counts,
    # which a finer radius changes
    for mode, flag in (("U", "-pmradius"), ("J", "-jsurf-radius")):
        for device in ("cpu", "gpu"):
            freeze = ["-beamfreeze"] if mode == "J" else []
            rc0, ref, g0, _ = render("own_%s_%s" % (mode, device), device, mode, freeze)
            rc1, fine, g1, log = render("fine_%s_%s" % (mode, device), device, mode, [flag, "1e-6"] + freeze)
            ran = rc1 == 0 and fine is not None and (device == "cpu" or g1)
            finite = ran and bool(np.isfinite(fine).all())
            m0 = float(ref.mean()) if ref is not None else float("nan")
            m1 = float(fine.mean()) if finite else float("nan")
            good = finite and rc0 == 0 and ref is not None and m0 > 0 and abs(m1 / m0 - 1.0) <= 0.05
            ok &= good
            print("%-4s mode %s on the %s at %s 1e-6 over a 10 m box: %s, mean %.4g of the own radius's"
                  % ("ok" if good else "FAIL", mode, device.upper(), flag,
                     ("exit %d" % rc1) if not ran else ("finite" if finite else "not finite"),
                     (m1 / m0) if (finite and m0 > 0) else float("nan")), flush=True)
            if not ran:
                print(log[-800:])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
