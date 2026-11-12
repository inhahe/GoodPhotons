"""Regression check: a rough interference coating interferes against the medium it sits in (ROUGH-IRID-AIR).

    python tools/check_irid_medium.py [--out png/check_irid_medium] [--exe ftrace.exe] [--no-gpu]

A coating's colour depends on the index of the medium the light meets it from. The smooth coating
(`thinfilm`, `multilayer`) takes it from the medium stack -- the enclosing medium's index where
both carry a `priority`, else air (exteriorIor) -- and since 0.439.6 the rough one does too: a hit
carries the medium its ray met it from (Hit::extMat, DHit::extMat), and iridescenceR reads it.
Before, a rough coating always interfered against air.

An anodized-titanium film (check_rough_irid's) on a sphere inside a larger sphere of water
(`dielectric ior 1.33`), both given a priority, under a uniform sky (whose reflection holds still,
so the near-mirror lobe is not a small light's high-variance image), seen from outside the
water. As the roughness goes to
0 the rough coating becomes the smooth one, so at roughness 0.003 the sphere must read the same
colour as at roughness 0 (the specular coating): each channel's mean over the sphere within 3 %,
in mode R on the CPU and on the GPU. The same in a sphere of air, as the control. Before the fix the
water case read the rough sphere's hue as the film's in air. Exits 1 on any failure.
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

W, H = 64, 64


def scene(rough, medium):
    film = ('material "ti" { type thinfilm  ior 2.5  substrate_k 3.0  film_ior 2.30  film_thickness 85'
            '  roughness %g  priority 2 }' % rough)
    box = ('material "water" { type dielectric  ior 1.33  priority 1 }' if medium == "water" else
           'material "water" { type dielectric  ior 1.0  priority 1 }')
    return NL.join([
        "scene { units meters  spectral 360 830 1 }",
        'material "floor" { type diffuse  reflect 0.5 }',
        film, box,
        "quad { origin -3 0 -3  u 6 0 0  v 0 0 6  material floor }",
        "sphere { center 0 0.5 0  radius 0.9  material water }",
        "sphere { center 0 0.5 0  radius 0.4  material ti }",
        "light env { spd preset:d65 }",
        'camera "c" { eye 0 1.2 3.2  look_at 0 0.5 0  up 0 1 0  fov_y 26  film { res %d %d } }' % (W, H),
    ]) + NL


def sphere_mask():
    """The pixels around the sphere's centre as the camera sees it through the water, well inside
    its rim (the water sphere magnifies it)."""
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H * 0.53
    return (xx - cx) ** 2 + (yy - cy) ** 2 < (W * 0.14) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_irid_medium"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    mask = sphere_mask()
    fails = []

    def mean(tag, rough, medium, dev):
        ftsl = os.path.join(d, tag + ".ftsl")
        with open(ftsl, "w", encoding="utf-8", newline="") as f:
            f.write(scene(rough, medium))
        png = os.path.join(d, tag + ".png")
        cmd = [exe, "-in", ftsl, "-mode", "R", "-device", dev, "-spp", "1024", "-seed", "1",
               "-hdr", "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        img = read_pfm(png[:-4] + ".pfm", top_down=True)
        return img[mask].mean(axis=0)

    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    for dev in devs:
        for medium in ("water", "air"):
            smooth = mean("%s_%s_smooth" % (dev, medium), 0.0, medium, dev)
            rough = mean("%s_%s_rough" % (dev, medium), 0.003, medium, dev)
            r = rough / smooth
            ok = bool(np.all(np.abs(r - 1.0) < 0.03))
            print("%-4s %-6s rough 0.003 / smooth, X Y Z: %.4f %.4f %.4f  %s"
                  % (dev, medium, r[0], r[1], r[2], "ok" if ok else "FAIL"))
            if not ok:
                fails.append("%s %s" % (dev, medium))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
