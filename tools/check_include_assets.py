"""Regression check: an `include`d part finds the assets it names beside itself.

    python -u tools/check_include_assets.py [--out <dir>] [--exe ftrace.exe]

A scene includes `parts/part.ftsl`, a part kept in a directory of its own, which names a mesh
(`tri.obj`) and a measured reflectance (`spectrum ... = file:refl.csv`) relative to itself; neither
file exists beside the scene or in the working directory. The scene must load and render (a 16x16
mode-W frame on the CPU) with the part's triangle in view and lit by the CSV's reflectance -- its mean
above the background's. Assets resolve beside the file a block came from before the scene's directory
(assetbytes::blockDirRef, the loader's BlockDirScope). Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from pfm import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENE = [
    'scene { units meters }',
    'camera "c" { eye 0 0 3  look_at 0 0 0  up 0 1 0  fov_y 40  film { res 16 16 } }',
    'light area { origin -1 2 2  u 2 0 0  v 0 0 1  normal 0 -1 0  power 50  spd preset:d65 }',
    'include "parts/part.ftsl"',
]
PART = [
    'spectrum "refl" = file:refl.csv',
    'material "m" { type diffuse  reflect spectrum:refl }',
    'mesh { file "tri.obj"  material m }',
]
OBJ = ["v -1 -1 0", "v 1 -1 0", "v 0 1 0", "f 1 2 3"]
CSV = ["# wavelength_nm,value", "380,0.8", "780,0.8"]


def write(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_include_assets"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    write(os.path.join(out, "scene.ftsl"), SCENE)
    write(os.path.join(parts, "part.ftsl"), PART)
    write(os.path.join(parts, "tri.obj"), OBJ)
    write(os.path.join(parts, "refl.csv"), CSV)
    png = os.path.join(out, "frame.png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    # Run from the repository root, so neither file is in the working directory either.
    cmd = [exe, "-in", os.path.join(out, "scene.ftsl"), "-mode", "W", "-device", "cpu", "-spp", "1",
           "-hdr", "-no-meter", "-o", png]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and os.path.exists(pfm)
    if not ok:
        print(r.stdout[-1500:], r.stderr[-1500:])
        print("FAIL: the scene with an included part in its own directory did not render (exit %d)" % r.returncode)
    else:
        img = read_pfm(pfm, top_down=True).sum(-1)
        h, w = img.shape
        mid, corner = float(img[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2].mean()), float(img[0:3, 0:3].mean())
        ok = mid > 0.0 and mid > 4.0 * corner
        print("%s: the included part's mesh and reflectance load beside it (centre %.4g, corner %.4g)"
              % ("ok" if ok else "FAIL", mid, corner))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
