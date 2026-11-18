"""Regression check: an `include`d part finds the assets it names beside itself.

    python -u tools/check_include_assets.py [--out <dir>] [--exe ftrace.exe]

A scene includes two parts, each kept in a directory of its own and naming a mesh (`tri.obj`) and a
measured reflectance (`file:refl.csv`) relative to itself -- the same names, different files: the
left part's reflects 0.8, the right part's 0.2. The left part's mesh sits inside a `prefer` branch.
Neither file exists beside the scene or in the working directory. The scene must load and render (a
32x16 mode-W frame on the CPU) with each part's triangle in view, lit by its own CSV: the left half at
least 2.5 times as bright as the right. Assets resolve beside the file a block came from (Block::dir,
BlockDirScope), and the loader's spectrum caches key a file by where it resolves. Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

from pfm import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENE = [
    'scene { units meters }',
    'camera "c" { eye 0 0 4  look_at 0 0 0  up 0 1 0  fov_y 40  film { res 32 16 } }',
    'light area { origin -2 2 2  u 4 0 0  v 0 0 1  normal 0 -1 0  power 80  spd preset:d65 }',
    'include "left/part.ftsl"',
    'include "right/part.ftsl"',
]
LEFT = [
    'material "ml" { type diffuse  reflect file:refl.csv }',
    'prefer { mesh { file "tri.obj"  material ml } } else { mesh { file "tri.obj"  material ml } }',
]
RIGHT = [
    'material "mr" { type diffuse  reflect file:refl.csv }',
    'mesh { file "tri.obj"  material mr }',
]


def obj(cx):
    return ["v %g -0.6 0" % (cx - 0.6), "v %g -0.6 0" % (cx + 0.6), "v %g 0.6 0" % cx, "f 1 2 3"]


def csv(v):
    return ["# wavelength_nm,value", "380,%g" % v, "780,%g" % v]


def write(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_include_assets"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    for side, part, cx, refl in (("left", LEFT, -0.9, 0.8), ("right", RIGHT, 0.9, 0.2)):
        d = os.path.join(out, side)
        os.makedirs(d, exist_ok=True)
        write(os.path.join(d, "part.ftsl"), part)
        write(os.path.join(d, "tri.obj"), obj(cx))
        write(os.path.join(d, "refl.csv"), csv(refl))
    write(os.path.join(out, "scene.ftsl"), SCENE)
    png = os.path.join(out, "frame.png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    # Run from the repository root, so neither part's files are in the working directory either.
    cmd = [exe, "-in", os.path.join(out, "scene.ftsl"), "-mode", "W", "-device", "cpu", "-spp", "1",
           "-hdr", "-no-meter", "-o", png]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and os.path.exists(pfm)
    if not ok:
        print(r.stdout[-1500:], r.stderr[-1500:])
        print("FAIL: the scene with included parts in directories of their own did not render (exit %d)"
              % r.returncode)
    else:
        img = read_pfm(pfm, top_down=True).sum(-1)
        h, w = img.shape
        left, right = float(img[:, :w // 2].mean()), float(img[:, w // 2:].mean())
        ok = right > 0.0 and left > 2.5 * right
        print("%s: each part's mesh and its own reflectance load beside it (left %.4g, right %.4g, x%.2f)"
              % ("ok" if ok else "FAIL", left, right, left / right if right > 0 else float("inf")))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
