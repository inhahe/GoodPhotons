"""Regression check: a pattern that steps exactly where a surface lies reads one side of the
step on every hit, on the CPU and the GPU alike.

    python tools/check_pattern_seam.py [--exe ftrace.exe] [--cpu-only]

A floor at y = 0 under a hand-written checker, fract(0.5 (floor(4x) + floor(4y) + floor(4z))) * 2,
lies on the checker's own cell walls: floor(4y) steps at y = 0. A hit point taken along the ray
(o + d t) rounds to either side from ray to ray; the GPU's did, so its pixels averaged the two
cells and the floor rendered grey. A triangle's hit point is now taken from its vertices
(geometry.h intersectTri and its device twin), which puts every hit at exactly y = 0. Two
floors: a quad, and the same square as a mesh_asset modelled at y = -1 and lifted to y = 0 by
mesh_instance (the instance hit path, whose world point now comes from the local one through
the instance's transform). Each renders from above on the CPU and the GPU; the check is the
floor's contrast -- the checker's two cells against each other, which a grey floor lacks --
and the CPU against the GPU tile by tile. Exits 0 when everything holds, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'scraps', '_pattern_seam')
RES = 64
CHECKER = 'fract(0.5 * (floor(4*x) + floor(4*y) + floor(4*z))) * 2'

HEAD = '''scene { units meters }
pattern "p" { expr "%s" }
material "pale"  { type diffuse reflect rgb 0.8 0.8 0.8 }
material "dark"  { type diffuse reflect rgb 0.05 0.05 0.05 }
material "floor" { type mix  layer "pale" 0.5  layer "dark" 0.5  weight_map pattern:p }
light area { origin -0.5 2.5 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:bb6500 }
camera "top" { eye 0 2.2 0.001  look_at 0 0 0  up 0 0 1  fov_y 50  film { res %d %d } }
''' % (CHECKER, RES, RES)


def read_pfm(p):
    with open(p, 'rb') as f:
        ch = 3 if f.readline().strip() == b'PF' else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.fromfile(f, '<f4' if scale < 0 else '>f4').reshape(h, w, ch)
    a = np.flipud(a).astype(np.float64)
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722 if ch == 3 else a[..., 0]


def scenes():
    os.makedirs(OUT, exist_ok=True)
    obj = os.path.join(OUT, 'floor_low.obj')
    with open(obj, 'w', encoding='utf-8', newline='\n') as f:
        f.write('v -1 -1 -1\nv 1 -1 -1\nv 1 -1 1\nv -1 -1 1\nf 1 2 3\nf 1 3 4\n')
    quad = HEAD + 'quad { origin -1 0 -1  u 2 0 0  v 0 0 2  material floor }\n'
    inst = HEAD + ('mesh_asset "low" { file "%s"  material floor }\n'
                   'mesh_instance { of "low"  translate 0 1 0 }\n') % obj.replace('\\', '/')
    paths = {}
    for name, text in (('quad', quad), ('instance', inst)):
        p = os.path.join(OUT, name + '.ftsl')
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        paths[name] = p
    return paths


def render(exe, scene, name, dev):
    out = os.path.join(OUT, f'{name}_{dev}.png')
    cmd = [exe, '-in', scene, '-hdr', '-no-meter', '-seed', '3', '-mode', 'R', '-spp', '64', '-device', dev,
           '-o', out]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    pfm = out[:-4] + '.pfm'
    if r.returncode != 0 or not os.path.exists(pfm):
        print(f'  render failed: {" ".join(cmd)}\n{r.stdout[-600:]}{r.stderr[-600:]}')
        return None
    return read_pfm(pfm)


def contrast(img):
    """The checker's two cells against each other: the 90th over the 10th percentile of the
    middle of the floor (its cells are 0.8 and 0.05; averaged, both read the same grey)."""
    m = img[RES // 4: 3 * RES // 4, RES // 4: 3 * RES // 4]
    lo, hi = np.percentile(m, 10), np.percentile(m, 90)
    return hi / max(lo, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--cpu-only', action='store_true')
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    devices = ['cpu'] if a.cpu_only else ['cpu', 'gpu']
    ok = True
    for name, scene in scenes().items():
        imgs = {}
        for dev in devices:
            img = render(exe, scene, name, dev)
            if img is None:
                ok = False
                continue
            imgs[dev] = img
            c = contrast(img)
            good = c > 4.0
            ok = ok and good
            print(f'  {name:9s} {dev.upper()}: floor contrast {c:6.2f} {"ok" if good else "FAIL (grey: hits on both sides of the step)"}')
        if 'cpu' in imgs and 'gpu' in imgs:
            q = []
            for i in range(4):
                for j in range(4):
                    sl = (slice(i * RES // 4, (i + 1) * RES // 4), slice(j * RES // 4, (j + 1) * RES // 4))
                    q.append(imgs['gpu'][sl].mean() / max(imgs['cpu'][sl].mean(), 1e-30))
            worst = max(abs(x - 1.0) for x in q)
            good = worst < 0.03
            ok = ok and good
            print(f'  {name:9s} GPU against CPU by tile: worst {worst * 100:.2f}% {"ok" if good else "FAIL"}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
