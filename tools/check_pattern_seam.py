"""Regression check: a pattern that steps exactly where a surface lies reads one side of the
step on every hit, on the CPU and the GPU alike.

    python tools/check_pattern_seam.py [--out png/check_pattern_seam] [--exe ftrace.exe] [--cpu-only]

A floor at y = 0 under a hand-written checker, fract(0.5 (floor(4x) + floor(4y) + floor(4z))) * 2,
lies on the checker's own cell walls: floor(4y) steps at y = 0. A hit point taken along the ray
(o + d t) rounds to either side from ray to ray, and the floor renders grey; a triangle's hit
point taken from its vertices (geometry.h intersectTri and its device twin) puts every hit at
exactly y = 0. Four floors, one square at y = 0: a quad; a mesh_asset modelled at y = -1 and
lifted by mesh_instance; one modelled upright in z = 2 at twice the size and stood on the floor
by a quarter turn about x, a half scale and a lift (the instance's transform applied the wrong
way round would put it at y = 2); and one modelled upright in z = 0 and only turned (a quarter
turn that is not exact, cos 90 = 6e-17, would leave its halves either side of y = 0: two
checker phases with a seam between). Each renders from above on the CPU and the GPU; the check
is the floor's contrast -- the checker's two cells against each other, which a grey floor
lacks -- and every image against the quad's on the CPU, tile by tile, which a seam or a
misplaced floor fails. Exits 0 when everything holds, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def write(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def scenes(out):
    os.makedirs(out, exist_ok=True)
    low = os.path.join(out, 'floor_low.obj').replace('\\', '/')
    upright = os.path.join(out, 'floor_upright.obj').replace('\\', '/')
    write(low, 'v -1 -1 -1\nv 1 -1 -1\nv 1 -1 1\nv -1 -1 1\nf 1 2 3\nf 1 3 4\n')
    write(upright, 'v -2 -2 2\nv 2 -2 2\nv 2 2 2\nv -2 2 2\nf 1 2 3\nf 1 3 4\n')
    upright0 = os.path.join(out, 'floor_upright0.obj').replace('\\', '/')
    write(upright0, 'v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3\nf 1 3 4\n')
    texts = {
        'quad': HEAD + 'quad { origin -1 0 -1  u 2 0 0  v 0 0 2  material floor }\n',
        'instance': HEAD + ('mesh_asset "low" { file "%s"  material floor }\n'
                            'mesh_instance { of "low"  translate 0 1 0 }\n') % low,
        # Rx(90) carries z = 1 (after the half scale) to y = -1; the move lifts it to y = 0.
        'turned': HEAD + ('mesh_asset "up" { file "%s"  material floor }\n'
                          'mesh_instance { of "up"  scale 0.5  rotate 90 0 0  translate 0 1 0 }\n') % upright,
        'upright': HEAD + ('mesh_asset "up0" { file "%s"  material floor }\n'
                           'mesh_instance { of "up0"  rotate 90 0 0 }\n') % upright0,
    }
    paths = {}
    for name, text in texts.items():
        paths[name] = os.path.join(out, name + '.ftsl')
        write(paths[name], text)
    return paths


def render(exe, out, scene, name, dev):
    png = os.path.join(out, f'{name}_{dev}.png')
    pfm = png[:-4] + '.pfm'
    for stale in (png, pfm):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [exe, '-in', scene, '-hdr', '-no-meter', '-seed', '3', '-mode', 'R', '-spp', '64', '-device', dev,
           '-o', png]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0 or not os.path.exists(pfm):
        print(f'  render failed: {" ".join(cmd)}\n{r.stdout[-600:]}{r.stderr[-600:]}')
        return None
    if dev == 'gpu' and '-> GPU' not in r.stdout + r.stderr:
        print(f'  did not run on the GPU: {" ".join(cmd)}')
        return None
    return read_pfm(pfm)


def contrast(img):
    """The checker's two cells against each other: the 90th over the 10th percentile of the
    middle of the floor (its cells are 0.8 and 0.05; averaged, both read the same grey)."""
    m = img[RES // 4: 3 * RES // 4, RES // 4: 3 * RES // 4]
    lo, hi = np.percentile(m, 10), np.percentile(m, 90)
    return hi / max(lo, 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'png', 'check_pattern_seam'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--cpu-only', action='store_true')
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    devices = ['cpu'] if a.cpu_only else ['cpu', 'gpu']
    ok = True
    ref = None                                           # the quad on the CPU
    for name, scene in scenes(a.out).items():
        imgs = {}
        for dev in devices:
            img = render(exe, a.out, scene, name, dev)
            if img is None:
                ok = False
                continue
            imgs[dev] = img
            c = contrast(img)
            good = c > 4.0
            ok = ok and good
            print(f'  {name:9s} {dev.upper()}: floor contrast {c:6.2f} {"ok" if good else "FAIL (grey: hits on both sides of the step)"}')
        if ref is None and 'cpu' in imgs:
            ref = imgs['cpu']
        for dev, img in imgs.items():
            if ref is None or (name == 'quad' and dev == 'cpu'):
                continue
            # Pixel by pixel: a cell of the other colour reads 16x off (0.8 against 0.05), noise
            # at 64 spp a few tens of per cent, so a seam -- half the floor in the other phase --
            # or a misplaced floor shows as many pixels 3x off. Tile means would not: a tile
            # spans whole pairs of cells, whose mean a swap of their colours leaves alone.
            ratio = (img + 1e-30) / (ref + 1e-30)
            off = float(np.mean((ratio > 3.0) | (ratio < 1.0 / 3.0)))
            good = off < 0.02
            ok = ok and good
            print(f'  {name:9s} {dev.upper()} against the quad on the CPU: {off * 100:.2f}% of pixels 3x off '
                  f'{"ok" if good else "FAIL (a seam, or the floor misplaced)"}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
