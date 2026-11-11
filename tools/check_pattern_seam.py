"""Regression check: a pattern that steps exactly where a surface lies reads one side of the
step on every hit, on the CPU and the GPU alike.

    python tools/check_pattern_seam.py [--out png/check_pattern_seam] [--exe ftrace.exe] [--cpu-only]

A floor at y = 0 under a hand-written checker, fract(0.5 (floor(4x) + floor(4y) + floor(4z))) * 2,
lies on the checker's own cell walls: floor(4y) steps at y = 0. A pattern read at a hit point
that rounds to either side of the step from ray to ray renders grey, or splits into two phases
along a seam. Patterns read a hit a hair outside the surface, along its normal (pattern.h
patNudgedPoint), hit points are taken from the surface's own coordinates rather than along the
ray, and quarter turns are exact; this renders the cases those cover.

Floors seen from 2.2 m, all at y = 0 but one: a quad (facing down, as u x v has it); a
mesh_asset modelled at y = -1 and lifted by mesh_instance; one modelled upright in z = 2 at
twice the size and stood on the floor by a quarter turn about x, a half scale and a lift (the
instance's transform applied the wrong way round would put it at y = 2); one modelled upright
in z = 0 and only turned; one whose vertices carry float noise, y = +-1e-12; the quad turned to
face up; and a 100 m floor facing up whose vertices all sit 1e-6 below 0 -- an export's float
noise at that size, which a hair sized by the hit's own distance does not clear within a metre
of the origin (it is sized by the triangle's reach). Each must show the checker's contrast (a
grey floor lacks it) and match its quad -- facing the same way -- on the CPU pixel by pixel (a
seam, speckle or a misplaced surface does not). The rasterizer's preview (-raster) of the noisy
floor is held to the quad's the same way, on each device (which must really be the GPU's
rasterizer, not a silent fall back to the CPU's), and the quad's preview must show the cells the
CPU trace shows: the floor lies on a cell wall, and a preview reading its other side shows the
other phase.

And surfaces of radius 0.5 under a step at that radius, hit along the ray: a sphere seen from
2.2 m, and a sphere, an isosurface sphere and a curve's tube seen from about 3000 m and 30 km
off, along a direction with no symmetry, where a float hit point computed along the ray rounds
relative to the camera's distance, far past the nudge, so the GPU must take these hits from the
surfaces themselves. They glow, seen directly,
against the same surface with the pattern replaced by 1 and the same seed: a pixel's ratio of
the two is exactly the fraction of its hits that read the outside of the step, which must be
all of them. (A mix of two colours would not do: a pixel reads 3x off only once most of its
hits are wrong.)

And which side a pattern reads (FTSL 6.1), in the backward and bidirectional tracers and the
preview on each device: under step(0, ny), a floor facing up and over its middle a square
facing down, seen from above. An opaque pair reads the side the eye sees, so its two windings
show the same cell; a pair light can cross reads its own side's, so the square reads ny = -1
-- from behind -- and shows the other cell: a mix of translucents, a mix with one translucent
layer, and a mix whose opaque layer carries the pattern itself (every pattern on a surface
follows its material's rule, whichever layer a renderer picked there).

Exits 0 when everything holds, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from pfm import read_pfm, luma  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = 64
CHECKER = 'fract(0.5 * (floor(4*x) + floor(4*y) + floor(4*z))) * 2'
RADIAL = 'step(0.5, r)'
TUBE = 'step(0.5, sqrt(y*y + z*z))'
NEAR_CAM = 'eye 0 2.2 0.001  look_at 0 0 0  up 0 0 1  fov_y 50'
# About 3000 m and 30 km off along a direction with no symmetry, the field narrowed to the
# same 1.2 m. Not straight above, where fl(dy t) is exactly -t and the march's points land on
# the surface's own; nor along the diagonal, where the three coordinates round alike and the
# error lies along the ray, where it does no harm.
FAR_CAM = 'eye 2400.37 1300.91 1151.23  look_at 0 0 0  up 0 1 0  fov_y 0.023'
FAR30K_CAM = 'eye 24003.7 13009.1 11512.3  look_at 0 0 0  up 0 1 0  fov_y 0.0023'


def head(expr=CHECKER, cam=NEAR_CAM):
    # The lamp hangs over the floor; for the far cameras, whose line of sight it would block,
    # well out of frame (the far surfaces glow by themselves; the loader wants a light).
    lamp = 'origin -0.5 2.5 -0.5' if cam == NEAR_CAM else 'origin 30 -40 10'
    return '''scene { units meters }
pattern "p" { expr "%s" }
material "pale"  { type diffuse reflect rgb 0.8 0.8 0.8 }
material "dark"  { type diffuse reflect rgb 0.05 0.05 0.05 }
material "floor" { type mix  layer "pale" 0.5  layer "dark" 0.5  weight_map pattern:p }
material "glow"  { type diffuse  reflect 0.0  emit blackbody 6500  emit_map pattern:p }
light area { %s  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:bb6500 }
camera "top" { %s  film { res %d %d } }
''' % (expr, lamp, cam, RES, RES)


HEAD = head()
# The floors, checked for contrast and against their reference on the CPU: the quad, facing
# down (u x v), or for the floors facing up, the quad turned up.
FLOOR_REF = {'quad': 'quad', 'instance': 'quad', 'turned': 'quad', 'upright': 'quad', 'noisy': 'quad',
             'quad_up': 'quad_up', 'big': 'quad_up'}
FLOORS = tuple(FLOOR_REF)
# The glowing surfaces: each '<name>' (emit_map the step) against '<name>_one' (emit_map 1).
SPHERE = 'sphere { center 0 0 0  radius 0.5  material glow }'
# Long enough that its end balls, on which the step reads inside, stay out of the 1.2 m field.
TUBE_BODY = 'curve { material glow  basis linear  radius 0.5  point -4 0 0  point 4 0 0 }'
GLOW = {
    'sphere': (RADIAL, SPHERE, NEAR_CAM),
    'sphere_far': (RADIAL, SPHERE, FAR_CAM),
    'iso_far': (RADIAL, 'isosurface { material glow  sphere { center 0 0 0  radius 0.5 } }', FAR_CAM),
    'curve_far': (TUBE, TUBE_BODY, FAR_CAM),
    'sphere_far30k': (RADIAL, SPHERE, FAR30K_CAM),
    'iso_far30k': (RADIAL, 'isosurface { material glow  sphere { center 0 0 0  radius 0.5 } }', FAR30K_CAM),
    'curve_far30k': (TUBE, TUBE_BODY, FAR30K_CAM),
}
# Which side a pattern reads (FTSL 6.1): step(0, ny) picks a mix's first, pale layer where ny
# is +1 and its dark one where it is -1. Seen from above, an opaque floor reads the side the eye
# sees whichever way it faces; a floor light can cross reads its own side's, so facing down it
# reads ny = -1 from above. The crossable layers reflect and transmit alike, so which hemisphere
# a renderer calls the front changes nothing but the cell. Both windings in one frame -- a
# floor facing up, and over its middle, a millimetre above it, a square facing down -- since
# the preview meters each frame for itself: alone, a dark floor previews as grey as a pale one.
SIDE = 'step(0, ny)'
# Each pair's material, and whether light crosses it: the opaque `floor`; `cross`, both layers
# translucent; `mixed`, only its second (the preview bakes its opaque first in); `child`, a
# constant mix whose opaque first layer's own reflect_map reads ny -- under the mix's rule, as
# every pattern on the surface is, however the renderer picked the layer (a BDPT connection
# rebuilds the hit from the layer's vertex).
SIDE_MATS = '''material "tpale" { type translucent  reflect 0.45  transmit 0.45 }
material "tdark" { type translucent  reflect 0.03  transmit 0.03 }
material "cross" { type mix  layer "tpale" 0.5  layer "tdark" 0.5  weight_map pattern:p }
material "mixed" { type mix  layer "pale" 0.5  layer "tdark" 0.5  weight_map pattern:p }
material "pd"    { type diffuse  reflect rgb 0.8 0.8 0.8  reflect_map pattern:p }
material "child" { type mix  layer "pd" 0.5  layer "tdark" 0.5 }
'''
SIDE_BODY = ('quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material %s }\n'
             'quad { origin -0.4 0.001 -0.4  u 0.8 0 0  v 0 0 0.8  material %s }\n')
SIDE_CROSSES = {'floor': False, 'cross': True, 'mixed': True, 'child': True}
SIDE_NAMES = tuple(f'side_{m}' for m in SIDE_CROSSES)


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
    noisy = os.path.join(out, 'floor_noisy.obj').replace('\\', '/')
    write(noisy, 'v -1 1e-12 -1\nv 1 -1e-12 -1\nv 1 1e-12 1\nv -1 -1e-12 1\nf 1 2 3\nf 1 3 4\n')
    # 100 m across, facing up, 1e-6 below y = 0 -- an export's float noise at that size. Hit
    # within a metre of the origin, a hair sized by the hit's distance would not clear it.
    big = os.path.join(out, 'floor_big.obj').replace('\\', '/')
    write(big, 'v -50 -1e-6 -50\nv 50 -1e-6 -50\nv 50 -1e-6 50\nv -50 -1e-6 50\nf 1 3 2\nf 1 4 3\n')
    texts = {
        'quad': HEAD + 'quad { origin -1 0 -1  u 2 0 0  v 0 0 2  material floor }\n',
        'quad_up': HEAD + 'quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material floor }\n',
        'big': HEAD + 'mesh { file "%s"  material floor }\n' % big,
        'instance': HEAD + ('mesh_asset "low" { file "%s"  material floor }\n'
                            'mesh_instance { of "low"  translate 0 1 0 }\n') % low,
        # Rx(90) carries z = 1 (after the half scale) to y = -1; the move lifts it to y = 0.
        'turned': HEAD + ('mesh_asset "up" { file "%s"  material floor }\n'
                          'mesh_instance { of "up"  scale 0.5  rotate 90 0 0  translate 0 1 0 }\n') % upright,
        'upright': HEAD + ('mesh_asset "up0" { file "%s"  material floor }\n'
                           'mesh_instance { of "up0"  rotate 90 0 0 }\n') % upright0,
        'noisy': HEAD + 'mesh { file "%s"  material floor }\n' % noisy,
    }
    for name, (expr, body, cam) in GLOW.items():
        texts[name] = head(expr, cam) + body + '\n'
        texts[name + '_one'] = head('1', cam) + body + '\n'
    for mat in SIDE_CROSSES:
        texts[f'side_{mat}'] = head(SIDE) + SIDE_MATS + SIDE_BODY % (mat, mat)
    paths = {}
    for name, text in texts.items():
        paths[name] = os.path.join(out, name + '.ftsl')
        write(paths[name], text)
    return paths


def render(exe, out, scene, name, dev, mode='R'):
    png = os.path.join(out, f'{name}_{dev}.png' if mode == 'R' else f'{name}_{mode}_{dev}.png')
    pfm = png[:-4] + '.pfm'
    for stale in (png, pfm):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [exe, '-in', scene, '-hdr', '-no-meter', '-seed', '3', '-mode', mode, '-spp', '64', '-device', dev,
           '-o', png]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0 or not os.path.exists(pfm):
        print(f'  render failed: {" ".join(cmd)}\n{r.stdout[-600:]}{r.stderr[-600:]}')
        return None
    if dev == 'gpu' and '-> GPU' not in r.stdout + r.stderr:
        print(f'  did not run on the GPU: {" ".join(cmd)}')
        return None
    return luma(read_pfm(pfm, top_down=True))


def render_raster(exe, out, scene, name, dev):
    """The rasterizer's preview of `scene` on `dev`, as linear luminance (it writes an sRGB PNG)."""
    from PIL import Image
    png = os.path.join(out, f'{name}_raster_{dev}.png')
    if os.path.exists(png):
        os.remove(png)
    cmd = [exe, '-in', scene, '-raster', '-device', dev, '-o', png]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0 or not os.path.exists(png):
        print(f'  raster failed: {" ".join(cmd)}\n{r.stdout[-600:]}{r.stderr[-600:]}')
        return None
    log = r.stdout + r.stderr
    if dev == 'gpu' and ('GPU rasterizer' not in log or 'rendering on the CPU' in log or 'using CPU' in log):
        print(f'  the preview did not run on the GPU: {" ".join(cmd)}\n{log[-600:]}')
        return None
    a = np.asarray(Image.open(png).convert('RGB'), dtype=np.float64) / 255.0
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def contrast(img):
    """The checker's two cells against each other: the 90th over the 10th percentile of the
    middle of the floor (its cells are 0.8 and 0.05; averaged, both read the same grey)."""
    h, w = img.shape
    m = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    lo, hi = np.percentile(m, 10), np.percentile(m, 90)
    return hi / max(lo, 1e-30)


def same_cells(img, ref):
    """The fraction of `ref`'s pixels whose checker cell (pale or dark) `img` shows too, `img`
    sampled at `ref`'s pixel centres (both frame the same view), each split at the geometric
    mean of its own 10th and 90th percentiles."""
    h, w = ref.shape
    ys = ((np.arange(h) + 0.5) * img.shape[0] / h).astype(int)
    xs = ((np.arange(w) + 0.5) * img.shape[1] / w).astype(int)
    s = img[np.ix_(ys, xs)]

    def pale(a):
        return a > np.sqrt(np.percentile(a, 10) * np.percentile(a, 90))
    return float(np.mean(pale(s) == pale(ref)))


def off3x(img, ref, floor=1e-30):
    """The fraction of the pixels the reference covers fully -- lit, and not on its rim, which
    it only partly covers -- that `img` has 3x off either way (a floor larger than the
    reference's fills the background and rim the reference leaves)."""
    lit = ref > 1e-3 * ref.max()
    m = lit.copy()
    m[1:, :] &= lit[:-1, :]; m[:-1, :] &= lit[1:, :]; m[:, 1:] &= lit[:, :-1]; m[:, :-1] &= lit[:, 1:]
    m[0, :] = m[-1, :] = False; m[:, 0] = m[:, -1] = False
    ratio = (img[m] + floor) / (ref[m] + floor)
    return float(np.mean((ratio > 3.0) | (ratio < 1.0 / 3.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'png', 'check_pattern_seam'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--cpu-only', action='store_true')
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    devices = ['cpu'] if a.cpu_only else ['cpu', 'gpu']
    ok = True
    imgs = {}
    paths = scenes(a.out)
    for name, scene in paths.items():
        for dev in devices:
            img = render(exe, a.out, scene, name, dev)
            if img is None:
                ok = False
            else:
                imgs[(name, dev)] = img

    def report(good, line):
        nonlocal ok
        ok = ok and good
        print(f'  {line} {"ok" if good else "FAIL"}')

    # The floors: the checker's contrast, and every pixel against the quad on the CPU. A cell of
    # the other colour reads 16x off (0.8 against 0.05), noise at 64 spp a few tens of per cent,
    # so grey speckle, a seam -- half the floor in the other phase -- or a misplaced surface
    # shows as many pixels 3x off. Tile means would not: a tile spans whole pairs of cells,
    # whose mean a swap of their colours leaves alone.
    for name in FLOORS:
        refname = FLOOR_REF[name]
        ref = imgs.get((refname, 'cpu'))
        for dev in devices:
            img = imgs.get((name, dev))
            if img is None:
                continue
            c = contrast(img)
            report(c > 4.0, f'{name:10s} {dev.upper()}: floor contrast {c:6.2f}')
            if (name, dev) == (refname, 'cpu'):
                continue
            if ref is None:
                report(False, f'{name:10s} {dev.upper()}: no reference ({refname} on the CPU did not render)')
                continue
            report(off3x(img, ref) < 0.02, f'{name:10s} {dev.upper()} against {refname} on the CPU: '
                                           f'{off3x(img, ref) * 100:.2f}% of pixels 3x off')
    ref = imgs.get(('quad', 'cpu'))
    # The glowing surfaces: per pixel, the fraction of its hits that read the inside of the step.
    for name in GLOW:
        for dev in devices:
            img, one = imgs.get((name, dev)), imgs.get((name + '_one', dev))
            if img is None or one is None:
                continue
            m = one > 1e-3 * one.max() if one.max() > 0 else np.zeros_like(one, dtype=bool)
            inside = 1.0 - img[m] / one[m] if m.any() else np.ones(1)
            report(m.sum() > RES * RES // 8 and inside.max() < 1e-3,
                   f'{name:10s} {dev.upper()}: {m.sum()} pixels, {np.mean(inside > 1e-3) * 100:.2f}% with hits '
                   f'reading inside the step (worst {inside.max() * 100:.1f}% of its hits)')
    # The rasterizer reads its patterns at interpolated vertex positions: the noisy floor's
    # straddle y = 0 as its vertices do, and must read the eye's side of floor(4y) as the exact
    # quad does. Each device's preview against its own quad (the two rasterizers shade alike
    # but not identically); deterministic, so any pixel 3x off is a phase swapped.
    # And the preview must show the cells the tracer shows: the quad lies on the checker's
    # wall, so a preview reading the other side of it would show the other phase.
    for dev in devices:
        rq = render_raster(exe, a.out, paths['quad'], 'quad', dev)
        rn = render_raster(exe, a.out, paths['noisy'], 'noisy', dev)
        if rq is None or rn is None:
            ok = False
            continue
        c, off = contrast(rn), off3x(rn, rq, 1e-6)
        report(c > 4.0 and off < 0.02, f'noisy raster {dev.upper()}: floor contrast {c:6.2f}, '
                                       f'{off * 100:.2f}% of pixels 3x off the quad\'s')
        if ref is not None:
            agree = same_cells(rq, ref)
            report(agree > 0.75, f'quad raster {dev.upper()}: {agree * 100:.1f}% of pixels in the cell the CPU '
                                f'trace shows there')
        # The big floor's preview against the up-facing quad's: its triangles' reach sizes the
        # hair. The middle of the frame only: the preview's background is not black, so the
        # border the quad leaves and the big floor fills cannot be masked out by its level.
        ru = render_raster(exe, a.out, paths['quad_up'], 'quad_up', dev)
        rb = render_raster(exe, a.out, paths['big'], 'big', dev)
        if ru is None or rb is None:
            ok = False
            continue
        h, w = ru.shape
        mid = (slice(h // 10, h - h // 10), slice(w // 10, w - w // 10))
        off = off3x(rb[mid], ru[mid], 1e-6)
        report(off < 0.02, f'big raster {dev.upper()}: {off * 100:.2f}% of pixels 3x off the up-facing quad\'s')

    # Which side a pattern reads, in each renderer -- the backward tracer (R), the bidirectional
    # one (D), whose connections rebuild a hit from its vertex, and the preview: the square
    # facing down (seen from behind) over the floor round it facing up (from the front). The
    # square spans the middle 39% of the frame each way, the floor 97%: the middle 24% against
    # a ring from 55% to 85% out. An opaque pair reads the same cell; a crossable pair its two
    # cells, the square the dark (0.03 against 0.45; `child`'s square has only its translucent
    # half's 0.03 to reflect).
    def down_over_up(img):
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.maximum(np.abs((yy + 0.5) / h * 2 - 1), np.abs((xx + 0.5) / w * 2 - 1))
        return float(img[r < 0.24].mean()) / max(float(img[(r > 0.55) & (r < 0.85)].mean()), 1e-30)
    for dev in devices:
        for kind in ('R', 'D', 'raster'):
            for n in SIDE_NAMES:
                if kind == 'R':
                    img = imgs.get((n, dev))
                elif kind == 'D':
                    img = render(exe, a.out, paths[n], n, dev, 'D')
                else:
                    img = render_raster(exe, a.out, paths[n], n, dev)
                label = f'side {kind:6s} {dev.upper()} {n[5:]:6s}'
                if img is None:
                    report(False, f'{label}: did not render')
                    continue
                ratio = down_over_up(img)
                if SIDE_CROSSES[n[5:]]:
                    report(ratio < 1 / 3, f'{label}: facing down over facing up {ratio:.3f} '
                                          f'(its own side\'s cell, the dark)')
                else:
                    report(1 / 1.5 < ratio < 1.5, f'{label}: facing down over facing up {ratio:.3f} (the same cell)')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
