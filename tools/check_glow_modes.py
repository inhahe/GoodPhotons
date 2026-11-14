"""Regression check: a glow no emitter samples looks the same in every mode, and lights nothing.

    python tools/check_glow_modes.py [--out png/check_glow_modes] [--exe ftrace.exe] [--quick]

A surface whose material emits but that no emitter samples (`glow_only on`, Material::glowOnly; until
0.442.0 every bare emissive quad, isosurface and CSG solid) is seen, and lights nothing: it shows to a
camera ray and through the specular surfaces after one -- a mirror, glass, a glossy reflection, the
surfaces mode R's camera walk arrives at "specularly" (scene.h isSpecularType) -- and no diffuse,
translucent or medium scattering sees it. Every mode is to agree on that. Until 0.442.0 modes D, J and
U counted it on any arrival (all of itself, lighting the room), the forward modes did not show it at
all, and mode P's composite ran away on it (x368).

The scene: a room lit by a ceiling panel, a glowing panel standing on the floor facing the camera,
a mirror ball and a glossy ball beside it that reflect it. Rendered with and without the glow in
mode R, the floor and walls must agree (it lights nothing); every other mode, on the CPU and the
GPU, is compared with mode R tile by tile over a 4 x 3 grid. The forward modes (B, A) see a glow only
directly, as they see the sky (main.cpp prepareSky): the tiles holding the two balls are left out of
their comparison. And mode P under a camera shutter, whose split files each sample by its first
surface, on a glossy-type `glow_only` panel seen directly: its glow is the camera side's (0.442.4).
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

W, H = 96, 72
GX, GY = 4, 3                                    # the comparison grid
BALL_TILES = {(0, 1), (0, 2), (3, 1), (3, 2)}    # (column, row) of the two balls' tiles, top row 0
# Each mode at the bounce cap check_bump_modes gives it (M's and S's deposits run deep; mode R is
# rendered at both), with its own sample budget (CPU; the GPU takes 4x the samples).
DEPTH = {"M": 32, "S": 32}
# (M and S see the glow in the mirror ball as a few pixels twenty times the room's brightness, whose
# share of their tile M reads within 15 % of R at 64 samples: hence their larger budgets)
ARGS = {"R": ["-spp", "1024"], "D": ["-spp", "512"], "J": ["-spp", "256"], "U": ["-spp", "512"],
        "M": ["-spp", "512", "-n", "2000000"], "S": ["-spp", "1024", "-n", "100000"],
        "B": ["-n", "40000000"], "P": ["-n", "40000000", "-spp", "256"]}


def budget(mode, gpu, k):
    """The mode's arguments, its sample counts times 4 on the GPU and over k for --quick."""
    out, it = [], iter(ARGS[mode])
    for key in it:
        v = int(next(it)) * (4 if gpu and key == "-spp" else 1) // k
        out += [key, str(max(1, v))]
    return out


def scene(glow, flag):
    # emit is per nm: 0.05 puts the glow's radiance beside the ceiling panel's, so a glow that lit
    # the room would light the floor before it as brightly as the panel lights the far wall.
    emit = "emit 0.05" if glow else ""
    gf = "glow_only on" if (glow and flag) else ""
    return NL.join([
        "scene { units meters }",
        'camera "c" { eye 0 1.1 3.6  look_at 0 0.8 0  up 0 1 0  fov_y 50  film { res %d %d } }' % (W, H),
        'material "wall"  { type diffuse reflect 0.6 }',
        'material "floor" { type diffuse reflect 0.5 }',
        'material "glow"  { type diffuse reflect 0.3 %s %s }' % (emit, gf),
        'material "chrome" { type mirror reflect 0.9 }',
        'material "gloss"  { type glossy reflect 0.8 roughness 0.15 }',
        'quad { origin -2 0 -2  u 4 0 0  v 0 0 6  material floor }',
        'quad { origin -2 0 -2  u 0 2.6 0  v 4 0 0  material wall }',
        'quad { origin -2 0 -2  u 0 0 6  v 0 2.6 0  material wall }',
        'quad { origin 2 0 -2  u 0 2.6 0  v 0 0 6  material wall }',
        'quad { origin -2 2.6 -2  u 0 0 6  v 4 0 0  material wall }',
        'light area { origin -0.4 2.59 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }',
        # the glow: a panel facing the camera
        'quad { origin -0.35 0.4 -0.6  u 0.7 0 0  v 0 0.6 0  material glow }',
        # wholly inside the grid's outer columns (BALL_TILES)
        'sphere { center -1.4 0.35 0.5  radius 0.35  material chrome }',
        'sphere { center 1.4 0.35 0.5  radius 0.35  material gloss }',
        "",
    ])


def lacquer_scene(glow):
    """A diffuse floor, a lacquered ball (a Fresnel coat over a dark body) and a glow_only panel
    whose glow the ball's coat throws onto the floor; the camera looks down at that floor."""
    emit = "emit 0.5  glow_only on" if glow else ""
    return NL.join([
        "scene { units meters }",
        # straight down at the floor in front of the ball: neither the ball nor the panel in view
        'camera "c" { eye 0 2.0 1.0  look_at 0 0 0.9  up 0 0 -1  fov_y 30  film { res 64 48 } }',
        'material "floor" { type diffuse reflect 0.6 }',
        'material "body"  { type diffuse reflect 0.05 }',
        'material "lacquer" { type layered  coat { reflectance fresnel  ior 1.5  roughness 0 }  layer "body" 1 }',
        'material "glow"  { type diffuse reflect 0.0 %s }' % emit,
        "quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material floor }",
        'light area { origin -0.2 2.4 -0.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500  power 6 }',
        "sphere { center 0 0.3 0  radius 0.3  material lacquer }",
        # the panel faces the ball, out of the camera's view behind it
        "quad { origin -0.5 0.1 -1.2  u 1 0 0  v 0 0.8 0  material glow }",
        "",
    ])


def blur_scene():
    """Mode P's blurred split (main.cpp addBlurredSkyRows, glowsOnly): a glossy-type `glow_only` panel
    the camera sees directly, which the split's camera side files in its specular-first film; a
    ball moving across the room, so the camera sees motion and P takes the split."""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'camera "c" { eye 0 1.1 3.6  look_at 0 0.8 0  up 0 1 0  fov_y 50  film { res %d %d }'
        "  shutter_open -0.25  shutter_close 0.25 }" % (W, H),
        'material "wall"  { type diffuse reflect 0.6 }',
        'material "screen" { type glossy reflect 0.3  roughness 0.3  emit 0.05  glow_only on }',
        'quad { origin -2 0 -2  u 4 0 0  v 0 0 6  material wall }',
        'quad { origin -2 0 -2  u 0 2.6 0  v 4 0 0  material wall }',
        'quad { origin -2 0 -2  u 0 0 6  v 0 2.6 0  material wall }',
        'quad { origin 2 0 -2  u 0 2.6 0  v 0 0 6  material wall }',
        'quad { origin -2 2.6 -2  u 0 0 6  v 4 0 0  material wall }',
        'light area { origin -0.4 2.59 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }',
        # the panel faces the camera, in the middle two columns' lower tiles
        'quad { origin -0.35 0.4 -0.6  u 0.7 0 0  v 0 0.6 0  material screen }',
        "group { motion { key { translate -1.4 0.3 0.8 } key { translate -1.0 0.3 0.8 } }",
        "        sphere { center 0 0 0  radius 0.25  material wall } }",
        "",
    ])


def write(path, txt):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(txt)


def render(exe, scn, mode, out, gpu, extra):
    pfm = out[:-4] + ".pfm"
    for f in (out, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scn, "-mode", mode, "-device", "gpu" if gpu else "cpu", "-hdr", "-no-meter",
           "-seed", "1", "-window-min", "-o", out, "-max-bounce", str(DEPTH.get(mode, 8))] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-2000:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[..., 1]   # Y


def tiles(img):
    h, w = img.shape
    return np.array([[img[r * h // GY:(r + 1) * h // GY, c * w // GX:(c + 1) * w // GX].mean()
                      for c in range(GX)] for r in range(GY)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glow_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--quick", action="store_true", help="fewer samples (looser tolerance)")
    ap.add_argument("--no-flag", action="store_true",
                    help="leave `glow_only` out (a binary before 0.442.0, where no glow had an emitter)")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    flag = not a.no_flag
    s_on, s_off = os.path.join(out, "glow.ftsl"), os.path.join(out, "noglow.ftsl")
    write(s_on, scene(True, flag))
    write(s_off, scene(False, flag))
    k = 4 if a.quick else 1
    tol = 0.04 if a.quick else 0.025
    ok = True

    refs = {}
    for depth in sorted(set(DEPTH.values()) | {8}):
        DEPTH["R"] = depth
        refs[depth] = tiles(render(exe, s_on, "R", os.path.join(out, "R_ref_%d.png" % depth), False,
                                   ["-spp", str(4096 // k)]))
    DEPTH["R"] = 8
    ref = refs[8]
    dark = tiles(render(exe, s_off, "R", os.path.join(out, "R_off.png"), False, ["-spp", str(4096 // k)]))
    # It lights nothing: away from the glow itself (the middle two columns' lower tiles) and the
    # balls, which show it, the room reads the same with and without it.
    lit = [(c, r) for r in range(GY) for c in range(GX)
           if (c, r) not in BALL_TILES and not (c in (1, 2) and r >= 1)]
    worst = max(abs(ref[r, c] / dark[r, c] - 1.0) for c, r in lit)
    good = worst <= tol
    ok = ok and good
    print("%s: mode R, the room with and without the glow: worst tile off by %.4f (tolerance %.3f)"
          % ("ok" if good else "FAIL", worst, tol), flush=True)
    print("     the glow's tiles read %.3g with it, %.3g without"
          % (ref[1, 1] + ref[1, 2], dark[1, 1] + dark[1, 2]))

    # ...nor through a lacquered ball's coat: the floor the coat would throw the glow on reads the
    # same with the glow and without, in mode R on both backends (0.442.0's host R counted it).
    for gpu in (False, True):
        fl = []
        for glow in (True, False):
            scn = os.path.join(out, "lacquer_%s.ftsl" % ("on" if glow else "off"))
            write(scn, lacquer_scene(glow))
            img = render(exe, scn, "R", os.path.join(out, "lacquer_%s_%s.png"
                                                        % ("on" if glow else "off", "gpu" if gpu else "cpu")),
                         gpu, ["-spp", str(4096 // k)])
            fl.append(img[:, :].mean())
        d = abs(fl[0] / fl[1] - 1.0)
        good = d <= tol / 2
        ok = ok and good
        print("%s: mode R on the %s, the floor by a lacquered ball with and without a glow_only panel: "
              "off by %.4f" % ("ok" if good else "FAIL", "GPU" if gpu else "CPU", d), flush=True)

    # Mode P under a blur files each sample by its first surface: the glossy panel's glow is its camera
    # side's, and the forward layer must not draw it as well (until 0.442.4 it did: twice as bright).
    s_blur = os.path.join(out, "blur.ftsl")
    write(s_blur, blur_scene())
    rb = tiles(render(exe, s_blur, "R", os.path.join(out, "blur_R.png"), False, ["-spp", str(4096 // k)]))
    for gpu in (False, True):
        t = tiles(render(exe, s_blur, "P", os.path.join(out, "blur_P_%s.png" % ("gpu" if gpu else "cpu")),
                         gpu, budget("P", gpu, k)))
        w = max(abs(t[r, c] / rb[r, c] - 1.0) for c in (1, 2) for r in (1, 2))
        good = w <= tol
        ok = ok and good
        print("%s: mode P on the %s under a blur, the glossy glow's tiles: off mode R's by %.4f"
              % ("ok" if good else "FAIL", "GPU" if gpu else "CPU", w), flush=True)

    runs = [("R", True), ("D", False), ("D", True), ("J", False), ("J", True), ("U", False), ("U", True),
            ("M", False), ("M", True), ("S", False), ("S", True), ("B", False), ("B", True),
            ("P", False), ("P", True)]
    for mode, gpu in runs:
        tag = "%s_%s" % (mode, "gpu" if gpu else "cpu")
        t = tiles(render(exe, s_on, mode, os.path.join(out, tag + ".png"), gpu, budget(mode, gpu, k)))
        rm = refs[DEPTH.get(mode, 8)]
        skip = BALL_TILES if mode in ("B", "A") else set()
        rel = [(abs(t[r, c] / rm[r, c] - 1.0), c, r) for r in range(GY) for c in range(GX)
               if (c, r) not in skip and rm[r, c] > 0]
        w, c, r = max(rel)
        good = w <= tol
        ok = ok and good
        print("%s: mode %s on the %s: worst tile (%d, %d) %.4f of mode R's, off by %.4f"
              % ("ok" if good else "FAIL", mode, "GPU" if gpu else "CPU", c, r, t[r, c] / rm[r, c], w),
              flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
