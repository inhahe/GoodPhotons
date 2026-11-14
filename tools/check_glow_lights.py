"""Regression check: an emissive surface lights the scene, the same in every mode, whatever its shape.

    python tools/check_glow_lights.py [--out png/check_glow_lights] [--exe ftrace.exe] [--quick] [--no-gpu]

Until 0.442.0 only a light block and an emissive mesh block registered an emitter, so a bare emissive
quad, a sphere or an isosurface lit nothing in modes R, M, S and B and all of itself in D, J and U
(GLOW-LIGHT-MODES). Every emissive surface now gets one at load (ftsl.h registerGlows): triangles a
mesh light, a sphere a sphere light, an isosurface or CSG solid -- or a sphere under an emission
pattern -- a Glow, sampled by lines through its box (scene.h glowSamplePoint, the Crofton draw).

SAME LIGHT. The floor of a room lit by one glowing ball of radius 0.25, written three ways: an
analytic sphere (a sphere light), the same ball as an isosurface `sphere` leaf (a Glow), and the
analytic sphere under an emission pattern that is 1 everywhere (a Glow of a sphere). Mode R's floor
must read the same under all three (the Glow's crossings weigh its draws exactly), and the room is
lit: the floor under the ball against the same room whose ball is `glow_only` (dark) by far.

EVERY MODE. A room lit by a glowing wire frame (an isosurface CSG box frame, as gallery_rain's
stands), a grid of glowing lines on the floor (a quad under an emission pattern) and a dim ceiling
panel: every mode, on the CPU and the GPU, against mode R tile by tile over a 4 x 3 grid.

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
GX, GY = 4, 3
DEPTH = {"M": 32, "S": 32}
# (the frame's thin glowing lines are a few bright pixels each, which modes M and S, a sample a pixel
# a pass, resolve within a few per cent of R only at these budgets; J's CPU merges likewise)
ARGS = {"R": ["-spp", "1024"], "D": ["-spp", "512"], "J": ["-spp", "512"], "U": ["-spp", "512"],
        "M": ["-spp", "512", "-n", "2000000"], "S": ["-spp", "1024", "-n", "100000"],
        "B": ["-n", "40000000"], "P": ["-n", "40000000", "-spp", "256"]}

ROOM = [
    "scene { units meters }",
    'camera "c" { eye 0 1.3 3.6  look_at 0 0.5 0  up 0 1 0  fov_y 50  film { res %d %d } }' % (W, H),
    'material "wall"  { type diffuse reflect 0.6 }',
    'material "floor" { type diffuse reflect 0.5 }',
    'quad { origin -2 0 -2  u 4 0 0  v 0 0 6  material floor }',
    'quad { origin -2 0 -2  u 0 2.6 0  v 4 0 0  material wall }',
    'quad { origin -2 0 -2  u 0 0 6  v 0 2.6 0  material wall }',
    'quad { origin 2 0 -2  u 0 2.6 0  v 0 0 6  material wall }',
    'quad { origin -2 2.6 -2  u 0 0 6  v 4 0 0  material wall }',
]


def ball_scene(kind):
    """One glowing ball, written as `kind`: sphere, iso, pattern, or dark (glow_only)."""
    lines = list(ROOM)
    # a faint panel, the same in every variant: a scene whose one emitter is a glow_only ball has
    # no light, which a render refuses
    lines.append('light area { origin -0.1 2.59 -0.1  u 0.2 0 0  v 0 0 0.2  normal 0 -1 0  '
                 'spd preset:bb6500  power 0.02 }')
    glow = 'material "glow" { type diffuse reflect 0.2  emit 0.05 %s }'
    if kind == "pattern":
        lines += ['pattern "one" { expr "1" }', glow % "emit_map pattern:one"]
    elif kind == "dark":
        lines.append(glow % "glow_only on")
    else:
        lines.append(glow % "")
    if kind == "iso":
        lines.append("isosurface { material glow  sphere { center 0 0.5 0  radius 0.25 } }")
    else:
        lines.append("sphere { center 0 0.5 0  radius 0.25  material glow }")
    # the camera must not see the ball (its own glow would swamp the floor's light in a tile)
    return NL.join(lines + [""])


def frame_scene():
    lines = list(ROOM)
    lines += [
        'light area { origin -0.3 2.59 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  spd preset:bb6500  power 4 }',
        'pattern "gridlines" { expr "max(1 - smoothstep(0.01, 0.03, abs(fract(x * 2 + 0.5) - 0.5)),'
        '1 - smoothstep(0.01, 0.03, abs(fract(z * 2 + 0.5) - 0.5)))" }',
        'material "grid" { type diffuse reflect 0.05  emit gaussian 528 34 0.05  emit_map pattern:gridlines }',
        'material "cage" { type diffuse reflect 0.05  emit gaussian 600 40 0.4 }',
        # u x v = +y: it glows upward (emission is one-sided), a centimetre over the floor
        'quad { origin -1.5 0.01 -1.5  u 0 0 3  v 3 0 0  material grid }',
        # a box frame of 3 cm bars, 0.8 m on a side, as gallery_rain's stands are made
        "isosurface { material cage  difference {",
        "    box { center 0 0.6 -0.2  size 0.8 0.8 0.8 }",
        "    box { center 0 0.6 -0.2  size 0.86 0.74 0.74 }",
        "    box { center 0 0.6 -0.2  size 0.74 0.86 0.74 }",
        "    box { center 0 0.6 -0.2  size 0.74 0.74 0.86 }",
        "} }",
    ]
    return NL.join(lines + [""])


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
    return read_pfm(pfm, top_down=True)[..., 1]


def budget(mode, gpu, k):
    out, it = [], iter(ARGS[mode])
    for key in it:
        v = int(next(it)) * (4 if gpu and key == "-spp" else 1) // k
        out += [key, str(max(1, v))]
    return out


def tiles(img):
    h, w = img.shape
    return np.array([[img[r * h // GY:(r + 1) * h // GY, c * w // GX:(c + 1) * w // GX].mean()
                      for c in range(GX)] for r in range(GY)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glow_lights"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    k = 4 if a.quick else 1
    tol = 0.04 if a.quick else 0.02
    ok = True

    # SAME LIGHT: the floor (the image's bottom third) under the three balls.
    floors = {}
    for kind in ("sphere", "iso", "pattern", "dark"):
        scn = os.path.join(out, "ball_%s.ftsl" % kind)
        write(scn, ball_scene(kind))
        img = render(exe, scn, "R", os.path.join(out, "ball_%s_R.png" % kind), False,
                     ["-spp", str(2048 // k)])
        floors[kind] = img[2 * H // 3:, :].mean()
    for kind in ("iso", "pattern"):
        r = floors[kind] / floors["sphere"]
        good = abs(r - 1.0) <= tol
        ok = ok and good
        print("%s: mode R, the floor under a glowing ball written as %s: %.4f of the sphere light's"
              % ("ok" if good else "FAIL", {"iso": "an isosurface", "pattern": "a patterned sphere"}[kind], r),
              flush=True)
    lit = floors["sphere"] / max(floors["dark"], 1e-30)
    good = lit > 3.0
    ok = ok and good
    print("%s: the ball lights the floor: %.1f times its glow_only twin's" % ("ok" if good else "FAIL", lit),
          flush=True)

    # EVERY MODE, on the frame and the grid.
    scn = os.path.join(out, "frame.ftsl")
    write(scn, frame_scene())
    refs = {}
    for depth in sorted(set(DEPTH.values()) | {8}):
        DEPTH["R"] = depth
        refs[depth] = tiles(render(exe, scn, "R", os.path.join(out, "R_ref_%d.png" % depth), False,
                                   ["-spp", str(4096 // k)]))
    DEPTH["R"] = 8
    devs = [False] + ([] if a.no_gpu else [True])
    for mode in ("R", "D", "J", "U", "M", "S", "B", "P"):
        for gpu in devs:
            if mode == "R" and not gpu:
                continue
            tag = "%s_%s" % (mode, "gpu" if gpu else "cpu")
            t = tiles(render(exe, scn, mode, os.path.join(out, tag + ".png"), gpu, budget(mode, gpu, k)))
            rm = refs[DEPTH.get(mode, 8)]
            w, c, r = max((abs(t[r, c] / rm[r, c] - 1.0), c, r) for r in range(GY) for c in range(GX)
                          if rm[r, c] > 0)
            good = w <= tol
            ok = ok and good
            print("%s: mode %s on the %s: worst tile (%d, %d) %.4f of mode R's"
                  % ("ok" if good else "FAIL", mode, "GPU" if gpu else "CPU", c, r, t[r, c] / rm[r, c]),
                  flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
