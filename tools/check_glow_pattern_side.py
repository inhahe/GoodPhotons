"""Regression check: a Glow's emission pattern is read on the side it emits from, at every draw.

    python tools/check_glow_pattern_side.py [--out png/check_glow_pattern_side] [--exe ftrace.exe]
                                            [--no-gpu]

A Glow (an emissive isosurface, CSG solid or patterned sphere: scene.h glowSamplePoint) is drawn
at the crossings of lines through its box. About half the crossings are where a line LEAVES the
solid; the intersector reports those from behind, with the normal and curvature turned inward.
Emission is toward the outside, so a pattern reading the normal, the curvature or the cavity must
be read from outside there too, as a camera ray that sees the glow reads it. Until 0.443.0 the
draws read it from behind: a sphere glowing only on its top cap (`step(0.5, ny)`) emitted from
its bottom in every light-sampling and light-tracing estimate (Fable's review of 0.442.0).

The scene: an open floor and, above it, a black sphere glowing on its top cap only, as an
isosurface and as a patterned `sphere`. No point of the cap sees the floor directly beneath the
sphere (the sphere hides it, and the cap faces up), and nothing else returns light there, so that
patch must read black while the floor beyond the sphere's shadow is lit. Modes B (light tracing
alone) and R (light sampling with the camera paths' own hits), CPU and GPU: the patch must read
under 0.5 % of the lit floor. Exits 1 on any failure.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

W, H = 128, 96
TOL = 0.005
ARGS = {"B": ["-n", "20000000"], "R": ["-spp", "256"]}   # CPU; the GPU takes 4x
EYE, AT, FOV = np.array([0.0, 0.9, 1.1]), np.array([0.0, 0.0, 0.0]), 60.0
C, R = np.array([0.0, 1.5, 0.0]), 0.3                      # the sphere


def masks():
    """Pixels whose camera ray meets the floor straight under the sphere (within 0.2 of its axis,
    two thirds of its radius), and those meeting it lit, 0.9 to 1.6 from the axis, the sphere not in the way."""
    fwd = AT - EYE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = math.tan(math.radians(FOV) / 2.0)
    under = np.zeros((H, W), bool)
    lit = np.zeros((H, W), bool)
    for py in range(H):
        for px in range(W):
            d = fwd + right * ((2.0 * (px + 0.5) / W - 1.0) * th * W / H) + up * ((1.0 - 2.0 * (py + 0.5) / H) * th)
            d /= np.linalg.norm(d)
            if d[1] >= 0.0:
                continue
            t = -EYE[1] / d[1]
            oc = EYE - C
            b = float(np.dot(oc, d))
            disc = b * b - (float(np.dot(oc, oc)) - R * R)
            if disc > 0.0 and -b - math.sqrt(disc) < t:   # the sphere first (or the ray inside it)
                continue
            q = EYE + d * t
            rho = math.hypot(q[0], q[2])
            under[py, px] = rho < 0.2
            lit[py, px] = 0.9 < rho < 1.6
    return under, lit


def scene(kind):
    """The floor and the cap-glowing sphere, as an isosurface or a patterned `sphere`."""
    body = ("isosurface { material glow  sphere { center 0 1.5 0  radius 0.3 } }" if kind == "iso"
            else "sphere { center 0 1.5 0  radius 0.3  material glow }")   # C, R
    return NL.join([
        "scene { units meters }",
        # below the sphere's bottom, so the floor under it is in view (the masks follow it)
        'camera "c" { eye %g %g %g  look_at %g %g %g  up 0 1 0  fov_y %g  film { res %d %d } }'
        % (*EYE, *AT, FOV, W, H),
        'material "floor" { type diffuse reflect 0.5 }',
        'pattern "cap" { expr "step(0.5, ny)" }',
        'material "glow" { type diffuse reflect 0.0  emit 0.5  emit_map pattern:cap }',
        "quad { origin -4 0 -4  u 0 0 8  v 8 0 0  material floor }",
        body,
        "",
    ])


def render(exe, scn, mode, gpu, out):
    pfm = out[:-4] + ".pfm"
    for f in (out, pfm):
        if os.path.exists(f):
            os.remove(f)
    budget, it = [], iter(ARGS[mode])
    for key in it:
        budget += [key, str(int(next(it)) * (4 if gpu else 1))]
    cmd = [exe, "-in", scn, "-mode", mode, "-device", "gpu" if gpu else "cpu", "-hdr", "-no-meter",
           "-seed", "1", "-window-min", "-o", out] + budget
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-2000:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[..., 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glow_pattern_side"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    under_m, lit_m = masks()
    if under_m.sum() < 20 or lit_m.sum() < 100:
        raise SystemExit("internal error: the masks hold %d and %d pixels" % (under_m.sum(), lit_m.sum()))
    for kind in ("iso", "sphere"):
        scn = os.path.join(out, kind + ".ftsl")
        with open(scn, "w", encoding="utf-8", newline="") as f:
            f.write(scene(kind))
        for mode in ("B", "R"):
            for gpu in ([False] if a.no_gpu else [False, True]):
                img = render(exe, scn, mode, gpu, os.path.join(out, "%s_%s_%s.png"
                                                                % (kind, mode, "gpu" if gpu else "cpu")))
                under, lit = float(img[under_m].mean()), float(img[lit_m].mean())
                r = under / lit if lit > 0 else float("inf")
                good = lit > 0 and r <= TOL
                ok = ok and good
                print("%s: %s, mode %s on the %s: the floor under the sphere reads %.4f of the lit floor"
                      % ("ok" if good else "FAIL", kind, mode, "GPU" if gpu else "CPU", r), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
