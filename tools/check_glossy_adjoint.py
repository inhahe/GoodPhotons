"""Regression check: a walk from the light carries the glossy lobe's adjoint (GLOSSY-FLAT-DJU).

    python tools/check_glossy_adjoint.py [--out png/check_glossy_adjoint] [--exe ftrace.exe]
                                         [--no-gpu] [--modes D,J,U,B,M,S] [--seeds 3]

The glossy BSDF divides its lobe by the cosine on the light's side (render.h glossyF), so it is not
reciprocal, and a walk from the light must carry its adjoint: drawing its way out of the lobe about
its arrival, it multiplies r by cos(out)/cos(in) (glossyAdjointRatio), where a walk from the camera
takes r alone. Until 0.440.2 the light walks took r too -- connections had the adjoint (0.276.0),
continuations did not -- so every glossy bounce on a light subpath was off by cos(in)/cos(out).

Two scenes, every mode against mode R on the same device at -max-bounce 32 (modes M and S deposit
over 32 bounces whatever the flag says):
  * "bounce": a glossy floor lit steeply by a panel that faces it, throwing its light at a grazing
    angle onto a diffuse wall, which is the camera's view -- light that arrives steep and leaves
    grazing, the ratio far from 1. Before the fix, on the host, B read 1.32 of R, M 1.31, D 1.11.
  * "room": five glossy walls (roughness 0.4), lit at a grazing angle by a tilted panel -- many
    glossy bounces at every angle. Before the fix D read 1.011 of R, J 1.008, U 1.012.
Each mode's frame mean over `--seeds` seeds must be within TOL of mode R's. Exits 1 on any failure.
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

SPP = {"R": 4096, "D": 1024, "J": 512, "U": 1024, "B": 1, "M": 128, "S": 256}   # CPU; the GPU takes 4x
NPHOT = {"B": 40000000, "M": 2000000, "S": 100000}   # photons (mode S's per pass)
DEPTH = 32

SCENES = {
    "bounce": (0.02, [
        "scene { units meters }",
        'material "g" { type glossy reflect 0.9  roughness 0.5 }',
        'material "w" { type diffuse reflect 0.8 }',
        "quad { origin -2 0 -2  u 0 0 6  v 4 0 0  material g }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
        "light area { origin -0.6 0.3 2.4  u 1.2 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500  power 40 }",
        'camera "c" { eye 0 1.5 3  look_at 0 1.2 -2  up 0 1 0  fov_y 40  film { res 64 48 } }',
    ]),
    "room": (0.01, [
        "scene { units meters }",
        'material "w" { type glossy reflect 0.8  roughness 0.4 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w }",
        "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w }",
        "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w }",
        "light area { origin -1.9 0.2 -1.5  u 0 0 0.6  v 0 2.4 0  normal 1 0 0.2  spd preset:bb6500  power 40 }",
        'camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
    ]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glossy_adjoint"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--modes", default="D,J,U,B,M,S")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    fails = []
    for name, (tol, lines) in SCENES.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        scene = os.path.join(d, "scene.ftsl")
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)

        def frame(mode, dev, seed):
            png = os.path.join(d, "%s_%s_%d.png" % (mode, dev, seed))
            pfm = png[:-4] + ".pfm"
            for f in (png, pfm):
                if os.path.exists(f):
                    os.remove(f)
            cmd = [exe, "-in", scene, "-mode", mode, "-device", dev,
                   "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)),
                   "-n", str(NPHOT.get(mode, 200000)),
                   "-max-bounce", str(DEPTH), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
                   "-o", png]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            return float(read_pfm(pfm, top_down=True).sum(-1).mean())

        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            r = [frame("R", dev, s) for s in seeds]
            rm, rse = np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r))
            for mode in a.modes.split(","):
                v = [frame(mode, dev, s) for s in seeds]
                m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
                ratio = m / rm
                err = np.hypot(se / rm, m * rse / rm ** 2)
                good = abs(ratio - 1.0) <= tol
                print("%s: %-6s mode %s on the %s: %.4f of mode R (+-%.4f)"
                      % ("ok" if good else "FAIL", name, mode, "host" if dev == "cpu" else "device",
                         ratio, err), flush=True)
                if not good:
                    fails.append("%s %s %s" % (name, mode, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
