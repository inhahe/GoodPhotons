"""Regression check: modes M and S gather a translucent surface's transmitted light (TRANSLUCENT-MS).

    python tools/check_translucent_gather.py [--out png/check_translucent_gather] [--exe ftrace.exe]
                                             [--no-gpu] [--modes M,S] [--seeds 3]

A `translucent` sheet (DiffuseTransmit) reflects into its lit side and transmits into the other,
and a photon it transmits is deposited on its far face, facing away from a visible point on the
near one. Until 0.440.1 the gathers of modes M and S kept only photons facing the visible point's
own side (gatherplane.h onGatherSurface) and weighed them by the reflect albedo alone, so a
translucent sheet lit from behind rendered without its transmitted light: 0.912 of mode R in M,
0.901 in S, on the room below. Now a translucent visible point takes the far side's photons too,
at its transmit albedo (onGatherSurfaceSided, gatherAlbedo).

The room: five diffuse walls (reflect 0.8), a translucent panel (reflect 0.4, transmit 0.4)
standing across it, lit at a grazing angle by a tilted panel on the far side, 64x48. Every mode
against mode R on the same device at -max-bounce 32 (modes M and S deposit over 32 bounces
whatever the flag says); each mode's frame mean over `--seeds` seeds must be within 1 % of R's.
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

SPP = {"R": 2048, "D": 512, "B": 1, "M": 64, "S": 256}   # CPU; the GPU takes 4x
# (mode S: passes; progressive photon mapping's bias shrinks with them)
NPHOT = {"B": 40000000, "M": 2000000, "S": 100000}   # photons (mode S's per pass)
DEPTH = 32
TOL = 0.01

SCENE = NL.join([
    "scene { units meters }",
    'material "w" { type diffuse reflect 0.8 }',
    'material "t" { type translucent reflect 0.4  transmit 0.4 }',
    "quad { origin -1 0 -0.5  u 2 0 0.5  v 0 2 0  material t }",
    "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
    "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
    "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w }",
    "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w }",
    "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w }",
    "light area { origin -1.9 0.2 -1.5  u 0 0 0.6  v 0 2.4 0  normal 1 0 0.2  spd preset:bb6500  power 40 }",
    'camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_translucent_gather"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--modes", default="M,S")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    os.makedirs(out, exist_ok=True)
    scene = os.path.join(out, "room.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)

    def frame(mode, dev, seed):
        png = os.path.join(out, "%s_%s_%d.png" % (mode, dev, seed))
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", scene, "-mode", mode, "-device", dev,
               "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)), "-n", str(NPHOT.get(mode, 200000)),
               "-max-bounce", str(DEPTH), "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
        return float(read_pfm(pfm, top_down=True).sum(-1).mean())

    fails = []
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        r = [frame("R", dev, s) for s in seeds]
        rm, rse = np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r))
        for mode in a.modes.split(","):
            v = [frame(mode, dev, s) for s in seeds]
            m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
            ratio = m / rm
            err = np.hypot(se / rm, m * rse / rm ** 2)
            good = abs(ratio - 1.0) <= TOL
            print("%s: mode %s on the %s: %.4f of mode R (+-%.4f)"
                  % ("ok" if good else "FAIL", mode, "host" if dev == "cpu" else "device", ratio, err),
                  flush=True)
            if not good:
                fails.append("%s %s" % (mode, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
