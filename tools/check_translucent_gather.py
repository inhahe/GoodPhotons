"""Regression check: modes M and S gather a translucent surface's transmitted light (TRANSLUCENT-MS).

    python tools/check_translucent_gather.py [--out png/check_translucent_gather] [--exe ftrace.exe]
                                             [--no-gpu] [--modes M,S] [--seeds 3]

A `translucent` sheet (DiffuseTransmit) reflects into its lit side and transmits into the other,
and a photon it transmits is deposited on its far face, facing away from a visible point on the
near one. A translucent visible point takes the far side's photons too, at its transmit albedo
(gatherplane.h onGatherSurfaceSided, photonmap_render.h gatherAlbedo); a gather that keeps only the
photons facing its own side, at the reflect albedo, renders a sheet lit from behind without its
transmitted light, about 0.91 of mode R in M and S on the room below.

The room: five diffuse walls (reflect 0.8), a translucent panel (reflect 0.4, transmit 0.4)
standing across it, lit at a grazing angle by a tilted panel on the far side, 64x48. Every mode
against mode R on the same device at -max-bounce 32 (modes M and S deposit over 32 bounces
whatever the flag says); each mode's frame mean over `--seeds` seeds must be within 1 % of R's.

A far-side photon must lie on the sheet itself: the back face of a thin solid behind it, parallel
and a thickness away, holds the light arriving at that face, not the light the sheet transmits. The
second room puts an opaque diffuse plate 5 mm behind the sheet, lit from behind, and gathers mode M
at a pinned radius of 0.15 (30 times the gap; the adaptive radius stays far smaller here): on the
host and device the panel's tiles must read within 2 % of mode R. A far-side test with the band's
flat allowance gathered the plate's back and read them 2.4 and 4.6 (0.440.4); with the far side's
fixed, the near side's allowance (a tenth of the distance) still gathered the plate's lit inner
face as a same-facing surface wherever the distance passed ten times the gap, and read them 1.14
and 1.36 (0.440.12, THIN-GAP-GATHER). The band has no flat allowance now. Exits 1 on any failure.
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


# The plate: opaque, 5 mm behind the sheet (along the sheet's normal), lit from behind.
_N = np.array([-1.0, 0.0, 4.0]) / np.sqrt(17.0)
PLATE = SCENE.replace(
    "quad { origin -1 0 -0.5  u 2 0 0.5  v 0 2 0  material t }",
    "quad { origin -1 0 -0.5  u 2 0 0.5  v 0 2 0  material t }" + NL +
    "quad { origin %.6f %.6f %.6f  u 2 0 0.5  v 0 2 0  material w }" % tuple(np.array([-1.0, 0.0, -0.5]) - _N * 0.005))
PLATE_TOL = 0.02


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
    plate = os.path.join(out, "plate.ftsl")
    with open(plate, "w", encoding="utf-8", newline="") as f:
        f.write(PLATE)

    def img(mode, dev, seed, extra=()):
        png = os.path.join(out, "plate_%s_%s_%d.png" % (mode, dev, seed))
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", plate, "-mode", mode, "-device", dev,
               "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)), "-n", str(NPHOT.get(mode, 200000)),
               "-max-bounce", str(DEPTH), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
               "-o", png] + list(extra)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
        return read_pfm(pfm, top_down=True).sum(-1)

    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        r = np.mean([img("R", dev, s) for s in seeds[:2]], 0)
        m = np.mean([img("M", dev, s, ["-pmradius", "0.15"]) for s in seeds[:2]], 0)
        h, w = r.shape
        tiles = [m[h * i // 4:h * (i + 1) // 4, w * 2 // 6:w * 4 // 6].sum() /
                 r[h * i // 4:h * (i + 1) // 4, w * 2 // 6:w * 4 // 6].sum() for i in (1, 2)]
        good = max(abs(t - 1.0) for t in tiles) <= PLATE_TOL
        print("%s: plate, mode M at radius 0.15 on the %s: the panel's tiles %s of mode R"
              % ("ok" if good else "FAIL", "host" if dev == "cpu" else "device",
                 " and ".join("%.3f" % t for t in tiles)), flush=True)
        if not good:
            fails.append("plate M %s" % dev)
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
