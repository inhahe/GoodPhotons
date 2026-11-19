"""Regression check: a merge map's query is exact at any size of its hashed table.

    python -u tools/check_merge_table.py [--exe ftrace.exe] [--out png/check_merge_table] [--modes UJMS]

The merge maps -- mode U's light vertices, mode J's surface photons, the photon map of modes M and
S -- bin their points in a hashed lattice (merge_grid.h): cells the merge radius wide in hashed
blocks of 3x3x3, buckets about two a point. A bucket holds a cell of every block that hashes to its
slot, and a query reads the 27 cells around it, so it must never read one bucket twice, or the
points of two of its cells sharing it would count twice; a bucket's low part is its cell's place in
its block, which the 27 never share. FTRACE_MERGE_TABLE=1 holds the table to one slot, 27 buckets
that every block shares, so every bucket holds thousands of cells and a query's blocks all collide:
an exact query still accepts each point once, in the order it was binned, so the render is the
default table's. A glass ball's caustic on a diffuse floor in a room, each mode rendered with the
default table and with the 27 buckets:

  - on the CPU, byte for byte the same HDR output;
  - on the GPU, the same to 1e-5 of each pixel in modes U and J (a warp's share of a merge list
    moves with the strays among it, which regroups the sums; nothing more), and to 3e-4 in modes M
    and S, whose photons are deposited by atomics and so differ that much from run to run.

And `-checkpmgrid`'s photon map, its radius queries compared with brute force as exact photon
sets, at the 27 buckets too.

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

SCENE = """scene { units meters }
material "w" { type diffuse reflect 0.7 }
material "r" { type diffuse reflect rgb 0.7 0.15 0.1 }
material "g" { type dielectric ior 1.5 }
quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }
quad { origin -1 0 -1  u 2 0 0  v 0 2 0  material w }
quad { origin -1 0 -1  u 0 2 0  v 0 0 2  material r }
quad { origin 1 0 -1  u 0 0 2  v 0 2 0  material w }
quad { origin -1 2 -1  u 2 0 0  v 0 0 2  material w }
sphere { center 0.1 0.35 0.1  radius 0.35  material g }
light area { origin -0.2 1.98 -0.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0  power 30  spd preset:d65 }
camera "c" { eye 0 1 3.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  film { res 64 48 } }
"""
# Mode J redraws its light side at epochs paced by the work it counts, the gather's entries among it,
# which a smaller table's strays raise (those whose block's tag matches the cell's, one in 256 of
# them): -beamfreeze holds it to one realization. Mode M chooses its radius at the automatic setting,
# from a median over the cells the geometry alone picks (PhotonMap::medianNeighborCount), so the
# table must not move that either.
MODES = {"U": ["-mode", "U", "-spp", "4"], "J": ["-mode", "J", "-spp", "4", "-beamfreeze"],
         "M": ["-mode", "M", "-n", "400000", "-spp", "2", "-beamfreeze"],
         "S": ["-mode", "S", "-spp", "4"]}
GPU_TOL = {"U": 1e-5, "J": 1e-5, "M": 3e-4, "S": 3e-4}
GPU_TOKENS = ("-> GPU", "[gpu]", "VCM/UPS on ", "SPPM on ")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_merge_table"))
    ap.add_argument("--modes", default="".join(MODES))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(d, "room.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    fails = []

    def render(tag, device, args, table):
        png = os.path.join(d, "%s_%s_%s.png" % (tag, device, table or "default"))
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        env = dict(os.environ)
        env.pop("FTRACE_MERGE_TABLE", None)
        if table:
            env["FTRACE_MERGE_TABLE"] = table
        p = subprocess.run([exe, "-in", scene, "-device", device, "-seed", "3", "-hdr", "-no-meter", "-window-min",
                            "-o", png] + args, cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        log = p.stdout + p.stderr
        if p.returncode != 0 or not os.path.exists(pfm):
            return None, None, log
        with open(pfm, "rb") as f:
            raw = f.read()
        return raw, read_pfm(pfm, top_down=True), log

    for tag in a.modes:
        for device in ("cpu", "gpu"):
            name = "mode %s on the %s, 27 buckets against the default table" % (tag, device.upper())
            r0, i0, log0 = render(tag, device, MODES[tag], None)
            r1, i1, log1 = render(tag, device, MODES[tag], "1")
            if r0 is None or r1 is None:
                fails.append(name)
                print("FAIL %s: a render failed%s%s" % (name, NL, (log0 if r0 is None else log1)[-800:]), flush=True)
                continue
            if device == "gpu" and not all(any(t in lg for t in GPU_TOKENS) for lg in (log0, log1)):
                fails.append(name)
                print("FAIL %s: a render did not run on the GPU" % name, flush=True)
                continue
            m0 = float(i0.mean())
            if device == "cpu":
                ok = r0 == r1
                detail = "byte-identical" if ok else "differs: mean x%.6f" % (float(i1.mean()) / max(m0, 1e-30))
            else:
                rel = np.abs(i1 - i0) / np.maximum(np.abs(i0), 1e-6 * max(m0, 1e-30))
                worst = float(rel.max())
                ok = worst <= GPU_TOL[tag]
                detail = "worst pixel %.2e of itself, mean x%.8f" % (worst, float(i1.mean()) / max(m0, 1e-30))
            if not ok:
                fails.append(name)
            print("%-4s %s: %s" % ("ok" if ok else "FAIL", name, detail), flush=True)
    env = dict(os.environ)
    env["FTRACE_MERGE_TABLE"] = "1"
    p = subprocess.run([exe, "-checkpmgrid"], cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    line = next((l for l in p.stdout.splitlines() if "brute force" in l and "probes" in l), "")
    ok = p.returncode == 0 and "(ok)" in line
    if not ok:
        fails.append("-checkpmgrid at 27 buckets")
    print("%-4s -checkpmgrid at 27 buckets: %s" % ("ok" if ok else "FAIL", line.strip() or (p.stdout + p.stderr)[-400:]),
          flush=True)
    print("PASS" if not fails else "FAILED: " + "; ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
