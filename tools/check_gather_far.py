"""Regression check: a photon gather far from the origin keeps its own surface's photons (device).

    python tools/check_gather_far.py [--out png/check_gather_far] [--exe ftrace.exe] [--far 3000]

Modes M and S keep a photon only if it lies on the gather point's surface: near its tangent plane,
off it by at most d sin(theta) plus a slack for rounding (gatherplane.h withinGatherBand,
gatherBandSlack). The band has no flat allowance (THIN-GAP-GATHER), so on a flat surface the slack
alone has to cover the rounding of the two positions. The host computes them in double; the device,
under FTRACE_GPU_FP32, in float, whose rounding is about 1e-7 of the coordinates -- far from the
origin that outgrows a thousandth of a small radius, and the gather and its footprint probe dropped
the very surface they were on: the room below, 3 km out, read 3.6 (mode S) and 4.2 (mode M) of the
same room at the origin. The slack now takes the device's rounding at the hit's coordinate scale
(Hit::pscale).

check_bump_modes' flat room (five diffuse walls, a lamp) at the origin and moved `--far` metres
out along x and z, camera and lamp with it, modes M and S on the device: the far room must read
within 1 % of the near one. A render that did not run on the device (a fallback to the host, which
computes in double and so tests nothing here) fails the check. Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402
import check_bump_modes as cbm  # noqa: E402

TOL = 0.01
MODES = (("S", ["-spp", "1024", "-n", "100000"]), ("M", ["-spp", "256", "-n", "2000000"]))


def shift(txt, t):
    """Every point of the room (quad origins, the lamp, the eye and its target) moved by (t, 0, t)."""
    def mv(m):
        return "%s %.6f %.6f %.6f" % (m.group(1), float(m.group(2)) + t, float(m.group(3)),
                                      float(m.group(4)) + t)
    return re.sub(r"\b(origin|eye|look_at)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", mv, txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gather_far"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--far", type=float, default=3000.0)
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    scenes = {}
    for t in (0.0, a.far):
        d = os.path.join(out, "t%g" % t)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "room.ftsl")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(shift(cbm.room(d, 0.0), t))
        scenes[t] = p
    ok = True
    for mode, args in MODES:
        means = {}
        for t, sc in scenes.items():
            v = []
            for s in (1, 2):
                png = os.path.join(os.path.dirname(sc), "%s_%d.png" % (mode, s))
                for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                    if os.path.exists(f):
                        os.remove(f)
                cmd = [exe, "-in", sc, "-mode", mode, "-device", "gpu", "-max-bounce", "32",
                       "-seed", str(s), "-hdr", "-no-meter", "-window-min", "-o", png] + args
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if p.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
                    print((p.stdout + p.stderr)[-1500:])
                    raise SystemExit("render failed: " + " ".join(cmd))
                log = p.stdout + p.stderr
                banner = [ln for ln in log.splitlines() if ln.startswith("mode %s:" % mode)]
                if "-> GPU:" not in log or "using CPU" in log or any("CPU thread" in ln for ln in banner):
                    print(log[-1500:])
                    raise SystemExit("not rendered on the device: " + " ".join(cmd))
                v.append(read_pfm(png[:-4] + ".pfm", top_down=True).sum(-1).mean())
            means[t] = np.mean(v)
        ratio = means[a.far] / means[0.0]
        good = abs(ratio - 1.0) <= TOL
        ok &= good
        print("%s: mode %s on the device: the room %g m out reads %.4f of the room at the origin"
              % ("ok" if good else "FAIL", mode, a.far, ratio), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
