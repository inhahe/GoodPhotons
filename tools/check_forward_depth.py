"""Regression check: the GPU's photon walks take their bounce caps as the CPU's do (B-GPU-MAXBOUNCE).

    python tools/check_forward_depth.py [--out png/check_forward_depth] [--exe ftrace.exe]

`-max-bounce` caps how many surfaces a photon meets in modes A, B and C. The CPU's renderForward
always took it; GPU kernels launched with a literal 32 render `-max-bounce 7` and `8` alike, and a
closed room of albedo 0.8 2 % brighter than CPU mode B at 8. Modes M and S cap their deposits' photons by `-photon-bounce` instead (photonMaxBounce,
default 32), which both backends must take. The room here keeps a lot of its light in long paths, so a cap shows: mode B at
`-max-bounce 2` and `8`, and modes M and S at `-photon-bounce 2` and `8`, on the GPU must each read
within 1 % of the CPU at the same cap (frame mean, two seeds; 2 % for S, whose progressive radius
the backends shrink apart), and 2 must read clearly darker than 8 on both. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

SCENE = NL.join([
    "scene { units meters }",
    'material "w" { type diffuse reflect 0.8 }',
    "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
    "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
    "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w }",
    "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w }",
    "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w }",
    "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 40 }",
    'camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_forward_depth"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(d, "room.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)

    def mean(mode, dev, cap):
        v = []
        for seed in (1, 2):
            png = os.path.join(d, "%s_%s_%d_%d.png" % (mode.lower(), dev, cap, seed))
            budget = {"B": ["-n", "40000000", "-max-bounce", str(cap)],
                      "M": ["-n", "2000000", "-spp", "64", "-photon-bounce", str(cap)],
                      "S": ["-n", "100000", "-spp", "128", "-photon-bounce", str(cap)]}[mode]
            cmd = [exe, "-in", scene, "-mode", mode, "-device", dev] + budget + [
                   "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png]
            for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            v.append(float(read_pfm(pfm, top_down=True).sum(-1).mean()))
        return sum(v) / len(v)

    ok = True
    for mode, flag, tol in (("B", "-max-bounce", 0.01), ("M", "-photon-bounce", 0.01),
                            ("S", "-photon-bounce", 0.02)):
        got = {}
        for cap in (2, 8):
            c, g = mean(mode, "cpu", cap), mean(mode, "gpu", cap)
            got[cap] = (c, g)
            good = abs(g / c - 1.0) <= tol
            ok &= good
            print("%s: mode %s at %s %d: GPU %.5g, CPU %.5g, x%.4f"
                  % ("ok" if good else "FAIL", mode, flag, cap, g, c, g / c), flush=True)
        for i, dev in enumerate(("CPU", "GPU")):
            drop = got[2][i] / got[8][i]
            good = drop < 0.9
            ok &= good
            print("%s: %s mode %s at %s 2 over 8: x%.4f"
                  % ("ok" if good else "FAIL", dev, mode, flag, drop), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
