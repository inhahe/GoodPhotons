"""Regression check: mode M counts a sun once at a fiber -- by its light sampling, not again on escape.

    D:/python314/python.exe -u tools/check_hair_nee_sun.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                           [--photons N] [--no-gpu]

Mode M's camera walk takes a fiber's direct light by light sampling (HAIR-NEE), as mode R does, and
must then not count an emitter its continuation reaches. The emitter-hit site skipped it after a
fiber (`hairArrival`); the escape into the sun's disc did not, so a continuation that left the
scene inside the disc added the sun a second time -- in expectation doubling the fibers' direct
sunlight, as fireflies -- on the CPU (photonGather) and the GPU (dPhotonGather) alike.

The scene is a fur ball on a floor under a wide sun (20 degrees: the double count then lands
often and small enough to measure; a real sun's lands rarely, as fireflies). On 0.377.0 the
first comparison read 18 % bright over the frame (15 standard errors). Two comparisons, two seeds
each:
  1. CPU mode M with the fiber light sampling on against off (FTRACE_HAIR_NEE=0: every fiber
     continuation then takes what it reaches at full weight) -- one estimator's two strategies;
  2. GPU mode M against CPU mode M, both with it on (the device has no switch).
A tile of a 4 x 4 grid fails when the two differ by more than 5 standard errors (from the seed
pairs, tile- and pixel-wise, the larger kept) AND 2 % (3 % device against host); the frame when
by more than 5 standard errors AND 1 %. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.45 -1.4  look_at 0 0.3 0  up 0 1 0  fov_y 32  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.5 }',
    'material "core"  { type diffuse  reflect 0.3 }',
    'material "fur"   { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'sphere "ball" { center 0 0.3 0  radius 0.15  material core }',
    'fur "coat" { on "ball"  material fur  count 20000  length 0.12  length_jitter 0.3  radius 0.0015  droop 0.1  jitter 0.25  seed 7 }',
    'light sun { dir 0.3 0.8 -0.5  angle 20  spd blackbody 5800  intensity 9.2649e-14 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_hair_nee_sun"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--photons", type=int, default=2000000)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "s.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)

    def render(tag, device, nee):
        """Two seeds of mode M; returns the pair of images."""
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            env = dict(os.environ)
            if not nee:
                env["FTRACE_HAIR_NEE"] = "0"
            cmd = [a.exe, "-in", "s.ftsl", "-mode", "M", "-device", device, "-spp", str(a.spp),
                   "-n", str(a.photons), "-seed", str(seed), "-hdr", "-no-meter", "-o", png]
            p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    on = render("cpu_on", "cpu", True)
    pairs = [("CPU fiber light sampling on / off", on, render("cpu_off", "cpu", False), 0.02)]
    if not a.no_gpu:
        pairs.append(("GPU / CPU", render("gpu_on", "gpu", True), on, 0.03))
    ok = True
    for name, x1, x0, rel in pairs:
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), rel)
        fr, fz, fbad = tile_verdict(tile_stats(*x1, 1), tile_stats(*x0, 1), 0.01)
        good = not bad.any() and not fbad.any()
        ok &= good
        i, j = np.unravel_index(int(np.argmax(z)), z.shape)
        print("%s: mode M, %s, frame x%.4f (%.1f sigma); worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, fr[0, 0], fz[0, 0], ratio[i, j], z[i, j]), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
