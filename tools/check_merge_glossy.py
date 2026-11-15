"""Regression check: modes J's and U's merges on a concave glossy surface read as mode R does.

    python tools/check_merge_glossy.py [--out png/check_merge_glossy] [--exe ftrace.exe] [--no-gpu]

A merge divides by the share of its disc its own surface covers (MergeFootprint), measured by probe
rays that stop on the faces merge vertices lie on (holdsMergeVertex) -- a glossy face among them, which
holds no photon of mode M's. `scenes/_merge_glossy_bowl.ftsl` is a room inside a rough glossy sphere
(radius 0.3, roughness 1.2) lit by a small lamp: at the pinned merge radius of 0.05 every merge's
surface rises above its tangent plane by up to r^2 / 2R, more than the probe's 1e-3 r, so a probe that
passed through glossy faces would find nothing and read the merge up to 9 times too bright (J and U
read 8.0 times mode R so). Modes J and U, CPU and GPU, against mode R on the GPU: the frame's mean
within 3 % and 5 standard errors over the seeds, every mode at one depth (`-max-bounce 24`: the
walls reflect 0.7, so J's and U's default of 8 edges would leave 6 % of the light out, R's 32
none). Exits 1 on any failure.
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

SCENE = os.path.join(ROOT, "scenes", "_merge_glossy_bowl.ftsl")
DEPTH = ["-max-bounce", "24"]
REF = ["-mode", "R", "-device", "gpu", "-spp", "4096"]
CASES = [("J", "cpu", ["-spp", "128"]), ("U", "cpu", ["-spp", "128"]),
         ("J", "gpu", ["-spp", "512"]), ("U", "gpu", ["-spp", "128"])]
SEEDS = (1, 2, 3)
TOL = 0.03


def render(exe, out, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", SCENE, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-o", png] + DEPTH + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return float(read_pfm(pfm, top_down=True)[..., 1].mean())


def stats(exe, out, tag, args):
    v = np.array([render(exe, out, tag, args, s) for s in SEEDS])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(SEEDS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_merge_glossy"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ref = REF if not a.no_gpu else ["-mode", "R", "-device", "cpu", "-spp", "2048"]
    rm, rse = stats(exe, out, "R_ref", ref)
    print("mode R: %.6f" % rm, flush=True)
    ok = True
    for mode, dev, args in CASES:
        if a.no_gpu and dev == "gpu":
            continue
        m, se = stats(exe, out, "%s_%s" % (mode, dev),
                      ["-mode", mode, "-device", dev, "-pmradius", "0.05"] + args)
        q, err = m / rm, np.hypot(se / rm, rse * m / (rm * rm))
        bad = abs(q - 1.0) > TOL and abs(q - 1.0) > 5.0 * err
        ok = ok and not bad
        print("%s: mode %s on the %s: %.4f of mode R (+- %.4f)"
              % ("FAIL" if bad else "ok", mode, "GPU" if dev == "gpu" else "CPU", q, err), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
