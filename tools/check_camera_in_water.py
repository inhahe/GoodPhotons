"""Regression check: a camera inside a dielectric sees through it, and every connection to it is absorbed.

    python tools/check_camera_in_water.py [--out png/check_camera_in_water] [--exe ftrace.exe] [--no-gpu]

CAMERA-EMPTY-STACK. A camera walk keeps the dielectric solids it is inside on a stack, and one from a
camera inside a solid must start with it there (Camera::enclosure, Scene::eyeStack); every connection
-- a light sample, a forward splat to the camera, a BDPT or VCM connection -- runs inside one medium
and must take its absorption (legAbsorption). `scenes/_camera_in_water.ftsl` is a white room under
water absorbing 0.3 per metre at every wavelength, the camera and a lamp in it, the lamp facing the
camera 2.8 m away. Two parts:
  * the closed form: mode R, CPU and GPU, reads the lamp's middle exp(-0.3 * 2.8) = 0.4317 of the
    same room dry, within 1 %;
  * the modes: B, D, J, U, M and S, CPU and GPU, against mode R on the GPU, the walls' mean and the
    lamp's within 3 % and 5 standard errors over the seeds.
Without the camera's stack mode R reads the lamp as if dry; without the legs' absorption the light
side's splats and connections read the walls too bright. Exits 1 on any failure.
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

SCENE = os.path.join(ROOT, "scenes", "_camera_in_water.ftsl")
DEPTH = ["-max-bounce", "12"]
REF = ["-mode", "R", "-device", "gpu", "-spp", "4096"]
CASES = [("B", ["-n", "60000000"]), ("D", ["-spp", "256"]), ("J", ["-spp", "128"]), ("U", ["-spp", "256"]),
         ("M", ["-spp", "64", "-n", "2000000"]), ("S", ["-spp", "256", "-n", "200000"])]
SEEDS = (1, 2)
# Mode S's gather blurs neighbouring pixels together, so two seeds' scatter is one degree of freedom
# under-estimating its real seed-to-seed spread (check_rough_irid's reason): eight.
SEEDS_OF = {"S": tuple(range(1, 9))}
TOL = 0.03
EXPECT = math.exp(-0.3 * 2.8)
# The lamp, centred in the 96 x 72 frame (the camera looks at its middle), about 4 pixels in radius.
_Y, _X = np.mgrid[0:72, 0:96]
_R2 = (_X - 47.5) ** 2 + (_Y - 35.5) ** 2
LAMP = _R2 <= 2.5 ** 2
WALLS = _R2 >= 7.0 ** 2


def render(exe, out, scene, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scene, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-o", png] + DEPTH + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    img = read_pfm(pfm, top_down=True)[..., 1]
    return np.array([img[WALLS].mean(), img[LAMP].mean()])


def stats(exe, out, scene, tag, args, seeds=SEEDS):
    v = np.array([render(exe, out, scene, tag, args, s) for s in seeds])
    return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(seeds))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_camera_in_water"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    dry = os.path.join(out, "dry.ftsl")
    with open(SCENE, encoding="utf-8") as f, open(dry, "w", encoding="utf-8", newline="") as g:
        g.write("".join(l for l in f if "material water" not in l))
    ok = True
    for dev in devs:
        # One wavelength a path in both (a bundle under water is one), so wet and dry draw alike and
        # the ratio is the absorption's alone, not two renders' colour noise.
        r = ["-mode", "R", "-device", dev, "-spp", "1024", "-heroc", "1"]
        wet = render(exe, out, SCENE, "R_%s_wet" % dev, r, 1)[1]
        dr = render(exe, out, dry, "R_%s_dry" % dev, r, 1)[1]
        q = wet / dr
        good = abs(q / EXPECT - 1.0) <= 0.01
        ok = ok and good
        print("%s: mode R on the %s, the lamp under water: %.4f of it dry (exp(-0.84) = %.4f)"
              % ("ok" if good else "FAIL", dev.upper(), q, EXPECT), flush=True)
    ref = REF if not a.no_gpu else ["-mode", "R", "-device", "cpu", "-spp", "2048"]
    rm, rse = stats(exe, out, SCENE, "R_ref", ref)
    print("mode R: walls %.6f, lamp %.6f" % (rm[0], rm[1]), flush=True)
    for mode, args in CASES:
        for dev in devs:
            m, se = stats(exe, out, SCENE, "%s_%s" % (mode, dev), ["-mode", mode, "-device", dev] + args,
                          SEEDS_OF.get(mode, SEEDS))
            q = m / rm
            err = np.hypot(se / rm, rse * m / (rm * rm))
            bad = (np.abs(q - 1.0) > TOL) & (np.abs(q - 1.0) > 5.0 * err)
            ok = ok and not bad.any()
            print("%s: mode %s on the %s: walls %.4f (+- %.4f), lamp %.4f (+- %.4f) of mode R"
                  % ("FAIL" if bad.any() else "ok", mode, dev.upper(), q[0], err[0], q[1], err[1]), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
