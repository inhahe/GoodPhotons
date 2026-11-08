"""Regression check: modes D and J light an unbounded sunlit haze as mode R does (HAZE-BLUR-DOMAIN).

    python tools/check_haze_domain.py [--out png/check_haze_domain] [--exe ftrace.exe]
                                      [--no-gpu] [--only still_D_cpu,...]

A white ball of radius 1 under `light sun` in `medium { sigma_t 0.04  albedo 0.5 }` with no
bounds, beside a 2x2 floor; and, held still, the same with a bounded cloud the sun's first leg
crosses, and with the haze a density field instead (that also held 32 m out, in mode D). Mode R's NEE leg dims the sun by the haze out
to the instant's sphere (Scene::mediaExitDistance: |centre - p| + sceneRadius); a light subpath
of modes D and J walks it from the entry disc's plane, never as far. Before the subpaths took the
NEE leg's haze (sunLegFactor, the user's option (a)), D read the ball 1.75 % bright still,
0.3 % with the ball held by a shutter at x = 6.4 outside the instant's sphere, and 2.75 % at
x = 32. Each case renders mode R and the mode under test, two seeds each, and requires the ball's
mean and the haze's (the top corners) to agree within 0.6 % (D) or 1.5 % (J, whose beam
merges are noisier), and within four of their standard errors from the seeds' spread, whichever
is wider.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
FIELD = ('medium { sigma_t 0.06  albedo 0.5  density "0.3 + 0.7*saturate(1.0 - y/8.0)"  '
         'density_max 1 }')


def scene(still, keys=8.0, at=6.4, medium="medium { sigma_t 0.04  albedo 0.5 }", extra=""):
    move = ("sphere { center %g 1 0  radius 1  material w }" % at) if still else (
        "group { motion { key { translate %g 0 0 } key { translate %g 0 0 } }  "
        "sphere { center 0 1 0  radius 1  material w } }" % (-keys, keys))
    shutter = "" if still else "shutter_open 0.4  shutter_close 0.4  "
    return NL.join([
        "scene { units meters%s }" % ("" if still else "  motion_interval -0.5 0.5"),
        'material "w" { type diffuse reflect 0.6 }',
        "light sun { dir 0.3 1 0.2  angle 0.53  spd preset:d65 }",
        medium,
        "quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }",
        move] + ([extra] if extra else []) + [
        'camera "c" { eye %g 1.5 6  look_at %g 1 0  up 0 1 0  fov_y 30  %sfilm { res 96 72 } }'
        % (at, at, shutter)]) + NL


def load(p):
    with open(p, "rb") as f:
        f.readline()
        w, h = (int(x) for x in f.readline().split())
        s = float(f.readline())
        return np.frombuffer(f.read(w * h * 12), dtype="<f4" if s < 0 else ">f4") \
                 .reshape(h, w, 3)[::-1].astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_haze_domain"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scenes = {"still": scene(True), "near": scene(False), "far": scene(False, 40.0, 32.0),
              # a bounded cloud the sun's first leg crosses inside the haze: its beams are clipped
              # to the cloud, the leg having begun outside it (BeamMis::leadIn)
              "cloud": scene(True, extra="medium { sigma_t 0.6  albedo 0.9  "
                                         "bounds { min 5.4 2.6 -1  max 7.4 3.4 1 } }"),
              # a density field with no bounds: the factor drawn by the medium's own estimator
              "field": scene(True, medium=FIELD),
              # ...and held 32 m out under a blur, where the walk began further out than the NEE
              # leg reaches: the factor's inverse, by quadrature (mode D only: J-FIELD-SLOW)
              "farfield": scene(False, 40.0, 32.0, medium=FIELD)}
    for name, text in scenes.items():
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    fails = []

    def render(name, mode, dev, seed):
        tag = "%s_%s_%s_s%d" % (name, mode, dev, seed)
        png = os.path.join(d, tag + ".png")
        spp = {"R": "512", "D": "512", "J": "1024"}[mode]
        if name == "field" and mode == "J":
            spp = "32"   # mode J in an unbounded density field runs ~8 s a sample here (J-FIELD-SLOW)
        for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", os.path.join(d, name + ".ftsl"), "-mode", mode, "-device", dev,
               "-spp", spp, "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1200:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return load(png[:-4] + ".pfm")[..., 1]

    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    only = set(x for x in a.only.split(",") if x)
    refs = {}
    for name in scenes:
        for dev in devs:
            for mode in ("D", "J"):
                case = "%s_%s_%s" % (name, mode, dev)
                if only and case not in only:
                    continue
                if name == "farfield" and mode == "J":
                    continue
                if name == "field" and mode == "J" and dev == "gpu":
                    continue   # ~10 min a sample on the GPU (J-FIELD-SLOW)
                key = (name, dev)
                if key not in refs:
                    refs[key] = [render(name, "R", dev, s) for s in (1, 2)]
                R = refs[key]
                X = [render(name, mode, dev, s) for s in (1, 2)]
                h, w = R[0].shape
                yy, xx = np.mgrid[0:h, 0:w]
                ball = (yy - h * 0.5) ** 2 + (xx - w * 0.5) ** 2 < (0.25 * h) ** 2
                haze = (yy < h * 0.15) & ((xx < w * 0.2) | (xx > w * 0.8))
                tol = 0.006 if mode == "D" else 0.015
                for region, m in (("ball", ball), ("haze", haze)):
                    r = [x[m].mean() for x in R]
                    v = [x[m].mean() for x in X]
                    ratio = np.mean(v) / np.mean(r)
                    # the ratio's standard error from the two seeds of each
                    se = np.hypot(abs(v[1] - v[0]) / 2 / np.mean(v), abs(r[1] - r[0]) / 2 / np.mean(r))
                    ok = abs(ratio - 1.0) < max(tol, 4.0 * se)
                    print("%-22s %-5s %s/R %.4f (+-%.4f)  %s" % (case, region, mode, ratio, se,
                                                              "ok" if ok else "FAIL"))
                    if not ok:
                        fails.append(case + " " + region)
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
