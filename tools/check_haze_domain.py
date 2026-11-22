"""Regression check: modes D and J light an unbounded sunlit haze as mode R does (HAZE-BLUR-DOMAIN).

    python tools/check_haze_domain.py [--out png/check_haze_domain] [--exe ftrace.exe]
                                      [--no-gpu] [--only still_D_cpu,...]

A white ball of radius 1 under `light sun` in `medium { sigma_t 0.04  albedo 0.5 }` with no
bounds, beside a 2x2 floor; and, held still, the same with a bounded cloud the sun's first leg
crosses, and with the haze a density field instead (that also held 32 m out under a blur). Mode
R's NEE leg dims the sun by the haze out to the instant's
sphere (Scene::mediaExitDistance: |centre - p| + sceneRadius); a light subpath of modes D and J
walks it from the entry disc's plane, never as far. Before the subpaths took the NEE leg's haze
(sunLegFactor, the user's option (a)), D read the ball 1.75 % bright still, 0.3 % with the ball
held by a shutter at x = 6.4 outside the instant's sphere, and 2.75 % at x = 32.

`stilldom` is `still` in mode J with its beam domain shrunk to half the scene's radius
(FTRACE_JBEAMDOMAIN=0.5; Scene::jBeamMerges), the ball and nearly all the haze outside it: the
beams end there, and the MIS weights must leave the merges out past it, the connections rendering
the haze alone. A weight that still counted them read dark by their share (J-BEAM-FARLIMIT).

Each case renders mode R and the mode under test, two seeds each (mode J in `farfield` six), and requires the ball's mean and
the haze's (the top corners) to agree within 0.6 % (D) or 1.5 % (J, whose beam merges are
noisier), or within four of their standard errors from the seeds' spread, whichever is wider.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from pfm import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
FIELD = ('medium { sigma_t 0.06  albedo 0.5  density "0.3 + 0.7*saturate(1.0 - y/8.0)"  '
         'density_max 1 }')
# Per-case environment, and the case whose mode R renders stand for its own (R ignores ENV).
ENV = {"stilldom": {"FTRACE_JBEAMDOMAIN": "0.5"}}
REF_OF = {"stilldom": "still"}
# Seeds of each mode a case renders, and how many of the ratio's standard errors it may stray. Two
# seeds give the error one degree of freedom, a statistic as heavy-tailed as Cauchy's: harmless
# while every case's noise is under the tolerance, which then decides. A case noisier than that
# needs more seeds and a bound to match (six and 5: P(|t5| > 5) = 0.4 % a region).
SEEDS, Q = 2, 4.0
# Such a case: mode J in the blurred density field, each seed's own -jbeams auto knee and beam map,
# whose haze reads +-1.5 % from seed to seed (seeds 1 and 2 of 0.451.13 agreed to 0.2 % and read
# 0.983, seeds 3 and 4 1.026 and 0.994): six seeds and Q 5.
NOISY = {"farfield_J": (6, 5.0)}


def scene(still, keys=8.0, at=6.4, medium="medium { sigma_t 0.04  albedo 0.5 }", extra="",
          res=(96, 72)):
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
        'camera "c" { eye %g 1.5 6  look_at %g 1 0  up 0 1 0  fov_y 30  %sfilm { res %d %d } }'
        % (at, at, shutter, res[0], res[1])]) + NL


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
              # leg reaches: the factor's inverse, by quadrature (HAZE-BLUR-INVERSE-TR)
              "farfield": scene(False, 40.0, 32.0, medium=FIELD),
              # `still` with mode J's beam domain shrunk inside the scene (ENV); R is still's
              "stilldom": scene(True)}
    for name, text in scenes.items():
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    fails = []

    def spp_for(name, mode, dev):
        """Samples per pixel for one case, or None to skip it. Mode J in the blurred density
        field, its beam domain holding the 40 m sweep, is the slow one: ~0.6 s a sample on the GPU
        and 0.5 s on the CPU (0.434.40), so 256. `stilldom` exists for mode J alone."""
        if mode == "R":
            return 512
        if name == "stilldom" and mode != "J":
            return None
        if name == "farfield" and mode == "J":
            return 256
        return 512 if mode == "D" else 1024

    def render(name, mode, dev, seed):
        name = REF_OF.get(name, name) if mode == "R" else name   # R ignores ENV: share it
        tag = "%s_%s_%s_s%d" % (name, mode, dev, seed)
        png = os.path.join(d, tag + ".png")
        spp = str(spp_for(name, mode, dev))
        for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
            if os.path.exists(stale):
                os.remove(stale)
        # Mode J at -jbeams auto: the knee-sized map, whose merges carry the haze -- the sparse
        # fixed default (0.435.0) would leave the merge weights under test a small share
        # (`stilldom`'s proof, 0.434.40, was made at this map). Modes R and D ignore it.
        cmd = [exe, "-in", os.path.join(d, name + ".ftsl"), "-mode", mode, "-device", dev,
               "-jbeams", "auto",
               "-spp", spp, "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, **ENV.get(name, {})))
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1200:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return read_pfm(png[:-4] + ".pfm", top_down=True)[..., 1]

    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    only = set(x for x in a.only.split(",") if x)
    refs = {}
    for name in scenes:
        for dev in devs:
            for mode in ("D", "J"):
                case = "%s_%s_%s" % (name, mode, dev)
                if (only and case not in only) or spp_for(name, mode, dev) is None:
                    continue
                key = (REF_OF.get(name, name), dev)
                if key not in refs:
                    refs[key] = [render(name, "R", dev, s) for s in range(1, SEEDS + 1)]
                R = refs[key]
                nSeeds, q = NOISY.get("%s_%s" % (name, mode), (SEEDS, Q))
                X = [render(name, mode, dev, s) for s in range(1, nSeeds + 1)]
                h, w = R[0].shape
                yy, xx = np.mgrid[0:h, 0:w]
                ball = (yy - h * 0.5) ** 2 + (xx - w * 0.5) ** 2 < (0.25 * h) ** 2
                haze = (yy < h * 0.15) & ((xx < w * 0.2) | (xx > w * 0.8))
                tol = 0.006 if mode == "D" else 0.015
                for region, m in (("ball", ball), ("haze", haze)):
                    r = [x[m].mean() for x in R]
                    v = [x[m].mean() for x in X]
                    ratio = np.mean(v) / np.mean(r)
                    # The ratio's standard error from the seeds' spread (SEEDS, Q; NOISY's).
                    se = np.hypot(np.std(v, ddof=1) / np.sqrt(len(v)) / np.mean(v),
                                  np.std(r, ddof=1) / np.sqrt(len(r)) / np.mean(r))
                    ok = abs(ratio - 1.0) < max(tol, q * se)
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
