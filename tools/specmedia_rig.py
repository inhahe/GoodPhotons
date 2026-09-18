"""Spectral media, HETEROGENEOUS tier: the end-to-end number that was missing (TODO item 6).

    python tools/specmedia_rig.py [--keep] [--spp N] [--dspp N]

Result (0.344.0): **110.1 % channel spread against mode D -> 2.0 %**, flat control 3.3 %.

The three-tier spectral transmittance fix (v0.322.0) was validated end to end for the flat tier and
the homogeneous one -- coloured fog went from a 113 % channel spread against mode D to 1.0 %. The
COLOURED-AND-HETEROGENEOUS combination never got a number, because the scene it was attempted on ran
20-40 minutes and was stopped. That cost was never the spectral vector (measured: the same scene
with a FLAT spectrum, taking the scalar fast path, is equally slow) and it was not the density field
either -- it was the BEAM BUDGET, see BEAMS below. So what was missing was a CHEAP scene, not a
redesign.

This is that scene: a single smooth analytic blob -- `density` as a squared radial falloff, so the
majorant is tight and ratio tracking terminates fast -- in a small box in front of a lit diffuse
wall, at 120x96, measured in HDR so the ratio is scale-free. No VDB, no dense noise field, nothing
to load, and the beam budget cut to validation size.

The metric is the one the homogeneous case used: per-channel mode-M(`-beams`)-vs-mode-D ratios over
an ROI seen THROUGH the fog, and the SPREAD across channels, (max-min)/min. A spread near zero means
the three tiers agree with the reference; a large one is a wrong colour, not a brightness shift.

  0. CONTROL IS FLAT   -- the same scene with a flat `sigma_a` must already agree with mode D. This
                          is what makes any coloured result attributable to the spectral mechanism
                          rather than to the beam estimator's own bias.
  1. RIG CAN SEE IT    -- `FTRACE_SPECMEDIA=0` forces the scalar tier on a coloured medium, i.e. the
                          pre-0.322.0 bug. The spread must blow up. Without this an "after" number
                          from an instrument that cannot see the "before" proves nothing -- which is
                          the trap this file's own history is full of, and which this check caught
                          in the act: the rig's first scene (an emissive panel seen directly through
                          the fog) returned 5.1 % before AND after, because a directly-viewed
                          emitter samples one wavelength end to end and was never biased. The ROI
                          has to be a surface GATHERED FROM THE PHOTON MAP -- that is the only place
                          a camera segment's transmittance scales a sum over many wavelengths.
  2. FIXED             -- with the tier active the spread returns to the few-percent noise floor.
"""
import argparse
import io
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = os.path.join("scraps", "_specmedia")
EXE = os.path.join(ROOT, "ftrace.exe")

# A lit back wall, viewed through one soft blob. The blob is a squared radial falloff, so its
# majorant is tight (density <= 1, smooth) and the stochastic tier's null-collision count stays
# low -- that is the whole difference between this and the scene that had to be abandoned.
SCENE = """camera "cam" {{ eye 0 1.15 4.6  look_at 0 1.15 0  up 0 1 0  fov_y 42  mode R  film {{ res 120 96 }} }}
# A DIFFUSE wall, lit from outside the fog box, seen THROUGH the blob. The measured pixels must be
# gathered from the photon map, because that is where the bug lives: the camera segment's scalar
# transmittance at one wavelength multiplying a sum over photons of MANY wavelengths. A glowing
# panel seen directly through the same fog is NOT a test of it -- that path samples one wavelength
# end to end and was always unbiased -- which this rig discovered the hard way when its
# FTRACE_SPECMEDIA=0 control came back identical to the fixed build.
material "wall" {{ type diffuse  reflect 0.75 }}
quad {{ origin -2.4 0 -0.6  u 4.8 0 0  v 0 3.2 0  material wall }}
material "floor" {{ type diffuse  reflect 0.4 }}
quad {{ origin -2.4 0 -0.6  u 4.8 0 0  v 0 0 3.4  material floor }}
light area {{ origin -0.9 2.9 1.5  u 1.8 0 0  v 0 0 -1.2  normal 0 -1 0
              spd preset:d65  power 2400 }}
medium {{
    sigma_a {sa}   sigma_s 0.35   g 0.0
    bounds {{ min -1.3 0.25 0.1  max 1.3 2.35 1.95 }}
    density "pow( saturate( 1 - sqrt(x^2 + (y-1.25)^2 + (z-1.0)^2) / 1.15 ), 2 )"
}}
"""


# THE REASON THIS SCENE IS CHEAP AND THE OLD ONE WAS NOT. At the shipping defaults mode M builds
# ~1 M photon beams, splits them to 7.9 M, and a probe ray then gathers **4235** of them -- 20 s per
# spp at 200x160, which is the 20-40 minutes the earlier attempt hit and abandoned. Those are flyby
# budgets, not validation budgets. Cutting the photon and beam counts leaves 174 beams per probe and
# costs ~13 s/spp at this size, with no effect on the ratio being measured (the beam budget scales
# the volume estimator's noise, not its mean). The engine says so itself in its own log line:
# "lower -beamcount to get there for free".
BEAMS = ["-beams", "-n", "150000", "-beamcount", "40000", "-beamsplitmax", "60000000"]


def write(tag, sa):
    p = os.path.join(OUT, tag + ".ftsl")
    io.open(p, "w", encoding="utf-8", newline="\n").write(SCENE.format(sa=sa))
    return p


def render(scene, tag, mode, spp, spec_media=True, extra=None):
    out = os.path.join(OUT, tag + ".png")
    env = dict(os.environ)
    env["FTRACE_SPECMEDIA"] = "1" if spec_media else "0"
    cmd = [EXE, "-in", scene, "-mode", mode, "-device", "cpu", "-r", "120", "96",
           "-spp", str(spp), "-hdr", "-o", out, "-window-min", "-interval", "60"] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if not os.path.exists(out):
        raise RuntimeError("render failed for %s:\n%s" % (tag, (r.stdout + r.stderr)[-1500:]))
    return out


def channels(png):
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from grade_hdr import read_pfm
    a = np.asarray(read_pfm(png[:-4] + ".pfm")).astype(np.float64)
    h, w = a.shape[:2]
    # the wall seen THROUGH the thickest part of the blob
    roi = a[int(h * 0.30):int(h * 0.62), int(w * 0.34):int(w * 0.66)]
    return roi.reshape(-1, 3).mean(axis=0)


def spread(m, d):
    r = [m[i] / max(d[i], 1e-9) for i in range(3)]
    return r, (max(r) - min(r)) / max(min(r), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--dspp", type=int, default=512)
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-14s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    cases = [("flat", "1.05"), ("coloured", "rgb 0.25 1.0 2.6")]
    res = {}
    for tag, sa in cases:
        sc = write(tag, sa)
        d = channels(render(sc, tag + "_D", "D", args.dspp))
        m = channels(render(sc, tag + "_M", "M", args.spp, True, BEAMS))
        r, sp = spread(m, d)
        res[tag] = (d, m, r, sp)
        print("  %-9s  mode D %6.2f %6.2f %6.2f | mode M %6.2f %6.2f %6.2f | ratio %.4f %.4f %.4f  spread %5.1f %%"
              % (tag, d[0], d[1], d[2], m[0], m[1], m[2], r[0], r[1], r[2], 100 * sp))
    # the control: forced scalar tier on the coloured medium == the pre-0.322.0 bug
    sc = os.path.join(OUT, "coloured.ftsl")
    dC = res["coloured"][0]
    mB = channels(render(sc, "coloured_M_before", "M", args.spp, False, BEAMS))
    rB, spB = spread(mB, dC)
    print("  %-9s  mode D %6.2f %6.2f %6.2f | mode M %6.2f %6.2f %6.2f | ratio %.4f %.4f %.4f  spread %5.1f %%"
          % ("BEFORE", dC[0], dC[1], dC[2], mB[0], mB[1], mB[2], rB[0], rB[1], rB[2], 100 * spB))

    check("control flat", res["flat"][3] < 0.06,
          "a flat sigma_a already agrees with mode D to %.1f %% -- so a coloured result is the spectral mechanism" % (100 * res["flat"][3]))
    check("rig sees it", spB > 0.20,
          "FTRACE_SPECMEDIA=0 (the pre-0.322.0 scalar tier) gives %.1f %% spread" % (100 * spB))
    check("tier fixed", res["coloured"][3] < 0.08,
          "with the three tiers active: %.1f %% (from %.1f %%)" % (100 * res["coloured"][3], 100 * spB))

    print("\n-> spectral media (heterogeneous) rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep and ok:
        shutil.rmtree(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
