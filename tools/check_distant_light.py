#!/usr/bin/env python3
"""Device-parity gate for DISTANT lights (known-issues.md: GPU-NEE-EPS).

The GPU build is float32 and the CPU build is double. That is normally a sub-percent
difference, but shadow rays that must stop *short* of a sampled emitter point were
shortened by a hard-coded absolute epsilon, and an absolute epsilon is meaningless once
it drops below one ulp of the distance it is subtracted from::

    RAY_EPS = 1e-4f, shortening = 2e-4
    one ulp of a float at magnitude d  =  d * 2^-23
    the subtraction rounds back to d   when  2e-4 < 0.5 ulp,  i.e.  d > ~3360

Past that the ray reached the emitter point exactly, re-hit the emitter it had just been
sampled from, and the sample was thrown away as occluded -- so the GPU rendered a distant
light ~24% too dark. It went unnoticed for six weeks because it was mis-filed as a
participating-media bug (it reproduces with no medium at all) and because no scene in the
repo put a light thousands of units from its geometry. This gate is that missing scene.

It renders three scenes:

  * ``scenes/_distant_light.ftsl``      -- light ~6325 units away (the regime GPU-NEE-EPS broke)
  * ``scenes/_distant_light_near.ftsl`` -- the same scene scaled by 0.1 (the control)
  * ``scenes/_distant_geometry.ftsl``   -- the near scene translated to |p| ~ 5000: quad,
    camera and light all far from the origin, with the light NEARER than the shading
    point's coordinate magnitude (the regime GPU-ORIGIN-EPS broke, twice -- first the
    absolute origin offset rounding away, then un-re-aimed shadow rays overshooting the
    sampled emitter point after a scale-correct push)

and compares ``-device gpu`` against ``-device cpu`` for each. Comparing the two devices
rather than against a stored golden image is deliberate: the double CPU build is the
reference, so the gate keeps working when sampling changes legitimately alter both.

The control matters. If all three fail, something generic is broken (the device build, the
emitter sampler). If only the far-light scene fails, a far-end shortening has gone absolute
again (``connMaxT``). If only the far-geometry scene fails, a ray ORIGIN has gone absolute
again, or an occlusion query is no longer re-aimed from its moved origin (``dOffsetAlong`` /
``occludedTo``). A single scene could not distinguish these.

Usage::

    python tools/check_distant_light.py [--time SEC] [--tol PCT] [--mode M]

Exit status is non-zero if either half exceeds the tolerance, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTRACE = os.path.join(_REPO, "ftrace.exe")
_OUT = os.path.join(_REPO, "png")

# (scene, label, why it is here)
_PAIR = (
    ("_distant_light",      "far light (~6325 units)",  "the regime GPU-NEE-EPS broke"),
    ("_distant_light_near", "near      (~632 units)",   "control: same image, smaller floats"),
    ("_distant_geometry",   "far geometry (|p|~5000)",  "the regime GPU-ORIGIN-EPS broke, twice"),
)


def read_pfm(path):
    """Minimal PFM reader. The repo writes scene-linear XYZ, pre-tone-map."""
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        if hdr not in (b"PF", b"Pf"):
            raise ValueError("not a PFM: %r" % hdr)
        nc = 3 if hdr == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        scale = float(f.readline().strip())
        n = w * h * nc
        data = struct.unpack("%s%df" % ("<" if scale < 0 else ">", n), f.read(4 * n))
    return w, h, nc, data


def luminance(data, nc):
    """Channel 1 = Y. A shift in Y is a real brightness error; one confined to X or Z
    would be chromatic, which this bug is not (it measured -23.68/-23.69/-23.68)."""
    if nc == 1:
        return list(data)
    return [data[i + 1] for i in range(0, len(data), nc)]


def render(scene, device, mode, secs):
    out = os.path.join(_OUT, "chkdl_%s_%s.png" % (scene, device))
    cmd = [_FTRACE, "-in", os.path.join(_REPO, "scenes", scene + ".ftsl"),
           "-mode", mode, "-device", device, "-time", str(secs), "-hdr",
           "-o", out, "-window-min", "-interval", str(secs)]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise SystemExit("render failed: %s on %s (exit %d)" % (scene, device, r.returncode))
    return out[:-4] + ".pfm"


def median_ratio(ref_pfm, test_pfm):
    wr, hr, cr, dr = read_pfm(ref_pfm)
    wt, ht, ct, dt = read_pfm(test_pfm)
    if (wr, hr) != (wt, ht):
        raise SystemExit("size mismatch: %dx%d vs %dx%d" % (wr, hr, wt, ht))
    yr, yt = luminance(dr, cr), luminance(dt, ct)
    # Only compare where the reference carries real signal: a ratio against near-zero is
    # dominated by the reference's own noise and would swamp the median with meaningless
    # outliers.
    ys = sorted(v for v in yr if v > 0.0)
    if not ys:
        raise SystemExit("reference %s is entirely zero" % ref_pfm)
    floor = ys[len(ys) // 2] * 1e-3
    ratios = sorted(t / r for r, t in zip(yr, yt) if r > floor)
    if not ratios:
        raise SystemExit("no pixels above the signal floor in %s" % ref_pfm)
    # A median, not a mean: forward/bidirectional modes are firefly-prone and a handful
    # of outliers move a mean by several percent on their own.
    return ratios[len(ratios) // 2], len(ratios)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", type=int, default=25, help="seconds per render (default 25)")
    ap.add_argument("--tol", type=float, default=3.0,
                    help="max |median GPU/CPU - 1| in percent (default 3)")
    ap.add_argument("--mode", default="R", help="render mode (default R, the backward reference)")
    a = ap.parse_args()

    if not os.path.exists(_FTRACE):
        raise SystemExit("no ftrace.exe at %s -- build first" % _FTRACE)
    os.makedirs(_OUT, exist_ok=True)

    print("GPU-vs-CPU parity for distant lights (mode %s, %ds/render, tol +-%.1f%%)"
          % (a.mode, a.time, a.tol))
    print()
    print("%-26s %10s %8s   %s" % ("scene", "GPU/CPU", "verdict", "role"))
    bad = 0
    for scene, label, why in _PAIR:
        cpu = render(scene, "cpu", a.mode, a.time)
        gpu = render(scene, "gpu", a.mode, a.time)
        ratio, npx = median_ratio(cpu, gpu)
        off = 100.0 * (ratio - 1.0)
        ok = abs(off) <= a.tol
        bad += not ok
        print("%-26s %+9.2f%% %8s   %s" % (label, off, "ok" if ok else "FAIL", why))

    print()
    if bad:
        print("FAILED: the float32 device disagrees with the double CPU reference.")
        print("Only the far-LIGHT scene: a shadow-ray end-shortening has gone absolute again")
        print("  (connMaxT, known-issues GPU-NEE-EPS).")
        print("Only the far-GEOMETRY scene: a ray origin has gone absolute again, or an")
        print("  occlusion query is no longer re-aimed from its moved origin (dOffsetAlong /")
        print("  occludedTo, known-issues GPU-ORIGIN-EPS).")
        print("All three: suspect something generic (the device build, the emitter sampler).")
    else:
        print("PASS: all three magnitude regimes agree across devices.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
