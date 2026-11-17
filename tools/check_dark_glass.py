"""Regression check: modes M and S see through glass that absorbs some wavelengths almost entirely.

    python tools/check_dark_glass.py [--out png/check_dark_glass] [--exe ftrace.exe]

A camera looks through a 0.5 m slab of glass that absorbs blue at 200 per metre (it passes red) at
a lit floor. A walk whose own wavelength is blue arrives with a throughput near exp(-100), while its
spectral carrier (SpecThr, mode M's SPECGATHER; mode S's per-pixel copy) re-expresses it at the red
photons it gathers by a ratio near exp(+100): finite in double, past a float's range (3.4e+38). Every
gather that stored or summed that ratio in float made it infinite, and the conversion to RGB made
the pixel NaN (known-issues MODEM-COMPOTE-NAN: GPU mode M on gallery_rain's gem compote). They now
take ratio / peak in float and fold the peak into the double throughput (SpecThr::peak).

Modes M and S on the CPU and the GPU must write no non-finite pixel, and the floor seen through the
glass must read mode R per colour channel, over six seeds against a reference over four: within 6 %
or 5 standard errors -- of the channel, or of a tenth of the three together where the channel is
smaller (the glass leaves little blue to measure). Both sides are noisy here: one seed of mode M
spreads some 10 % a channel, and mode R, which reaches the floor through the glass only by its walk
(no light sample crosses a dielectric), 2-4 %. On 0.433.0 GPU mode M wrote 2433 NaN pixels and mode S
2933 on the GPU and 2037 on the CPU. Exits 0 when every case holds, 1 otherwise.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_motion as cm  # noqa: E402
from check_gather_base import box  # noqa: E402

NL = chr(10)
SCENE = NL.join([
    "scene { units meters }",
    'camera "c" { eye 0 1.2 -1.6  look_at 0 0 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
    'material "floor" { type diffuse reflect 0.6 }',
    'material "dark"  { type dielectric  ior 1.5  absorb rgb 0 0 200 }',
    "quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material floor }",
    "light area { origin -0.5 2.5 -0.5  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 400  spd preset:d65 }",
    # the slab, between the camera and the floor: a closed box of six quads, normals out (a
    # dielectric made of quads reads entering and leaving off them)
] + box(-0.6, 0.3, -0.6, 0.6, 0.8, 0.6, "dark")) + NL
REF = ["-mode", "R", "-device", "gpu", "-spp", "1024"]
CASES = [("M_gpu", ["-mode", "M", "-device", "gpu", "-n", "8000000", "-spp", "32"]),
         ("M_cpu", ["-mode", "M", "-device", "cpu", "-n", "8000000", "-spp", "16"]),
         ("S_gpu", ["-mode", "S", "-device", "gpu", "-spp", "64"]),
         ("S_cpu", ["-mode", "S", "-device", "cpu", "-spp", "16"])]
TOL = 0.06
SEEDS, REF_SEEDS = tuple(range(1, 7)), (1, 2, 3, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_dark_glass"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "dark.ftsl")
    cm.write(path, SCENE)

    def image(tag, args, gpu, seeds):
        return [cm.render(exe, out, path, "%s_%d" % (tag, s), args, seed=s, want_gpu=gpu)[0] for s in seeds]

    # the floor seen through the slab: the middle of the frame, where the camera looks down through it
    win = (slice(18, 34), slice(22, 42))

    def chans(ims):
        v = np.array([[np.nan_to_num(im)[win][..., c].mean() for c in range(3)] for im in ims])
        return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))

    rc, rse = chans(image("R_gpu", REF, True, REF_SEEDS))
    floor = 0.1 * rc.sum()
    ok = True
    for name, args in CASES:
        ims = image(name, args, "gpu" in name, SEEDS)
        bad = int(sum((~np.isfinite(im)).any(axis=-1).sum() for im in ims))
        mc, mse = chans(ims)
        q = mc / rc
        err = np.hypot(mse, rse * mc / rc)          # of the difference, in the channel's units
        held = [abs(mc[c] - rc[c]) <= max(TOL * max(rc[c], floor), 5.0 * err[c]) for c in range(3)]
        good = bad == 0 and all(held)
        ok &= good
        print("%-6s non-finite px %d; through the glass, of mode R: %.3f %.3f %.3f  %s" % (
            name, bad, q[0], q[1], q[2], "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
