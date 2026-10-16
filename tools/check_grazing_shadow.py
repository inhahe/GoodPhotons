"""Regression check: a shadow ray aimed at a sampled light point must not strike that light.

    D:/python314/python.exe -u tools/check_grazing_shadow.py [--out <dir>] [--exe ftrace.exe]
                                                             [--spp N] [--no-gpu]

A connection's shadow ray starts a hair off the surface it leaves (1e-6 along its normal), so it
cannot hit that surface. Aimed along the direction from the UNMOVED point, the moved ray ends up
to the move over cos(a) past the far end's surface, a being the angle it arrives at, and once
that exceeded the ray's far-end shortening, a grazing connection struck the very light it was
sampling and counted as shadowed. Modes D and U lost the floor that way more than 3 m from a
1 m-high quad light, down to 0.1 % of its light in the far corner, on the host only: the device
already aimed from the moved origin (render_cuda.cu, occludedTo). Mode R's relative shortening
covered all but a grazing sliver. Since 0.384.1 every host connection that ends on a sampled
point is aimed from the moved origin (Scene::occludedTo).

The scene is check_tilted_light.py's floor with the light's normal straight down: a 0.2 x 0.2
light 1 m above a diffuse floor that runs out to 3.6 m from it, seen from above. Each render is
compared, tile by tile, with the floor's direct irradiance integrated over the light, scaled to
the render over the floor within 2 m of the light; a tile of a 4 x 4 grid fails when the two
differ by more than 5 standard errors (two seeds) AND 2 %. Modes R, D and U on the host and
(unless --no-gpu) the device. Exits 1 on any failure.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402
import check_tilted_light as ctl  # noqa: E402

NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_grazing_shadow"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "floor.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(ctl.floor_scene("%g %g %g" % tuple(ctl.FACE))) + NL)

    p, blocked = ctl.floor_points()
    keep = ~blocked
    ef = ctl.irradiance(p, lambda wd: np.maximum(wd @ ctl.FACE, 0.0)) * keep
    centre = ctl.LIGHT_O + (ctl.LIGHT_U + ctl.LIGHT_V) / 2
    reach = np.sqrt(((p - centre) ** 2).sum(-1))
    near = keep & (reach < 2.0)
    ok = True
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        for m in "RDU":
            label = "mode %s, %s" % (m, "device" if dev == "gpu" else "host")
            imgs = [x * keep for x in render_seeds(a.exe, d, "floor.ftsl", "%s_%s" % (m, dev),
                                                   ["-mode", m], dev, a.spp)]
            s = sum(x[near].sum() for x in imgs) / (2.0 * ef[near].sum())
            r, z, bad = tile_verdict(tile_stats(*imgs), tile_stats(ef * s, ef * s), 0.02)
            i, j = np.unravel_index(int(np.argmax(z)), z.shape)
            good = not bad.any()
            ok &= good
            print("%s: %s, floor / analytic irradiance: lowest tile x%.4f; worst x%.4f "
                  "(%.1f sigma)" % ("ok" if good else "FAIL", label, r.min(), r[i, j], z[i, j]),
                  flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
