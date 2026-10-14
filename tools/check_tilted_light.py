"""Regression check: a quad light whose authored `normal` leaves its plane lights fog and a floor as
the same light with its normal in the plane does.

    D:/python314/python.exe -u tools/check_tilted_light.py [--out <dir>] [--exe ftrace.exe]
                                                           [--spp N] [--no-gpu]

A light's `normal` decides which side emits; the solid angle the patch subtends belongs to the
patch's own orientation (Emitter::nGeom), whatever the normal says. Two scenes, identical but for
the light's `normal` -- cross(u, v), or tilted 45 degrees out of the plane to the same side -- each
a 1x1 light above a box of fog over a floor. Every point of the fog and the floor sees the whole
light inside both hemispheres, so the two must render the same: mode R, host and (unless
--no-gpu) device, tile by tile. Until 0.381.1 the device's fog light sampling (bkNeeVolumeHero)
measured the tilted light by its authored normal, x0.71 straight below it. Fails when a tile
differs by more than 5 standard errors (two seeds each) AND by more than 2 %. Exits 1 on failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)


def scene(normal):
    return [
        'camera "c" { eye 0 0.4 -3  look_at 0 0.3 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
        'material "floor" { type diffuse  reflect 0.6 }',
        'quad { origin -0.6 0 -0.6  u 0 0 1.2  v 1.2 0 0  material floor }',
        'medium { sigma_t 0.8  albedo 0.8  g 0.2  bounds { min -0.6 0 -0.6  max 0.6 0.6 0.6 } }',
        'light area { origin -0.5 2 -0.5  u 1 0 0  v 0 0 1  normal %s  power 40  '
        'spd preset:d65 }' % normal,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_tilted_light"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    for name, normal in (("flat", "0 -1 0"), ("tilted", "0.70710678 -0.70710678 0")):
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(scene(normal)) + NL)
    ok = True
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        g = "g" if dev == "gpu" else ""
        flat = render_seeds(a.exe, d, "flat.ftsl", "flat" + g, ["-mode", "R"], dev, a.spp)
        tilt = render_seeds(a.exe, d, "tilted.ftsl", "tilted" + g, ["-mode", "R"], dev, a.spp)
        ratio, z, bad = tile_verdict(tile_stats(*tilt), tile_stats(*flat), 0.02)
        good = not bad.any()
        ok &= good
        frame = (tilt[0].sum() + tilt[1].sum()) / max(flat[0].sum() + flat[1].sum(), 1e-30)
        worst = "worst tile x%.4f (%.1f sigma)" % (ratio.flat[int(z.argmax())], z.max()) \
            if z.max() > 0.0 else "every tile identical"
        print("%s: mode R%s, tilted / flat normal: frame x%.4f; %s" % (
            "ok" if good else "FAIL", " (device)" if dev == "gpu" else "", frame, worst),
            flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
