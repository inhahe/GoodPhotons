"""Regression check: mode M's footprint correction does not stop on glass above the surface it gathers on.

    D:/python314/python.exe -u tools/check_gather_glass.py [--out <dir>] [--exe ftrace.exe]
                                                          [--spp N] [--no-gpu]

Mode M divides each photon-map estimate by the fraction of its gather disc that is surface
(M-GATHERAREA), measured by probes dropped from the disc's radius above the tangent plane. A probe
that met a transparent face first -- glass a centimetre above the floor, closer than the radius --
counted that face and then rejected it for lying above the plane, so the coverage read low and the
estimate high, though no photon is ever stored on glass and the floor beneath holds them all (known-
issues GATHERAREA-TRANSPARENT). The probes now pass through such faces.

The scene: a floor lit only by the bounce off a white wall (the lamp faces the wall), a glass slab
standing 1 cm above the floor, and the camera looking down through the slab -- so the floor under it
is main-map light (lamp, wall, glass, floor: no caustic-map path), gathered at a radius above the
gap. Mode M (host, and device unless --no-gpu) against mode R, per tile, two seeds each; a tile
fails when they differ by more than 5 standard errors AND by more than 5 %. Exits 1 on failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402
from check_radiance_eta2 import box  # noqa: E402

NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.9 -1.0  look_at 0 0 0.05  up 0 1 0  fov_y 34  film { res 64 48 } }',
    'material "white" { type diffuse  reflect 0.8 }',
    'material "glass" { type dielectric  ior 1.5 }',
    'quad { origin -2 0 -2   u 0 0 4   v 4 0 0   material white }',
    'quad { origin -2 0 0.8  u 4 0 0   v 0 2 0   material white }',
    'light area { origin -0.4 0.4 0.2  u 0.8 0 0  v 0 0.8 0  normal 0 0 1  power 200  '
    'spd preset:d65 }',
] + box(-0.35, 0.01, -0.35, 0.35, 0.03, 0.35, "glass")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gather_glass"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "slab.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    ref = render_seeds(a.exe, d, "slab.ftsl", "R", ["-mode", "R"], "cpu", a.spp)
    ok = True
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        m = render_seeds(a.exe, d, "slab.ftsl", "M" + ("g" if dev == "gpu" else ""),
                         ["-mode", "M", "-n", "20000000"], dev, a.spp)
        ratio, z, bad = tile_verdict(tile_stats(*m), tile_stats(*ref), 0.05)
        good = not bad.any()
        ok &= good
        frame = (m[0].sum() + m[1].sum()) / (ref[0].sum() + ref[1].sum())
        print("%s: mode M%s / mode R, floor under a glass slab: frame x%.4f; worst tile x%.4f "
              "(%.1f sigma)" % ("ok" if good else "FAIL", " (device)" if dev == "gpu" else "",
                                frame, ratio.flat[int(z.argmax())], z.max()), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
