"""Regression check: the sky seen through a fog, in the forward modes and mode M, against mode R.

    python tools/check_sky_fog.py [--out png/check_sky_fog] [--exe ftrace.exe] [--no-gpu] [--spp 256]
                                  [--fogs absorb,scatter,field_absorb]

A uniform sky (`light env { spd 1.0 }`) over a 10 m floor (small, so the forward modes' env photons, spread
over the scene's bounding sphere, cross the fog often), seen through a fog: a box 2 m deep absorbing only
(`albedo 0`) and scattering (`albedo 0.5`), and an absorbing density field with no bounds, falling off
with height, which an escaping ray takes out to infinity, as mode R's camera ray does: the sky behind it
is gone, and the ratio tracking such a field runs ends by Russian roulette (kTrRoulette) -- the case
holds that it ends. (An unbounded medium that scatters is no case here: the forward modes light one only
near the scene, warnHazeOnlyNearScene, where mode R lights all of it.) The frame's top quarter is sky
alone, so it holds the sky's transmittance and the fog's in-scatter. Each mode on the CPU and the GPU against mode R on the CPU, that quarter in X, Y and Z over
4 x 4 tiles:

  - mode B, whose sky pass (main.cpp prepareSky, skySeenThroughXYZ) takes the media along the ray;
  - mode A through a physical lens, the same pass's lens samples;
  - mode M with -beams, whose camera ray takes the media's extinction whether or not a beam was stored
    (a medium that scatters nothing stores none);
  - mode P, whose still composite takes the sky seen through a medium from its forward layer
    (classifyComposite), the sky pass's glows-only form drawing it there.

Exits 1 on any failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import SRGB_TO_XYZ, render_seeds, tile_stats, tile_verdict  # noqa: E402

TOL = 0.03
W, H = 64, 48
SKY_ROWS = H // 4        # the top quarter: sky alone
SCENE = [
    'camera "c" { eye 0 0.6 -2.5  look_at 0 0.9 0  up 0 1 0  fov_y 30  %s film { res %d %d } }',
    'material "white" { type diffuse  reflect 0.7 }',
    'quad { origin -5 0 -5   u 0 0 10   v 10 0 0   material white }',
    'light env { spd 1.0 }',
    '%s',
]
PHYS = "focus 4.0  lens { preset achromat  focal 50  fstop 2.8 }"
BOX = "medium { sigma_t 0.5  albedo %s  g 0.0  bounds { min -5 0 -1  max 5 6 1 } }"
FIELD = 'medium { sigma_t 0.06  albedo %s  density "0.3 + 0.7*saturate(1.0 - y/8.0)"  density_max 1 }'
FOGS = [   # (name, the medium, the modes that take it)
    ("absorb", BOX % "0.0", "BAMP"),
    ("scatter", BOX % "0.5", "BAMP"),
    ("field_absorb", FIELD % "0.0", "BAMP"),
]
MODES = [   # (tag, the camera's extra statement, the mode's arguments)
    ("B", "", ["-mode", "B", "-n", "4000000"]),
    ("A", PHYS, ["-mode", "A", "-n", "4000000"]),
    ("M", "", ["-mode", "M", "-beams", "-n", "400000"]),
    ("P", "", ["-mode", "P", "-n", "4000000"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_sky_fog"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--fogs", default="", help="comma-separated fog names (default: all)")
    a = ap.parse_args()
    names = [f[0] for f in FOGS]
    want = [w for w in a.fogs.split(",") if w] or names
    if [w for w in want if w not in names]:
        ap.error("the fogs are %s" % ", ".join(names))
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]

    def sky(name, cam, medium, args, dev):
        """Each seed's render, in XYZ, cut to the sky's rows (the PFM stores rows bottom-up)."""
        lines = list(SCENE)
        lines[0] = lines[0] % (cam, W, H)
        lines[-1] = lines[-1] % medium
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        tag = name + "_" + args[1] + ("g" if dev == "gpu" else "")
        return [(x @ SRGB_TO_XYZ.T)[-SKY_ROWS:] for x in
                render_seeds(exe, d, name + ".ftsl", tag, args, dev, a.spp, rgb=True)]

    ok = True
    for fog, medium, modes in FOGS:
        if fog not in want:
            continue
        for tag, cam, margs in MODES:
            if tag not in modes:
                continue
            name = "%s_%s" % (fog, tag)
            rr = sky(name, cam, medium, ["-mode", "R"], "cpu")
            for dev in devs:
                bb = sky(name, cam, medium, margs, dev)
                good, parts = True, []
                for c, cname in enumerate("XYZ"):
                    ratio, z, bad = tile_verdict(tile_stats(bb[0][..., c], bb[1][..., c]),
                                                 tile_stats(rr[0][..., c], rr[1][..., c]), TOL)
                    good &= not bad.any()
                    parts.append("%s worst x%.3g (%.1f sigma)" % (cname, ratio.flat[int(z.argmax())], z.max()))
                ok &= good
                print("%s: %s fog, mode %s%s / mode R: %s" % ("ok" if good else "FAIL", fog, tag,
                      " (device)" if dev == "gpu" else "", ", ".join(parts)), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
