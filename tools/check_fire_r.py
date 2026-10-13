"""Regression check: mode R renders a fire as the forward tracer does.

    D:/python314/python.exe -u tools/check_fire_r.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                     [--photons N]

A medium with a `temperature` grid emits j(x, lambda) = Medium::emissionAt per unit length; the
forward tracer births photons at that density, and mode R reaches it two ways -- light sampling of
the fire at every vertex (fireNee) and the emission each path segment collects (fireSegment),
combined as an area light's two strategies are. Writes a fire (loom.vdbio: a soot `density` blob,
a hot `temperature` core; the soot scatters) lighting a diffuse floor and wall and a glossy
sphere, the fire the only light, and compares tile by tile, two seeds each:
  1. mode R with its fire light sampling on against off (FTRACE_FIRE_NEE=0: every segment then
     takes its whole emission) -- two strategies of one estimator, so they must agree tightly;
  2. mode R at one wavelength per path (-heroc 1) against the bundle (-heroc 4);
  3. mode R against mode B, the forward tracer (an independent estimator), with the sphere made
     diffuse: B cannot splat a glossy (specular-first) pixel to its pinhole at all (REFERENCE.md),
     so on the glossy sphere the two are not comparable;
  4. a fire wider than the scene -- its `temperature` grid reaching far past the small `density`
     grid that bounds the medium, over a small floor -- light sampling on against off: until
     0.377.3 the light sampling's integral stopped at the scene's bounding sphere, which the
     density grid and the floor alone set, and the floor lost most of the fire's light.
A tile fails when two renders differ by more than 5 standard errors (from the seed pairs, tile-
and pixel-wise, the larger kept) AND by more than the pair's tolerance (1 %, 1 %, 3 %, 1 %).
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "loom"))
from check_glossy_nee_media import read_pfm, tile_stats, tile_summary  # noqa: E402
from loom.vdbio import VolumeGrid, write_vdb  # noqa: E402

NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.7 -3.2  look_at 0 0.45 0  up 0 1 0  fov_y 42  film { res 96 72 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "wall"  { type diffuse  reflect rgb 0.55 0.6 0.7 }',
    'material "gloss" { type glossy  reflect 0.8  roughness 0.2 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'quad { origin -3 0 1.5  u 6 0 0   v 0 3 0   material wall }',
    'sphere { center 1.15 0.4 0.2  radius 0.4  material gloss }',
    'medium {',
    '    sigma_t 4   albedo 0.5   g 0.3',
    # relative names, resolved from the output directory the renders run in (a `vdb:` path
    # ends at the first space)
    '    density vdb:fire.vdb',
    '    temperature vdb:fire.vdb',
    '    emission blackbody   emission_kelvin 1500   emission_scale 4',
    '}',
]


def write_fire(d):
    """Write the check's fire (fire.vdb) and its two scenes into directory `d`: fire.ftsl, with
    the glossy sphere, and fire_diffuse.ftsl, with it made diffuse."""
    n = 24
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    write_vdb(os.path.join(d, "fire.vdb").replace(chr(92), "/"),
              [VolumeGrid("density", np.clip(1.0 - r, 0.0, None), (-0.5, 0.0, -0.5, 0.5, 1.0, 0.5)),
               VolumeGrid("temperature", np.clip(1.0 - 1.6 * r, 0.0, None),
                          (-0.5, 0.0, -0.5, 0.5, 1.0, 0.5))])
    with open(os.path.join(d, "fire.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    with open(os.path.join(d, "fire_diffuse.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(l.replace("material gloss }", "material floor }") for l in SCENE) + NL)


def write_wide_fire(d):
    """Write fire_wide.vdb and fire_wide.ftsl into `d`: a 0.5-unit soot blob (the `density` grid,
    which bounds the medium) inside a 4-unit hot `temperature` grid, over a 1-unit floor."""
    n = 24
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    write_vdb(os.path.join(d, "fire_wide.vdb").replace(chr(92), "/"),
              [VolumeGrid("density", np.clip(1.0 - r, 0.0, None), (-0.25, 0.0, -0.25, 0.25, 0.5, 0.25)),
               VolumeGrid("temperature", np.clip(1.0 - r, 0.0, None), (-2.0, 0.0, -2.0, 2.0, 4.0, 2.0))])
    scene = [
        'camera "c" { eye 0 0.8 -0.9  look_at 0 0 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
        'material "floor" { type diffuse  reflect 0.6 }',
        'quad { origin -0.5 0 -0.5   u 0 0 1   v 1 0 0   material floor }',
        'medium {',
        '    sigma_t 4   albedo 0.5   g 0.3',
        '    density vdb:fire_wide.vdb',
        '    temperature vdb:fire_wide.vdb',
        '    emission blackbody   emission_kelvin 1500   emission_scale 4',
        '}',
    ]
    with open(os.path.join(d, "fire_wide.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(scene) + NL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_fire_r"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--photons", type=int, default=100000000)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    write_fire(d)
    write_wide_fire(d)

    def render(tag, args, env=None, scene="fire.ftsl"):
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            cmd = [a.exe, "-in", scene, "-device", "cpu", "-seed", str(seed), "-hdr",
                   "-no-meter", "-o", png] + args
            e = dict(os.environ)
            e.update(env or {})
            p = subprocess.run(cmd, cwd=d, env=e, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return tile_stats(imgs[0], imgs[1])

    spp = ["-spp", str(a.spp)]
    r4 = render("r_c4", ["-mode", "R", "-heroc", "4"] + spp)
    r4off = render("r_c4_noneed", ["-mode", "R", "-heroc", "4"] + spp, {"FTRACE_FIRE_NEE": "0"})
    r1 = render("r_c1", ["-mode", "R", "-heroc", "1"] + spp)
    rd = render("rd_c4", ["-mode", "R", "-heroc", "4"] + spp, scene="fire_diffuse.ftsl")
    bd = render("bd", ["-mode", "B", "-n", str(a.photons)], scene="fire_diffuse.ftsl")
    w4 = render("w_c4", ["-mode", "R", "-heroc", "4"] + spp, scene="fire_wide.ftsl")
    w4off = render("w_c4_noneed", ["-mode", "R", "-heroc", "4"] + spp, {"FTRACE_FIRE_NEE": "0"},
                   scene="fire_wide.ftsl")
    ok = True
    for name, x1, x0, tol in (("light sampling on / off", r4, r4off, 0.01),
                              ("-heroc 4 / -heroc 1", r4, r1, 0.01),
                              ("mode R / mode B (diffuse sphere)", rd, bd, 0.03),
                              ("a fire wider than the scene, light sampling on / off", w4, w4off,
                               0.01)):
        good, frame, worst, zw = tile_summary(x1, x0, tol)
        ok &= good
        print("%s: %s, frame x%.4f; worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, frame, worst, zw), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
