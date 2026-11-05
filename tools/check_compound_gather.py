"""Regression check: modes M and S gather on a compound surface (a mix, a layered coat) as mode R shades it.

    python tools/check_compound_gather.py [--out png/check_compound_gather] [--exe ftrace.exe]
                                          [--only mix,mix2,clearcoat,thinfilm]

A wall of a compound material, lit from above over a white floor, seen face on:
  mix        `mix { layer white 0.5 }`, the rest absorbed;
  mix2       half white diffuse, half rough glossy;
  clearcoat  `layered`: a Fresnel coat of index 1.5 over white;
  thinfilm   `layered`: a thin-film coat (index 2.4, 300 nm) over white, whose reflectance swings
             across the band, so the body under it keeps the complement of the film's colour.
Modes M and S on the CPU and the GPU must read the wall's mean colour within 3 % of mode R (CPU)
per channel, of the channel or of a tenth of the three together where it is smaller. Two seeds.

A camera walk resolves a compound to one lobe with the lobe's probability (a mix's weight, the
coat's Fresnel R or the body's 1 - R), and so does a photon. Until 0.433.10 the photon pass
deposited after its own pick, so the photons on a diffuse lobe already carried that probability and
the camera's pick took it a second time (known-issues COMPOUND-DOUBLE-PICK): mix read 0.50 of mode R,
mix2 0.56, clearcoat 0.89, all four modes alike. And until 0.433.9 neither walk folded a layered
coat's coin into its spectral carrier, which the thin film needs. Exits 0 when every case holds.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_motion as cm  # noqa: E402

NL = chr(10)
WALLS = {
    "mix": 'material "wall" { type mix  layer "white" 0.5 }',
    "mix2": 'material "wall" { type mix  layer "white" 0.5  layer "gloss" 0.5 }',
    "clearcoat": 'material "wall" { type layered  ior 1.5  coat { }  layer "white" 1.0 }',
    "thinfilm": 'material "wall" { type layered  ior 1.5  coat { reflectance thinfilm  film_ior 2.4  '
                'film_thickness 300 }  layer "white" 1.0 }',
}
REF = ["-mode", "R", "-device", "cpu", "-spp", "256"]
CASES = [("M_cpu", ["-mode", "M", "-device", "cpu", "-n", "4000000", "-spp", "16"]),
         ("M_gpu", ["-mode", "M", "-device", "gpu", "-n", "8000000", "-spp", "32"]),
         ("S_cpu", ["-mode", "S", "-device", "cpu", "-spp", "32"]),
         ("S_gpu", ["-mode", "S", "-device", "gpu", "-spp", "64"])]
# Modes U and J merge too (J's surface merges with -jsurf), and their light walks pick a compound's
# lobe before they store a vertex as the photon pass did (EXTRA_CASES, --extra).
EXTRA_CASES = [("U_cpu", ["-mode", "U", "-device", "cpu", "-spp", "64"]),
               ("U_gpu", ["-mode", "U", "-device", "gpu", "-spp", "256"]),
               ("J_gpu", ["-mode", "J", "-device", "gpu", "-spp", "256", "-jsurf"])]
TOL = 0.03


def scene(wall):
    return NL.join([
        "scene { units meters }",
        'camera "c" { eye 0 0.6 -2.5  look_at 0 0.9 1.5  up 0 1 0  fov_y 24  film { res 64 48 } }',
        'material "white" { type diffuse  reflect 0.7 }',
        'material "gloss" { type glossy  reflect 0.7  roughness 0.3 }',
        wall,
        "quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material white }",
        "quad { origin -3 0 1.5  u 0 3 0   v 6 0 0   material wall }",
        "light area { origin -0.5 2.4 0.3  u 1 0 0  v 0 0 1  normal 0 -1 0  power 150  spd preset:d65 }",
    ]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_compound_gather"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--only", default=",".join(WALLS))
    ap.add_argument("--extra", action="store_true", help="modes U and J too (EXTRA_CASES)")
    ap.add_argument("--cases", default="", help="only these cases, by name")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for name in a.only.split(","):
        path = os.path.join(out, name + ".ftsl")
        cm.write(path, scene(WALLS[name]))

        def wall(tag, args, gpu):
            ims = [cm.render(exe, out, path, "%s_%s_%d" % (name, tag, s), args, seed=s, want_gpu=gpu)[0]
                   for s in (1, 2)]
            return np.mean([im[:24].reshape(-1, 3).mean(axis=0) for im in ims], axis=0)   # the upper half

        r = wall("R", REF, False)
        floor = 0.1 * r.sum()
        cases = CASES + (EXTRA_CASES if a.extra else [])
        if a.cases:
            cases = [c for c in CASES + EXTRA_CASES if c[0] in a.cases.split(",")]
        for tag, args in cases:
            x = wall(tag, args, "gpu" in tag)
            good = all(abs(xi - ri) <= TOL * max(abs(ri), floor) for xi, ri in zip(x, r))
            ok &= good
            print("%-9s %-5s wall, of mode R: %s  %s" % (
                name, tag, " ".join("%.3f" % (xi / ri) for xi, ri in zip(x, r)), "ok" if good else "FAIL"),
                flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
