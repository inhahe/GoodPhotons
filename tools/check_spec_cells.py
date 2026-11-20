"""Regression check: mode M's and mode S's spectral carrier reads a spectral step inside one of its bins.

    python tools/check_spec_cells.py [--out png/check_spec_cells] [--exe ftrace.exe] [--no-gpu] [--spp 128]
                                     [--cases gel_centre,tank]

SPECGATHER-BIN-RESOLUTION. The camera walk of modes M and S carries its spectral factors over a grid
(SpecThr / DSpecThr) and a photon reads the cell its wavelength falls in. On the 24 uniform bins alone, a
step inside a bin gave one side's value to the other side's photons and the bin's whole walk density to
one side of a support edge: behind a gel opaque below 546.5 nm (just past bin 9's centre) mode M read
mode R's green 0.72 and its red 1.05; a quarter into the bin, green 1.12 and red 0.96; off a half mirror
stepping there, green 0.75. The scene's tables now split the bins at their steps and support edges
(SpecCells). Each case, mode M and mode S, host and device, against host mode R, in X, Y and Z (linear
sRGB's channels cancel behind a gel, where XYZ's do not):

  - a gel opaque below 546.5 nm, clear above it (a step just past bin 9's centre);
  - the same step a quarter into the bin (541 nm);
  - a gel ramping from 0 at 555 nm to 0.9 at 585 nm (a support edge inside bin 9);
  - a half mirror whose reflectance steps from 0.05 to 0.9 at 546.5 nm;
  - check_coat_modes' tank, its thin-film ball's substrate absorbing above 546.5 nm only: a coin whose
    branch is impossible on one side of the step.

Every render must report the carrier split (its `[specgather]` line). Exits 1 on any failure.
"""
import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import tile_stats, tile_verdict  # noqa: E402
from check_radiance_eta2 import TANK  # noqa: E402
import check_glossy_nee_media as gnm  # noqa: E402

SPLIT = re.compile(r"\[specgather\] the spectral carrier's 24 bins split at")
TOL = 0.03
TANK_TOL = 0.05   # check_coat_modes' for the same tank: its upper tiles are dark
# linear sRGB (D65) to XYZ
M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def gel(table):
    return [
        'camera "c" { eye 0 0.6 -2.5  look_at 0 0.6 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
        'material "white" { type diffuse  reflect 0.7 }',
        'material "gel" { type filter  transmit table { %s } }' % table,
        'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material white }',
        'quad { origin -3 0 1.5  u 6 0 0   v 0 3 0   material white }',
        'quad { origin -0.6 0.05 -0.5  u 1.2 0 0  v 0 1.2 0  material gel }',
        'light area { origin -0.5 2.4 0.3  u 1 0 0  v 0 0 1  normal 0 -1 0  power 150  spd preset:d65 }',
    ]


MIRROR = [
    'camera "c" { eye 0 0.6 -2.5  look_at 0 0.6 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
    'material "white" { type diffuse  reflect 0.7 }',
    'material "hm" { type halfmirror  reflect table { 380:0.05 546.5:0.05 546.52:0.9 780:0.9 } }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material white }',
    'quad { origin 1.5 0 -3  u 0 3 0   v 0 0 6   material white }',
    'quad { origin -0.6 0.05 -0.6  u 1.2 0 1.2  v 0 1.2 0  material hm }',
    'light area { origin -0.5 2.4 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  power 150  spd preset:d65 }',
]


def tank():
    """check_coat_modes' partial tank with the substrate's step at 546.5 nm, inside bin 9."""
    out = []
    for ln in TANK:
        if ln.startswith('camera'):
            out.append('camera "c" { eye 0 0.45 -1.5  look_at 0 0.4 0  up 0 1 0  fov_y 26  film { res 64 48 } }')
        elif ln.startswith('material "water"'):
            out += ['material "water" { type dielectric  ior 1.333  priority 1 }',
                    'material "redcoat" { type thinfilm  ior 1.5  film_ior 1.38  film_thickness 300'
                    '  substrate_k table { 380:0 546.5:0 546.52:2 780:2 }  priority 1 }']
        else:
            out.append(ln.replace("radius 0.25  material white", "radius 0.25  material redcoat"))
    return out


CASES = [
    ("gel_centre", gel("380:0 546.5:0 546.52:0.9 780:0.9")),
    ("gel_quarter", gel("380:0 541.0:0 541.02:0.9 780:0.9")),
    ("gel_ramp", gel("380:0 555:0 585:0.9 780:0.9")),
    ("mirror", MIRROR),
    ("tank", tank()),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_spec_cells"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--cases", default=",".join(n for n, _ in CASES), help="the cases to run, comma-separated")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]
    logs = {}
    real_run = gnm.subprocess.run

    def run_logged(cmd, *args, **kw):   # render_seeds' renders, their logs kept for the split line
        p = real_run(cmd, *args, **kw)
        logs[cmd[cmd.index("-o") + 1]] = (p.stdout or "") + (p.stderr or "")
        return p
    gnm.subprocess.run = run_logged
    ok = True
    for name, lines in [c for c in CASES if c[0] in a.cases.split(",")]:
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        # The tank's upper tiles are dark and lit through water and the ball's coat: mode R wants 8192
        # samples there (check_coat_modes' own reference), on the GPU where there is one.
        if name != "tank":
            ref = (["-mode", "R"], "cpu", 4 * a.spp)
        else:
            ref = (["-mode", "R"], "cpu", 1024) if a.no_gpu else (["-mode", "R"], "gpu", 8192)
        rr = gnm.render_seeds(exe, d, name + ".ftsl", name + "_R", ref[0], ref[1], ref[2], rgb=True)
        rx = [x @ M.T for x in rr]
        for mode, args in (("M", ["-mode", "M", "-n", "20000000"]), ("S", ["-mode", "S", "-n", "2000000"])):
            for dev in devs:
                tag = name + "_" + mode + ("g" if dev == "gpu" else "")
                # Four seeds where two would leave a dark tile's error one degree of freedom: mode S's
                # correlated noise everywhere, and the tank's upper tiles in both modes.
                seeds = (1, 2) if mode == "M" and name != "tank" else (1, 2, 3, 4)
                mm = gnm.render_seeds(exe, d, name + ".ftsl", tag, args, dev, a.spp, seeds=seeds, rgb=True)
                said = all(SPLIT.search(logs.get("%s_s%d.png" % (tag, s), "")) for s in seeds)
                good = said
                parts = []
                for c, cname in enumerate("XYZ"):
                    xs = [x @ M.T for x in mm]
                    x1 = tile_stats(xs[0][..., c], xs[1][..., c]) if len(xs) == 2 else gnm.tile_stats_k([x[..., c] for x in xs])
                    x0 = tile_stats(rx[0][..., c], rx[1][..., c])
                    ratio, z, bad = tile_verdict(x1, x0, TANK_TOL if name == "tank" else TOL)
                    good &= not bad.any()
                    parts.append("%s worst x%.3f (%.1f sigma)" % (cname, ratio.flat[int(z.argmax())], z.max()))
                ok &= good
                print("%s: %s, mode %s%s / mode R: %s%s" % ("ok" if good else "FAIL", name, mode,
                      " (device)" if dev == "gpu" else "", ", ".join(parts), "" if said else "; NO split line"),
                      flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
