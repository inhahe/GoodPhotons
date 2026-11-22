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
  - a gel ramping from 0 at 555 nm to 0.9 at 585 nm (a support edge inside bin 9, a bend inside bin 11);
  - the first gel before a glossy wall (the spectral light sampling at the glossy vertex);
  - a half mirror whose reflectance steps from 0.05 to 0.9 at 546.5 nm;
  - a half mirror reflecting a 3 nm Gaussian band at 550 nm, and one with a 1 nm shortpass edge at
    546.5 nm (analytic spectra, sampled rather than read from knots; the edge's reflectance falls to
    1e-123 in the red, a coin as good as impossible there: known-issues HOST-HALFMIRROR-ANALYTIC);
  - check_coat_modes' tank, its thin-film ball's substrate absorbing above 546.5 nm only: a coin whose
    branch is impossible on one side of the step;
  - a 1 nm line at 589 nm behind a gel ramping over 570-600 nm, before the wall and before a glossy one:
    a cell is read at the light's centroid in it (SPECGATHER-CELL-CENTROID; its middle read the ramp
    18 % low), and the spectral NEE at a glossy vertex weighs a cell by the light's own emission in it
    (Emitter::cellEmit; the cell's width times the SPD at its middle missed the line, x0.10).

Every render must report the carrier split (its `[specgather]` line). Exits 1 on any failure.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import SRGB_TO_XYZ as M, tile_stats, tile_verdict  # noqa: E402
from check_coat_modes import filter_scene, half_mirror, partial_tank  # noqa: E402
import check_glossy_nee_media as gnm  # noqa: E402

SPLIT = re.compile(r"\[specgather\] the spectral carrier's 24 bins split at")
TOL = 0.03
TANK_TOL = 0.05   # check_coat_modes' for the same tank: its upper tiles are dark


def line_lit(lines):
    """`lines` lit by a 1 nm line at 589 nm in place of the area light's D65."""
    return [ln.replace("spd preset:d65", "spd gaussian center=589 sigma=1") for ln in lines]


CASES = [
    ("gel_centre", filter_scene("table { 380:0 546.5:0 546.52:0.9 780:0.9 }")),
    ("gel_quarter", filter_scene("table { 380:0 541.0:0 541.02:0.9 780:0.9 }")),
    ("gel_ramp", filter_scene("table { 380:0 555:0 585:0.9 780:0.9 }")),
    ("gel_glossy", filter_scene("table { 380:0 546.5:0 546.52:0.9 780:0.9 }",
                                "type glossy  reflect 0.7  roughness 0.3")),
    ("mirror", half_mirror("table { 380:0.05 546.5:0.05 546.52:0.9 780:0.9 }")),
    ("mir_gauss", half_mirror("gaussian center=550 sigma=3 amp=0.9")),
    ("mir_short", half_mirror("shortpass edge=546.5 slope=1 amp=0.9")),
    # check_coat_modes' tank, its substrate stepping inside bin 9: a coin impossible on one side.
    ("tank", partial_tank("table { 380:0 546.5:0 546.52:2 780:2 }")),
    # A line light behind a gel ramping across its cell: read where the line is, not at the cell's middle.
    ("line_ramp", line_lit(filter_scene("table { 380:0 570:0 600:0.9 780:0.9 }"))),
    ("line_glossy", line_lit(filter_scene("table { 380:0 570:0 600:0.9 780:0.9 }",
                                          "type glossy  reflect 0.7  roughness 0.3"))),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_spec_cells"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--cases", default=",".join(n for n, _ in CASES), help="the cases to run, comma-separated")
    a = ap.parse_args()
    known = [n for n, _ in CASES]
    asked = [c for c in a.cases.split(",") if c]
    if not asked or any(c not in known for c in asked):
        ap.error("--cases: one or more of " + ", ".join(known))
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
    for name, lines in [c for c in CASES if c[0] in asked]:
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        # The tank's upper tiles are dark and lit through water and the ball's coat: mode R wants 8192
        # samples there (check_coat_modes' own reference), on the GPU where there is one.
        if name != "tank":
            ref = (["-mode", "R"], "cpu", 4 * a.spp)
        else:
            ref = (["-mode", "R"], "cpu" if a.no_gpu else "gpu", 8192)
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
