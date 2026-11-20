"""Regression check: mode B sees through a gel and through a half mirror.

    python tools/check_passthrough.py [--out png/check_passthrough] [--exe ftrace.exe] [--no-gpu] [--spp 256]

MODEB-PASSTHROUGH. Mode B connects each photon vertex to the camera along a straight leg. A `filter` gel and
a `halfmirror` pass what they pass unbent, so a leg through one is a light path carrying the gel's T(lam) or
the mirror's 1 - R(lam) (camLegPass / dCamLegThrough); until 0.451.3 any surface on the leg cut it, and the
wall behind a gel rendered black. Each case, mode B on the CPU and the GPU against mode R on the CPU, the
frame in X, Y and Z (tiles under 2 % of a channel's brightest carry no verdict):

  - check_coat_modes' red gel before a lit wall (nothing below 595 nm);
  - check_coat_modes' half mirror reflecting red above 595 nm (the blue it passes);
  - a half mirror with a soft shortpass edge at 546.5 nm (the red it passes).

Exits 1 on any failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import SRGB_TO_XYZ, render_seeds, tile_stats, tile_verdict  # noqa: E402
from check_coat_modes import FILTER, HALF_MIRROR, half_mirror  # noqa: E402

TOL = 0.05
CASES = [
    ("gel", FILTER),
    ("mirror", HALF_MIRROR),
    ("mir_short", half_mirror("shortpass edge=546.5 slope=1 amp=0.9")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_passthrough"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]
    ok = True
    for name, lines in CASES:
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        rr = [x @ SRGB_TO_XYZ.T for x in
              render_seeds(exe, d, name + ".ftsl", name + "_R", ["-mode", "R"], "cpu", a.spp, rgb=True)]
        for dev in devs:
            bb = [x @ SRGB_TO_XYZ.T for x in
                  render_seeds(exe, d, name + ".ftsl", name + "_B" + ("g" if dev == "gpu" else ""),
                               ["-mode", "B", "-n", "20000000"], dev, a.spp, rgb=True)]
            good, parts = True, []
            for c, cname in enumerate("XYZ"):
                ratio, z, bad = tile_verdict(tile_stats(bb[0][..., c], bb[1][..., c]),
                                             tile_stats(rr[0][..., c], rr[1][..., c]), TOL)
                good &= not bad.any()
                parts.append("%s worst x%.3f (%.1f sigma)" % (cname, ratio.flat[int(z.argmax())], z.max()))
            ok &= good
            print("%s: %s, mode B%s / mode R: %s" % ("ok" if good else "FAIL", name,
                  " (device)" if dev == "gpu" else "", ", ".join(parts)), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
