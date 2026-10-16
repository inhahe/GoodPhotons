"""Regression check: an area light emits on the side its `normal` picks, for every strategy.

    D:/python314/python.exe -u tools/check_light_side.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                         [--no-gpu]

`light area { ... normal ... }`: the normal picks the side that emits and defaults to u x v
(FTSL.md). Light sampling honoured it, but every strategy that HITS the light -- a camera ray, a
reflection, a BSDF continuation -- read the side from the quad's triangles, which were wound by
u x v whatever the normal said. A light whose normal opposed u x v was therefore lit for light
sampling and black to everything that saw it, and the two halves of each MIS pair disagreed
(mode R's glossy light sampling on / off differed by 2.6x on such a lamp; eight lights in seven
of the repository's scenes are written that way).

Renders the same lamp written two ways -- its normal against u x v, and with u and v swapped so
they agree -- seen directly and in a glossy floor, and requires the two images to match: mode R on
the host and the device, and mode B (the forward tracer). Two seeds each; a tile of a 4 x 4 grid
fails when they differ by more than 5 standard errors (from the seed pairs, tile- and pixel-wise,
the larger kept) AND 2 %, the frame when by more than 5 standard errors AND 1 %.

And from BEHIND: the lamp's back, seen directly, must be black, as must everything else mode R
renders black (the void past the floor's end, which stops at the lamp's plane, so that nothing
behind the lamp can be lit). Every pixel mode R renders exactly black in both seeds, less a
one-pixel margin, is summed in each mode; the sum must stay under 0.1 % of mode R's whole frame.
Modes R W D U S M B P on the host and the device (A images through a lens whose blur spreads
light across edges). Until 0.384.2 mode S added a light's emission on whichever side its camera
path met it, so the back of every light glowed in mode S on both backends. Exits 1 on any
failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)
BASE = [
    'scene { units meters }',
    'camera "c" { eye 0 0.45 -1  look_at 0 0.35 1.5  up 0 1 0  fov_y 50  film { res 96 72 } }',
    'material "floor" { type glossy  reflect 0.8  roughness 0.1 }',
    'quad { origin -3 0 -3   u 0 0 8   v 6 0 0   material floor }',
]
LAMPS = {
    "against": 'light area { origin -0.8 0.3 2.8  u 1.6 0 0  v 0 0.8 0  normal 0 0 -1  power 400  spd preset:d65 }',
    "with":    'light area { origin -0.8 0.3 2.8  u 0 0.8 0  v 1.6 0 0  normal 0 0 -1  power 400  spd preset:d65 }',
}
# The lamp seen from behind, over a floor that ends at the lamp's plane.
BEHIND = [
    'scene { units meters }',
    'camera "c" { eye 0.2 0.9 4.6  look_at 0 0.4 1.0  up 0 1 0  fov_y 50  film { res 96 72 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'quad { origin -3 0 -3   u 0 0 5.8   v 6 0 0   material floor }',
    LAMPS["with"],
]
BEHIND_MODES = "RWDUSMBP"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_light_side"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    for name, lamp in LAMPS.items():
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(BASE + [lamp]) + NL)

    def render(name, tag, args):
        imgs = []
        for seed in (1, 2):
            png = "%s_%s_s%d.png" % (name, tag, seed)
            cmd = [a.exe, "-in", name + ".ftsl", "-seed", str(seed), "-hdr", "-no-meter",
                   "-o", png] + args
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    cases = [("mode R, host", "rcpu", ["-mode", "R", "-device", "cpu", "-spp", str(a.spp)]),
             ("mode B, host", "bcpu", ["-mode", "B", "-device", "cpu", "-n", "20000000"])]
    if not a.no_gpu:
        cases.append(("mode R, device", "rgpu", ["-mode", "R", "-device", "gpu", "-spp", str(a.spp)]))
    ok = True
    for label, tag, args in cases:
        x1, x0 = render("against", tag, args), render("with", tag, args)
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), 0.02)
        fr, fz, fbad = tile_verdict(tile_stats(*x1, 1), tile_stats(*x0, 1), 0.01)
        good = not bad.any() and not fbad.any()
        ok &= good
        i, j = np.unravel_index(int(np.argmax(z)), z.shape)
        print("%s: %s, normal against u x v / with it, frame x%.4f (%.1f sigma); worst tile x%.4f "
              "(%.1f sigma)" % ("ok" if good else "FAIL", label, fr[0, 0], fz[0, 0], ratio[i, j],
                                z[i, j]), flush=True)

    with open(os.path.join(d, "behind.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(BEHIND) + NL)
    ref = None                                    # mode R on the host: (black core, frame sum)
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        for m in BEHIND_MODES:
            n = 5000000 // a.spp if m == "S" else 5000000     # mode S: -n is per pass
            imgs = render("behind", m + dev, ["-mode", m, "-device", dev, "-spp", str(a.spp),
                                               "-n", str(n)])
            if ref is None:
                black = (imgs[0] == 0.0) & (imgs[1] == 0.0)
                core = black.copy()                       # less a one-pixel margin
                core[1:, :] &= black[:-1, :]
                core[:-1, :] &= black[1:, :]
                core[:, 1:] &= black[:, :-1]
                core[:, :-1] &= black[:, 1:]
                ref = (core, imgs[0].sum() + imgs[1].sum())
            core, total = ref
            leak = (imgs[0][core].sum() + imgs[1][core].sum()) / max(total, 1e-30)
            good = bool(core.any()) and leak < 1e-3
            ok &= good
            print("%s: mode %s, %s, seen from behind: %.4f %% of the frame's light where mode R "
                  "is black (%d pixels)" % ("ok" if good else "FAIL", m,
                                            "device" if dev == "gpu" else "host", 100 * leak,
                                            int(core.sum())), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
