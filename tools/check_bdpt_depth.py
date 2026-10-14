"""Regression check: modes D and J's paths of exactly the maximum depth carry their full weight.

    D:/python314/python.exe -u tools/check_bdpt_depth.py [--out <dir>] [--exe ftrace.exe]
                                                         [--spp N] [--no-gpu]

BDPT weights every path over all the strategies that could make it, s = 0 (the eye walk landing
on a light) included -- so the eye walk has to be able to make each one: a path of depth d
(d scattering vertices) needs d + 2 eye vertices for s = 0. Mode D generated maxDepth + 1, so a
path of depth exactly maxDepth was made by every strategy but s = 0 while still weighted as if
s = 0 shared it, and lost that share (known-issues BDPT-LAST-DEPTH). Mode J adds point merges
to the same weights, and its merge refused a path of depth exactly maxDepth too (its cap was one
too strict: a photon at light vertex j merged at eye vertex k makes depth j + k - 1, not j + k).
Both show where the last bounce carries a large share of the light -- a low -max-bounce, and a
lamp as big as the ceiling -- so this renders a closed box lit that way at -max-bounce 1 and 2,
modes D and J against mode R (whose -max-bounce counts the same scattering vertices), on the
host and, unless --no-gpu, the device. Mode D read x0.936 and x0.987 of R; mode J x0.83 and
x0.94 at the default merge radius. Mode J merges at a quarter of it here -- at the default, its
point merges' own density-estimation bias reads ~1 % bright near the corners, which is not what
this checks -- where it read x0.984 and x0.995 (14 and 6 sigma), their share being smaller.
A tile fails when the two differ by more than 5 standard errors (two seeds each) AND by more
than 2 %; the frame, by more than 5 standard errors AND 0.5 %. Exits 1 on any failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)

# A closed 2 x 2 x 2 box, a square lamp filling most of its ceiling, the camera inside looking
# at the back wall. Every quad's u x v points into the room.
BOX = [
    'camera "c" { eye 0 1 -0.9  look_at 0 1 1  up 0 1 0  fov_y 70  film { res 64 48 } }',
    'material "white" { type diffuse  reflect 0.7 }',
    'material "red"   { type diffuse  reflect rgb 0.7 0.2 0.2 }',
    'material "green" { type diffuse  reflect rgb 0.2 0.7 0.2 }',
    'quad { origin -1 0 -1   u 0 0 2   v 2 0 0   material white }',     # floor, +y
    'quad { origin -1 2 -1   u 2 0 0   v 0 0 2   material white }',     # ceiling, -y
    'quad { origin -1 0 1    u 2 0 0   v 0 2 0   material white }',     # back, -z
    'quad { origin -1 0 -1   u 0 2 0   v 2 0 0   material white }',     # front, +z
    'quad { origin -1 0 -1   u 0 0 2   v 0 2 0   material red }',       # left, +x
    'quad { origin 1 0 -1    u 0 2 0   v 0 0 2   material green }',     # right, -x
    'light area { origin -0.8 1.999 -0.8  u 1.6 0 0  v 0 0 1.6  normal 0 -1 0  power 50  '
    'spd preset:d65 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_bdpt_depth"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "box.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(BOX) + NL)

    def render(tag, args, device="cpu"):
        """Two seeds of one configuration (render_seeds)."""
        return render_seeds(a.exe, d, "box.ftsl", tag, args, device, a.spp)

    pairs = []
    for depth in ("1", "2"):
        r = render("r_" + depth, ["-mode", "R", "-max-bounce", depth])
        for mode in ("D", "J"):
            # Mode J's point merges (and mode U's) at a quarter of the default radius: at the
            # default, their O(r^2) density-estimation bias reads ~1 % bright near the corners,
            # up to x1.03 on a tile, which this check is not about.
            args = ["-mode", mode, "-max-bounce", depth] + (
                ["-pmradiusfrac", "0.005"] if mode in "JU" else [])
            pairs.append(("-max-bounce %s: mode %s / mode R" % (depth, mode),
                          render("%s_%s" % (mode.lower(), depth), args), r))
            if not a.no_gpu:
                pairs.append(("-max-bounce %s: mode %s on the device / mode R" % (depth, mode),
                              render("%sg_%s" % (mode.lower(), depth), args, "gpu"), r))
    ok = True
    for name, x1, x0 in pairs:
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), 0.02)
        fr, fz, fbad = tile_verdict(tile_stats(*x1, 1), tile_stats(*x0, 1), 0.005)
        good = not bad.any() and not fbad.any()
        ok &= good
        print("%s: %s, frame x%.4f (%.1f sigma); worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, fr[0, 0], fz[0, 0], ratio.flat[int(z.argmax())], z.max()),
            flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
