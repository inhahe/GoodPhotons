"""Regression check: a GPU connection reads a vertex's patterns where the walk read them.

    python -u tools/check_vertex_hit_state.py [--out <dir>] [--exe ftrace.exe] [--modes DJU]

The device's bidirectional modes keep a path's vertices as DVertex and rebuild a hit from one
(dVertHit) for every BSDF, pdf and coat a connection or a merge evaluates there. The rebuilt hit
must carry what the walk's hit read its patterns by (DVertHitState, GPU-DVERTEX-HIT-STATE), among
it where a moving primitive's point sat at the instant. A ball striped by a pattern in its own
space slides across the shutter interval and is held sharp at a time when it has travelled half a
stripe (`shutter_open 0.105`): a rebuilt hit that did not carry its point back to the instant read
the stripes inverted, so the light a connection brought striped the ball out of phase with the rest
(on 0.450.2, 5-6 blocks out at z 20-28 in each mode). It renders on the GPU in modes D, J and U
(two seeds each), which must agree with mode R on the CPU, whose walk reads every pattern at its own
hit, over 8x8 blocks of luminance (each within 5 standard errors or 3 %).

(The state's other half, the hit's cavity, has no case here: a connection at a layered body read its
coat's cavity-driven film as 0 before, but the coat's factor there is a ratio of the coat's
transmissions at one thickness, and a titania film whose thickness follows `cavity` moved it by
under the noise -- the walk's own coin and the coat lobe, which read the film themselves, carry the
look. What storing it buys is one probe a vertex where a connection probed in every call.)

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_motion import agree, lum, stripe_scene  # noqa: E402
from pfm import read_pfm  # noqa: E402
import check_motion  # noqa: E402

W, H = 96, 72
MODES = {"D": ["-spp", "256"], "J": ["-spp", "256"], "U": ["-spp", "256"]}


def render(exe, out, path, tag, args, seed):
    png = os.path.join(out, tag + ".png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", path, "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png] + args
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-2000:])
        raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
    return lum(read_pfm(pfm, top_down=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vertex_hit_state"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--modes", default="DJU")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    check_motion.W, check_motion.H = W, H
    scenes = {"motion": stripe_scene(shutter=(0.105, 0.105))}
    ok = True
    for name, text in scenes.items():
        path = os.path.join(out, name + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        ref = [render(exe, out, path, "%s_R_cpu_s%d" % (name, s), ["-mode", "R", "-device", "cpu", "-spp", "256"], s)
               for s in (1, 2)]
        for m in a.modes:
            g = [render(exe, out, path, "%s_%s_gpu_s%d" % (name, m, s), ["-mode", m, "-device", "gpu"] + MODES[m], s)
                 for s in (1, 2)]
            good, detail = agree(g, ref)
            ok &= good
            print("%-4s %s: GPU mode %s agrees with mode R on the CPU   %s" % ("ok" if good else "FAIL", name, m, detail),
                  flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
