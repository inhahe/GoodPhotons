"""Regression check: a wavelength split's sub-paths keep the glossy MIS record across a fur pass-through.

    D:/python314/python.exe -u tools/check_split_mis.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                        [--no-gpu]

`-herosplit` fans a wavelength bundle out at a dispersive vertex into one sub-path per wavelength;
a fiber goes through that split in mode W, on the device, and on the host under -dual-scatter. A
coverage pass-through (see-through fur) is no vertex: the glossy MIS record of the last real vertex
must survive it, so light reached through the fur after a glossy bounce is weighed against the
glossy vertex's own light sampling (GLOSSY-NEE) instead of counted again. The hero restored it; the
secondaries' sub-paths started with a fresh record (and spent a bounce on the pass-through), on
the host and the device alike, so they counted that light twice.

The camera sees a large lamp reflected in a near-mirror floor, through a cloud of `opacity 0.3`
fur. Mode R with the glossy light sampling on against off (`-no-glossy-nee`: the reflection then
takes what it hits at full weight) must agree: on the host with `-herosplit -dual-scatter`, on the
device with `-herosplit`. Two seeds each; a tile of a 4 x 4 grid fails when the two differ by more
than 5 standard errors (from the seed pairs, tile- and pixel-wise, the larger kept) AND 2 %, the
frame when by more than 5 standard errors AND 1 %. Exits 1 on any failure. (Mode W itself cannot
be held to this: its fixed lattice of glossy directions misses a lamp that light sampling finds.)
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
SCENE = [
    'scene { units meters }',
    'camera "c" { eye 0 0.3 -1  look_at 0 0.05 0.6  up 0 1 0  fov_y 40  film { res 96 72 } }',
    'material "floor" { type glossy  reflect 0.8  roughness 0.08 }',
    'material "core"  { type diffuse  reflect 0.3 }',
    'material "fur"   { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3  opacity 0.3 }',
    'quad { origin -3 0 -3   u 0 0 8   v 6 0 0   material floor }',
    'light area { origin -0.8 0.3 2.8  u 0 0.8 0  v 1.6 0 0  normal 0 0 -1  power 400  spd preset:d65 }',
    'sphere "ball" { center 0 0.28 1.5  radius 0.02  material core }',
    'fur "coat" { on "ball"  material fur  count 6000  length 0.25  length_jitter 0.2  radius 0.0015  droop 0.0  jitter 0.25  seed 3 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_split_mis"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "s.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)

    def render(tag, args, gnee):
        """Two seeds of mode R; returns the pair of images."""
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            cmd = [a.exe, "-in", "s.ftsl", "-mode", "R", "-spp", str(a.spp), "-seed", str(seed),
                   "-hdr", "-no-meter", "-o", png] + args + ([] if gnee else ["-no-glossy-nee"])
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    cases = [("host, -herosplit -dual-scatter", "cpu", ["-device", "cpu", "-herosplit", "-dual-scatter"])]
    if not a.no_gpu:
        cases.append(("device, -herosplit", "gpu", ["-device", "gpu", "-herosplit"]))
    ok = True
    for name, tag, args in cases:
        on, off = render(tag + "_on", args, True), render(tag + "_off", args, False)
        ratio, z, bad = tile_verdict(tile_stats(*on), tile_stats(*off), 0.02)
        fr, fz, fbad = tile_verdict(tile_stats(*on, 1), tile_stats(*off, 1), 0.01)
        good = not bad.any() and not fbad.any()
        ok &= good
        i, j = np.unravel_index(int(np.argmax(z)), z.shape)
        print("%s: mode R %s, glossy light sampling on / off, frame x%.4f (%.1f sigma); worst tile "
              "x%.4f (%.1f sigma)" % ("ok" if good else "FAIL", name, fr[0, 0], fz[0, 0], ratio[i, j],
                                      z[i, j]), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
