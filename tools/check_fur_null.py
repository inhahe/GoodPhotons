"""Regression check: fur at `opacity 0` is invisible -- to mode R's light in a medium and at a cap, to mode S.

    D:/python314/python.exe -u tools/check_fur_null.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                       [--device cpu|gpu]

`opacity` is coverage, the fraction of rays a fiber intercepts; at 0 a fur must leave the render
exactly as if it were not there -- the null FTSL.md uses to validate coverage. Every shadow ray
therefore takes the fibers' TRANSMITTANCE (Scene::shadowTransmittance), not a yes/no; light
sampling from a MEDIUM vertex (neeVolume, and the bundle's neeVolumeHero) once took the binary
`occluded` instead, so fog behind see-through fur was shadowed as if the fur were solid.

A fog box under an area light, a ball of `opacity 0` fur between them, rendered in mode R at one
wavelength per path (-heroc 1) and with the bundle (-heroc 4), against the same scene without the
fur block (its core sphere kept, so only the fur differs). Compared tile by tile; a tile fails when
the two differ by more than 5 standard errors (from the seed pairs, tile- and pixel-wise, the
larger kept) AND by more than 1 %.

And the same null at a bounce cap, under a sky: a floor with a furred ball low over it, lit by a
constant sky alone, mode R at -max-bounce 1. The floor's sky reaches it by its light sampling and by
its own escape, MIS-weighted, and the escape at the cap is the tail of the walk; a strand it meets
there is a coverage pass-through (no vertex) and must be passed, as everywhere else in the walk, or
see-through fur darkens the floor under it (BackwardRenderer::tailPassesStrand).

And mode S's camera walk through the same null: a rough glossy floor reflecting an area lamp, the
furred ball on the reflected path. The light the floor's lobe reaches through the fur is weighed by
that glossy vertex's MIS record against its light sampling, and softened by it; a strand the lobe
passes is no vertex and must keep both, or the lamp's reflection is counted twice behind the fur
(0.440.8 read it 1.359 of the bare scene); nor may it spend a bounce, for a ray crosses about a
hundred strands of this coat (keeping the record alone read 0.973). Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats  # noqa: E402

NL = chr(10)
BASE = [
    'scene { units meters }',
    'camera "c" { eye 0 0.8 -2.4  look_at 0 0.5 0.5  up 0 1 0  fov_y 40  film { res 96 64 } }',
    'material "ground" { type diffuse  reflect 0.4 }',
    'material "core"   { type diffuse  reflect 0.3 }',
    'material "fur"    { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3  opacity 0 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material ground }',
    'light area { origin -0.5 2.0 0.0  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 300  spd preset:d65 }',
    'medium { sigma_t 1.5  albedo 0.9  g 0.0  bounds { min -1 0 0  max 1 1 1 } }',
    'sphere "ball" { center 0 1.45 0.5  radius 0.01  material core }',
]
FUR = 'fur "coat" { on "ball"  material fur  count 30000  length 0.25  length_jitter 0.3  radius 0.001  droop 0.1  jitter 0.25  seed 7 }'
# The cap's case: no fog, no lamp, a sky; the ball low over the floor, its fur long and dense.
SKY = [l for l in BASE if not l.startswith(("light area", "medium", "sphere"))] + [
    'light env { spd preset:d65 }',
    'sphere "ball" { center 0 0.35 0.5  radius 0.05  material core }',
]
SKY_FUR = 'fur "coat" { on "ball"  material fur  count 60000  length 0.3  length_jitter 0.3  radius 0.001  droop 0.0  jitter 0.25  seed 7 }'
# Mode S's case: no fog, a rough glossy floor, the lamp ahead, the ball on its reflection's way.
GLOSSY = [
    'scene { units meters }',
    'camera "c" { eye 0 0.8 -2.4  look_at 0 0.2 0.5  up 0 1 0  fov_y 40  film { res 96 64 } }',
    'material "floor" { type glossy  reflect 0.8  roughness 0.25 }',
    'material "core"  { type diffuse  reflect 0.3 }',
    'material "fur"   { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3  opacity 0 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'light area { origin -0.6 2.0 2.2  u 1.2 0 0  v 0 0 0.6  normal 0 -1 0  power 300  spd preset:d65 }',
    'sphere "ball" { center 0 1.0 0.75  radius 0.05  material core }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_fur_null"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--spp", type=int, default=256)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for name, lines in (("fur", BASE + [FUR]), ("bare", BASE), ("skyfur", SKY + [SKY_FUR]), ("skybare", SKY),
                        ("glossyfur", GLOSSY + [SKY_FUR]), ("glossybare", GLOSSY)):
        with open(os.path.join(out, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
    ok = True
    for mode, heroc, pre, extra in (("R", 1, "", []), ("R", 4, "", []), ("R", 1, "sky", ["-max-bounce", "1"]),
                                    ("R", 4, "sky", ["-max-bounce", "1"]),
                                    ("S", 1, "glossy", ["-n", "100000"])):
        if mode == "S" and a.device == "gpu":
            continue   # the device's mode S takes no hair (cudaSppmSupported): it would run the host's
        stats = {}
        for name in (pre + "fur", pre + "bare"):
            imgs = []
            for seed in (1, 2):
                png = os.path.join(out, "%s_%s_c%d_s%d_n%d.png" % (name, mode, heroc, seed, a.spp))
                for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                    if os.path.exists(f):
                        os.remove(f)
                cmd = [a.exe, "-in", os.path.join(out, name + ".ftsl"), "-mode", mode, "-device", a.device,
                       "-spp", str(a.spp), "-heroc", str(heroc), "-seed", str(seed), "-hdr",
                       "-no-meter", "-window-min", "-o", png] + extra
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                pfm = png[:-4] + ".pfm"
                if r.returncode != 0 or not os.path.exists(pfm):
                    print(r.stdout[-1500:], r.stderr[-1500:])
                    raise SystemExit("render failed: " + " ".join(cmd))
                imgs.append(read_pfm(pfm).sum(-1))
            stats[name] = tile_stats(imgs[0], imgs[1])
        (m1, e1), (m0, e0) = stats[pre + "fur"], stats[pre + "bare"]
        z = np.abs(m1 - m0) / np.maximum(np.hypot(e1, e0), 1e-30)
        ratio = m1 / np.maximum(m0, 1e-30)
        live = m0 > 0.02 * m0.max()
        bad = live & (z > 5.0) & (np.abs(ratio - 1.0) > 0.01)
        good = not bad.any()
        ok &= good
        zl = np.where(live, z, 0.0)
        i, j = np.unravel_index(int(np.argmax(zl)), zl.shape)
        where = {"": "", "sky": "sky at -max-bounce 1, ", "glossy": "a glossy floor's lamp, "}[pre]
        print("%s: mode %s, %sheroc %d: opacity-0 fur / no fur, frame x%.4f; worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", mode, where, heroc, m1.sum() / m0.sum(), ratio[i, j], zl[i, j]),
            flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
