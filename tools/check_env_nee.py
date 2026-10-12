"""Regression check: sky light sampling and sky escapes split the sky between them exactly (mode R).

    D:/python314/python.exe -u tools/check_env_nee.py [--out <dir>] [--exe ftrace.exe] [--spp N]

Mode R reaches an environment light two ways, balance-heuristic weighted: light sampling at a
vertex (neeEnv and its bundle, fiber and volume twins) and a continuation that escapes the scene.
The weights sum to one only if both use the same continuation density. With FTRACE_ENV_NEE=0 the
light sampling is off and every escape takes its whole radiance, the BSDF strategy alone, so the
two must render the same image.

The scene hangs a wide burst of see-through fur (`opacity 0.6`) over a diffuse floor and frames
the floor beneath it, which sees much of its sky through the fur. The light-sampling weight once
took the cosine density from a cosine already multiplied by the fur's transmittance, so the
weights summed above one wherever the sky was partly hidden. A ray through this fur crosses a
few strands, which leaves its transmittance well inside (0, 1), where that error is largest; a
denser coat hides the sky outright and shows almost nothing. Measured at the default 2048 spp,
the old weight reads 1.5 % bright over the frame (23 standard errors) and 2 % in most tiles (5
to 9); the fixed one 0.05 % (0.7).

Compared at one wavelength per path (-heroc 1) and with the bundle (-heroc 4), per tile of a 4 x
4 grid and over the whole frame. A tile fails when on and off differ by more than 5 standard
errors (from the seed pairs, tile- and pixel-wise, the larger kept) AND by more than 0.5 %; the
frame when by more than 5 standard errors AND 0.2 %. Exits 1 on any failure.
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
SCENE = [
    'camera "c" { eye 0 0.4 -0.45  look_at 0 0 0.45  up 0 1 0  fov_y 45  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "core"  { type diffuse  reflect 0.3 }',
    'material "fur"   { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3  opacity 0.6 }',
    'quad { origin -4 0 -4   u 0 0 8   v 8 0 0   material floor }',
    'sphere "ball" { center 0 1.15 0.4  radius 0.02  material core }',
    'fur "coat" { on "ball"  material fur  count 5000  length 1.0  length_jitter 0.1  radius 0.002  droop 0.0  jitter 0.25  seed 9 }',
    'light env { spd 1.0 }',
]


def verdict(on, off, z_max, rel_min):
    """Mean ratio on / off, z-scores, and which cells fail: both differ by more than `z_max`
    standard errors and by more than `rel_min` of the off image. `on` and `off` are
    (mean, standard error) array pairs from tile_stats."""
    (m1, e1), (m0, e0) = on, off
    z = np.abs(m1 - m0) / np.maximum(np.hypot(e1, e0), 1e-30)
    ratio = m1 / np.maximum(m0, 1e-30)
    live = m0 > 0.02 * m0.max()
    return ratio, np.where(live, z, 0.0), live & (z > z_max) & (np.abs(ratio - 1.0) > rel_min)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_env_nee"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=2048)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "s.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    ok = True
    for heroc in (1, 4):
        imgs = {}
        for flag in ("on", "off"):
            for seed in (1, 2):
                png = "c%d_%s_s%d.png" % (heroc, flag, seed)
                env = dict(os.environ)
                if flag == "off":
                    env["FTRACE_ENV_NEE"] = "0"
                cmd = [a.exe, "-in", "s.ftsl", "-mode", "R", "-device", "cpu", "-spp", str(a.spp),
                       "-heroc", str(heroc), "-seed", str(seed), "-hdr", "-no-meter", "-o", png]
                p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                pfm = os.path.join(d, png[:-4] + ".pfm")
                if p.returncode != 0 or not os.path.exists(pfm):
                    print(p.stdout[-1500:], p.stderr[-1500:])
                    raise SystemExit("render failed: " + " ".join(cmd))
                imgs[flag, seed] = read_pfm(pfm).sum(-1)
        stats = {n: [tile_stats(imgs[f, 1], imgs[f, 2], n) for f in ("on", "off")] for n in (4, 1)}
        ratio, z, bad = verdict(*stats[4], 5.0, 0.005)
        fr, fz, fbad = verdict(*stats[1], 5.0, 0.002)
        good = not bad.any() and not fbad.any()
        ok &= good
        i, j = np.unravel_index(int(np.argmax(z)), z.shape)
        print("%s: heroc %d: sky light sampling on / off, frame x%.4f (%.1f sigma); worst tile x%.4f "
              "(%.1f sigma)" % ("ok" if good else "FAIL", heroc, fr[0, 0], fz[0, 0], ratio[i, j], z[i, j]),
              flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
