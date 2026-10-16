"""Regression check: a glossy bounce's MIS record lasts exactly as long as it should (mode R).

    D:/python314/python.exe -u tools/check_glossy_nee_media.py [--out <dir>] [--exe ftrace.exe]
                                                               [--spp N] [--device cpu|gpu] [--mode R|M]
                                                               [--only name,...]

Glossy NEE (on by default; `-no-glossy-nee` turns it off) connects a glossy vertex to the lights
and records the lobe's density, so that the next vertex can weigh a light the continuation hits
against that connection. Both estimators are unbiased, so the image must not depend on the flag
-- which holds only if the record is cleared at the next real vertex and kept across a vertex
that is not one. Scenes, each a way to get that wrong:
  fog   -- glossy floor -> fog scatter -> sky: a medium scatter is a vertex the material code never
           sees, so it must clear the record itself (else the sky escape is weighted by the floor's
           lobe instead of the phase function the fog vertex's own sky NEE used: too bright).
  coat  -- glossy floor -> clearcoat sheen -> area light: the sheen does no NEE of its own and
           skips the material code too, so it must clear the record (else too dark).
  fiber -- glossy floor -> straight through see-through fur -> area light: a coverage pass-through
           is no vertex at all, so it must KEEP the record (the floor's NEE already counted the
           light through the fur's transmittance; else too bright).
Each is rendered in mode R (or `--mode M`, whose camera walks keep the same record) with glossy NEE
on and off, two seeds each, at one wavelength per path (-heroc 1) and with the bundle (-heroc 4),
and compared tile by tile: the paths at fault light only part of the frame. A tile fails when the
two differ by more than 5 standard errors (from the seed pairs, tile- and pixel-wise, the larger
kept) AND by more than 1 %. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENES = {
    "fog": [
        'scene { units meters }',
        'camera "c" { eye 0 0.6 -2.0  look_at 0 0.1 0.6  up 0 1 0  fov_y 45  film { res 96 64 } }',
        'material "floor" { type glossy  reflect 0.9  roughness 0.15 }',
        'material "wall"  { type diffuse  reflect 0.5 }',
        'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
        'quad { origin -3 0 2.5  u 6 0 0   v 0 3 0   material wall }',
        'light env { spd 1.0 }',
        'medium { sigma_t 0.9  albedo 0.95  g 0.3  bounds { min -3 0 -3  max 3 1.2 2.5 } }',
    ],
    "coat": [
        # The floor seen at a grazing angle mirrors the sphere's upper front, where the coat's
        # normal leans 45 degrees up and turns that near-horizontal ray straight up into a light
        # overhead. From the floor that light is a small target, so its light-sampling density is
        # well above the lobe's, and a stale floor weight darkens the reflected sheen.
        'scene { units meters }',
        'camera "c" { eye 0 0.25 -2.5  look_at 0 0.15 0.6  up 0 1 0  fov_y 30  film { res 96 64 } }',
        'material "floor" { type glossy  reflect 0.9  roughness 0.2 }',
        'material "paint" { type diffuse  reflect 0.05 }',
        'material "clear" { type layered  ior 2.5  coat { roughness 0.05 }  layer "paint" 1.0 }',
        'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
        'sphere { center 0 0.35 1.0  radius 0.35  material clear }',
        'light area { origin -0.4 2.0 0.6  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 200  spd preset:d65 }',
    ],
    "fiber": [
        # The floor at the frame centre mirrors into an area light tilted to face it, through a
        # ball of see-through fur sitting on that mirror ray; light out of frame.
        'scene { units meters }',
        'camera "c" { eye 0 1.0 -2.0  look_at 0 0 0  up 0 1 0  fov_y 36  film { res 96 64 } }',
        'material "floor" { type glossy  reflect 0.9  roughness 0.35 }',
        'material "core"  { type diffuse  reflect 0.3 }',
        'material "fur"   { type hair  reflect rgb 0.8 0.6 0.4  beta_m 0.3  beta_n 0.3  opacity 0.3 }',
        'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
        'sphere "ball" { center 0 0.5 1.0  radius 0.06  material core }',
        'fur "coat" { on "ball"  material fur  count 20000  length 0.18  length_jitter 0.3  radius 0.001  droop 0.2  jitter 0.25  seed 5 }',
        'light area { origin -0.5 1.468 2.266  u 1.0 0 0  v 0 -0.536 0.268  normal 0 -0.447 -0.894  power 300  spd preset:d65 }',
    ],
}


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data.astype(np.float64)


# What a device render prints to say where it ran: the `-> GPU` device line (modes R, W, D, J and
# the forward modes), `[gpu]` lines (mode M), and the device modes U and S name (`VCM/UPS on`,
# `SPPM on`). One list, so every check asks the same question.
GPU_TOKENS = ("-> GPU", "[gpu]", "VCM/UPS on ", "SPPM on ")


def render_seeds(exe, cwd, scene, tag, args, device="cpu", spp=256, seeds=(1, 2), rgb=False):
    """Render `scene` (a path relative to `cwd`, or absolute) once per seed with the extra
    command-line `args`, as linear PFMs `cwd/<tag>_s<seed>.pfm`, and return the images: summed
    over the colour channels, or h x w x 3 with `rgb`. An earlier run's output is deleted first,
    so a render that fails to write can never be scored, and a device render that does not say
    it ran on the GPU fails the check."""
    imgs = []
    for seed in seeds:
        png = "%s_s%d.png" % (tag, seed)
        pfm = os.path.join(cwd, png[:-4] + ".pfm")
        for f in (os.path.join(cwd, png), pfm):
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", scene, "-device", device, "-seed", str(seed), "-hdr", "-no-meter",
               "-spp", str(spp), "-o", png] + list(args)
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print(p.stdout[-1500:], p.stderr[-1500:])
            raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
        if device == "gpu" and not any(t in p.stdout + p.stderr for t in GPU_TOKENS):
            raise SystemExit("did not run on the GPU: " + " ".join(cmd))
        img = read_pfm(pfm)
        imgs.append(img if rgb else img.sum(-1))
    return imgs


def tile_stats(a, b, n=4):
    """Per-tile mean of (a + b) / 2 and its standard error, the larger of two estimates: from
    the two tile means (one degree of freedom) and from the per-pixel seed differences (many,
    blind to correlation between pixels)."""
    h, w = a.shape
    m = np.zeros((n, n)); e = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ta = a[h * i // n:h * (i + 1) // n, w * j // n:w * (j + 1) // n]
            tb = b[h * i // n:h * (i + 1) // n, w * j // n:w * (j + 1) // n]
            d = ta - tb
            m[i, j] = (ta.mean() + tb.mean()) / 2
            e[i, j] = max(abs(ta.mean() - tb.mean()) / 2, np.sqrt((d * d).sum()) / (2.0 * d.size))
    return m, e


def tile_verdict(x1, x0, rel, z_max=5.0):
    """Compare two tile_stats (mean, standard error) pairs: the ratio x1 / x0 per tile, the
    z-scores (0 on a tile under 2 % of x0's brightest, too dark to carry information), and the
    failing tiles -- those that differ by more than `z_max` standard errors AND by more than
    `rel` of x0."""
    (m1, e1), (m0, e0) = x1, x0
    z = np.abs(m1 - m0) / np.maximum(np.hypot(e1, e0), 1e-30)
    ratio = m1 / np.maximum(m0, 1e-30)
    live = m0 > 0.02 * m0.max()
    return ratio, np.where(live, z, 0.0), live & (z > z_max) & (np.abs(ratio - 1.0) > rel)


def tile_summary(x1, x0, rel, z_max=5.0):
    """tile_verdict as one line's worth: whether no tile fails, the frame ratio x1 / x0, and the
    worst tile's ratio and z-score."""
    ratio, z, bad = tile_verdict(x1, x0, rel, z_max)
    i, j = np.unravel_index(int(np.argmax(z)), z.shape)
    return (not bad.any()), x1[0].sum() / x0[0].sum(), ratio[i, j], z[i, j]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glossy_nee_media"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--mode", default="R", choices=("R", "M"),
                    help="mode M's camera walks keep the same record (and a photon map)")
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for name in SCENES:
        if a.only and name not in a.only.split(","):
            continue
        scene = os.path.join(out, name + ".ftsl")
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(SCENES[name]) + NL)
        for heroc in (1, 4):
            stats = {}
            for flag in ("on", "off"):
                args = ["-mode", a.mode, "-heroc", str(heroc)]
                if flag == "off":
                    args.append("-no-glossy-nee")
                if a.mode == "M":
                    args += ["-n", "5000000"]
                imgs = render_seeds(a.exe, out, scene, "%s_%s_c%d_%s_n%d" % (
                    name, a.mode, heroc, flag, a.spp), args, a.device, a.spp)
                stats[flag] = tile_stats(imgs[0], imgs[1])
            good, frame, worst, zw = tile_summary(stats["on"], stats["off"], 0.01)
            ok &= good
            print("%s: %s heroc %d: glossy NEE on / off, frame x%.4f; worst tile x%.4f (%.1f sigma)"
                  % ("ok" if good else "FAIL", name, heroc, frame, worst, zw), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
