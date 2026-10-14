"""Regression check: the hero-wavelength bundle stays unbiased through fibers and grey media.

    D:/python314/python.exe -u tools/check_hero_spectral.py [--out <dir>] [--exe ftrace.exe]
                                                            [--spp N] [--only name,...] [--modes RD]
                                                            [--device cpu|gpu]

A path carries a BUNDLE of wavelengths (`-heroc 4`) wherever the light it follows does not depend
on wavelength in DIRECTION, and falls back to one wavelength (`-heroc 1` everywhere) otherwise.
Keeping the bundle where the direction does depend on wavelength, but smoothly -- a hair fiber's
lobes, a rainbow droplet's phase -- needs spectral MIS (Wilkie et al. 2014), and carrying it
through a medium needs the medium's extinction to be the same at every wavelength. All of that
changes only the NOISE: every scene here must render the same image with the bundle as without.

For each scene and each mode it runs in (mode R; mode D where the scene has no sky, which BDPT
refuses), the CPU renders linear PFMs (`-hdr`) at `-heroc 1` and `-heroc 4`, two seeds each. The
frame is cut into tiles; a tile fails when the bundle's mean is off the single-wavelength mean by
more than 5 standard errors (estimated from the seed pairs, tile- and pixel-wise, the larger kept)
AND by more than 1 %. The chroma noise (the spread of R/G and B/G between the seed pairs) is
reported, which is what the bundle exists to lower. A scene whose `-heroc 4` render is bit-identical
to its `-heroc 1` render also fails: its bundle was switched off. With `--device gpu` the renders
are mode R's on the device (a render that fell back to the CPU fails), and the device's `-heroc 1`
render is also held against the host's, by the same test. Scenes:
  hair  -- two fur balls (pale: long multiple-scattering chains; dark), a glass sphere the fur
           sees through (a fiber vertex followed by a single-wavelength fallback), sun + area
           light + sky.
  hairsun -- the same without the sky, for mode D too.
  hairback -- red and pale fur rim-lit by a large area light behind them: built to favour a joint
           (strategy x hero wavelength) MIS weight in mode D, measured and not kept (known-issues
           SPECTRAL-BUNDLE); fibers seen by both camera and light strategies.
  hairmed -- fur whose medulla scatters only above 550 nm, so a bundle's wavelengths differ in
           whether their fiber is solid: the per-lobe fold of a solid fiber must not be applied
           to a wavelength whose own medulla is live (it once took the hero's word for it).
  fog   -- diffuse spheres in a bounded grey Henyey-Greenstein fog, sun + area light.
  fogcli -- the same lit by the command line's global `-fog`, added after the scene is built.
  coatfog -- an iridescent coat (which splits the bundle per wavelength, re-tracing the segment
           that reached it) on a sphere inside the fog.
  rain  -- a box of rainbow-phase droplets lit by a sun behind the camera, the bow in frame.
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENES = {
    "hair": [
        'scene { units meters }',
        'camera "c" { eye 0.45 0.32 -0.75  look_at 0.45 0.20 0.55  up 0 1 0  fov_y 34  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.45 }',
        'material "skin"   { type diffuse  reflect rgb 0.35 0.26 0.22 }',
        'material "pale"   { type hair  reflect rgb 0.85 0.70 0.40  beta_m 0.25  beta_n 0.30 }',
        'material "dark"   { type hair  reflect rgb 0.30 0.17 0.09  beta_m 0.30  beta_n 0.30 }',
        'material "glass"  { type dielectric  ior 1.5 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'light sun { dir -0.4 0.6 -0.7  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'light area { origin 0.0 0.9 0.2  u 0.9 0 0  v 0 0 0.6  normal 0 -1 0  power 120  spd preset:d65 }',
        'light env { spd 0.02 }',
        'sphere "b1" { center 0.22 0.13 0.55  radius 0.10  material skin }',
        'fur "b1_coat" { on "b1"  material pale  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 1 }',
        'sphere "b2" { center 0.62 0.13 0.62  radius 0.10  material skin }',
        'fur "b2_coat" { on "b2"  material dark  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 2 }',
        'sphere { center 0.43 0.07 0.25  radius 0.07  material glass }',
    ],
    "hairsun": [
        'scene { units meters }',
        'camera "c" { eye 0.45 0.32 -0.75  look_at 0.45 0.20 0.55  up 0 1 0  fov_y 34  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.45 }',
        'material "skin"   { type diffuse  reflect rgb 0.35 0.26 0.22 }',
        'material "pale"   { type hair  reflect rgb 0.85 0.70 0.40  beta_m 0.25  beta_n 0.30 }',
        'material "dark"   { type hair  reflect rgb 0.30 0.17 0.09  beta_m 0.30  beta_n 0.30 }',
        'material "glass"  { type dielectric  ior 1.5 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'light sun { dir -0.4 0.6 -0.7  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'light area { origin 0.0 0.9 0.2  u 0.9 0 0  v 0 0 0.6  normal 0 -1 0  power 120  spd preset:d65 }',
        'sphere "b1" { center 0.22 0.13 0.55  radius 0.10  material skin }',
        'fur "b1_coat" { on "b1"  material pale  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 1 }',
        'sphere "b2" { center 0.62 0.13 0.62  radius 0.10  material skin }',
        'fur "b2_coat" { on "b2"  material dark  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 2 }',
        'sphere { center 0.43 0.07 0.25  radius 0.07  material glass }',
    ],
    "hairback": [
        # Strongly coloured fur RIM-LIT by a large area light behind it: the fibers' TT/TRT
        # weights differ by orders of magnitude across the band, and the light is big enough that
        # sampling it and hitting it off the fiber lobe are both live strategies.
        'scene { units meters }',
        'camera "c" { eye 0 0.22 -0.8  look_at 0 0.16 0.3  up 0 1 0  fov_y 30  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.2 }',
        'material "skin"   { type diffuse  reflect rgb 0.30 0.20 0.18 }',
        'material "red"    { type hair  reflect rgb 0.70 0.10 0.04  beta_m 0.15  beta_n 0.30 }',
        'material "pale"   { type hair  reflect rgb 0.85 0.75 0.50  beta_m 0.20  beta_n 0.30 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'light area { origin -0.5 0.8 1.0  u 0 0.45 -0.5  v 1.0 0 0  normal 0 -0.743 -0.669  power 300  spd preset:d65 }',
        'sphere "b1" { center -0.13 0.13 0.3  radius 0.09  material skin }',
        'fur "b1_coat" { on "b1"  material red  count 25000  length 0.06  length_jitter 0.3  radius 0.0006  droop 0.4  jitter 0.25  seed 3 }',
        'sphere "b2" { center 0.13 0.13 0.35  radius 0.09  material skin }',
        'fur "b2_coat" { on "b2"  material pale  count 25000  length 0.06  length_jitter 0.3  radius 0.0006  droop 0.4  jitter 0.25  seed 4 }',
    ],
    "hairmed": [
        'scene { units meters }',
        'camera "c" { eye 0 0.22 -0.6  look_at 0 0.14 0.3  up 0 1 0  fov_y 30  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.45 }',
        'material "skin"   { type diffuse  reflect rgb 0.35 0.26 0.22 }',
        'material "furm"   { type hair  reflect rgb 0.80 0.65 0.45  beta_m 0.25  beta_n 0.30  medulla 0.6'
        '  medulla_sigma_s table { 360:0 549:0 551:4 830:4 }  medulla_sigma_a table { 360:0 549:0 551:0.5 830:0.5 }'
        '  medulla_g 0.3 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'light sun { dir -0.4 0.6 -0.7  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'light area { origin -0.4 0.9 0.0  u 0.8 0 0  v 0 0 0.6  normal 0 -1 0  power 120  spd preset:d65 }',
        'sphere "b1" { center 0 0.13 0.3  radius 0.10  material skin }',
        'fur "b1_coat" { on "b1"  material furm  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 5 }',
    ],
    "fog": [
        'scene { units meters }',
        'camera "c" { eye 0.5 0.4 -1.2  look_at 0.5 0.25 0.6  up 0 1 0  fov_y 36  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.5 }',
        'material "red"    { type diffuse  reflect rgb 0.7 0.15 0.1 }',
        'material "blue"   { type diffuse  reflect rgb 0.1 0.2 0.7 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'sphere { center 0.3 0.15 0.5  radius 0.15  material red }',
        'sphere { center 0.75 0.15 0.7  radius 0.15  material blue }',
        'light sun { dir 0.5 0.7 -0.4  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'light area { origin 0.0 1.2 0.0  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 200  spd preset:d65 }',
        'medium { sigma_t 2.0  albedo 0.85  g 0.5  bounds { min -0.2 0 0.0  max 1.2 0.8 1.2 } }',
    ],
    "fogcli": [
        # The fog scene's geometry with the CLI's `-fog` instead of a medium block (EXTRA): a
        # medium added after the scene is built, which must still be baked as grey.
        'scene { units meters }',
        'camera "c" { eye 0.5 0.4 -1.2  look_at 0.5 0.25 0.6  up 0 1 0  fov_y 36  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.5 }',
        'material "red"    { type diffuse  reflect rgb 0.7 0.15 0.1 }',
        'material "blue"   { type diffuse  reflect rgb 0.1 0.2 0.7 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'sphere { center 0.3 0.15 0.5  radius 0.15  material red }',
        'sphere { center 0.75 0.15 0.7  radius 0.15  material blue }',
        'light sun { dir 0.5 0.7 -0.4  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'light area { origin 0.0 1.2 0.0  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 200  spd preset:d65 }',
    ],
    "coatfog": [
        # An iridescent coat -- a high-index film, its reflectance swinging across the band --
        # splits the bundle into one sub-path per wavelength at most hits, each RE-TRACING the
        # segment that reached the coat. In fog that segment's free flight must not be drawn a
        # second time; the fog is a pure absorber so that a second draw can only darken (a
        # scattering fog would hand some of it back as in-scatter and mask the error).
        'scene { units meters }',
        'camera "c" { eye 0.5 0.3 -1.0  look_at 0.5 0.2 0.5  up 0 1 0  fov_y 24  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.5 }',
        'material "base"   { type diffuse  reflect 0.05 }',
        'material "irid"   { type layered  ior 1.5  coat { reflectance thinfilm  film_ior 2.4  film_thickness 300  roughness 0.02 }  layer "base" 1.0 }',
        'quad { origin -2 0 -2   u 0 0 5   v 5 0 0   material ground }',
        'sphere { center 0.5 0.2 0.5  radius 0.2  material irid }',
        'light env { spd 1.0 }',
        'medium { sigma_t 1.5  albedo 0.0  g 0.0  bounds { min -0.5 0 -0.5  max 1.5 0.9 1.5 } }',
    ],
    "rain": [
        'scene { units meters }',
        'camera "c" { eye 0 1.0 -4  look_at 0 1.6 4  up 0 1 0  fov_y 60  film { res 160 120 } }',
        'material "ground" { type diffuse  reflect 0.08 }',
        'quad { origin -20 0 -20   u 0 0 40   v 40 0 0   material ground }',
        'light sun { dir 0 0.55 -0.83  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
        'medium { sigma_t 0.35  albedo 0.99  bounds { min -8 0 2  max 8 8 5 }  phase rainbow { droplet_um 300  dispersion 0.577 } }',
    ],
}


# A tile fails when its bundle mean is off the single-wavelength mean by more than Z_MAX standard
# errors (a real difference, not noise) AND by more than REL_MAX (one big enough to matter).
Z_MAX = 5.0
REL_MAX = 0.01

# The modes each scene runs in: BDPT (mode D) refuses an environment light.
MODES = {"hair": "R", "hairsun": "RD", "hairback": "RD", "hairmed": "RD", "fog": "RD", "fogcli": "R",
         "coatfog": "R", "rain": "RD"}
# Command-line arguments a scene renders with.
EXTRA = {"fogcli": ["-fog", "1.5"]}


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data[::-1].astype(np.float64)     # PFM rows run bottom to top


def render(exe, scene, out, mode, heroc, seed, spp, device="cpu"):
    png = os.path.join(out, "%s_%s%s_c%d_s%d_n%d.png" % (
        os.path.splitext(os.path.basename(scene))[0], mode, "g" if device == "gpu" else "", heroc,
        seed, spp))
    pfm = png[:-4] + ".pfm"
    if os.path.exists(pfm) and os.path.getmtime(pfm) > os.path.getmtime(scene) and \
            os.path.getmtime(pfm) > os.path.getmtime(exe):
        return read_pfm(pfm)
    if os.path.exists(pfm):
        os.remove(pfm)                       # never score an earlier run's image
    cmd = [exe, "-in", scene, "-mode", mode, "-device", device, "-spp", str(spp), "-heroc", str(heroc),
           "-seed", str(seed), "-hdr", "-no-meter", "-o", png]
    cmd += EXTRA.get(os.path.splitext(os.path.basename(scene))[0], [])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(pfm):
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("render failed: " + " ".join(cmd))
    if device == "gpu" and "-> GPU" not in r.stdout + r.stderr:
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    return read_pfm(pfm)


def compare(x1, x2, y1, y2):
    """Tile verdict of the mean of renders x1, x2 (two seeds) against the mean of y1, y2: the
    failing tiles, the whole-frame luminance ratio and the worst tile as (sigma, ratio). The
    standard error of the difference is taken two ways and the larger kept: from each seed
    pair's tile means (one degree of freedom: can come out small by luck) and from their pixels
    (many, but blind to correlation between pixels)."""
    lumx = ((tiles(x1) + tiles(x2)) / 2).sum(-1)
    lumy = ((tiles(y1) + tiles(y2)) / 2).sum(-1)
    se_pair = np.sqrt(((tiles(y1) - tiles(y2)).sum(-1) / 2) ** 2 +
                      ((tiles(x1) - tiles(x2)).sum(-1) / 2) ** 2)
    se_pix = np.sqrt(tile_se(y1, y2) ** 2 + tile_se(x1, x2) ** 2)
    selum = np.maximum(se_pair, se_pix)
    floor = 0.02 * lumy.max()                      # tiles this dark carry no information
    bad = []
    worst = (0.0, 1.0)
    for i in range(lumy.shape[0]):
        for j in range(lumy.shape[1]):
            if lumy[i, j] < floor:
                continue
            ratio = lumx[i, j] / lumy[i, j]
            z = abs(lumx[i, j] - lumy[i, j]) / max(selum[i, j], 1e-30)
            if z > worst[0]:
                worst = (z, ratio)
            if z > Z_MAX and abs(ratio - 1.0) > REL_MAX:
                bad.append("tile (%d,%d): x%.3f, %.1f sigma" % (i, j, ratio, z))
    return bad, lumx.sum() / lumy.sum(), worst


def tiles(img, n=4):
    h, w, _ = img.shape
    return np.array([[img[h * i // n:h * (i + 1) // n, w * j // n:w * (j + 1) // n].mean(axis=(0, 1))
                      for j in range(n)] for i in range(n)])          # n x n x 3


def tile_se(a, b, n=4):
    """Standard error of each tile's luminance in the mean of two seeds, (a + b) / 2: from the
    per-pixel seed differences, whose variance is twice a pixel's, summed over the tile -- far
    more degrees of freedom than the one the two tile means give. Pixels are taken as
    independent, which a light-tracing splat (mode D's t = 1) breaks a little; the caller also
    keeps the tile-mean estimate and uses the larger."""
    d2 = ((a - b).sum(-1)) ** 2
    h, w = d2.shape
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            blk = d2[h * i // n:h * (i + 1) // n, w * j // n:w * (j + 1) // n]
            out[i, j] = np.sqrt(blk.sum()) / (2.0 * blk.size)
    return out


def chroma_noise(a, b):
    """Mean |difference| of the chromaticities (r, b over r+g+b) between two realizations, over
    the pixels bright enough in both to have a colour (linear sRGB can go negative on a
    saturated colour, so no logarithms)."""
    sa, sb = a.sum(-1), b.sum(-1)
    floor = 0.05 * np.median(np.concatenate([sa[sa > 0], sb[sb > 0]]))
    m = (sa > floor) & (sb > floor)
    if not m.any():
        return 0.0
    ca = a[m] / sa[m][:, None]; cb = b[m] / sb[m][:, None]
    return float(np.abs(ca[:, [0, 2]] - cb[:, [0, 2]]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_hero_spectral"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--only", default="")
    ap.add_argument("--modes", default="RD", help="the modes to run, of those each scene allows")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="gpu: render mode R on the device, and also hold its -heroc 1 render "
                         "against the host's")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    names = [n for n in SCENES if not a.only or n in a.only.split(",")]
    ok = True
    for name in names:
        scene = os.path.join(out, name + ".ftsl")
        text = NL.join(SCENES[name]) + NL
        if not os.path.exists(scene) or open(scene, encoding="utf-8").read() != text:
            with open(scene, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        for mode in MODES[name]:
            # The device runs mode R's bundle; its mode D still keeps media to one wavelength.
            if mode not in a.modes or (a.device == "gpu" and mode != "R"):
                continue
            r = {(c, sd): render(a.exe, scene, out, mode, c, sd, a.spp, a.device)
                 for c in (1, 4) for sd in (1, 2)}
            bad, whole, worst = compare(r[(4, 1)], r[(4, 2)], r[(1, 1)], r[(1, 2)])
            cn1 = chroma_noise(r[(1, 1)], r[(1, 2)]); cn4 = chroma_noise(r[(4, 1)], r[(4, 2)])
            if np.array_equal(r[(1, 1)], r[(4, 1)]):
                bad.append("-heroc 4 rendered bit-identically to -heroc 1: the bundle is off")
            good = not bad
            ok &= good
            print("%s: %s mode %s%s  whole frame x%.4f; worst tile %.1f sigma (x%.3f); chroma noise "
                  "%.3f -> %.3f (x%.2f)%s" % (
                      "ok" if good else "FAIL", name, mode, " (device)" if a.device == "gpu" else "",
                      whole, worst[0], worst[1], cn1, cn4,
                      cn1 / max(cn4, 1e-30), ("" if good else "; " + "; ".join(bad))), flush=True)
            if a.device == "gpu":
                # The device's single-wavelength walk against the host's: the two share the
                # volume NEE with the bundle, so a fault there would pass the comparison above.
                h = {sd: render(a.exe, scene, out, mode, 1, sd, a.spp) for sd in (1, 2)}
                bad, whole, worst = compare(r[(1, 1)], r[(1, 2)], h[1], h[2])
                good = not bad
                ok &= good
                print("%s: %s mode %s  device / host at -heroc 1: whole frame x%.4f; worst tile "
                      "%.1f sigma (x%.3f)%s" % (
                          "ok" if good else "FAIL", name, mode, whole, worst[0], worst[1],
                          ("" if good else "; " + "; ".join(bad))), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
