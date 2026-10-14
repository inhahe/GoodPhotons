"""Regression check: modes M and S render a coated body as one -- refracting, in its own colour.

    D:/python314/python.exe -u tools/check_coat_modes.py [--out <dir>] [--exe ftrace.exe]
                                                         [--spp N] [--no-gpu]

A thin-film or multilayer body with a clear substrate is a dielectric body with a coating: it
reflects with the coating's interference reflectance -- its whole colour -- and refracts the rest
(Renderer::interfaceStep). Until 0.378.5 the camera walks of modes M and S bounced off one as off
a half mirror (the material's unset `reflect` slot), so a soap bubble or a dichroic ball rendered
grey and opaque there. This renders such a bubble and a dichroic ball, each around a diffuse core,
over a floor and in front of a wall, and requires against mode R, per tile:
  * mode M, each colour channel alike (its spectral gather re-expresses the coating's coin at
    every photon's wavelength -- Renderer::stepFactorAt);
  * mode S, the sum of the channels (mode S folds no spectral factor per photon for any
    material: known-issues SPPM-SPECTRAL);
  * unless --no-gpu, both on the device against the host.
A tile fails when the two differ by more than 5 standard errors (two seeds each) AND by more
than 5 %. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.45 -2.2  look_at 0 0.35 0  up 0 1 0  fov_y 34  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "wall"  { type diffuse  reflect rgb 0.7 0.6 0.5 }',
    'material "core"  { type diffuse  reflect rgb 0.3 0.5 0.8 }',
    'material "bubble" { type thinfilm  ior 1.33  film_ior 1.28  film_thickness 380 }',
    'material "dichroic" { type multilayer  ior 1.5  layer 2.3 0.0 60  layer 1.38 0.0 100'
    '  layer 2.3 0.0 60  layer 1.38 0.0 100 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'quad { origin -3 0 1.2  u 6 0 0   v 0 3 0   material wall }',
    'sphere { center -0.4 0.3 0  radius 0.28  material bubble }',
    'sphere { center -0.4 0.3 0  radius 0.14  material core }',
    'sphere { center 0.4 0.3 0  radius 0.28  material dichroic }',
    'sphere { center 0.4 0.3 0  radius 0.14  material core }',
    'light area { origin -0.5 2.0 -1.0  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 120  '
    'spd preset:d65 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_coat_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "coats.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)

    def render(tag, args, device="cpu"):
        """Two seeds of one configuration (RGB images); a device render must say it ran there."""
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            pfm = os.path.join(d, png[:-4] + ".pfm")
            for f in (os.path.join(d, png), pfm):       # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            cmd = [a.exe, "-in", "coats.ftsl", "-device", device, "-seed", str(seed), "-hdr",
                   "-no-meter", "-spp", str(a.spp), "-o", png] + args
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            log = p.stdout + p.stderr
            if device == "gpu" and not any(t in log for t in ("-> GPU", "[gpu]", "SPPM on ")):
                raise SystemExit("did not run on the GPU: " + " ".join(cmd))
            imgs.append(read_pfm(pfm))
        return imgs

    modeM = ["-mode", "M", "-n", "20000000"]
    modeS = ["-mode", "S", "-n", "2000000"]
    r = render("R", ["-mode", "R"])
    m = render("M", modeM)
    s = render("S", modeS)
    pairs = [("mode M / mode R, channel %d" % c, [x[..., c] for x in m], [x[..., c] for x in r])
             for c in range(3)]
    pairs.append(("mode S / mode R, luminance", [x.sum(-1) for x in s], [x.sum(-1) for x in r]))
    if not a.no_gpu:
        mg = render("Mg", modeM, "gpu")
        sg = render("Sg", modeS, "gpu")
        pairs += [("mode M device / host, channel %d" % c, [x[..., c] for x in mg],
                   [x[..., c] for x in m]) for c in range(3)]
        pairs.append(("mode S device / host, luminance", [x.sum(-1) for x in sg],
                      [x.sum(-1) for x in s]))
    ok = True
    for name, x1, x0 in pairs:
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), 0.05)
        good = not bad.any()
        ok &= good
        print("%s: %s; worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, ratio.flat[int(z.argmax())], z.max()), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
