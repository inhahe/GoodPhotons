"""Regression check: mode D's light sampling does not depend on its light subpath getting started.

    D:/python314/python.exe -u tools/check_bdpt_s1.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                      [--no-gpu]

BDPT's s = 1 strategy -- an eye vertex connected to a point it samples on a light -- draws its own
light, but mode D ran it only when the frame's light subpath had come back with a first vertex,
and a light subpath gives up where its emitter is dark: at a point a `spd_map` pattern zeroes, or
at a wavelength outside the emitter's band. Every path the s = 1 strategy reaches was then weighted
for samples it never took, and the image came out dark. A light subpath also gave up when its
emitter was dark at the HERO wavelength alone, dropping the rest of the bundle with it.

Writes a box lit by two narrow-band lamps (a `table` band with exact zeros outside 440-480 nm and
600-660 nm) and a white lamp dark over half its face (`spd_map` of max(0, sin 2 pi u)), and
compares mode D with mode R -- an independent estimator -- tile by tile and over the frame, two
seeds each: at -heroc 4 and at -heroc 1; then mode D on the device against the host (`--no-gpu`
skips it). A tile fails when two renders differ by more than 5 standard errors AND by more than
2 %; the frame by more than 5 standard errors AND 0.5 %. Exits 1 on any failure.
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
    'camera "c" { eye 0.5 0.5 -1.1  look_at 0.5 0.45 0.5  up 0 1 0  fov_y 50  film { res 64 48 } }',
    'spectrum "blue" = table { 360:0 439.9:0 440:1 480:1 480.1:0 830:0 }',
    'spectrum "red"  = table { 360:0 599.9:0 600:1 660:1 660.1:0 830:0 }',
    'pattern "half" { expr "max(0, sin(2*pi*u))" }',
    'material "white" { type diffuse  reflect whitewall 0.75 }',
    'material "grey"  { type diffuse  reflect 0.35 }',
    'quad { origin 0 0 0   u 1 0 0   v 0 0 1   material white }',
    'quad { origin 0 1 0   u 0 0 1   v 1 0 0   material white }',
    'quad { origin 0 0 1   u 0 1 0   v 1 0 0   material grey }',
    'quad { origin 0 0 0   u 0 0 1   v 0 1 0   material white }',
    'quad { origin 1 0 0   u 0 1 0   v 0 0 1   material white }',
    'light area { origin 0.1 0.999 0.2  u 0.25 0 0  v 0 0 0.25  normal 0 -1 0  power 20  spd spectrum:blue }',
    'light area { origin 0.65 0.999 0.2  u 0.25 0 0  v 0 0 0.25  normal 0 -1 0  power 20  spd spectrum:red }',
    'light area { origin 0.3 0.999 0.6  u 0.4 0 0  v 0 0 0.3  normal 0 -1 0  power 30  spd preset:d65',
    '             spd_map pattern:half }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_bdpt_s1"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "box.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)

    def render(tag, args):
        """Two seeds of one configuration; a `-device gpu` render must say it ran there."""
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            for f in (png, png[:-4] + ".pfm"):     # never score an earlier run's image
                if os.path.exists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
            cmd = [a.exe, "-in", "box.ftsl", "-seed", str(seed), "-hdr", "-no-meter",
                   "-spp", str(a.spp), "-o", png] + args
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            if "gpu" in args and "-> GPU" not in p.stdout + p.stderr:
                raise SystemExit("did not run on the GPU: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    pairs, host = [], {}
    for c in ("4", "1"):
        host[c] = render("d_c" + c, ["-mode", "D", "-device", "cpu", "-heroc", c])
        rr = render("r_c" + c, ["-mode", "R", "-device", "cpu", "-heroc", c])
        pairs.append(("mode D / mode R, -heroc " + c, host[c], rr))
    if not a.no_gpu:
        dg = render("d_gpu", ["-mode", "D", "-device", "gpu", "-heroc", "4"])
        pairs.append(("mode D device / host, -heroc 4", dg, host["4"]))
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
