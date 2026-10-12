"""Regression check: mode W renders fur in its true colour at 1 spp.

    D:/python314/python.exe -u tools/check_modew_fiber.py [--out <dir>] [--exe ftrace.exe]

Mode W's wavelength lattice is a function of the sample index alone, shared by every pixel, so a
path that collapses its hero bundle onto one wavelength (a de-hero) renders the WHOLE frame at that
wavelength on that pass: a uniform colour cast that only more spp averages away. At a fiber, mode W
therefore splits the bundle into one sub-path per wavelength (`heroSplit`, always on in mode W),
as it does at glass. 0.375.0 de-heroed there instead, and every fur in the frame came out one
saturated hue per pass.

Renders a close-up of a fur ball (pale fur filling most of the frame) in mode W at 1 spp and at 64
spp and requires the frame's mean chromaticity (r and b over r+g+b) to agree within 0.05 (a
de-heroed fiber is off by 0.25; eight wavelengths at 1 spp leave ~0.02). Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENE = [
    'scene { units meters }',
    'camera "c" { eye 0 0.15 -0.42  look_at 0 0.13 0  up 0 1 0  fov_y 40  film { res 96 72 } }',
    'material "ground" { type diffuse  reflect 0.4 }',
    'material "skin"   { type diffuse  reflect rgb 0.35 0.26 0.22 }',
    'material "pale"   { type hair  reflect rgb 0.85 0.70 0.40  beta_m 0.25  beta_n 0.30 }',
    'quad { origin -2 0 -2   u 0 0 4   v 4 0 0   material ground }',
    'light sun { dir -0.4 0.6 -0.7  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
    'light area { origin -0.4 0.9 -0.3  u 0.8 0 0  v 0 0 0.6  normal 0 -1 0  power 120  spd preset:d65 }',
    'sphere "b" { center 0 0.13 0  radius 0.10  material skin }',
    'fur "coat" { on "b"  material pale  count 25000  length 0.05  length_jitter 0.3  radius 0.0006  droop 0.5  jitter 0.25  seed 1 }',
]


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_modew_fiber"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = os.path.join(out, "fur.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    chroma = {}
    for spp in (1, 64):
        png = os.path.join(out, "w_%d.png" % spp)
        cmd = [a.exe, "-in", scene, "-mode", "W", "-device", "cpu", "-spp", str(spp), "-hdr",
               "-no-meter", "-o", png]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        pfm = png[:-4] + ".pfm"
        if r.returncode != 0 or not os.path.exists(pfm):
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        m = read_pfm(pfm).reshape(-1, 3).mean(axis=0)
        chroma[spp] = m / m.sum()
    d = np.abs(chroma[1] - chroma[64])[[0, 2]].max()
    ok = d < 0.05
    print("%s: mode W fur chromaticity, 1 spp (r %.3f b %.3f) vs 64 spp (r %.3f b %.3f): off by %.3f"
          % ("ok" if ok else "FAIL", chroma[1][0], chroma[1][2], chroma[64][0], chroma[64][2], d))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
