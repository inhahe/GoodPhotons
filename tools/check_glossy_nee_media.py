"""Regression check: a glossy bounce's MIS state does not leak past a medium scatter (mode R).

    D:/python314/python.exe -u tools/check_glossy_nee_media.py [--out <dir>] [--exe ftrace.exe]
                                                               [--spp N]

Glossy NEE (on by default; `-no-glossy-nee` turns it off) connects a glossy vertex to the lights
and records the lobe's density, so that the next vertex can weigh a light the continuation hits
against that connection. The record is cleared by the next material interaction -- but a medium
scatter (and an aggregate-fur collision) is a vertex the material code never sees, and before
0.375.0 it left the record standing: a path glossy -> fog scatter -> sky had its sky escape
weighted by the GLOSSY lobe's density instead of the phase function's, which the fog vertex's own
sky NEE had used. Both estimators are unbiased, so the image must not depend on the flag.

A glossy floor under a grey fog, lit by the sky alone, rendered in mode R with glossy NEE on and
off, two seeds each, at one wavelength per path (-heroc 1) and with the bundle (-heroc 4). The
frame means must agree within 4 standard errors and 1 %. Exits 1 on any failure.
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
    'camera "c" { eye 0 0.6 -2.0  look_at 0 0.1 0.6  up 0 1 0  fov_y 45  film { res 96 64 } }',
    'material "floor" { type glossy  reflect 0.9  roughness 0.15 }',
    'material "wall"  { type diffuse  reflect 0.5 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'quad { origin -3 0 2.5  u 6 0 0   v 0 3 0   material wall }',
    'light env { spd 1.0 }',
    'medium { sigma_t 0.9  albedo 0.95  g 0.3  bounds { min -3 0 -3  max 3 1.2 2.5 } }',
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
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glossy_nee_media"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=512)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = os.path.join(out, "s.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    ok = True
    for heroc in (1, 4):
        means = {}
        for flag in ("on", "off"):
            vals = []
            for seed in (1, 2):
                png = os.path.join(out, "c%d_%s_s%d.png" % (heroc, flag, seed))
                cmd = [a.exe, "-in", scene, "-mode", "R", "-device", "cpu", "-spp", str(a.spp),
                       "-heroc", str(heroc), "-seed", str(seed), "-hdr", "-no-meter", "-o", png]
                if flag == "off":
                    cmd.append("-no-glossy-nee")
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                pfm = png[:-4] + ".pfm"
                if r.returncode != 0 or not os.path.exists(pfm):
                    print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit("render failed")
                vals.append(read_pfm(pfm).mean())
            means[flag] = (np.mean(vals), abs(vals[0] - vals[1]) / 2)
        (m1, e1), (m0, e0) = means["on"], means["off"]
        z = abs(m1 - m0) / max(np.hypot(e1, e0), 1e-30)
        ratio = m1 / m0
        good = not (z > 4.0 and abs(ratio - 1.0) > 0.01)
        ok &= good
        print("%s: heroc %d: glossy NEE on / off = x%.4f (%.1f sigma)" % ("ok" if good else "FAIL", heroc, ratio, z))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
