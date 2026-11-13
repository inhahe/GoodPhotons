"""Regression check: a photon gather on a curve keeps the photons on the neighbouring strands.

    python tools/check_fur_gather.py [--out png/check_fur_gather] [--exe ftrace.exe] [--record]

A ball in a coat of `type diffuse` fur (fur_creature's density and strand radius) on a floor under
a ceiling panel. A gather on a strand is no surface estimate: the photons in its ball lie on the
neighbouring strands. Since 0.433.3 modes M and S never gather there (known-issues CURVE-GATHER):
they light a strand by NEE and walk on along its diffuse lobe, and deposit no photon on one. Before
that the coat read, of mode R, M 0.71 (GPU) / 0.87 (CPU) and S 1.33 / 1.45; at 0.440.12, 0.989 /
1.014 and 0.985 / 0.981. Mode U merged on strands until 0.439.2 (VCM-THIN-CURVES), at 2.10 / 2.36;
it merges at no curve vertex now, 1.000 / 1.014. Mode M runs with its final gather (`-pmfg`) too,
whose sub-walk has taken the same step since 0.433.6: 0.992 / 1.063.

The coat is the mean over the ball's middle, two seeds, against mode R's on the GPU. In modes M, S
and U on the CPU and the GPU it must lie within 10 % of the value recorded below; `--record` prints
the values to paste in after a change that moves them. Exits 0 when every case holds, 1 otherwise.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_motion as cm  # noqa: E402

NL = chr(10)
SCENE = NL.join([
    'scene { units meters }',
    'camera "c" { eye 0 0.22 -0.42  look_at 0 0.09 0  up 0 1 0  fov_y 34  film { res 96 72 } }',
    'material "ground" { type diffuse  reflect 0.4 }',
    'material "skin"   { type diffuse  reflect 0.3 }',
    'material "coat"   { type diffuse  reflect 0.5 }',
    'quad { origin -1 0 -1   u 0 0 2   v 2 0 0   material ground }',
    'light area { origin -0.3 0.8 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  power 200  spd preset:d65 }',
    'sphere "ball" { center 0 0.09 0  radius 0.07  material skin }',
    'fur "coat" {',
    '    on "ball"  material coat  density 450000',
    '    points 4  segments 2',
    '    length 0.03  length_jitter 0.3',
    '    radius 0.0004',
    '    droop 0.3  jitter 0.3',
    '    seed 7',
    '}',
]) + NL
REF = ["-mode", "R", "-device", "gpu", "-spp", "512"]
# (name, arguments, the coat's value against mode R as recorded)
CASES = [("M_gpu", ["-mode", "M", "-device", "gpu", "-n", "4000000", "-spp", "16"], 0.989),
         ("M_cpu", ["-mode", "M", "-device", "cpu", "-n", "2000000", "-spp", "4"], 1.014),
         # the final gather, whose sub-walk meets the strands from the skin
         ("Mfg_gpu", ["-mode", "M", "-device", "gpu", "-n", "4000000", "-spp", "16", "-pmfg", "16"], 0.992),
         ("Mfg_cpu", ["-mode", "M", "-device", "cpu", "-n", "2000000", "-spp", "2", "-pmfg", "8"], 1.063),
         ("S_gpu", ["-mode", "S", "-device", "gpu", "-spp", "16"], 0.985),
         ("S_cpu", ["-mode", "S", "-device", "cpu", "-spp", "4"], 0.981),
         ("U_gpu", ["-mode", "U", "-device", "gpu", "-spp", "16"], 1.000),
         ("U_cpu", ["-mode", "U", "-device", "cpu", "-spp", "4"], 1.014)]
TOL = 0.10


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_fur_gather"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--record", action="store_true", help="print each case's value to record, and pass")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "fur_ball.ftsl")
    cm.write(path, SCENE)
    # The ball's middle: coat seen face-on, no floor and no silhouette.
    mask = np.zeros((72, 96), bool)
    mask[10:62, 28:68] = True

    def coat(tag, args, gpu):
        imgs = [cm.lum(cm.render(exe, out, path, "%s_%d" % (tag, s), args, seed=s, want_gpu=gpu)[0])
                for s in (1, 2)]
        return float(np.mean([im[mask].mean() for im in imgs]))

    ref = coat("R_gpu", REF, True)
    ok = True
    for name, args, want in CASES:
        got = coat(name, args, "gpu" in name) / ref
        if a.record or want is None:
            print("%-7s coat %.3f of mode R  (record)" % (name, got))
            continue
        good = abs(got / want - 1.0) <= TOL
        ok &= good
        print("%-7s coat %.3f of mode R, recorded %.3f  %s" % (name, got, want, "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
