"""Regression check: a MIS weight does not count a strategy that cannot happen (REMAP0-IMPOSSIBLE).

    python tools/check_glow_mis.py [--out png/check_glow_mis] [--exe ftrace.exe] [--spp N] [--no-gpu]

A bidirectional MIS weight sums, over every strategy that could have built the path, the ratio of
its density to the sampled one's, one vertex at a time. A density of 0 there means one of two
things: the vertex the strategy would generate this one from is a delta (a mirror, glass), whose
density cancels between the strategies and is read as 1; or the strategy cannot generate the
vertex at all, and every strategy that needs it is impossible. Before 0.439.7 the weights read
every 0 as the first kind (but a sun's NEE, special-cased in 0.391.5), so the strategy that did
fire was weighed against ones that never do.

The glow was the case to see it on: a surface whose material emits but that had no emitter (an
isosurface, until 0.442.0) could be reached only by the eye path landing on it (s = 0), and modes
D and J weighed that against light tracing from the glow at a ratio that depended on the scene's
units: the frame read 0.748 of mode U on both devices. Since 0.442.0 the same isosurface is an
emitter (a Glow, sampled by the lines through its box: ftsl.h registerGlows) and lights the floor
in every mode, so the check now holds the bidirectional modes' weights of a Glow's light -- its
light subpaths and light sampling against the eye path landing on it, which weigh the Glow's
crossing-weighted draws by its nominal density -- to mode U on the host: the camera sees a floor
lit by the glow sphere, standing behind the camera, and by a lamp. The frame's mean, both seeds
averaged, must be within 3 % of it in modes D and J on the host and the device, and in mode U on
the device (the seeds differ by about 0.5 %; two of them are too few to let a standard error widen
the band). (A glow that lights nothing, `glow_only`, is tools/check_glow_modes.py's.) Exits 1 on
any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

SCENE = [
    "scene { units meters  spectral 360 830 5 }",
    # The floor ahead; the glowing sphere stands behind the camera, out of its view, so every
    # pixel is the floor and the glow reaches it only through the floor's diffuse bounce.
    'camera "c" { eye 0 1.4 -1.0  look_at 0 0 1.0  up 0 1 0  fov_y 40  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.5 }',
    "quad { origin -3 0 -3   u 0 0 7   v 6 0 0   material floor }",
    'material "glow" { type diffuse  reflect 0.6  emit 0.05 }',
    "isosurface { material glow  sphere { center 0 0.7 -1.8  radius 0.4 } }",
    "light area { origin -0.3 1.8 0.7  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  power 20  spd preset:d65 }",
]
REF = ("U", "cpu")
ARMS = [("D", "cpu"), ("D", "gpu"), ("J", "cpu"), ("J", "gpu"), ("U", "gpu")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_glow_mis"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "glow.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)

    def frame(mode, dev):
        """The frame's mean (X+Y+Z) in each of two seeds."""
        out = []
        for seed in (1, 2):
            png = os.path.join(d, "%s_%s_s%d.png" % (mode, dev, seed))
            cmd = [exe, "-in", os.path.join(d, "glow.ftsl"), "-mode", mode, "-device", dev,
                   "-spp", str(a.spp), "-n", "200000", "-seed", str(seed), "-hdr", "-no-meter",
                   "-window-min", "-o", png]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            out.append(float(read_pfm(pfm, top_down=True).sum(-1).mean()))
        return out

    r = frame(*REF)
    ref, ref_err = (r[0] + r[1]) / 2, abs(r[0] - r[1]) / 2
    print("reference: mode %s on the %s, frame mean %.4g (seeds %.4g / %.4g)"
          % (REF[0], "host" if REF[1] == "cpu" else "device", ref, r[0], r[1]), flush=True)
    ok = True
    for mode, dev in ARMS:
        if dev == "gpu" and a.no_gpu:
            continue
        v = frame(mode, dev)
        mean, err = (v[0] + v[1]) / 2, abs(v[0] - v[1]) / 2
        ratio = mean / ref
        sigma = abs(mean - ref) / max((err ** 2 + ref_err ** 2) ** 0.5, 1e-30)
        good = abs(ratio - 1.0) <= 0.03
        ok &= good
        print("%s: mode %s on the %s, a floor a glow with no emitter lights: x%.4f of the "
              "reference (%.1f sigma)" % ("ok" if good else "FAIL", mode,
                                          "host" if dev == "cpu" else "device", ratio, sigma),
              flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
