"""Regression check: the modes agree with mode R on a normal-mapped surface (BUMP-MODES).

    python tools/check_bump_modes.py [--out png/check_bump_modes] [--exe ftrace.exe] [--no-gpu]
                                     [--modes D,J,U,B,M,S] [--seeds 3]

A shading normal can tilt so far that a ray meets the surface from the geometric front but below
it (ns.wo <= 0 < ng.wo). Mode R lights such a point: its integrand is a Lambertian lobe lit by
light from above ns, whichever side of ns the camera side lies on (bsdf_eval.h lambertLit). Every
other mode has to share that integrand, or it renders a bumped surface brighter or darker than R.
Before 0.439.9 mode U read the room below at 0.844 of R, D 0.959 (0.989 once 0.439.7 stopped its
weights counting strategies that cannot happen), J 0.964 (0.983); before 0.439.10 mode M 1.148, B
1.008.

The room: five walls under one ripple normal map, a tangent-space slope of 2 (the normal tilted up
to 63 deg), lit at a grazing angle by a tilted panel, 64x48. The same room with a flat map is the
control: any mode that misses it there is wrong for another reason. Every mode runs at -max-bounce 8
(mode R's default is 32, the bidirectional modes' 8, and an albedo-0.8 room keeps a lot of light in
its long paths). Each mode's frame mean over `--seeds` seeds must be within 1 % of mode R's on the
same device, in both rooms (the seeds spread about 0.1 %; three of them are too few to let a
standard error widen the band). Modes M and S deposit photons over 32 bounces whatever
-max-bounce says (photonMaxBounce), so they and their mode-R reference run at -max-bounce 32; at 8
they read the flat room 2 % bright. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

SPP = {"R": 2048, "D": 512, "J": 256, "U": 512, "B": 1, "M": 64, "S": 256}   # CPU; the GPU takes 4x
# (mode S: passes, and progressive photon mapping's bias shrinks with them: on the CPU the flat
# room reads 0.991 of R at 64, 0.997 at 256)
NPHOT = {"B": 40000000, "M": 2000000, "S": 100000}   # photons (mode S's per pass)
DEPTH = {"M": 32, "S": 32}   # -max-bounce: the photon deposits' own depth; 8 for the rest
TOL = 0.01


def ripple(path, amp, w=64):
    """A tangent-space normal map of ripples along u, slope `amp`, as a binary PPM."""
    u = (np.arange(w) + 0.5) / w
    slope = amp * np.sin(2 * np.pi * 4 * u)
    n = np.stack([slope, np.zeros(w), np.ones(w)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = np.round((n * 0.5 + 0.5) * 255).astype(np.uint8)
    img = np.broadcast_to(rgb[None, :, :], (w, w, 3))
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, w))
        f.write(np.ascontiguousarray(img).tobytes())


def room(d, amp):
    ppm = os.path.join(d, "rip.ppm")
    ripple(ppm, amp)
    p = ppm.replace(chr(92), "/")
    return NL.join([
        "scene { units meters }",
        'texture "rip" { file "%s"  encoding linear  filter bilinear  wrap repeat }' % p,
        'material "w" { type diffuse reflect 0.8  normal_map texture:rip }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w }",
        "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w }",
        "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w }",
        "light area { origin -1.9 0.2 -1.5  u 0 0 0.6  v 0 2.4 0  normal 1 0 0.2  spd preset:bb6500  power 40 }",
        'camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
    ]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_bump_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--modes", default="D,J,U,B,M,S")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    fails = []
    for name, amp in (("flat", 0.0), ("bumped", 2.0)):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        scene = os.path.join(d, "room.ftsl")
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(room(d, amp))

        def frame(mode, dev, seed, depth):
            png = os.path.join(d, "%s_%s_%d_%d.png" % (mode, dev, depth, seed))
            cmd = [exe, "-in", scene, "-mode", mode, "-device", dev,
                   "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)),
                   "-n", str(NPHOT.get(mode, 200000)),
                   "-max-bounce", str(depth), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
                   "-o", png]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            return float(read_pfm(pfm, top_down=True).sum(-1).mean())

        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            refs = {}                                # mode R at each depth asked for
            for mode in a.modes.split(","):
                depth = DEPTH.get(mode, 8)
                if depth not in refs:
                    r = [frame("R", dev, s, depth) for s in seeds]
                    refs[depth] = (np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r)))
                rm, rse = refs[depth]
                v = [frame(mode, dev, s, depth) for s in seeds]
                m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
                ratio = m / rm
                err = np.hypot(se / rm, m * rse / rm ** 2)
                good = abs(ratio - 1.0) <= TOL
                print("%s: %-6s mode %s on the %s: %.4f of mode R (+-%.4f)"
                      % ("ok" if good else "FAIL", name, mode, "host" if dev == "cpu" else "device",
                         ratio, err), flush=True)
                if not good:
                    fails.append("%s %s %s" % (name, mode, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
