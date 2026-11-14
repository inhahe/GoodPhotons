"""Regression check: every mode renders a `layered` material as mode R does (LAYERED-MODES).

    python tools/check_layered_modes.py [--out png/check_layered_modes] [--exe ftrace.exe] [--no-gpu]
                                        [--modes D,J,U,B,M,S] [--seeds 3] [--rooms fresnel,...]

A layered material is a clear coat over a body. Light that the body scatters crosses the coat twice,
in and out, and a physical coat transmits T = 1 - R each way: f = T(in) T(out) / T-bar * f_body, where
T-bar is T's cosine-weighted mean (the body's albedo already holds the averaged escape). Every walk
takes the coat at its own arrival by a coin (reflect with R, else enter), and the body's lobe takes it
on the other direction, S(w) = T(w) / T-bar (render.h coatExitS), so the integrand is reciprocal and
the modes agree by construction. Before 0.444.0 the body lobe took the coat only at the walk's
arrival: mode R's integrand held T toward the camera, the light walks' T toward the light, and on the
room below at coat roughness 0 the host read D 1.014, J 1.002, U 1.012, B 0.913, M 1.003 and S 1.004
of R.

The rooms: five walls of one material lit at a grazing angle by a tilted panel, 64x48, every mode at
-max-bounce 8 (M and S, which deposit photons over 32 bounces, and their mode-R reference at 32).
  fresnel   a smooth Fresnel coat (ior 1.5, roughness 0) over a diffuse body (0.8)
  rough     the same coat at roughness 0.3
  film      a thin-film coat (film_ior 1.33, 380 nm over ior 1.5): R and with it S swing across the
            spectrum, so the factor is read per wavelength
  glossy    the Fresnel coat over a glossy body (roughness 0.3): car paint
  sheet     the diffuse room with a coated translucent panel standing in it: the transmit lobe
            leaves through the coat on the far face, which takes it too
Each mode's frame mean over `--seeds` seeds must be within 1 % of mode R's on the same device (B's
within 0.5 %). Mode B is not scored where the coat is smooth (fresnel, glossy): it connects no delta
lobe to the camera, so it misses the coat's mirror image of the room (0.90 of R there), as it misses
any mirror's. Exits 1 on any failure.
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
NPHOT = {"B": 40000000, "M": 2000000, "S": 100000}   # photons (mode S's per pass)
DEPTH = {"M": 32, "S": 32}
# Mode J's point merges keep a fixed radius, and their O(r^2) bias reads +0.6 % of R on the rough room
# at its default (the host: 1.0061; 1.0056 at 0.05, 1.0025 at 0.02, 0.9981 with -nojsurf): pinned
# small, as check_bump_modes pins it, so the check reads J's integrand, not its radius.
MODE_ARGS = {"J": ["-jsurf-radius", "0.02"]}
TOL = 0.01
TOL_MODE = {"B": 0.005}
# Rooms whose coat is smooth: a delta sheen, which mode B cannot image (see the docstring).
SMOOTH = {"fresnel", "glossy"}

COAT = "reflectance fresnel  ior 1.5  roughness %s"
FILM = "reflectance thinfilm  ior 1.5  film_ior 1.33  film_thickness 380  roughness 0.1"
WALLS = {
    "fresnel": ['material "body" { type diffuse reflect 0.8 }',
                'material "w" { type layered  coat { %s }  layer "body" 1.0 }' % (COAT % "0")],
    "rough": ['material "body" { type diffuse reflect 0.8 }',
              'material "w" { type layered  coat { %s }  layer "body" 1.0 }' % (COAT % "0.3")],
    "film": ['material "body" { type diffuse reflect rgb 0.7 0.6 0.5 }',
             'material "w" { type layered  coat { %s }  layer "body" 1.0 }' % FILM],
    "glossy": ['material "body" { type glossy reflect 0.8  roughness 0.3 }',
               'material "w" { type layered  coat { %s }  layer "body" 1.0 }' % (COAT % "0")],
    "sheet": ['material "w" { type diffuse reflect 0.8 }',
              'material "tb" { type translucent reflect 0.4  transmit 0.4 }',
              'material "t" { type layered  coat { %s }  layer "tb" 1.0 }' % (COAT % "0")],
}


def room(kind):
    lines = ["scene { units meters }"] + WALLS[kind] + [
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w }",
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w }",
        "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w }",
        "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w }",
        "light area { origin -1.9 0.2 -1.5  u 0 0 0.6  v 0 2.4 0  normal 1 0 0.2  spd preset:bb6500  power 40 }",
        'camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  fov_y 50  film { res 64 48 } }',
    ]
    if kind == "sheet":   # a coated translucent panel standing in the room, lit from behind
        lines.append("quad { origin -1 0 -0.5  u 2 0 0.5  v 0 2 0  material t }")
    return NL.join(lines) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_layered_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--modes", default="D,J,U,B,M,S")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rooms", default="fresnel,rough,film,glossy,sheet")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    fails = []
    for name in a.rooms.split(","):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        scene = os.path.join(d, "room.ftsl")
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(room(name))

        def frame(mode, dev, seed, depth):
            png = os.path.join(d, "%s_%s_%d_%d.png" % (mode, dev, depth, seed))
            for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            cmd = [exe, "-in", scene, "-mode", mode, "-device", dev,
                   "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)),
                   "-n", str(NPHOT.get(mode, 200000)),
                   "-max-bounce", str(depth), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
                   "-o", png] + MODE_ARGS.get(mode, [])
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            return float(read_pfm(pfm, top_down=True).sum(-1).mean())

        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            refs = {}                                # mode R at each depth asked for
            for mode in a.modes.split(","):
                if mode == "B" and name in SMOOTH:
                    continue
                depth = DEPTH.get(mode, 8)
                if depth not in refs:
                    r = [frame("R", dev, s, depth) for s in seeds]
                    refs[depth] = (np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r)))
                rm, rse = refs[depth]
                v = [frame(mode, dev, s, depth) for s in seeds]
                m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
                ratio = m / rm
                err = np.hypot(se / rm, m * rse / rm ** 2)
                good = abs(ratio - 1.0) <= TOL_MODE.get(mode, TOL)
                print("%s: %-7s mode %s on the %s: %.4f of mode R (+-%.4f)"
                      % ("ok" if good else "FAIL", name, mode, "host" if dev == "cpu" else "device",
                         ratio, err), flush=True)
                if not good:
                    fails.append("%s %s %s" % (name, mode, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
