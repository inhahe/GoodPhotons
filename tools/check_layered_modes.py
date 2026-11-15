"""Regression check: every mode renders a `layered` material as mode R does (LAYERED-MODES).

    python tools/check_layered_modes.py [--out png/check_layered_modes] [--exe ftrace.exe] [--no-gpu]
                                        [--modes D,J,U,B,M,S] [--seeds 3] [--rooms fresnel,...]

A layered material is a clear coat over a body. Every walk takes the coat at its own arrival by a
coin (reflect with R, else enter: T = 1 - R), and the lobe it then shades with takes the coat on its
other direction (render.h coatF), so the integrand is reciprocal and the modes agree by
construction: a Lambertian body's light crosses the coat twice, T(in) T(out) / T-bar (T-bar the
cosine-weighted mean the body's albedo already holds); a directional body's (glossy, mirror) takes
sqrt(T(in) T(out)), its albedo holding its own angle's escape; and the coat's reflection takes its
Fresnel term at the pair's half vector. A body that takes the coat only at its walk's arrival holds
T toward the camera in mode R's integrand and toward the light in the light walks', and on the rough
room below reads D 1.0109, U 1.0111, B 1.0256, M 1.0120 and S 1.0112 of R.

The rooms: five walls of one material lit at a grazing angle by a tilted panel, 64x48, every mode at
-max-bounce 8 (M and S, which deposit photons over 32 bounces, and their mode-R reference at 32).
  fresnel   a smooth Fresnel coat (ior 1.5, roughness 0) over a diffuse body (0.8)
  rough     the same coat at roughness 0.3: the coat's reflection turns the light, so it must be
            reciprocal too
  film      a thin-film coat (film_ior 1.33, 380 nm over ior 1.5): R and with it the coat's factors
            swing across the spectrum, so they are read per wavelength
  glossy    the smooth coat over a glossy body (roughness 0.3): car paint
  sheet     the diffuse room with a translucent panel under a rough coat standing in it: the
            transmit lobe leaves through the coat on the far face, which takes it too
  furnace   a white furnace (a uniform sky, nothing else) holding three spheres under a smooth coat:
            a white diffuse body, a white mirror, and a white diffuse body under a high-index film
            (film_ior 2.4 over ior 1.5) whose thickness a map sets to a quarter of its authored 240
            nm, about a quarter wave: a lossless stack returns exactly what it receives, so mode R
            must read each sphere at the sky's radiance within 1 %, and modes M and S the frame as R
            does (D, J and U take no sky). A coat's averaged escape (T-bar) tabulated at the
            authored thickness alone reads the mapped sphere 0.911 of the sky.
  fluoro    a rough high-index film (film_ior 2.4, 140 nm) over a fluorescent body that absorbs blue
            and glows red, where the film's averaged escape is about 0.6 and 0.9: mode B only (the
            modes that render a Stokes shift are R and the forward ones). A mode R that normalises
            the shifted light's way out at the absorbed wavelength reads B 0.916 of itself.
Each mode's frame mean over `--seeds` seeds must be within 1 % of mode R's on the same device (B's
within 0.5 %). Mode B images a smooth coat's mirror reflection through its closed-form mirror
connections, the coat a mirror of its own reflectance R(theta): fresnel reads 0.997 and glossy 0.999 of
R (without, 0.90: the coat's image of the room missing). It is not scored in the furnace, whose coats
reflect a sky -- a light seen in a mirror, with no vertex between for a connection to join. Exits 1 on
any failure.
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
TOL = 0.01
TOL_MODE = {"B": 0.005}
# Rooms mode B cannot image: the furnace's smooth coats reflect its sky, a light seen in a mirror with
# no vertex between for the closed-form connection to join (see the docstring).
SMOOTH = {"furnace"}
# Rooms lit by a sky, which modes D, J and U do not render.
SKY = {"furnace"}
# Rooms whose light is fluorescent, which only mode R and the forward modes shift: mode B alone.
FLUORO = {"fluoro"}

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
              'material "t" { type layered  coat { %s }  layer "tb" 1.0 }' % (COAT % "0.3")],
    "fluoro": ['material "body" { type fluorescent reflect 0.05  absorb shortpass edge=480 slope=0.2 amp=1  '
               'emit gaussian center=620 sigma=25 amp=1  yield 0.9 }',
               'material "w" { type layered  coat { reflectance thinfilm  ior 1.5  film_ior 2.4  '
               'film_thickness 140  roughness 0.3 }  layer "body" 1.0 }'],
}


# The furnace's spheres, left to right, each in its own third of the frame.
FURNACE_SPHERES = ("diffuse body", "mirror body", "film-mapped diffuse body")


def gray_texture(path):
    # A uniform 2x2 map of 64/255 (a quarter): the mapped film's thickness everywhere, whatever the
    # sphere's texture coordinates.
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write("P3" + NL + "2 2" + NL + "255" + NL + " ".join(["64 64 64"] * 4) + NL)


def room(kind, black=False, tex=None):
    if kind == "furnace":   # `black`: the spheres black, for the mask that finds them; `tex`: the map
        return NL.join([
            "scene { units meters }",
            "light env { spd preset:bb6500 }",
            'texture "q" { file "%s"  encoding linear  filter bilinear  wrap repeat }' % tex.replace(chr(92), "/"),
            'material "wd" { type diffuse reflect 1.0 }',
            'material "wm" { type mirror reflect 1.0 }',
            'material "cd" { type layered  coat { %s }  layer "wd" 1.0 }' % (COAT % "0"),
            'material "cm" { type layered  coat { %s }  layer "wm" 1.0 }' % (COAT % "0"),
            'material "cf" { type layered  coat { reflectance thinfilm  ior 1.5  film_ior 2.4  '
            'film_thickness 240  film_thickness_map texture:q  roughness 0 }  layer "wd" 1.0 }',
            'material "k" { type diffuse reflect 0 }',
            "sphere { center -1.1 0 0  radius 0.45  material %s }" % ("k" if black else "cd"),
            "sphere { center 0 0 0  radius 0.45  material %s }" % ("k" if black else "cm"),
            "sphere { center 1.1 0 0  radius 0.45  material %s }" % ("k" if black else "cf"),
            'camera "c" { eye 0 0 5  look_at 0 0 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
        ]) + NL
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
    ap.add_argument("--rooms", default="fresnel,rough,film,glossy,sheet,furnace,fluoro")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    fails = []
    for name in a.rooms.split(","):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        scene = os.path.join(d, "room.ftsl")
        tex = os.path.join(d, "quarter.ppm")
        gray_texture(tex)
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(room(name, tex=tex))

        def image(mode, dev, seed, depth, scn=scene, tag=""):
            png = os.path.join(d, "%s%s_%s_%d_%d.png" % (mode, tag, dev, depth, seed))
            for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            cmd = [exe, "-in", scn, "-mode", mode, "-device", dev,
                   "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1)),
                   "-n", str(NPHOT.get(mode, 200000)),
                   "-max-bounce", str(depth), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
                   "-o", png]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            return read_pfm(pfm, top_down=True).sum(-1)

        def frame(mode, dev, seed, depth):
            return float(image(mode, dev, seed, depth).mean())

        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            where = "host" if dev == "cpu" else "device"
            if name == "furnace":
                # Energy: each sphere at the sky's radiance in mode R. The mask is the same frame
                # with the spheres black; the sky is everything the mask leaves.
                bscn = os.path.join(d, "black.ftsl")
                with open(bscn, "w", encoding="utf-8", newline="") as f:
                    f.write(room(name, black=True, tex=tex))
                k = image("R", dev, 1, 8, bscn, "black")
                img = np.mean([image("R", dev, s, 8) for s in seeds], axis=0)
                sky = float(np.median(k))
                inside = k < 0.05 * sky
                third = img.shape[1] // 3
                for j, label in enumerate(FURNACE_SPHERES):
                    cols = slice(j * third, (j + 1) * third)
                    m = inside[:, cols]
                    ratio = float(img[:, cols][m].mean()) / sky
                    good = abs(ratio - 1.0) <= TOL
                    print("%s: furnace mode R on the %s: coated white %s %.4f of the sky"
                          % ("ok" if good else "FAIL", where, label, ratio), flush=True)
                    if not good:
                        fails.append("furnace %s R %s" % (label, dev))
            refs = {}                                # mode R at each depth asked for
            for mode in a.modes.split(","):
                if mode == "B" and name in SMOOTH:
                    continue
                if mode in ("D", "J", "U") and name in SKY:
                    continue
                if mode != "B" and name in FLUORO:
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
                      % ("ok" if good else "FAIL", name, mode, where, ratio, err), flush=True)
                if not good:
                    fails.append("%s %s %s" % (name, mode, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
