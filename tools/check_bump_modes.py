"""Regression check: the modes agree with mode R on a normal-mapped surface (BUMP-MODES).

    python tools/check_bump_modes.py [--out png/check_bump_modes] [--exe ftrace.exe] [--no-gpu]
                                     [--modes D,J,U,B,M,S] [--seeds 3] [--rooms flat,...,translucent]

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
they read the flat room 2 % bright.

A third room carries the map on a `layered` material instead -- a rough coat (roughness 0.3) over a
diffuse body -- and holds modes M and S to R there: a compound takes its photon deposit before its
lobe pick, under the arrival rule of the Lambertian lobes it is gathered for
(Renderer::compoundDepositWeight); a deposit without it reads M 1.087 of R and S 1.081. The modes
miss a layered material by about 1 % even on flat geometry (LAYERED-MODES in known-issues.md), so
there M and S are held within 2 %, and the others not at all.

A fourth carries the map on glossy walls (roughness 0.4), every mode held to R within 1 %, and
mode R to itself with glossy NEE off. The terminator's softening belongs to the vertex next to the
light, whichever strategy reaches it, so the light the lobe's own sample reaches takes it as the
light sampling does (contSoftening): an unsoftened lobe read R 1.044 of itself with its lobe alone,
device mode S 1.043. A fifth, the diffuse room open to a sky (no ceiling, no lamp), holds modes M
and S to R (D, J and U render no environment light, and mode B reads such walls ~1 % bright flat or
bumped, B-ENV-WALLS in known-issues.md) and R to itself with sky NEE off (FTRACE_ENV_NEE=0, host): the sky reached by a diffuse
vertex's escape takes the vertex's softening too, where without it the escape alone read 1.022.
A sixth, the diffuse room with a normal-mapped translucent panel standing in it, holds every mode
to R: a translucent lobe pair is lit by light from above ns on the light's own side, in its reflect
lobe on the camera's geometric side and its transmit lobe on the far one, and in neither between the
planes (bsdf_eval.h translucentLobe), softened on the light's side. 0.440.7 read it D 1.090, J 1.091,
U 1.090, B 1.082, M 1.026 and S 1.021 of R on the host. Mode J runs at a pinned merge radius
(MODE_ARGS, where the why is).
Exits 1 on any failure.
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
# Mode J's point merges carry an O(r^2) bias over a rippled surface, which its default starting
# radius leaves at +0.7 % of R on the translucent room (+0.4 % at 0.04, -0.15 % at 0.01; J without
# them 0.999): pinned where that floor is ~0.1 %, so the check reads J's integrand, not its radius.
MODE_ARGS = {"J": ["-jsurf-radius", "0.02"]}
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


def room(d, amp, kind="diffuse"):
    if kind == "sky":   # the diffuse room open to a sky: no ceiling, no lamp
        lines = room(d, amp, "diffuse").rstrip(NL).split(NL)
        lines = [l for l in lines if not l.startswith("quad { origin -2 3 -2") and not l.startswith("light area")]
        lines.insert(1, "light env { spd preset:bb6500 }")
        return NL.join(lines) + NL
    if kind == "translucent":   # the diffuse room with a normal-mapped translucent panel standing in it
        lines = room(d, amp, "diffuse").rstrip(NL).split(NL)
        lines.insert(3, 'material "t" { type translucent reflect 0.4  transmit 0.4  normal_map texture:rip }')
        lines.insert(4, "quad { origin -1 0 -0.5  u 2 0 0.5  v 0 2 0  material t }")
        return NL.join(lines) + NL
    ppm = os.path.join(d, "rip.ppm")
    ripple(ppm, amp)
    p = ppm.replace(chr(92), "/")
    wall = {"layered": ['material "body" { type diffuse reflect 0.8 }',
                        'material "w" { type layered  coat { reflectance fresnel  ior 1.5  roughness 0.3 }  '
                        'layer "body" 1.0  normal_map texture:rip }'],
            "glossy": ['material "w" { type glossy reflect 0.8  roughness 0.4  normal_map texture:rip }'],
            "diffuse": ['material "w" { type diffuse reflect 0.8  normal_map texture:rip }']}[kind]
    return NL.join([
        "scene { units meters }",
        'texture "rip" { file "%s"  encoding linear  filter bilinear  wrap repeat }' % p,
    ] + wall + [
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
    ap.add_argument("--rooms", default="flat,bumped,layered,glossy,sky,translucent")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    seeds = list(range(1, a.seeds + 1))
    fails = []
    for name, amp, kind in (("flat", 0.0, "diffuse"), ("bumped", 2.0, "diffuse"),
                            ("layered", 2.0, "layered"), ("glossy", 2.0, "glossy"),
                            ("sky", 2.0, "sky"), ("translucent", 2.0, "translucent")):
        if name not in a.rooms.split(","):
            continue
        layered = kind == "layered"
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        scene = os.path.join(d, "room.ftsl")
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(room(d, amp, kind))

        def frame(mode, dev, seed, depth, extra=(), tag="", env=None):
            png = os.path.join(d, "%s%s_%s_%d_%d.png" % (mode, tag, dev, depth, seed))
            for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            cmd = [exe, "-in", scene, "-mode", mode, "-device", dev,
                   "-spp", str(SPP[mode] * (4 if dev == "gpu" else 1) * (4 if (extra or env) else 1)),
                   "-n", str(NPHOT.get(mode, 200000)),
                   "-max-bounce", str(depth), "-seed", str(seed), "-hdr", "-no-meter", "-window-min",
                   "-o", png] + MODE_ARGS.get(mode, []) + list(extra)
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=dict(os.environ, **(env or {})))
            pfm = png[:-4] + ".pfm"
            if p.returncode != 0 or not os.path.exists(pfm):
                print((p.stdout + p.stderr)[-1500:])
                raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
            return float(read_pfm(pfm, top_down=True).sum(-1).mean())

        for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
            refs = {}                                # mode R at each depth asked for
            for mode in a.modes.split(","):
                if layered and mode not in ("M", "S"):
                    continue
                if kind == "sky" and mode not in ("M", "S"):   # D, J, U take no sky; B: B-ENV-WALLS
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
                good = abs(ratio - 1.0) <= (2 * TOL if layered else TOL)
                print("%s: %-6s mode %s on the %s: %.4f of mode R (+-%.4f)"
                      % ("ok" if good else "FAIL", name, mode, "host" if dev == "cpu" else "device",
                         ratio, err), flush=True)
                if not good:
                    fails.append("%s %s %s" % (name, mode, dev))
            # R's two strategies for the same light, one integrand: its lobe (or sky escape) alone
            if kind in ("glossy", "sky") and (kind == "glossy" or dev == "cpu"):
                rm, rse = refs.get(8) or (None, None)
                if rm is None:
                    r = [frame("R", dev, s, 8) for s in seeds]
                    rm, rse = np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r))
                if kind == "glossy":
                    v = [frame("R", dev, s, 8, ["-no-glossy-nee"], "lobe") for s in seeds]
                else:
                    v = [frame("R", dev, s, 8, (), "esc", {"FTRACE_ENV_NEE": "0"})
                         for s in seeds]
                ratio = np.mean(v) / rm
                good = abs(ratio - 1.0) <= TOL
                print("%s: %-6s mode R with %s NEE off, on the %s: %.4f of mode R"
                      % ("ok" if good else "FAIL", name, "glossy" if kind == "glossy" else "sky",
                         "host" if dev == "cpu" else "device", ratio), flush=True)
                if not good:
                    fails.append("%s R-%s %s" % (name, kind, dev))
    print("FAILED: " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
