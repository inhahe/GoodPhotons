"""Regression check: every leg a camera under water sees by takes the water's absorption, in the paths 0.445.8 missed.

    python tools/check_camera_legs.py [--out png/check_camera_legs] [--exe ftrace.exe] [--no-gpu]

CAMERA-EMPTY-STACK starts a camera's walks inside the solids around its eye, and every connection leg
takes its medium's absorption (legAbsorption). The reviews of 0.445.8 (REVIEW-0445B) found the legs it
missed; each case here is `scenes/_camera_in_water.ftsl` (a white room under water absorbing 0.3 per
metre, the camera and a lamp in it) or a variant, a mode against mode R, the walls' mean (and where
named, a region's) within the tolerance and 5 standard errors over the seeds:
  * mode C, CPU and GPU, against mode A (mode C's own scale) through a wide aperture and through a
    physical lens: a photon caught by the aperture or the front element took no absorption on its last
    leg;
  * mode M with a glossy ball, CPU and GPU: the ball's spectral glossy light sample took none;
  * mode R with `-rgb` on the GPU: its light samples took none;
  * mode B with a `glow_only` panel seen directly, CPU and GPU: the forward modes' view of a glow took
    none;
  * mode B with the camera inside a glass ball in the water (the two ranked by `priority`, so the
    ball refracts against the water's index), CPU and GPU, the lamp seen through it, in the water and
    in air: the leg outside the ball took the ball's absorption instead of the water's, the ball
    refracted against vacuum and its inside lacked n^2; and in water ranked above the ball, which
    suppresses the ball's boundary: the connection lensed through a ball the walks pass straight
    through;
  * mode R with `-radcache`: the cache's update rays started in air;
  * mode W at one sample: the hero bundle of a camera inside a solid collapsed to one wavelength for the
    whole frame, where `-herosplit` (which mode W forces) splits it -- one sample's lamp, against the
    same at 64 samples, keeps its colour (the walls are dark in mode W, which takes no indirect light).
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

SCENE = os.path.join(ROOT, "scenes", "_camera_in_water.ftsl")
DEPTH = ["-max-bounce", "12"]
_Y, _X = np.mgrid[0:72, 0:96]
_R2 = (_X - 47.5) ** 2 + (_Y - 35.5) ** 2
LAMP = _R2 <= 2.5 ** 2
WALLS = _R2 >= 7.0 ** 2
# The glossy ball (GLOSSY) seen left of the lamp, and the glow panel (GLOW) right of it.
BALL = (_X - 19.0) ** 2 + (_Y - 42.0) ** 2 <= 6.0 ** 2
GLOW = (_X >= 70) & (_X <= 75) & (_Y >= 34) & (_Y <= 38)

GLOSSY = 'material "gold" { type glossy  reflect 0.9  roughness 0.2 }' + NL + \
         "sphere { center -0.9 1.3 -0.6  radius 0.3  material gold }"
GLOWING = 'material "sign" { type diffuse  reflect 0.0  emit preset:bb6500  glow_only on }' + NL + \
          "quad { origin 0.8 1.35 -1.2  u 0.3 0 0  v 0 0.3 0  material sign }"
# A glass ball around the camera (eye 0 1.5 1.8), clear of the room's walls, absorbing 0.6 per metre,
# inside the water.
BUBBLE = 'material "glass" { type dielectric  ior 1.5  absorb 0.6  priority 2 }' + NL + \
         "sphere { center 0 1.5 1.8  radius 0.15  material glass }"


def wide_aperture(out):
    # Mode C catches what crosses the aperture's disc: a wide one (as check_motion's), focused on the
    # lamp, so enough photons land.
    with open(SCENE, encoding="utf-8") as f:
        src = f.read()
    a = "film { res 96 72 }"
    if src.count(a) != 1:
        raise SystemExit("the scene's camera is not the one this check widens")
    path = os.path.join(out, "wide.ftsl")
    with open(path, "w", encoding="utf-8", newline="") as g:
        g.write(src.replace(a, "aperture 0.25  focus 2.8  " + a))
    return path


def lens_camera(out):
    # The same room seen through a physical lens (an achromat), whose front element the mode-C catch
    # tests -- the hero bundle's catch as well as the scalar one.
    with open(SCENE, encoding="utf-8") as f:
        src = f.read()
    a = "film { res 96 72 }"
    if src.count(a) != 1:
        raise SystemExit("the scene's camera is not the one this check fits a lens to")
    path = os.path.join(out, "lensed.ftsl")
    with open(path, "w", encoding="utf-8", newline="") as g:
        g.write(src.replace(a, "focus 2.8  lens { preset achromat  focal 50  fstop 2.8 }  " + a))
    return path


def variant(out, name, extra):
    with open(SCENE, encoding="utf-8") as f:
        src = f.read()
    path = os.path.join(out, name + ".ftsl")
    with open(path, "w", encoding="utf-8", newline="") as g:
        g.write(src + NL + extra + NL)
    return path


def render(exe, out, scene, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scene, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-o", png] + DEPTH + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)


def stats(exe, out, scene, tag, args, region, seeds):
    v = np.array([render(exe, out, scene, tag, args, s)[..., 1][region].mean() for s in seeds])
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_camera_legs"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    refdev = "gpu" if not a.no_gpu else "cpu"
    glossy = variant(out, "glossy", GLOSSY)
    glowing = variant(out, "glowing", GLOWING)
    bubble = variant(out, "bubble", BUBBLE)
    with open(bubble, encoding="utf-8") as f:
        ranked = f.read()
    wa = 'material "water" { type dielectric  ior 1.33  absorb 0.3 }'
    if ranked.count(wa) != 1:
        raise SystemExit("the scene's water is not the one this check ranks")
    with open(bubble, "w", encoding="utf-8", newline="") as g:
        g.write(ranked.replace(wa, wa[:-1] + " priority 1 }"))
    lensed = lens_camera(out)
    wide = wide_aperture(out)
    # (name, scene, mode args, region, region name, devices, tolerance, seeds[, reference args])
    cases = []
    for dev in devs:
        cases += [("mode C", wide, ["-mode", "C", "-device", dev, "-n", "2000000000" if dev == "gpu" else "400000000"],
                   WALLS, "walls", dev, 0.05, (1, 2), ["-mode", "A", "-device", refdev, "-n", "40000000"]),
                  ("mode C, a physical lens", lensed,
                   ["-mode", "C", "-device", dev, "-n", "2000000000" if dev == "gpu" else "400000000"],
                   WALLS, "walls", dev, 0.05, (1, 2), ["-mode", "A", "-device", refdev, "-n", "40000000"]),
                  ("mode M, glossy ball", glossy, ["-mode", "M", "-device", dev, "-spp", "64", "-n", "2000000"],
                   BALL, "ball", dev, 0.04, (1, 2, 3)),
                  ("mode B, glow seen", glowing, ["-mode", "B", "-device", dev, "-n", "60000000"], GLOW, "glow",
                   dev, 0.03, (1, 2))]
    cases += [("mode R -rgb", SCENE, ["-mode", "R", "-device", refdev, "-spp", "1024", "-rgb"], WALLS, "walls",
               refdev, 0.04, (1, 2)),
              ("mode R -radcache", SCENE, ["-mode", "R", "-device", "cpu", "-spp", "256", "-radcache"], WALLS,
               "walls", "cpu", 0.03, (1, 2))]
    refs = {}
    ok = True
    for case in cases:
        name, scene, args, region, rname, dev, tol, seeds = case[:8]
        ref = case[8] if len(case) > 8 else ["-mode", "R", "-device", refdev, "-spp", "4096"]
        key = (scene, rname, tuple(ref))
        if key not in refs:
            refs[key] = stats(exe, out, scene, "%s_ref_%s_%s" % (ref[1], os.path.basename(scene)[:-5], rname),
                              ref, region, (1, 2))
        rm, rse = refs[key]
        tag = "%s_%s_%s" % (name.replace(" ", "_").replace(",", "").replace("-", ""), dev, rname)
        m, se = stats(exe, out, scene, tag, args, region, seeds)
        q = m / rm
        err = np.hypot(se / rm, rse * m / (rm * rm))
        bad = not (np.isfinite(q) and np.isfinite(err)) or (abs(q - 1.0) > tol and abs(q - 1.0) > 5.0 * err)
        ok = ok and not bad
        print("%s: %s on the %s: %s %.4f (+- %.4f) of mode %s" % ("FAIL" if bad else "ok", name, dev.upper(), rname,
                                                               q, err, ref[1]), flush=True)
    # The camera inside a small glass ball in the water: the lamp it sees through the ball, mode B
    # against mode R, in the water and in air. (The walls are left out: a camera walk from inside a
    # glass ball draws a halo round the lamp that mode B does not, known-issues BALL-EYE-HALO.)
    with open(bubble, encoding="utf-8") as f:
        wet_src = f.read()
    dry = os.path.join(out, "bubble_dry.ftsl")
    with open(dry, "w", encoding="utf-8", newline="") as g:
        g.write("".join(l + NL for l in wet_src.split(NL) if "material water" not in l))
    # The water ranked above the ball: the priority rule suppresses the ball's boundary, which is then
    # water to the walks and to the sphere connection alike (no lens, no n^2, the water's absorption).
    ranked_low = "absorb 0.3  priority 1 }"
    if wet_src.count(ranked_low) != 1:
        raise SystemExit("the bubble's water is not the one this check re-ranks")
    sunk = os.path.join(out, "bubble_suppressed.ftsl")
    with open(sunk, "w", encoding="utf-8", newline="") as g:
        g.write(wet_src.replace(ranked_low, "absorb 0.3  priority 3 }"))
    for sc, tag in ((bubble, "in the water"), (dry, "in air"), (sunk, "in water ranked above it")):
        rm, rse = stats(exe, out, sc, "R_ref_bubble_%s" % tag[3:], ["-mode", "R", "-device", refdev, "-spp", "1024"],
                        LAMP, (1, 2))
        for dev in devs:
            m, se = stats(exe, out, sc, "B_bubble_%s_%s" % (tag[3:], dev),
                          ["-mode", "B", "-device", dev, "-n", "30000000"], LAMP, (1, 2))
            q = m / rm
            err = np.hypot(se / rm, rse * m / (rm * rm))
            bad = not (np.isfinite(q) and np.isfinite(err)) or (abs(q - 1.0) > 0.03 and abs(q - 1.0) > 5.0 * err)
            ok = ok and not bad
            print("%s: mode B on the %s, the lamp from inside a glass ball %s: %.4f (+- %.4f) of mode R"
                  % ("FAIL" if bad else "ok", dev.upper(), tag, q, err), flush=True)
    # Mode W at one sample against itself at 64: the lamp's colour (r/g and b/g), every device.
    for dev in devs:
        one = render(exe, out, SCENE, "W1_%s" % dev, ["-mode", "W", "-device", dev, "-spp", "1"], 1)[LAMP].mean(0)
        many = render(exe, out, SCENE, "W64_%s" % dev, ["-mode", "W", "-device", dev, "-spp", "64"], 1)[LAMP].mean(0)
        c1 = np.array([one[0] / one[1], one[2] / one[1]])
        c64 = np.array([many[0] / many[1], many[2] / many[1]])
        d = np.abs(c1 / c64 - 1.0)
        bad = not bool(np.isfinite(d).all()) or bool((d > 0.10).any())
        ok = ok and not bad
        print("%s: mode W on the %s, one sample's lamp r/g %.3f b/g %.3f of 64 samples'" %
              ("FAIL" if bad else "ok", dev.upper(), c1[0] / c64[0], c1[1] / c64[1]), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
