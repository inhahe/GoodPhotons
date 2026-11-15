"""Regression check: a `two_sided on` area light is two one-sided panels back to back, in every mode.

    python tools/check_two_sided.py [--out png/check_two_sided] [--exe ftrace.exe] [--no-gpu]

COINCIDENT-LAMP-FACES. Two `light area` quads on one plane, back to back, leave the closest-hit test
either face, so a camera walk saw such a panel at about half its light. `two_sided on` makes one
surface of two lights, whose back face a hit from behind meets (Tri::emitsBack).
`scenes/_two_sided_lamp.ftsl` hangs one in a white room, its `normal` up and `power 20`, the camera
below it: the camera sees the back face directly and the ceiling, lit by the front face, beyond it.
Three parts:
  * the modes: R, B, D, J, U, M and S, CPU and GPU, each against itself on the reference -- two
    one-sided panels of `power 10`, 1 mm apart -- the panel's face, the ceiling and the lower frame
    within 2 % and 5 standard errors over the seeds;
  * the warning: the same two panels on one plane, back to back and facing alike, warn at load,
    and the two-sided light and the reference do not;
  * the refusal: `two_sided on` with an aimed (leaning) normal fails to load.
Without the back face a camera walk sees the panel black from below. Exits 1 on any failure.
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

SCENE = os.path.join(ROOT, "scenes", "_two_sided_lamp.ftsl")
LIGHT = "light area { origin -0.25 1.0 -0.25  u 0.5 0 0  v 0 0 0.5  normal 0 1 0  two_sided on  spd preset:bb6500  power 20 }"


def panel(y, ny, power=10):
    return ("light area { origin -0.25 %s -0.25  u 0.5 0 0  v 0 0 0.5  normal 0 %d 0  spd preset:bb6500  power %s }"
            % (y, ny, power))


REFERENCE = panel("0.9995", -1) + NL + panel("1.0005", 1)
COINCIDENT = panel("1.0", -1) + NL + panel("1.0", 1)
SAME_FACING = panel("1.0", -1) + NL + panel("1.0", -1)
AIMED = LIGHT.replace("normal 0 1 0", "normal 0 1 0.3")
DEPTH = ["-max-bounce", "12"]
CASES = [("R", ["-spp", "1024"]), ("B", ["-n", "60000000"]), ("D", ["-spp", "256"]), ("J", ["-spp", "128"]),
         ("U", ["-spp", "256"]), ("M", ["-spp", "64", "-n", "2000000"]), ("S", ["-spp", "256", "-n", "200000"])]
SEEDS = (1, 2)
# Mode S's gather blurs neighbouring pixels together, so two seeds' scatter under-estimates its
# seed-to-seed spread (check_rough_irid's reason): eight. Mode J's lower frame moves 0.8 % from seed
# to seed on the CPU, which two seeds put anywhere within 2 %: four.
SEEDS_OF = {"S": tuple(range(1, 9)), "J": (1, 2, 3, 4)}
TOL = 0.02
REGIONS = ("panel", "ceiling", "lower frame")
WARNING = "overlap on one plane"


def variant(out, name, light):
    with open(SCENE, encoding="utf-8") as f:
        src = f.read()
    if src.count(LIGHT) != 1:
        raise SystemExit("the scene's light line is not the one this check replaces")
    path = os.path.join(out, name + ".ftsl")
    with open(path, "w", encoding="utf-8", newline="") as g:
        g.write(src.replace(LIGHT, light))
    return path


def run(exe, cmd):
    return subprocess.run([exe] + cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def render(exe, out, scene, tag, args, seed):
    png = os.path.join(out, "%s_s%d.png" % (tag, seed))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = ["-in", scene, "-hdr", "-no-meter", "-seed", str(seed), "-window-min", "-o", png] + DEPTH + args
    p = run(exe, cmd)
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[..., 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_two_sided"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    devs = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    ref = variant(out, "reference", REFERENCE)
    ok = True

    # The warning, and the refusal: a load each, rendering a pixel's worth.
    probe = ["-mode", "R", "-device", "cpu", "-spp", "1", "-r", "8", "6", "-no-meter"]
    for name, light, warns in (("two_sided", LIGHT, False), ("reference", REFERENCE, False),
                               ("coincident", COINCIDENT, True), ("same_facing", SAME_FACING, True)):
        p = run(exe, ["-in", variant(out, name, light), "-o", os.path.join(out, "load_%s.png" % name)] + probe)
        said = WARNING in p.stdout + p.stderr
        good = p.returncode == 0 and said == warns
        ok = ok and good
        print("%s: %s %s" % ("ok" if good else "FAIL", name, "warns" if said else "loads without a warning"),
              flush=True)
    p = run(exe, ["-in", variant(out, "aimed", AIMED), "-o", os.path.join(out, "load_aimed.png")] + probe)
    good = p.returncode != 0 and "two_sided" in p.stdout + p.stderr
    ok = ok and good
    print("%s: an aimed two-sided light %s" % ("ok" if good else "FAIL",
                                               "is refused" if p.returncode != 0 else "loads"), flush=True)

    # The regions, from a mode R image of the reference: the panel is far the brightest thing in frame.
    img = render(exe, out, ref, "mask", ["-mode", "R", "-device", "cpu", "-spp", "64"], 1)
    h = img.shape[0]
    pan = img > 0.5 * img.max()
    rows = np.arange(h)[:, None] + np.zeros_like(img, dtype=int)
    masks = [pan, (rows < h // 8) & ~pan, (rows >= h - h // 3) & ~pan]
    if not pan.any() or any(not m.any() for m in masks):
        raise SystemExit("a region is empty: the frame is not the one this check reads")

    def stats(scene, tag, args, seeds):
        v = np.array([[render(exe, out, scene, tag, args, s)[m].mean() for m in masks] for s in seeds])
        return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(seeds))

    for mode, args in CASES:
        for dev in devs:
            seeds = SEEDS_OF.get(mode, SEEDS)
            full = ["-mode", mode, "-device", dev] + args
            tm, tse = stats(SCENE, "%s_%s_two" % (mode, dev), full, seeds)
            rm, rse = stats(ref, "%s_%s_ref" % (mode, dev), full, seeds)
            q = tm / rm
            err = np.hypot(tse / rm, rse * tm / (rm * rm))
            bad = (np.abs(q - 1.0) > TOL) & (np.abs(q - 1.0) > 5.0 * err)
            ok = ok and not bad.any()
            print("%s: mode %s on the %s, two-sided of the reference: %s"
                  % ("FAIL" if bad.any() else "ok", mode, dev.upper(),
                     ", ".join("%s %.4f (+- %.4f)" % (r, x, e) for r, x, e in zip(REGIONS, q, err))), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
