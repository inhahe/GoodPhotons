"""Regression check: the forward modes warn that they light an unbounded haze only near the scene.

    python tools/check_haze_warning.py [--out png/check_haze_warning] [--exe ftrace.exe]

A medium with no bounds has no end, and the modes give it different ones (known-issues HAZE-SUN-B):
modes R, D and J light the haze along every camera ray out to the horizon, while the forward modes
(A, B, C) send the sun's and the sky's light in through a disc the size of the scene and light only
the haze behind it -- on the scene below mode B's haze is 0.15 of mode R's. The backward modes'
domain is the one kept, and the forward modes say so. The sun over a floor in such a haze (the
scene of scenes/_haze_sun_b_vs_r.ftsl, tiny here), rendered for a moment:

  - modes A, B and C print the warning once;
  - mode R does not;
  - nor does mode B with the medium given `bounds`, nor under a ball light in the scene instead.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
MARK = "a medium with no bounds, lit by the sun or the sky"


def scene(medium, light):
    return NL.join([
        "scene { units meters }",
        'material "w" { type diffuse reflect 0.6 }',
        light,
        medium,
        "quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }",
        'camera "c" { eye 0 1.6 2.2  look_at 0 0 0  up 0 1 0  fov_y 45  film { res 24 18 } }']) + NL


SUN = "light sun { dir 0.3 1 0.2  angle 0.53  spd preset:d65 }"
BALL = "light sphere { center 0 1.5 0  radius 0.05  spd preset:d65 }"
HAZE = "medium { sigma_t 0.04  albedo 0.5 }"
BOUNDED = "medium { sigma_t 0.04  albedo 0.5  bounds { min -1 0 -1  max 1 1 1 } }"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_haze_warning"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails = []

    def run(tag, text, mode):
        path = os.path.join(d, tag + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        budget = ["-spp", "1"] if mode in "RDJ" else ["-n", "20000"]
        cmd = [exe, "-in", path, "-mode", mode, "-device", "cpu", "-seed", "1", "-no-meter",
               "-o", os.path.join(d, "%s_%s.png" % (tag, mode))] + budget
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return (p.stdout + p.stderr).count(MARK)

    def check(name, ok, detail=""):
        print("%-60s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    for mode in "ABC":
        n = run("haze", scene(HAZE, SUN), mode)
        check("mode %s warns of the unbounded haze, once" % mode, n == 1, "said %d times" % n)
    n = run("haze", scene(HAZE, SUN), "R")
    check("mode R does not", n == 0, "said %d times" % n)
    n = run("bounded", scene(BOUNDED, SUN), "B")
    check("nor mode B with the medium bounded", n == 0, "said %d times" % n)
    n = run("ball", scene(HAZE, BALL), "B")
    check("nor mode B under a light inside the scene", n == 0, "said %d times" % n)
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
