"""Regression check: the modes that light an unbounded haze only near the scene say so.

    python tools/check_haze_warning.py [--out png/check_haze_warning] [--exe ftrace.exe]

A medium with no bounds has no end, and the modes give it different ones (known-issues HAZE-SUN-B):
mode R lights the haze along every camera ray out to the horizon (so do modes D and J under a
sun; they refuse a sky), while forward emission sends the sun's and the sky's light in through a
disc the size of the scene and lights only the haze behind it -- on the scene below mode B's haze
is 0.15 of mode R's. The backward modes' domain is the one kept, and every mode whose haze comes
from forward photons says so (main.cpp modeLightsHazeOnlyNearScene): A, B and C, P (its forward
film), V (which compares it), M under -beams. The sun over a floor in such a haze (the scene of
scenes/_haze_sun_b_vs_r.ftsl, tiny here), rendered for a moment:

  - modes A, B, C, P, V and M -beams print the warning once, on stderr;
  - under a sun it names R, D and J as lighting all of it; under a sky only R;
  - two mode-B cameras sharing one forward flight print it once;
  - modes R, D and J do not, nor mode M without -beams (which renders no medium);
  - nor does mode B with the medium given `bounds`, or a medium that does not attenuate, or
    under a ball light in the scene instead.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
MARK = "a medium with no bounds, lit by the sun or the sky"
SUN_KEPT = "Modes R, D and J light all of it."
SKY_KEPT = "Mode R lights all of it (modes D and J refuse a sky)."

CAM = 'camera "%s" { eye %s  look_at 0 0 0  up 0 1 0  fov_y 45  film { res 24 18 } }'


def scene(medium, light, cameras=(("c", "0 1.6 2.2"),)):
    return NL.join([
        "scene { units meters }",
        'material "w" { type diffuse reflect 0.6 }',
        light,
        medium,
        "quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }"] +
        [CAM % c for c in cameras]) + NL


SUN = "light sun { dir 0.3 1 0.2  angle 0.53  spd preset:d65 }"
SKY = "light env { spd preset:d65 }"
BALL = "light sphere { center 0 1.5 0  radius 0.05  spd preset:d65 }"
HAZE = "medium { sigma_t 0.04  albedo 0.5 }"
BOUNDED = "medium { sigma_t 0.04  albedo 0.5  bounds { min -1 0 -1  max 1 1 1 } }"
CLEAR = "medium { sigma_t 0  albedo 0.5 }"
TWO_CAMS = (("c", "0 1.6 2.2"), ("d", "1.2 1.4 1.8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_haze_warning"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails = []

    def run(tag, text, mode, extra=()):
        """Renders `text` in `mode`; returns (times stderr carries the warning, stderr, stdout)."""
        path = os.path.join(d, tag + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        budget = ["-spp", "1"] if mode in "RDJ" else ["-n", "20000", "-spp", "1"]
        cmd = [exe, "-in", path, "-mode", mode, "-device", "cpu", "-seed", "1", "-no-meter",
               "-o", os.path.join(d, "%s_%s.png" % (tag, mode))] + budget + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return p.stderr.count(MARK), p.stderr, p.stdout

    def check(name, ok, detail=""):
        print("%-66s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    for mode, extra in (("A", ()), ("B", ()), ("C", ()), ("P", ()), ("V", ()), ("M", ("-beams",))):
        name = "mode %s%s" % (mode, " -beams" if extra else "")
        n, err, out = run("haze", scene(HAZE, SUN), mode, extra)
        check("%s warns of the unbounded haze, once, on stderr" % name, n == 1,
              "said %d times%s" % (n, " (on stdout)" if MARK in out else ""))
        if mode == "B":
            check("under a sun it names R, D and J", SUN_KEPT in err)
    n, err, _ = run("sky", scene(HAZE, SKY), "B")
    check("mode B under a sky warns once", n == 1, "said %d times" % n)
    check("and names only R, which renders a sky", SKY_KEPT in err and SUN_KEPT not in err)
    n, _, _ = run("twocams", scene(HAZE, SUN, TWO_CAMS), "B")
    check("two mode-B cameras on one shared flight warn once", n == 1, "said %d times" % n)
    for mode in "RDJ":
        n, _, _ = run("haze", scene(HAZE, SUN), mode)
        check("mode %s does not" % mode, n == 0, "said %d times" % n)
    n, _, _ = run("haze", scene(HAZE, SUN), "M")
    check("nor mode M without -beams", n == 0, "said %d times" % n)
    n, _, _ = run("bounded", scene(BOUNDED, SUN), "B")
    check("nor mode B with the medium bounded", n == 0, "said %d times" % n)
    n, _, _ = run("clear", scene(CLEAR, SUN), "B")
    check("nor mode B with a medium that does not attenuate", n == 0, "said %d times" % n)
    n, _, _ = run("ball", scene(HAZE, BALL), "B")
    check("nor mode B under a light inside the scene", n == 0, "said %d times" % n)
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
