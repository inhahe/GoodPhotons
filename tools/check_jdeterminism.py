"""Regression check: CPU modes J and M at a fixed -spp render the same image every time.

    python tools/check_jdeterminism.py [--out png/check_jdeterminism] [--exe ftrace.exe]

`-device cpu` is the reference and validation device, so it must repeat itself. Mode J splits a
render into light-side epochs, each drawing a new beam map; on the CPU at a fixed -spp an epoch
ends on work counts (workcount.h: vertices traced, connections, beams and photons weighed, the
beam map's build), not on the clock, which made which samples met which map depend on the
machine's speed (known-issues J-CPU-NONDET). Mode M's loop does the same (MODEM-CPU-NONDET), and
its camera pass stratifies its wavelengths by a sample's absolute index rather than within each
call, whose size also followed the clock. The scene is scraps/jspec2/hetmed's -- a heterogeneous
fog, whose transmittance estimates draw from the camera's stream, so a sample that meets another
map takes other draws -- rendered at 32x32 in mode J (256 spp, beam merges only) and in mode M
(-beams, 64 spp, -beamrefresh 0.5), each:

  - twice, byte for byte the same;
  - and at one thread, the same again: the counts are sums over the frame, which no division of
    it among threads changes.

Each needs more than one epoch to mean anything, and says how many it got. And mode M resumed
from a checkpoint draws its first light side under a salt of its own, not the first session's.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENE = NL.join([
    "scene { units meters  spectral 360 830 1 }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium {",
    "    sigma_a rgb 0.15 0.45 0.9   sigma_s 0.5   g 0.0",
    "    bounds { min 0 0 0   max 1 1 1 }",
    '    density "0.4 + 0.6*saturate(1.2 - 2.0*sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2))"',
    "}",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    'camera "cam" {',
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 32 32 }",
    "}"]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jdeterminism"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "jdet.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    fails = []

    CASES = {"J": ["-mode", "J", "-spp", "256", "-beamcount", "100000", "-beamk", "2", "-nojsurf"],
             "M": ["-mode", "M", "-beams", "-n", "20000", "-spp", "64", "-beamrefresh", "0.5"]}

    def render(mode, tag, extra=()):
        png = os.path.join(d, "%s_%s.png" % (mode, tag))
        cmd = [exe, "-in", ftsl, "-device", "cpu", "-max-bounce", "8", "-no-meter", "-hdr",
               "-seed", "1", "-o", png] + CASES[mode] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        m = re.search(r"\[jstats\] (\d+) epochs|averaged (\d+) independent light-side",
                      p.stdout + p.stderr)
        with open(png[:-4] + ".pfm", "rb") as f:
            return f.read(), int(m.group(1) or m.group(2)) if m else 0

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    for mode in CASES:
        first, epochs = render(mode, "run1")
        check("mode %s: more than one light-side epoch" % mode, epochs > 1, "%d epochs" % epochs)
        second, _ = render(mode, "run2")
        check("mode %s: a second run is byte for byte the first" % mode, second == first)
        single, _ = render(mode, "t1", ("-t", "1"))
        check("mode %s: so is a run at one thread" % mode, single == first)

    # A resumed mode M render must not gather from a light side the checkpoint already averaged:
    # its first is drawn under a salt past the loaded samples', so it deposits another count.
    def deposited(spp, extra):
        png = os.path.join(d, "M_resume.png")
        cmd = [exe, "-in", ftsl, "-device", "cpu", "-max-bounce", "8", "-no-meter", "-hdr",
               "-seed", "1", "-o", png, "-mode", "M", "-beams", "-n", "20000", "-spp", str(spp),
               "-beamrefresh", "0.5"] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        m = re.search(r"mode M: deposited (\d+) photons", p.stdout)
        return int(m.group(1)) if m else -1
    d0 = deposited(16, ("-checkpoint",))
    d1 = deposited(32, ("-checkpoint", "-resume"))
    check("mode M: a resume draws its first light side afresh", d0 > 0 and d1 > 0 and d1 != d0,
          "%d photons, then %d" % (d0, d1))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
