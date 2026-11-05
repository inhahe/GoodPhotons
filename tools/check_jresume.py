"""Regression check: a resumed mode-J render goes on down the merge radii's shrink schedule.

    python tools/check_jresume.py [--out png/check_jresume] [--exe ftrace.exe] [--device gpu|cpu]

Mode J under the wall-clock refresh (a -time budget) shrinks its beam kernel and its surface
merge radius per light-side realization (surfRadiusFor, beamRadScaleFor). A render that is
stopped and resumed from its .ftbuf must carry on from the realizations its film holds, or every
piece it is resumed in starts again at the largest radius and the bias never shrinks past a
piece's worth. Renders a small fog box for a few seconds with -checkpoint, then resumes it twice:

  - the sidecar records the realizations (its FTLSEPO1 section), and each resume's count is
    larger than the one before;
  - each resume says it goes on after that many, with the kernel scale (K+1)^(alpha-1) at the
    default -vcmalpha 0.75 and the surface radius's square root of it;
  - a fresh render (no -resume) over the same sidecar starts from 0 and says nothing.

Exits 1 on any failure.
"""
import argparse
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
TAG = b"FTLSEPO1"
ALPHA = 0.75

SCENE = NL.join([
    "scene { units meters }",
    'camera "c" { eye 0 1 -3.2  look_at 0 1 0  up 0 1 0  fov_y 40  film { res 48 48 } }',
    'material "w" { type diffuse reflect 0.6 }',
    "quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }",
    "quad { origin -1 2 -1  u 2 0 0  v 0 0 2  material w }",
    "quad { origin -1 0 1  u 2 0 0  v 0 2 0  material w }",
    "quad { origin -1 0 -1  u 0 2 0  v 0 0 2  material w }",
    "quad { origin 1 0 -1  u 0 0 2  v 0 2 0  material w }",
    "light area { origin -0.3 1.98 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  power 60  spd preset:d65 }",
    "medium { sigma_t 0.4 albedo 0.8 g 0.0 }",
]) + NL


def sidecar_count(path):
    """The light-side realizations a sidecar records, or None when it has no such section."""
    if not os.path.exists(path):
        return None
    data = open(path, "rb").read()
    i = data.rfind(TAG)
    return struct.unpack_from("<q", data, i + 8)[0] if i >= 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jresume"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(d, "fog.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    png = os.path.join(d, "j.png")
    ftbuf = png + ".ftbuf"
    if os.path.exists(ftbuf):
        os.remove(ftbuf)
    fails = []

    def check(name, ok, detail=""):
        print("%-66s %s  %s" % (name, "ok  " if ok else "FAIL", detail), flush=True)
        if not ok:
            fails.append(name)

    def run(resume):
        cmd = [exe, "-in", scene, "-mode", "J", "-device", a.device, "-time", "6", "-checkpoint",
               "-max-bounce", "6", "-seed", "1", "-no-meter", "-o", png] + (["-resume"] if resume else [])
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-2000:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return p.stdout + p.stderr

    log = run(False)
    k0 = sidecar_count(ftbuf)
    # One or more: the CPU may draw a single realization in the budget, the resumes' growth is held below.
    check("a fresh render's sidecar records its realizations", k0 is not None and k0 >= 1, "count %s" % k0)
    check("  ... and it says nothing of a resume", "resuming after" not in log)
    prev = k0 or 0
    for piece in (1, 2):
        log = run(True)
        m = re.search(r"resuming after (\d+) light-side realizations: .*beam kernel x([0-9.e+-]+), "
                      r"surface radius x([0-9.e+-]+)", log)
        said = int(m.group(1)) if m else None
        check("resume %d goes on after the %d realizations recorded" % (piece, prev), said == prev,
              "said %s" % said)
        if m:
            want = (prev + 1) ** (ALPHA - 1.0)
            kb, ks = float(m.group(2)), float(m.group(3))
            check("  ... at kernel scale (K+1)^(alpha-1) and its square root",
                  abs(kb / want - 1) < 1e-3 and abs(ks / want ** 0.5 - 1) < 1e-3,
                  "x%.4g / x%.4g, want x%.4g / x%.4g" % (kb, ks, want, want ** 0.5))
        k = sidecar_count(ftbuf)
        check("  ... and its sidecar records more", k is not None and k > prev, "count %s" % k)
        prev = k or prev
    log = run(False)
    k = sidecar_count(ftbuf)
    check("a fresh render over the sidecar starts from 0", "resuming after" not in log and k is not None and k < prev,
          "count %s after %d" % (k, prev))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
