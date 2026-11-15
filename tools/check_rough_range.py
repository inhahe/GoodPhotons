"""Regression check: a roughness pattern reaches the glossy lobe's whole range, as a constant does.

    python tools/check_rough_range.py [--out png/check_rough_range] [--exe ftrace.exe] [--no-gpu]

The glossy lobe takes a roughness from 0.001 to 2 (ggx::kMaxRoughness), and every way of binding one
-- a constant, a texture, a pattern, a record -- must reach it (clampRoughness, dClampRoughness). A
rough glossy ball on a floor, its roughness 1.6 written as a constant and as `roughness pattern:p`
whose expression is 1.6: mode R on the CPU and the GPU must render the two byte for byte alike. A
pattern held to 1, as a constant is not, renders the ball at roughness 1, the frame 5.6 % brighter.
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

BASE = [
    "scene { units meters }",
    'material "white" { type diffuse reflect 0.6 }',
    "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
    "sphere { center 0 0.5 0  radius 0.5  material rough }",
    "light area { origin -0.5 2 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  power 60  spd preset:d65 }",
    'camera "c" { eye 0 0.8 2.5  look_at 0 0.45 0  up 0 1 0  fov_y 40  film { res 96 72 } }',
]
SCENES = {
    "constant": ['material "rough" { type glossy reflect 0.8 roughness 1.6 }'],
    "pattern": ['pattern "r16" { expr "1.6" }',
                'material "rough" { type glossy reflect 0.8 roughness pattern:r16 }'],
}


def render(exe, out, name, dev):
    scn = os.path.join(out, name + ".ftsl")
    with open(scn, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(BASE + SCENES[name]) + NL)
    png = os.path.join(out, "%s_%s.png" % (name, dev))
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scn, "-mode", "R", "-device", dev, "-spp", "256", "-seed", "1", "-hdr", "-no-meter",
           "-window-min", "-o", png]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return pfm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_rough_range"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        c, p = render(exe, out, "constant", dev), render(exe, out, "pattern", dev)
        same = open(c, "rb").read() == open(p, "rb").read()
        mc = float(read_pfm(c, top_down=True)[..., 1].mean())
        mp = float(read_pfm(p, top_down=True)[..., 1].mean())
        ok = ok and same
        print("%s: mode R on the %s: the pattern's frame %s the constant's (%.4f of it)"
              % ("ok" if same else "FAIL", "GPU" if dev == "gpu" else "CPU",
                 "is byte for byte" if same else "differs from", mp / mc), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
