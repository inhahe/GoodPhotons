"""Regression check: the device clears a big primitive's own surface near the world origin.

    D:/python314/python.exe -u tools/check_big_primitive.py [--out <dir>] [--exe ftrace.exe]

A ground sphere of radius 1000 centred 1000 below the origin (the "one weekend" layout), a ball on it at
the origin and a lamp above, rendered in mode W at 16 spp (deterministic) on the GPU and the CPU: the
device's frame within 1 % of the host's and every live tile of a 4 x 4 grid within 3 %. Then in mode D
(the bidirectional walks, whose continuations and connections leave the ground too), GPU at 1024 spp
against CPU at 256 (both fixed seeds, so deterministic): the frame within 0.5 % and every live tile
within 1 % -- 0.449.1, whose BDPT walk left the ground without its reach, read x0.9916 and 1.6 %.

A hit on that sphere carries a float error of ulps of its reach (2000), not of the hit's own coordinates
near the origin: an origin pushed off it by an amount sized at the hit, and a next trace whose tmin is
too, meets the sphere again, and the ground's light samples shadow themselves -- the GPU read the ground
dark in speckles after its tmin stopped being an absolute 1e-4 (0.448.6). The push is sized by the
primitive's reach (DHit::pscale, REVIEW-0448D). Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

from check_modew_fiber import compare, render_w  # noqa: E402
from pfm import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENE = [
    'scene { units meters }',
    'camera "c" { eye 0 1.2 -4  look_at 0 0.3 0  up 0 1 0  fov_y 45  film { res 96 72 } }',
    'material "ground" { type diffuse  reflect 0.6 }',
    'material "ball"   { type diffuse  reflect rgb 0.7 0.3 0.2 }',
    'sphere "g" { center 0 -1000 0  radius 1000  material ground }',
    'sphere "b" { center 0 0.5 0  radius 0.5  material ball }',
    'light area { origin -1 3 -1  u 2 0 0  v 0 0 2  normal 0 -1 0  power 200  spd preset:d65 }',
]


def render(exe, out, scene, tag, args):
    """Render `scene` with `args` to <out>/<tag>.png; its HDR image summed over the channels (a GPU render
    that fell back to the CPU fails)."""
    png = os.path.join(out, tag + ".png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scene, "-hdr", "-no-meter", "-o", png] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(pfm):
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    if "gpu" in args and "-> GPU" not in r.stdout + r.stderr:
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    return read_pfm(pfm, top_down=False).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_big_primitive"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = os.path.join(out, "ground.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    img = {dev: render_w(exe, out, scene, "ground_" + dev, dev) for dev in ("cpu", "gpu")}
    frame, worst = compare(img["gpu"], img["cpu"])
    ok = abs(frame - 1.0) <= 0.01 and worst <= 0.03
    print("%s: a radius-1000 ground under the origin, mode W device / host: frame x%.4f, worst tile off by "
          "%.2f %%" % ("ok" if ok else "FAIL", frame, 100.0 * worst))
    d = {dev: render(exe, out, scene, "ground_D_" + dev, ["-mode", "D", "-device", dev, "-spp", spp])
         for dev, spp in (("cpu", "256"), ("gpu", "1024"))}
    frame, worst = compare(d["gpu"], d["cpu"])
    good = abs(frame - 1.0) <= 0.005 and worst <= 0.01
    ok &= good
    print("%s: the same in mode D, device / host: frame x%.4f, worst tile off by %.2f %%"
          % ("ok" if good else "FAIL", frame, 100.0 * worst))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
