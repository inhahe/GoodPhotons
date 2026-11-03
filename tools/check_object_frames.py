"""Regression check: things that follow their object, and paths in any script.

    D:/python314/python.exe -u tools/check_object_frames.py [--out <dir>] [--exe ftrace.exe] [--no-gpu]

  * An analytic sphere's UV turns with its group: a ball textured half-and-half in u, in a group
    turned 180 degrees about y, renders as the unturned ball with its u shifted by half a turn
    (mode R on the CPU, and on the GPU unless --no-gpu: the device reads the same axes).
  * `pattern_space object` reads a group's patterns in its own space: a striped ball in a group
    moved 0.3 m along x renders as the same ball with its world-space stripes shifted by 0.3 m;
    and such a scene asked for the GPU says it stays on the CPU.
  * A scene and its image at a path outside any one code page (Latin, Han) open and write.
Each image test compares 4x4-pixel blocks of luminance, two seeds each: within 5 standard errors
or 3 %. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_motion import read_pfm, lum, blocks, block_err, W, H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def scene(group, pattern, extra=""):
    return NL.join([
        "scene { units meters }",
        'material "white" { type diffuse reflect 0.7 }',
        'pattern "p" { expr "%s" }' % pattern,
        'material "pat" { type diffuse reflect pattern:p }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
        "group { %s  sphere { center 0 0 0  radius 0.45  material pat } }" % group,
        extra,
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  film { res %d %d } }'
        % (W, H),
    ]) + NL


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def render(exe, path, png, args, seed):
    pfm = png[:-4] + ".pfm"
    cmd = [exe, "-in", path, "-seed", str(seed), "-hdr", "-no-meter", "-o", png] + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm), p.stdout + p.stderr


def same(exe, out, tag, a_text, b_text, args):
    """Whether scenes a and b render alike (4x4 blocks), and the worst block."""
    pa, pb = os.path.join(out, tag + "_a.ftsl"), os.path.join(out, tag + "_b.ftsl")
    write(pa, a_text)
    write(pb, b_text)
    la = [lum(render(exe, pa, os.path.join(out, "%s_a%d.png" % (tag, s)), args, s)[0]) for s in (1, 2)]
    lb = [lum(render(exe, pb, os.path.join(out, "%s_b%d.png" % (tag, s)), args, s)[0]) for s in (1, 2)]
    ma, mb = blocks((la[0] + la[1]) / 2, 4), blocks((lb[0] + lb[1]) / 2, 4)
    err = np.hypot(block_err(la[0], la[1], 4), block_err(lb[0], lb[1], 4))
    z = np.abs(ma - mb) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
    bad = (z > 5.0) & (rel > 0.03)
    return not bad.any(), "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_object_frames"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-58s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""))

    half = "0.2 + 0.6 * smoothstep(0.47, 0.53, %s)"
    turned = scene("translate 0 0.45 0  rotate 0 180 0", half % "u")
    shifted = scene("translate 0 0.45 0", half % "((u + 0.5) % 1)")
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        good, detail = same(exe, out, "uv_" + dev, turned, shifted,
                            ["-mode", "R", "-device", dev, "-spp", "128"])
        verdict("a sphere's UV turns with its group (mode R %s)" % dev.upper(), good, detail)

    stripes = "0.5 + 0.4 * sin(25 * %s)"
    obj = scene("translate 0.3 0.45 0  pattern_space object", stripes % "x")
    world = scene("translate 0.3 0.45 0", stripes % "(x - 0.3)")
    good, detail = same(exe, out, "patspace", obj, world, ["-mode", "R", "-device", "cpu", "-spp", "128"])
    verdict("pattern_space object reads the group's own space", good, detail)
    if not a.no_gpu:
        p = os.path.join(out, "patspace_a.ftsl")
        _, log = render(exe, p, os.path.join(out, "patspace_gpu.png"),
                        ["-mode", "R", "-device", "gpu", "-spp", "1"], 1)
        verdict("an object-space scene asked for the GPU stays on the CPU", "using CPU" in log,
                "" if "using CPU" in log else log.strip()[-300:])

    d = os.path.join(out, "zo" + chr(0xEB) + " " + chr(0x5C71))
    p = os.path.join(d, "x.ftsl")
    write(p, scene("translate 0 0.45 0", "0.5"))
    png = os.path.join(d, "x.png")
    q = subprocess.run([exe, "-in", p, "-o", png, "-spp", "1", "-no-meter"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    good = q.returncode == 0 and os.path.exists(png)
    verdict("a scene and image at a non-ASCII path", good, "" if good else (q.stdout + q.stderr)[-300:])

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
