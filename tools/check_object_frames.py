"""Regression check: things that follow their object, and paths in any script.

    D:/python314/python.exe -u tools/check_object_frames.py [--out <dir>] [--exe ftrace.exe] [--no-gpu]

  * An analytic sphere's UV turns with its group: a ball textured half-and-half in u, in a group
    turned 180 degrees about y, renders as the unturned ball with its u shifted by half a turn
    (mode R on the CPU, and on the GPU unless --no-gpu: the device reads the same axes).
  * `pattern_space object` reads a group's patterns in its own space: a striped ball in a group
    moved 0.3 m along x renders as the same ball with its world-space stripes shifted by 0.3 m;
    and such a scene asked for the GPU says it stays on the CPU.
  * The same turned half round, scaled, in millimetres and reading the normal; and on an
    isosurface. A sphere that falls back to a mesh wears its UV where the analytic one does.
  * ... and on a curve and on a light panel's emission (seen, and sampled).
  * A scene and its image at a path outside any one code page (Latin, Han) open and write, and a
    texture named in a scene file in a legacy code page (cp1252) opens.
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


def scene(group, pattern, extra="", units="meters", body=None):
    """A ball (or `body`) in `group`, its reflectance `pattern`; lengths in `units`."""
    k = {"meters": 1.0, "millimeters": 1000.0}[units]
    L = lambda *xs: " ".join(repr(x * k) for x in xs)
    body = body or "sphere { center 0 0 0  radius %r  material pat }" % (0.45 * k)
    return NL.join([
        "scene { units %s }" % units,
        'material "white" { type diffuse reflect 0.7 }',
        'pattern "p" { expr "%s" }' % pattern,
        'material "pat" { type diffuse reflect pattern:p }',
        "quad { origin %s  u %s  v %s  material white }" % (L(-2, 0, -2), L(0, 0, 4), L(4, 0, 0)),
        "quad { origin %s  u %s  v %s  material white }" % (L(-2, 0, -2), L(4, 0, 0), L(0, 3, 0)),
        "light area { origin %s  u %s  v %s  normal 0 -1 0  spd preset:bb6500  power 60 }"
        % (L(-0.4, 2.9, -0.4), L(0.8, 0, 0), L(0, 0, 0.8)),
        "group { %s  %s }" % (group, body),
        extra,
        'camera "c" { eye %s  look_at %s  up 0 1 0  fov_y 40  film { res %d %d } }'
        % (L(0, 1.0, 4.2), L(0, 0.5, 0), W, H),
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

    # A sphere that falls back to a mesh (a scale a hair off uniform) wears its UV where the
    # analytic one does.
    good, detail = same(exe, out, "uv_mesh", scene("translate 0 0.45 0  rotate 0 30 0", half % "u"),
                        scene("translate 0 0.45 0  rotate 0 30 0  scale 1 1.0001 1", half % "u"),
                        ["-mode", "R", "-device", "cpu", "-spp", "128"])
    verdict("a sphere built as a mesh wears its UV as the analytic one", good, detail)

    stripes = "0.5 + 0.4 * sin(25 * %s)"
    obj = scene("translate 0.3 0.45 0  pattern_space object", stripes % "x")
    world = scene("translate 0.3 0.45 0", stripes % "(x - 0.3)")
    cpu = ["-mode", "R", "-device", "cpu", "-spp", "128"]
    good, detail = same(exe, out, "patspace", obj, world, cpu)
    verdict("pattern_space object reads the group's own space", good, detail)
    # Turned half round, scaled twice, in millimetres, and reading the normal: object x is
    # -(1000 x - 300) / 2 and object nx is -nx.
    turned_obj = scene("translate 300 450 0  rotate 0 180 0  scale 2  pattern_space object",
                       "0.5 + 0.3 * sin(0.05 * x) + 0.15 * nx", units="millimeters",
                       body="sphere { center 0 0 0  radius 225  material pat }")
    turned_world = scene("translate 300 450 0  rotate 0 180 0  scale 2",
                         "0.5 + 0.3 * sin(0.05 * (-(1000 * x - 300) / 2)) - 0.15 * nx",
                         units="millimeters", body="sphere { center 0 0 0  radius 225  material pat }")
    good, detail = same(exe, out, "patspace_turned", turned_obj, turned_world, cpu)
    verdict("... turned, scaled, in mm, with its normal", good, detail)
    iso = "isosurface { material pat  sphere { radius 0.45 } }"
    good, detail = same(exe, out, "patspace_iso",
                        scene("translate 0.3 0.45 0  pattern_space object", stripes % "x", body=iso),
                        scene("translate 0.3 0.45 0", stripes % "(x - 0.3)", body=iso), cpu)
    verdict("... on an isosurface", good, detail)
    tube = "curve { material pat  basis linear  radius 0.1  point -0.45 0 0  point 0.45 0 0 }"
    good, detail = same(exe, out, "patspace_curve",
                        scene("translate 0.3 0.45 0  pattern_space object", stripes % "x", body=tube),
                        scene("translate 0.3 0.45 0", stripes % "(x - 0.3)", body=tube), cpu)
    verdict("... on a curve", good, detail)
    # A light panel facing the camera, its emission striped by the pattern: in the group's space
    # both where it is seen and where it is sampled from.
    panel = ("light area { origin -0.4 -0.3 0.3  u 0.8 0 0  v 0 0.6 0  normal 0 0 1  "
             "spd preset:bb6500  spd_map pattern:p  power 20 }")
    good, detail = same(exe, out, "patspace_light",
                        scene("translate 0.3 0.45 0  pattern_space object", stripes % "x", body=panel),
                        scene("translate 0.3 0.45 0", stripes % "(x - 0.3)", body=panel), cpu)
    verdict("... on a light's emission", good, detail)
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

    # A texture named in the file in a legacy code page (cp1252 "caf\xe9"), the file on disk named
    # in UTF-8: the path is re-read in the system's legacy code page.
    try:
        from PIL import Image
        dl = os.path.join(out, "legacy")
        os.makedirs(dl, exist_ok=True)
        Image.new("RGB", (4, 4), (200, 40, 30)).save(os.path.join(dl, "caf" + chr(0xE9) + ".png"))
        text = scene("translate 0 0.45 0", "0.5").replace(
            'material "pat" { type diffuse reflect pattern:p }',
            'texture "t" { file "caf' + chr(0xE9) + '.png" }' + NL +
            'material "pat" { type diffuse reflect texture:t }')
        pl = os.path.join(dl, "legacy.ftsl")
        with open(pl, "wb") as f:
            f.write(text.encode("cp1252"))
        q = subprocess.run([exe, "-in", pl, "-o", os.path.join(dl, "legacy.png"), "-spp", "1",
                            "-no-meter"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        good = q.returncode == 0 and os.path.exists(os.path.join(dl, "legacy.png"))
        verdict("a texture named in a legacy code page inside a scene", good,
                "" if good else (q.stdout + q.stderr).strip()[-300:])
    except ImportError:
        print("(legacy code page test skipped: needs Pillow)")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
