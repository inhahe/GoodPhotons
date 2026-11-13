"""Regression check: per-object render modes reach every object -- a lamp's surface and a bare instance too.

    python tools/check_route_tags.py [--out png/check_route_tags] [--exe ftrace.exe] [--no-gpu]

A `mode D` / `mode J` on an object routes each camera path by the surfaces its camera end meets
(FTSL.md, Per-object render modes), so a frame whose every object is tagged J renders, under a
mode-D camera, exactly as mode J does, and every object tagged D under a mode-J camera exactly as
mode D. Until 0.440.14 two kinds of object kept the camera's mode whatever they were tagged: an
emissive surface (a light is found by its material, and the loader would not copy it) and a
mesh_instance without a `material` override (its asset's materials are shared by every placement).
Each warned, and a frame with either in view was not the tagged mode's.

A room with a lamp in view (`light area`, tagged) and a box placed by a bare `mesh_instance`
(tagged), modes D and J at 16 spp on the host and the device: each tagged render must be
byte-identical to the untagged render in the other mode -- but for the device's tagged-D frame
under a mode-J camera: there mode J's renderer draws other random streams than mode D's in a scene
without media (known-issues ROUTE-D-GPU-STREAM), so that pair is held to agree in the mean, within
3 sigma over six seeds at 64 spp, until it is byte-identical too. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def box_obj(path):
    v = [(x, y, z) for x in (-0.3, 0.3) for y in (0.0, 0.6) for z in (-0.3, 0.3)]
    f = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for p in v:
            fh.write("v %g %g %g\n" % p)
        for q in f:
            fh.write("f %d %d %d\nf %d %d %d\n" % (q[0], q[1], q[2], q[0], q[2], q[3]))


def scene(d, tag):
    """The room, every object (the lamp and the instance among them) tagged `mode <tag>`, or none."""
    obj = os.path.join(d, "box.obj").replace(chr(92), "/")
    t = ("  mode %s" % tag) if tag else ""
    return NL.join([
        "scene { units meters }",
        'material "w" { type diffuse reflect 0.7 }',
        'material "r" { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'mesh_asset "box" { file "%s"  material r }' % obj,
        'mesh_instance { of "box"  translate 0.4 0 -0.5  rotate 0 30 0%s }' % t,
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w%s }" % t,
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material w%s }" % t,
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material w%s }" % t,
        "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material w%s }" % t,
        "quad { origin -2 3 -2  u 4 0 0  v 0 0 4  material w%s }" % t,
        "light area { origin -0.5 2.4 -1.2  u 1 0 0  v 0 0 0.5  normal 0 -1 0.3  spd preset:bb6500  power 40%s }" % t,
        'camera "c" { eye 0 1.4 3.5  look_at 0 1.1 -1  up 0 1 0  fov_y 55  film { res 64 48 } }',
    ]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_route_tags"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    box_obj(os.path.join(out, "box.obj"))
    paths = {}
    for tag in ("", "D", "J"):
        p = os.path.join(out, "room%s.ftsl" % tag)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(scene(out, tag))
        paths[tag] = p

    def render(tag, mode, dev, seed=1, spp=16):
        png = os.path.join(out, "%s_%s%s_%d.png" % (dev, mode, ("_tag" + tag) if tag else "", seed))
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):                                # never score an earlier run's image
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", paths[tag], "-mode", mode, "-device", dev, "-spp", str(spp), "-heroc", "1",
               "-seed", str(seed), "-hdr", "-no-meter", "-window-min", "-o", png]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        warn = [ln for ln in (p.stdout + p.stderr).splitlines() if "warning" in ln and "mode" in ln]
        with open(pfm, "rb") as f:
            return f.read(), warn, pfm

    ok = True
    for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
        for tag, cam, want in (("J", "D", "J"), ("D", "J", "D")):
            if dev == "gpu" and tag == "D":                 # ROUTE-D-GPU-STREAM: in the mean
                sys.path.insert(0, os.path.join(ROOT, "tools"))
                from pfm import read_pfm
                v = {k: [read_pfm(render(t, m, dev, s, 64)[2], top_down=True).sum(-1).mean()
                         for s in range(1, 7)] for k, t, m in (("tag", tag, cam), ("ref", "", want))}
                a, b = np.mean(v["tag"]), np.mean(v["ref"])
                z = (a - b) / np.hypot(np.std(v["tag"], ddof=1), np.std(v["ref"], ddof=1)) * np.sqrt(6)
                good = abs(z) <= 3.0
                ok &= good
                print("%s: every object tagged %s under a mode-%s camera, on the device: %.4f of mode %s "
                      "(%+.1f sigma; byte identity is ROUTE-D-GPU-STREAM)"
                      % ("ok" if good else "FAIL", tag, cam, a / b, want, z), flush=True)
                continue
            got, warn, _ = render(tag, cam, dev)
            ref, _, _ = render("", want, dev)
            good = got == ref and not warn
            ok &= good
            print("%s: every object tagged %s under a mode-%s camera, on the %s: %s mode %s%s"
                  % ("ok" if good else "FAIL", tag, cam, "host" if dev == "cpu" else "device",
                     "byte-identical to" if got == ref else "differs from", want,
                     ("; " + warn[0].strip()) if warn else ""), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
