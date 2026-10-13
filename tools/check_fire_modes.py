"""Regression check: a mode that cannot render volumetric emission refuses a fire, not renders it black.

    D:/python314/python.exe -u tools/check_fire_modes.py [--out <dir>] [--exe ftrace.exe]

Volumetric emission ("fire": a medium with a `temperature` grid) is rendered by the forward modes
A/B/C, by mode R (and mode W, which runs on it) and by mode D; before 0.375.1 every other mode
rendered a fire scene as if the fire were cold smoke, without a word -- black, when the fire was the
only light (known-issues EMISSIVE-VOLUME-SILENT). Writes a small fire as a two-grid VDB
(loom.vdbio: a soot `density` blob, a hot `temperature` core) whose glow is the scene's only light,
and requires:
  1. modes A, B, R, W and D render it (exit 0, a non-zero image), D through a physical lens too;
  2. modes V, P, U, J, M and S refuse it (non-zero exit, the reason naming volumetric emission);
  3. `-on-unsupported fallback` on a mode-M camera renders it in mode R (non-zero image);
  4. `-on-unsupported strip` on a mode-M camera renders the cold smoke (exit 0, says so);
  5. ...but not when another camera of the run (mode B) renders the fire: the scene is shared,
     so the mode-M camera falls back to R instead, and the B camera's image keeps its fire;
  6. a camera with a physical lens (routed to mode R) renders the fire;
  7. an exposure-locked mode-M path under `fallback` meters in mode R, the mode it renders in:
     its anchor equals a mode-R run's (a mode that cannot see the fire meters black frames, which
     the meter must not take for an anchor);
  8. strips are decided for the whole run: a fisheye and a rectilinear camera, both mode J, under
     `strip`, in either order -- the fisheye one cannot be fixed by a strip and falls back to R,
     which renders the fire, so the other must not strip it either (both images keep their fire);
  9. a fisheye camera in mode A (one rectilinear thin lens) falls back under `fallback`, and the
     refusal's advice names only the modes that render it, B and R;
 10. a prefer{} branch whose lens camera cannot render -- routed to R, which refuses a fire
     beside a GRIN medium -- yields to the else{} branch, rather than being kept and aborting;
 11. `-no-media` drops the fire with the medium it is made of: mode R renders the scene black
     (its only light gone), where until 0.377.3 the fire's sampling still indexed the dropped
     medium (undefined behaviour: a crash, or the fire drawn from freed memory);
 12. under `strip`, a lens camera in mode R that cannot render a fire beside a GRIN medium, and
     has no fallback, renders in mode R once a mode-M camera's strip has removed the fire -- and
     says so; until 0.377.3 it reported that it could not strip, and fell back to R;
 13. mode W's `-ambient` fills a floor the fire alone lights (its scale counts the fire as a
     light); until 0.377.3 it was zero there, so `-ambient 0.3` changed nothing.
(Mode C is left out: its physical catch through the aperture almost never sees a fire this
small at a test budget.) Exits 1 on any failure.
"""
import argparse
import re
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "loom"))
from loom.vdbio import VolumeGrid, write_vdb  # noqa: E402

NL = chr(10)


def read_pfm_mean(path):
    with open(path, "rb") as f:
        f.readline()
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        return float(np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_fire_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    n = 24
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    density = np.clip(1.0 - r, 0.0, None)
    temperature = np.clip(1.0 - 1.6 * r, 0.0, None)
    box = (-0.5, 0.0, -0.5, 0.5, 1.0, 0.5)
    vdb = os.path.join(d, "fire.vdb").replace(chr(92), "/")
    write_vdb(vdb, [VolumeGrid("density", density, box), VolumeGrid("temperature", temperature, box)])
    medium = [
        'medium {',
        '    sigma_t 4   albedo 0.5   g 0.3',
        # by a relative name, resolved from the output directory the renders run in: a
        # `vdb:` path ends at the first space, and this repository's path has several
        '    density vdb:fire.vdb',
        '    temperature vdb:fire.vdb',
        '    emission blackbody   emission_kelvin 1500   emission_scale 4',
        '}']
    view = 'eye 0 0.5 -3  look_at 0 0.5 0  up 0 1 0  fov_y 35  film { res 64 48 }'
    scenes = {
        "fire.ftsl": ['camera "cam" { %s }' % view],
        "fire2.ftsl": ['camera "b" { %s  mode B }' % view, 'camera "d" { %s  mode M }' % view],
        "firelens.ftsl": ['camera "cam" { %s  lens { preset achromat  focal 50  fstop 2.8  glass BK7 } }'
                          % view],
        "firepath.ftsl": ['camera_path "fly" { look_at 0 0.5 0  up 0 1 0  fov_y 35  mode M  frames 2',
                          '    film { res 64 48 }  key 0 0 0.5 -3  key 1 0.3 0.5 -3  exposure_lock }'],
        "fireorder.ftsl": ['camera "f" { %s  projection fisheye  mode J }' % view,
                           'camera "d" { %s  mode J }' % view],
        "fireorder2.ftsl": ['camera "d" { %s  mode J }' % view,
                            'camera "f" { %s  projection fisheye  mode J }' % view],
        "firefish.ftsl": ['camera "cam" { %s  projection fisheye  mode A }' % view],
        # A GRIN medium away from the view: mode R then refuses the fire beside it.
        "firepref.ftsl": ['prefer {',
                          '    camera "cam" { %s  mode B  lens { preset achromat  focal 50  fstop 2.8 } }'
                          % view,
                          '} else {',
                          '    camera "cam" { %s  mode B }' % view,
                          '}',
                          'medium { bounds { center 3 0.5 3  radius 0.2 }  sigma_t 0.1  albedo 0.5  '
                          'ior "1.0" }'],
        # The fire over a diffuse floor.
        "firefloor.ftsl": ['camera "cam" { %s }' % view,
                           'material "floor" { type diffuse  reflect 0.6 }',
                           'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }'],
        # The same GRIN medium; the lens camera has no fallback, the mode-M one strips the fire.
        "firestrip.ftsl": ['camera "a" { %s  mode R  lens { preset achromat  focal 50  fstop 2.8 } }'
                           % view,
                           'camera "m" { %s  mode M }' % view,
                           'medium { bounds { center 3 0.5 3  radius 0.2 }  sigma_t 0.1  albedo 0.5  '
                           'ior "1.0" }'],
    }
    for name, cams in scenes.items():
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(cams + medium) + NL)

    def run(tag, args, scene="fire.ftsl", meter=False, out=None):
        png = os.path.join(d, (out or tag) + ".png")
        for p in (png, png[:-4] + ".pfm"):
            if os.path.exists(p):
                os.remove(p)
        r = subprocess.run([a.exe, "-in", scene, "-device", "cpu", "-spp", "4", "-n", "2000000",
                            "-hdr", "-o", tag + ".png"] + ([] if meter else ["-no-meter"]) + args,
                           cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        pfm = png[:-4] + ".pfm"
        mean = read_pfm_mean(pfm) if os.path.exists(pfm) else None
        return r.returncode, r.stdout + r.stderr, mean

    ok = True
    for m in ("A", "B", "R", "W", "D"):
        rc, log, mean = run("mode_" + m, ["-mode", m])
        good = rc == 0 and mean is not None and mean > 0.0
        ok &= good
        print("%s: mode %s renders the fire (exit %d, mean %s)" % ("ok" if good else "FAIL", m, rc, mean))
    rc, log, mean = run("lens_D", ["-mode", "D"], scene="firelens.ftsl")
    good = rc == 0 and "-> mode R" not in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: mode D renders it through a physical lens (exit %d, mean %s)" % (
        "ok" if good else "FAIL", rc, mean))
    for m in ("V", "P", "U", "J", "M", "S"):
        rc, log, mean = run("mode_" + m, ["-mode", m])
        good = rc != 0 and "volumetric emission" in log
        ok &= good
        print("%s: mode %s refuses it (exit %d%s)" % ("ok" if good else "FAIL", m, rc,
              "" if good else ("; image mean %s" % mean)))
    rc, log, mean = run("nomedia", ["-mode", "R", "-no-media"])
    good = rc == 0 and mean is not None and mean == 0.0
    ok &= good
    print("%s: -no-media drops the fire with its medium (exit %d, mean %s)" % (
        "ok" if good else "FAIL", rc, mean))
    rc, log, mean = run("fallback", ["-mode", "M", "-on-unsupported", "fallback"])
    good = rc == 0 and "-> mode R" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: -on-unsupported fallback -> mode R (exit %d, mean %s)" % ("ok" if good else "FAIL", rc, mean))
    rc, log, mean = run("strip", ["-mode", "M", "-on-unsupported", "strip"])
    good = rc == 0 and "stripped" in log
    ok &= good
    print("%s: -on-unsupported strip renders the cold smoke (exit %d)" % ("ok" if good else "FAIL", rc))
    rc, log, mean = run("strip2", ["-on-unsupported", "strip"], scene="fire2.ftsl", out="strip2_b")
    good = rc == 0 and "cannot strip" in log and "-> mode R" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: strip with a mode-B camera in the run: M falls back, B keeps its fire (exit %d, "
          "B mean %s)" % ("ok" if good else "FAIL", rc, mean))
    rc, log, mean = run("lens", [], scene="firelens.ftsl")
    good = rc == 0 and mean is not None and mean > 0.0
    ok &= good
    print("%s: a lens camera renders the fire in mode R (exit %d, mean %s)" % ("ok" if good else "FAIL", rc, mean))
    def anchor(log):
        m = re.search(r"\(anchor ([0-9.eE+-]+)\)", log)
        return float(m.group(1)) if m else None
    rc, log, _ = run("meter", ["-on-unsupported", "fallback"], scene="firepath.ftsl", meter=True)
    rcB, logB, _ = run("meterR", ["-mode", "R"], scene="firepath.ftsl", meter=True)
    aF, aB = anchor(log), anchor(logB)
    # A fire-blind meter does not say so: its black frames give the no-signal anchor of 1.
    good = rc == 0 and rcB == 0 and aF is not None and aB is not None and \
        abs(aF - aB) <= 1e-3 * abs(aB) and aB != 1.0
    ok &= good
    print("%s: an exposure-locked fallback path meters in mode R (anchor %s, mode R's %s)" % (
        "ok" if good else "FAIL", aF, aB))
    for scene, tag in (("fireorder.ftsl", "order1"), ("fireorder2.ftsl", "order2")):
        stale = os.path.join(d, tag + "_d.pfm")   # run() clears only the `out` camera's image
        if os.path.exists(stale):
            os.remove(stale)
        rc, log, _ = run(tag, ["-on-unsupported", "strip"], scene=scene, out=tag + "_f")
        mf = read_pfm_mean(os.path.join(d, tag + "_f.pfm")) if rc == 0 else None
        md = read_pfm_mean(os.path.join(d, tag + "_d.pfm")) if rc == 0 else None
        good = (rc == 0 and "stripped" not in log and mf is not None and md is not None and
                mf > 0.0 and md > 0.0)
        ok &= good
        print("%s: strip decided for the run (%s): neither camera strips the fire the other "
              "renders (exit %d, means f %s, d %s)" % ("ok" if good else "FAIL", tag, rc, mf, md))
    rc, log, mean = run("fish_fb", ["-on-unsupported", "fallback"], scene="firefish.ftsl")
    good = rc == 0 and "-> mode R" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: a fisheye camera in mode A falls back to R (exit %d, mean %s)" % (
        "ok" if good else "FAIL", rc, mean))
    rc, log, _ = run("fish_err", [], scene="firefish.ftsl")
    good = rc != 0 and "use mode B/R," in log
    ok &= good
    print("%s: ...and the refusal's advice is mode B/R (exit %d)" % ("ok" if good else "FAIL", rc))
    rc, log, mean = run("pref", [], scene="firepref.ftsl")
    good = rc == 0 and "using branch 2 of 2" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: a prefer branch whose lens camera cannot render yields to else (exit %d, mean %s)"
          % ("ok" if good else "FAIL", rc, mean))
    rc, log, _ = run("stripother", ["-on-unsupported", "strip"], scene="firestrip.ftsl",
                     out="stripother_a")
    good = (rc == 0 and "camera 'a': volumetric emission" in log and
            "stripped for the other cameras; rendering in mode R" in log and "cannot strip" not in log)
    ok &= good
    print("%s: a camera the other cameras' strips made renderable says so (exit %d)" % (
        "ok" if good else "FAIL", rc))
    if not good:
        print(log[-1200:])
    rc0, _, m0 = run("amb0", ["-mode", "W"], scene="firefloor.ftsl")
    rc1, _, m1 = run("amb3", ["-mode", "W", "-ambient", "0.3"], scene="firefloor.ftsl")
    good = rc0 == 0 and rc1 == 0 and m0 and m1 and m1 > 1.05 * m0
    ok &= good
    print("%s: -ambient fills a fire-lit floor in mode W (means %s -> %s)" % (
        "ok" if good else "FAIL", m0, m1))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
