"""Regression check: a mode that cannot render volumetric emission refuses a fire, not renders it black.

    D:/python314/python.exe -u tools/check_fire_modes.py [--out <dir>] [--exe ftrace.exe]

Volumetric emission ("fire": a medium with a `temperature` grid) is born only by the forward tracer,
so only modes A/B/C render it. Before 0.375.1 the other modes rendered a fire scene as if the
fire were cold smoke, without a word -- black, when the fire was the only light (known-issues
EMISSIVE-VOLUME-SILENT). Writes a small fire as a two-grid VDB (loom.vdbio: a soot `density`
blob, a hot `temperature` core) whose glow is the scene's only light, and requires:
  1. modes A and B render it (exit 0, a non-zero image);
  2. modes R, W, V, P, D, U, J, M and S refuse it (non-zero exit, the reason naming volumetric
     emission) -- P because its composite takes the fire-blind mode R film as its reference;
  3. `-on-unsupported fallback` on a mode-R camera renders it in mode B (non-zero image);
  4. `-on-unsupported strip` on a mode-R camera renders the cold smoke (exit 0, says so);
  5. ...but not when another camera of the run (mode B) renders the fire: the scene is shared,
     so the mode-R camera falls back to B instead, and the B camera's image keeps its fire;
  6. a camera with a physical lens refuses even under `fallback` (no mode renders a fire
     through a lens; B would silently drop it), and `strip` renders it in R, cold;
  7. an exposure-locked mode-R path under `fallback` meters in mode B, the mode it renders in:
     its anchor equals a mode-B run's (the fire-blind mode R meters black frames, which the meter
     must not take for an anchor).
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
        "fire2.ftsl": ['camera "b" { %s  mode B }' % view, 'camera "r" { %s  mode R }' % view],
        "firelens.ftsl": ['camera "cam" { %s  lens { preset achromat  focal 50  fstop 2.8  glass BK7 } }'
                          % view],
        "firepath.ftsl": ['camera_path "fly" { look_at 0 0.5 0  up 0 1 0  fov_y 35  mode R  frames 2',
                          '    film { res 64 48 }  key 0 0 0.5 -3  key 1 0.3 0.5 -3  exposure_lock }'],
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
    for m in ("A", "B"):
        rc, log, mean = run("mode_" + m, ["-mode", m])
        good = rc == 0 and mean is not None and mean > 0.0
        ok &= good
        print("%s: mode %s renders the fire (exit %d, mean %s)" % ("ok" if good else "FAIL", m, rc, mean))
    for m in ("R", "W", "V", "P", "D", "U", "J", "M", "S"):
        rc, log, mean = run("mode_" + m, ["-mode", m])
        good = rc != 0 and "volumetric emission" in log
        ok &= good
        print("%s: mode %s refuses it (exit %d%s)" % ("ok" if good else "FAIL", m, rc,
              "" if good else ("; image mean %s" % mean)))
    rc, log, mean = run("fallback", ["-mode", "R", "-on-unsupported", "fallback"])
    good = rc == 0 and "-> mode B" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: -on-unsupported fallback -> mode B (exit %d, mean %s)" % ("ok" if good else "FAIL", rc, mean))
    rc, log, mean = run("strip", ["-mode", "R", "-on-unsupported", "strip"])
    good = rc == 0 and "stripped" in log
    ok &= good
    print("%s: -on-unsupported strip renders the cold smoke (exit %d)" % ("ok" if good else "FAIL", rc))
    rc, log, mean = run("strip2", ["-on-unsupported", "strip"], scene="fire2.ftsl", out="strip2_b")
    good = rc == 0 and "cannot strip" in log and "-> mode B" in log and mean is not None and mean > 0.0
    ok &= good
    print("%s: strip with a mode-B camera in the run: R falls back, B keeps its fire (exit %d, "
          "B mean %s)" % ("ok" if good else "FAIL", rc, mean))
    rc, log, mean = run("lens_fb", ["-on-unsupported", "fallback"], scene="firelens.ftsl")
    good = rc != 0 and "through a physical lens" in log
    ok &= good
    print("%s: a lens camera refuses under fallback (exit %d)" % ("ok" if good else "FAIL", rc))
    rc, log, mean = run("lens_strip", ["-on-unsupported", "strip"], scene="firelens.ftsl")
    good = rc == 0 and "stripped" in log
    ok &= good
    print("%s: a lens camera renders the cold smoke under strip (exit %d)" % ("ok" if good else "FAIL", rc))
    def anchor(log):
        m = re.search(r"\(anchor ([0-9.eE+-]+)\)", log)
        return float(m.group(1)) if m else None
    rc, log, _ = run("meter", ["-on-unsupported", "fallback"], scene="firepath.ftsl", meter=True)
    rcB, logB, _ = run("meterB", ["-mode", "B"], scene="firepath.ftsl", meter=True)
    aF, aB = anchor(log), anchor(logB)
    # A fire-blind meter does not say so: its black frames give the no-signal anchor of 1.
    good = rc == 0 and rcB == 0 and aF is not None and aB is not None and \
        abs(aF - aB) <= 1e-3 * abs(aB) and aB != 1.0
    ok &= good
    print("%s: an exposure-locked fallback path meters in mode B (anchor %s, mode B's %s)" % (
        "ok" if good else "FAIL", aF, aB))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
