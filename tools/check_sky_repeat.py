"""Regression check: the sky a forward mode sees directly is the same whenever the render writes it.

    python tools/check_sky_repeat.py [--out png/check_sky_repeat] [--exe ftrace.exe]

The forward modes draw the sky the camera sees directly in a pass of their own (main.cpp
addEnvBackground / prepareSky), summed per pixel and held for the camera while the render writes
it. Two cases, in the room open to a sky (check_bump_modes' fifth room, flat), mode B:

CADENCE. Through a physical lens that pass is a Monte Carlo average whose sample count grows with
the photons per pixel, computed in batches as the writes ask for more. Until 0.440.11 a batch's
random streams were seeded per batch, so the same render written at another cadence -- a live
window's refreshes fall where the clock puts them -- drew other samples and showed another sky.
Each sample is now a function of its pixel and its index alone, added in index order. Through a
35 mm f/4 achromat, written at the default interval and every 0.05 s, the two images must be
byte-identical; 0.440.9 differed in 363 of the 3072 pixels.

STOPPED. A render stopped before its first write still writes its image, and the sky has to be in
it: in 0.440.11 the pinhole pass gave up on the pending stop and the image had no sky at all. A
pinhole render on a long budget with no write due (and so without the live window, whose repaints
would compute the sky before the stop), stopped two seconds after it starts tracing with `ftrace
-stop`: the pure sky at the top of the frame must match a completed render's to 1e-4 (the stopped
render sums it at 4 points a pixel, which a pixel of pure sky resolves exactly).

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402
import check_bump_modes as cbm  # noqa: E402

LENS_CAM = ('camera "c" { eye 0 1.2 4  look_at 0 1.2 0  up 0 1 0  focus 4  '
            'film { res 64 48  format full-frame }  lens { preset achromat  focal 35  fstop 4 } }')


def fresh(png):
    for f in (png, png[:-4] + ".pfm"):          # never score an earlier run's image
        if os.path.exists(f):
            os.remove(f)


def render(cmd, png):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(png[:-4] + ".pfm", top_down=True)


def cadence(exe, out, txt):
    cam = [l for l in txt.split(NL) if l.startswith('camera "c"')]
    assert len(cam) == 1
    scene = os.path.join(out, "lens.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(txt.replace(cam[0], LENS_CAM))
    imgs = []
    for tag, extra in (("slow", []), ("fast", ["-interval", "0.05"])):
        png = os.path.join(out, "B_%s.png" % tag)
        fresh(png)
        imgs.append(render([exe, "-in", scene, "-mode", "B", "-n", "20000000", "-seed", "1", "-hdr",
                            "-no-meter", "-window-min", "-o", png] + extra, png))
    diff = imgs[0] != imgs[1]
    good = not diff.any()
    print("%s: lens camera, sky seen directly, written at two cadences: %d of %d pixels differ"
          % ("ok" if good else "FAIL", int(diff.any(-1).sum()), diff.shape[0] * diff.shape[1]), flush=True)
    return good


def stopped(exe, out, txt):
    scene = os.path.join(out, "pinhole.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(txt)
    full_png = os.path.join(out, "B_full.png")
    fresh(full_png)
    full = render([exe, "-in", scene, "-mode", "B", "-n", "2000000", "-seed", "1", "-hdr", "-no-meter",
                   "-window-min", "-o", full_png], full_png)
    png = os.path.join(out, "B_stopped.png")
    fresh(png)
    p = subprocess.Popen([exe, "-in", scene, "-mode", "B", "-time", "600", "-interval", "600", "-seed", "1",
                          "-hdr", "-no-meter", "-o", png],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace")
    lines, started = [], threading.Event()

    def read():                                  # its output, watching for the render's banner
        for ln in p.stdout:
            lines.append(ln)
            if ln.startswith("mode B:"):
                started.set()
        started.set()                            # it exited first: let the check go on and fail
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        started.wait(120)                        # past the scene load, then two seconds of tracing
        time.sleep(2.0)
    finally:
        subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, text=True)
        p.wait(timeout=300)
        reader.join(timeout=30)
        log = "".join(lines)
    if not os.path.exists(png[:-4] + ".pfm"):
        print(log[-1500:])
        raise SystemExit("the stopped render wrote no image")
    img = read_pfm(png[:-4] + ".pfm", top_down=True)
    w = img.shape[1]
    a = img[0, w // 3: 2 * w // 3].sum(-1)       # the top row's middle third: sky alone
    b = full[0, w // 3: 2 * w // 3].sum(-1)
    rel = float(np.max(np.abs(a - b) / np.maximum(b, 1e-30)))
    good = bool(b.min() > 0.0) and rel <= 1e-4
    print("%s: pinhole render stopped before its first write: its sky %.4f of a completed render's "
          "(worst pixel off by %.2g)" % ("ok" if good else "FAIL", a.sum() / max(b.sum(), 1e-30), rel),
          flush=True)
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_sky_repeat"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    txt = cbm.room(out, 0.0, "sky")
    ok = cadence(exe, out, txt)
    ok = stopped(exe, out, txt) and ok
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
