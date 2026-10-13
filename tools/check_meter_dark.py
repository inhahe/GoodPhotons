"""Regression check: the exposure meter does not take a frame that is nearly black for an anchor.

    D:/python314/python.exe -u tools/check_meter_dark.py [--out <dir>] [--exe ftrace.exe]

An `exposure_lock` path meters some of its frames and locks every frame to their auto-exposure
anchor, 0.9 over the 99th-percentile luminance. A frame with under ~1 % of its pixels lit has a
p99 of 0 and asks for no exposure at all; the image writer then shows it at exposure 1, and until
0.377.1 the meter took that 1 for a reading whenever ANY pixel was lit -- locking the whole path
at 1 without a word. It now skips such a frame (autoExposureOf), and a path whose meter frames
are all of that kind ends in the "produced no valid anchor" notice.

The scene is a black room lit by nothing but a small lamp seen directly, 15 of a 64 x 48 frame's
3072 pixels. Requires the notice and no anchor; 0.377.0 locked the path at 1. Exits 1 on failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENE = [
    'material "black" { type diffuse  reflect 0 }',
    'quad { origin -5 0 -5   u 0 0 10   v 10 0 0   material black }',
    'light area { origin -0.075 0.425 2  u 0 0.15 0  v 0.15 0 0  normal 0 0 -1  spd blackbody 3000 }',
    'camera_path "fly" { look_at 0 0.5 2  up 0 1 0  fov_y 35  mode R  frames 3',
    '    film { res 64 48 }  key 0 0 0.5 -2  key 1 0.2 0.5 -2  exposure_lock }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_meter_dark"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "s.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    p = subprocess.run([a.exe, "-in", "s.ftsl", "-device", "cpu", "-spp", "8", "-hdr", "-o", "o.png"],
                       cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = p.stdout + p.stderr
    m = re.search(r"\[meter\][^\n]*\(anchor ([0-9.eE+-]+)\)", log)
    anchor = float(m.group(1)) if m else None
    notice = "produced no valid anchor" in log
    good = p.returncode == 0 and notice and anchor is None
    print("%s: a nearly black meter frame is no anchor (exit %d; notice %s; anchor %s)" % (
        "ok" if good else "FAIL", p.returncode, notice, anchor))
    if not good:
        print(log[-1500:])
    print("PASS" if good else "FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
