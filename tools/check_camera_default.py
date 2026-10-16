"""Regression check: which cameras a render selects, with and without `-camera` / `-frames`.

    D:/python314/python.exe -u tools/check_camera_default.py [--exe ftrace.exe]

A scene with a still, a 3-frame `camera_path` and a 4-frame `camera_orbit`, and one with the
path alone, rendered at 16x16 with one sample, counting the images written:

  - no `-camera`: the still only, and each skipped path named with its frame count;
  - `-camera all`: everything; `-camera dolly`: the path's frames, nothing named as skipped;
  - a scene of paths alone, no `-camera`: its frames;
  - `-camera dolly -frames 1 2`: two frames;
  - `-frames` without a path named by `-camera` (none, or a still), or on a built-in scene:
    refused with a message.

Everything is written in a temporary directory of its own.

Exits 1 on any failure.
"""
import argparse
import glob
import os
import tempfile
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
COMMON = [
    'material "white" { type diffuse  reflect 0.7 }',
    'quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }',
    'light area { origin -0.3 1.5 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  power 50  spd preset:bb5500 }',
]
STILL = 'camera "still" { eye 0 1 3  look_at 0 0 0  up 0 1 0  fov_y 40  film { res 16 16 } }'
DOLLY = ('camera_path "dolly" { look_at 0 0 0  up 0 1 0  fov_y 40  frames 3  film { res 16 16 }' + NL
         + '    key 0 0 1 3' + NL + '    key 1 1 1 3 }')
SPIN = ('camera_orbit "spin" { center 0 0.5 0  radius 3  height 0.5  up 0 1 0  fov_y 40  frames 4'
        + '  film { res 16 16 } }')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="check_camera_default_") as d:   # its own, whole
        return check_all(os.path.abspath(a.exe), d)


def check_all(exe, d):
    for name, blocks in (("mixed", [STILL, DOLLY, SPIN]), ("paths", [DOLLY])):
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(COMMON + blocks) + NL)

    fails = []

    def run(tag, scene, args):
        """Render `scene` (a scene file's stem, or a built-in's `-scene` name prefixed '@')."""
        out = os.path.join(d, tag)
        os.makedirs(out)
        src = ["-scene", scene[1:], "-r", "16"] if scene.startswith("@") else ["-in", scene + ".ftsl"]
        cmd = [exe] + src + ["-mode", "R", "-device", "cpu", "-spp", "1", "-no-meter",
                             "-o", os.path.join(tag, "img.png")] + args
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        n = len(glob.glob(os.path.join(out, "**", "*.png"), recursive=True))
        return p.returncode, p.stdout + p.stderr, n

    def check(ok, what):
        print("%s: %s" % ("ok" if ok else "FAIL", what), flush=True)
        if not ok:
            fails.append(what)

    rc, log, n = run("default", "mixed", [])
    check(rc == 0 and n == 1, "no -camera renders the still alone (exit %d, %d images)" % (rc, n))
    check("path 'dolly' (3 frames) is not rendered by default" in log
          and "path 'spin' (4 frames) is not rendered by default" in log,
          "no -camera names both skipped paths with their frame counts")
    rc, log, n = run("all", "mixed", ["-camera", "all"])
    check(rc == 0 and n == 8, "-camera all renders the still and all 7 frames (exit %d, %d images)" % (rc, n))
    rc, log, n = run("dolly", "mixed", ["-camera", "dolly"])
    check(rc == 0 and n == 3 and "not rendered by default" not in log,
          "-camera dolly renders the path's 3 frames, skipping nothing by name (exit %d, %d images)" % (rc, n))
    rc, log, n = run("pathsonly", "paths", [])
    check(rc == 0 and n == 3 and "not rendered by default" not in log,
          "a scene of paths alone renders them without -camera (exit %d, %d images)" % (rc, n))
    rc, log, n = run("range", "mixed", ["-camera", "dolly", "-frames", "1", "2"])
    check(rc == 0 and n == 2, "-camera dolly -frames 1 2 renders 2 frames (exit %d, %d images)" % (rc, n))
    rc, log, n = run("framesbare", "mixed", ["-frames", "0", "1"])
    check(rc != 0 and n == 0 and "needs -camera" in log,
          "-frames without -camera is refused (exit %d, %d images)" % (rc, n))
    rc, log, n = run("framesstill", "mixed", ["-camera", "still", "-frames", "0", "1"])
    check(rc != 0 and n == 0 and "needs -camera" in log,
          "-frames with a still named is refused (exit %d, %d images)" % (rc, n))
    rc, log, n = run("framesbuiltin", "@cornell", ["-frames", "0", "1"])
    check(rc != 0 and n == 0 and "declares no cameras" in log,
          "-frames on a built-in scene is refused (exit %d, %d images)" % (rc, n))
    print("PASS" if not fails else "FAILED (%d)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
