"""Regression check: an unknown light preset says when the spectral library behind it is missing.

    python -u tools/check_preset_library.py [--exe ftrace.exe] [--out scraps/check_preset_library]

Named presets are engine data -- data/light's bundles (daylight, sun, led...), data/illuminant's
tables (the CIE F series...) -- found in data/ under the working directory or beside the executable
(speclib::categoryDir). A copy of the executable run where neither has them has none, and its
"unknown preset" must say which directory is missing, not read as a misspelling (ftsl.h,
resolvePreset). The copy runs from a scratch directory holding:

  - no data/ at all: `preset:daylight` names data/light and data/illuminant;
  - data/illuminant only: names data/light alone;
  - both (empty): a plain "unknown preset", no library named;
  - both, without data/glass: `glass:BK7` names data/glass, as every library lookup names its own;
  - and the repository's own binary, from the repository: a misspelt `preset:daylite` is a plain
    "unknown preset" too, and `preset:daylight` loads.

Exits 1 on any failure; the scratch directory (--out: one this check made, or a new one -- any other
is refused), the copy included, is deleted at the end.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
SCENE = """scene { units meters }
material "w" { type diffuse reflect 0.7 }%s
quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }
light area { origin -0.2 1.5 -0.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0  power 20  spd preset:%s }
camera "a" { eye 0 0.8 2  look_at 0 0 0  up 0 1 0  fov_y 45  film { res 16 12 } }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--out", default=os.path.join(ROOT, "scraps", "check_preset_library"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    # The scratch directory is emptied before and deleted after, so only one this check made -- marked
    # as its own -- is ever deleted; any other existing directory is refused, not emptied.
    mark = os.path.join(d, ".check_preset_library")
    if os.path.isdir(d) and os.listdir(d) and not os.path.exists(mark):
        print("--out %s exists and is not this check's scratch directory: name a new one" % d)
        return 1
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(mark, "w").close()
    copy = os.path.join(d, "ftrace.exe")
    shutil.copy2(exe, copy)
    fails = []

    def run(binary, cwd, preset, glass=None):
        scene = os.path.join(d, "s_%s_%s.ftsl" % (preset, glass))
        line = (NL + 'material "g" { type dielectric ior glass:%s }' % glass) if glass else ""
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(SCENE % (line, preset))
        p = subprocess.run([binary, "-in", scene, "-mode", "R", "-device", "cpu", "-spp", "1", "-no-meter",
                            "-window-min", "-o", os.path.join(d, "out.png")], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return p.returncode, p.stdout + p.stderr

    def check(name, ok, detail):
        print("%-62s %s  %s" % (name, "ok  " if ok else "FAIL", detail), flush=True)
        if not ok:
            fails.append(name)

    unknown = "unknown preset 'daylight'"
    data = os.path.join(d, "data")
    for have, want in (((), "data/light and data/illuminant are in neither"),
                       (("illuminant",), "data/light is in neither"),
                       (("light", "illuminant"), None)):
        shutil.rmtree(data, ignore_errors=True)
        for cat in have:
            os.makedirs(os.path.join(data, cat))
        rc, log = run(copy, d, "daylight")
        line = next((l for l in log.splitlines() if unknown in l), "")
        if want:
            ok = rc != 0 and want in line
        else:
            ok = rc != 0 and line != "" and "neither" not in line
        check("data/ holding %s: %s" % (" and ".join(have) or "nothing",
                                         "names the missing" if want else "a plain unknown preset"),
              ok, "rc %d: %s" % (rc, line.strip()[:110] or "no 'unknown preset' line"))
    # A library name other than a preset's -- a glass -- says so the same way (speclib::missingLibraryNote).
    shutil.rmtree(data, ignore_errors=True)
    os.makedirs(os.path.join(data, "light"))
    os.makedirs(os.path.join(data, "illuminant"))
    rc, log = run(copy, d, "daylight", glass="BK7")
    line = next((l for l in log.splitlines() if "unknown glass 'BK7'" in l), "")
    check("data/ without glass: an unknown glass names data/glass",
          rc != 0 and "data/glass is in neither" in line, "rc %d: %s" % (rc, line.strip()[:110]))
    rc, log = run(exe, ROOT, "daylite")
    line = next((l for l in log.splitlines() if "unknown preset 'daylite'" in l), "")
    check("the repository's binary: a misspelling is a plain unknown preset",
          rc != 0 and line != "" and "neither" not in line, "rc %d: %s" % (rc, line.strip()[:110]))
    rc, log = run(exe, ROOT, "daylight")
    check("the repository's binary: preset:daylight loads", rc == 0 and unknown not in log, "rc %d" % rc)
    shutil.rmtree(d, ignore_errors=True)
    print("PASS" if not fails else "FAILED: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
