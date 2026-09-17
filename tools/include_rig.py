"""The `include` rig: prove that splitting a scene across files changes nothing.

    python tools/include_rig.py [--device gpu] [--keep]

Writes a small scene five ways -- in one file (the reference); split with `include`; nested
through a subdirectory with a `../` path; included from inside a `prefer` branch; plus a cycle
and a missing file -- renders the renderable ones, and prints:

  * whether each split rendering is BYTE-IDENTICAL to the one-file reference (it must be:
    an include is a splice, and any pixel of difference means a name failed to cross the
    file boundary or a block landed in a different order);
  * that the cycle is refused naming the whole chain;
  * that the missing file is refused naming the INCLUDING file and line.

The `fur` in the main file grows `on` a sphere defined in the INCLUDED file, on purpose: that is
the cross-file reference the feature exists for (Alice's groom lives in its own file and is
grown on a scalp the main scene never sees).

Cited by FTSL.md section 1.5 and the 0.325.0 entries in design.md / TODO.md.
"""
import argparse
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

OUT = os.path.join("scraps", "_inc")
PNG = os.path.join("png", "include_rig")

COMMON = """camera "cam" { eye 0 0.4 1.6  look_at 0 0.1 0  up 0 1 0  fov_y 40  mode R  film { res 160 120 } }
light area { origin -0.5 1.2 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:d65  power 60 }
material "floor" { type diffuse reflect 0.5 }
quad { origin -2 0 -2  u 4 0 0  v 0 0 4  material floor }
"""
PART = """# the part: a material and a sphere the main scene refers to by name
material "red" { type diffuse reflect rgb 0.7 0.1 0.1 }
sphere "ball" { center 0 0.25 0  radius 0.25  material red }
"""
TAIL = """# grown ON the sphere defined in the included file -- the name must cross the file boundary
fur "fuzz" { on "ball"  material red  count 400  length 0.06  radius 0.002  seed 3 }
"""


def w(rel, text):
    p = os.path.join(OUT, rel)
    d = os.path.dirname(p)
    if not os.path.isdir(d):
        os.makedirs(d)
    io.open(p, "w", encoding="utf-8", newline="\n").write(text)


def write_scenes():
    for d in (OUT, PNG):
        if not os.path.isdir(d):
            os.makedirs(d)
    w("one.ftsl", "# ONE FILE: the reference\n" + COMMON + PART + TAIL)
    w("main.ftsl", "# SPLIT: the same scene, the part included\n" + COMMON + 'include "part.ftsl"\n' + TAIL)
    w("part.ftsl", PART)
    w("nested.ftsl", "# NESTED through a subdirectory, with a ../ path resolved beside the INCLUDING file\n"
      + COMMON + 'include "sub/mid.ftsl"\n' + TAIL)
    w("sub/mid.ftsl", 'include "../part.ftsl"\n')
    w("prefer.ftsl", COMMON + 'prefer {\n    include "part.ftsl"\n} else {\n'
      '    sphere "ball" { center 0 0.25 0 radius 0.25 material floor }\n}\n' + TAIL)
    w("cyc_a.ftsl", COMMON + 'include "cyc_b.ftsl"\n')
    w("cyc_b.ftsl", 'include "cyc_a.ftsl"\n')
    w("bad.ftsl", COMMON + 'include "no_such_file.ftsl"\n' + TAIL)     # the include is on line 5


def run(tag, device, spp):
    exe = os.path.join(ROOT, "ftrace.exe")
    out = os.path.join(PNG, tag + ".png")
    r = subprocess.run([exe, "-in", os.path.join(OUT, tag + ".ftsl"), "-mode", "R", "-device", device,
                        "-spp", str(spp), "-hdr", "-o", out, "-window-min", "-interval", "30"],
                       capture_output=True, text=True, check=False)
    return r.stdout + r.stderr, out[:-4] + ".pfm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    write_scenes()
    from grade_hdr import read_pfm
    import numpy as np

    ok = True
    _, ref = run("one", args.device, args.spp)
    a = read_pfm(ref)
    print("IDENTITY -- each split form must be byte-identical to the one-file reference:")
    for tag, label in (("main", "split with include"), ("nested", "nested via sub/ and ../"),
                       ("prefer", "include inside a prefer branch")):
        log, pfm = run(tag, args.device, args.spp)
        same = os.path.exists(pfm) and np.array_equal(a, read_pfm(pfm))
        ok = ok and same
        print("   %-32s %s" % (label, "IDENTICAL" if same else "DIFFERS  <-- FAIL"))
    print("\nREFUSALS -- and what they say:")
    log, _ = run("cyc_a", args.device, 4)
    cyc = [l for l in log.splitlines() if "include cycle" in l]
    good = bool(cyc) and "cyc_a" in cyc[0] and "cyc_b" in cyc[0]
    ok = ok and good
    print("   cycle:   %s" % (cyc[0].strip() if cyc else "NOT REFUSED  <-- FAIL"))
    log, _ = run("bad", args.device, 4)
    bad = [l for l in log.splitlines() if "cannot open" in l and "include" in l]
    good = bool(bad) and "bad.ftsl:5" in bad[0]
    ok = ok and good
    print("   missing: %s" % (bad[0].strip() if bad else "NOT REFUSED  <-- FAIL"))
    print("\n-> include rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep:
        for dp, _, fs in os.walk(OUT, topdown=False):
            for fn in fs:
                os.remove(os.path.join(dp, fn))
            os.rmdir(dp)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
