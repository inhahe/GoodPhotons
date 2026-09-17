"""The groom model's round-trip rig: a scene rewritten through the tool's curve model must
produce the SAME strands, exactly.

    python tools/groom_rig.py [--keep] [--alice]

  1. REWRITE   -- each rig scene (the curve rig's shapes: a blended pair, a level-3 product, a
                  closed ring, a by-name reference, a group under a transform, a spline knob, a
                  point with its own radius, a group of mixed children) is rewritten with
                  `ftrace -groom-rewrite`, and `-dumpcurves` of the original and the rewrite are
                  byte-identical.
  2. IDEMPOTENT-- rewriting the rewrite gives the same text.
  3. ALICE     -- (--alice) scenes/alice_guides.ftsl rewritten and included in place of the
                  original under scenes/alice_hair.ftsl: the 64 guides and the 13 280 fur strands
                  dump byte-identically (loads the doll twice, ~30 s).

Cited by REFERENCE.md ("The groom tool") and design.md.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT = os.path.join("scraps", "_groom")
EXE = os.path.join(ROOT, "ftrace.exe")

HEAD = """camera "cam" { eye 0.5 1 4  look_at 0.5 1 0  up 0 1 0  fov_y 40  mode R  film { res 96 72 } }
light area { origin -1 3 -1  u 2 0 0  v 0 0 2  normal 0 -1 0  spd preset:d65  power 60 }
material "m" { type diffuse reflect 0.6 }
"""
G1 = 'curve { point 0 0 0   point 0 1 0   point 0 2 0 }'
G2 = 'curve { point 1 0 0   point 1 1 0   point 1 2 0 }'

SCENES = {
    "mid":    'curve "pair" { material m  basis linear  segments 1  count 3\n    %s\n    %s\n}\n' % (G1, G2),
    "lvl3":   'curve "rows" { material m  basis linear  segments 1  count 2\n'
              '    curve { count 3\n        %s\n        %s\n    }\n'
              '    curve { count 3\n        curve { point 0 0 2  point 0 1 2  point 0 2 2 }\n        curve { point 1 0 2  point 1 1 2  point 1 2 2 }\n    }\n}\n' % (G1, G2),
    "closed": 'curve "loop" { material m  basis linear  segments 1  count 4\n    closed\n    %s\n    %s\n}\n' % (G1, G2),
    "byname": 'curve "g1" { basis linear  segments 1  point 0 0 0  point 0 1 0  point 0 2 0 }\n'
              'curve "g2" { basis linear  segments 1  point 1 0 0  point 1 1 0  point 1 2 0 }\n'
              'curve "pair" { material m  basis linear  segments 1  count 3\n    curve "g1"\n    curve "g2"\n}\n',
    "grpref": 'group "g" { translate 9.15 1.28 3.55  scale 3\n'
              '    curve "d" { basis linear  segments 1  point 0 0 0  point 0 0.1 0 }\n'
              '    curve "ring" { curve "d" }\n'
              '    curve "hair" { material m  basis linear  segments 1  curve "ring" }\n'
              '}\n',
    "spline": 'curve "s" { material m  spline centripetal  point 0 0 0  point 0 1 0  point 1 1.2 0  point 1 3 0 }\n',
    "radius": 'curve "r" { material m  basis linear  segments 1  radius 0.002  point 0 0 0 r=0.004  point 0 1 0  point 0 2 0 r=0.0005 }\n',
    "group":  'curve "grp" { material m  basis linear  segments 1\n'
              '    curve { count 3\n        %s\n        %s\n    }\n'
              '    curve { point 3 0 0  point 3 1 0  point 3 2 0 }\n}\n' % (G1, G2),
    "density": 'curve "dens" { material m  basis linear  segments 1  density 4\n    density_at 0 1\n    density_at 1 3\n    %s\n    %s\n}\n' % (G1, G2),
}


def w(path, body):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    io.open(path, "w", encoding="utf-8", newline="\n").write(body)
    return path


def run(args):
    return subprocess.run([EXE] + args, capture_output=True, text=True, check=False)


def dump(scene):
    txt = scene[:-5] + ".dump.txt"
    if os.path.exists(txt):
        os.remove(txt)
    r = run(["-in", scene, "-dumpcurves", txt])
    if not os.path.exists(txt):
        raise RuntimeError("dump failed for %s:\n%s" % (scene, r.stdout + r.stderr))
    return io.open(txt, encoding="utf-8").read()


def rewrite(src, dst):
    r = run(["-groom-rewrite", src, dst])
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("rewrite failed for %s:\n%s" % (src, r.stdout + r.stderr))
    return io.open(dst, encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--alice", action="store_true", help="also round-trip scenes/alice_guides.ftsl under the doll (~30 s)")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-10s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    for name, body in SCENES.items():
        src = w(os.path.join(OUT, name + ".ftsl"), HEAD + body)
        dst = os.path.join(OUT, name + "_rt.ftsl")
        text1 = rewrite(src, dst)
        d0 = dump(src)
        d1 = dump(dst)
        same = (d0 == d1) and ("strand" in d0)
        check(name, same, "%d strand(s); dump %s" % (d0.count("strand"), "identical" if same else "DIFFERS"))
        text2 = rewrite(dst, os.path.join(OUT, name + "_rt2.ftsl"))
        check(name + "/idem", text1 == text2, "rewrite of the rewrite is the same text")

    if args.alice:
        # the guides rewritten, and alice_hair.ftsl copied beside them so its `include` finds the rewrite
        rewrite(os.path.join("scenes", "alice_guides.ftsl"), os.path.join(OUT, "alice_guides.ftsl"))
        shutil.copy(os.path.join("scenes", "alice_hair.ftsl"), os.path.join(OUT, "alice_hair.ftsl"))
        harness = HEAD + 'include "%s"\n'
        a = w(os.path.join(OUT, "alice_orig.ftsl"), harness % "../../scenes/alice_hair.ftsl")
        b = w(os.path.join(OUT, "alice_rt.ftsl"), harness % "alice_hair.ftsl")
        da, db = dump(a), dump(b)
        check("alice", da == db and da.count("strand") > 13000, "%d strand(s) dump %s" % (da.count("strand"), "identical" if da == db else "DIFFERS"))

    print("\n-> groom rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep and ok:
        shutil.rmtree(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
