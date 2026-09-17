"""The curves-of-curves rig: prove the recursive `curve` does what its rule says, exactly.

    python tools/curve_rig.py [--keep]

Every check reads the EMITTED control points back through `ftrace -dumpcurves` (with `basis
linear` and `segments 1` the tessellated polyline is the control polygon, so there is no
approximation in the oracle) and compares them to what the rule predicts:

  1. MIDPOINT  -- two straight parallel guides under a parent with `count 3`: the middle instance
                  must be their average point for point, and the two ends must BE the children.
                  (Uniform Catmull-Rom through two points at g = 0.5 is exactly the mean.)
  2. IDENTITY  -- a parent with no `count`/`density` emits its children bit-for-bit, and renders
                  byte-identical to the children written out as separate top-level curves.
  3. RESAMPLE  -- children with 3 and 5 points blend at a common 5, ends preserved exactly.
  4. PRODUCT   -- a level-3 node (`count 2` over two level-2 nodes each `count 3`) emits 6
                  strands whose first three equal its first child's own instances.
  5. BY NAME   -- a material-less named curve is a definition; referencing it by name gives the
                  same strands as writing it inline.
  6. CLOSED    -- `closed` with `count 4` over two children places instance 2 exactly on child 1
                  (i/N spacing, so g = 1.0 lands on the second root).
  7. SPLINE    -- `spline centripetal` on an ordinary strand changes its shape (the knob is wired)
                  and `spline uniform` is byte-identical to no `spline` at all.

Cited by FTSL.md section 8.6 and the 0.326.0 entries in design.md / TODO.md.
"""
import argparse
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

OUT = os.path.join("scraps", "_curve")
PNG = os.path.join("png", "curve_rig")
EXE = os.path.join(ROOT, "ftrace.exe")

HEAD = """camera "cam" { eye 0.5 1 4  look_at 0.5 1 0  up 0 1 0  fov_y 40  mode R  film { res 96 72 } }
light area { origin -1 3 -1  u 2 0 0  v 0 0 2  normal 0 -1 0  spd preset:d65  power 60 }
material "m" { type diffuse reflect 0.6 }
"""
G1 = 'curve { point 0 0 0   point 0 1 0   point 0 2 0 }'
G2 = 'curve { point 1 0 0   point 1 1 0   point 1 2 0 }'


def w(name, body):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    p = os.path.join(OUT, name + ".ftsl")
    io.open(p, "w", encoding="utf-8", newline="\n").write(HEAD + body)
    return p


def dump(scene):
    txt = os.path.join(OUT, os.path.basename(scene)[:-5] + ".txt")
    r = subprocess.run([EXE, "-in", scene, "-dumpcurves", txt], capture_output=True, text=True, check=False)
    if not os.path.exists(txt):
        raise RuntimeError("dump failed for %s:\n%s" % (scene, r.stdout + r.stderr))
    strands, cur = [], None
    for line in io.open(txt, encoding="utf-8"):
        if line.startswith("#"):
            continue
        if line.startswith("strand"):
            cur = []; strands.append(cur); continue
        cur.append(tuple(float(x) for x in line.split()))
    return strands


def render(scene, tag):
    if not os.path.isdir(PNG):
        os.makedirs(PNG)
    out = os.path.join(PNG, tag + ".png")
    subprocess.run([EXE, "-in", scene, "-mode", "R", "-device", "gpu", "-spp", "8", "-hdr", "-o", out,
                    "-window-min", "-interval", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out[:-4] + ".pfm"


def close(a, b, tol=1e-9):
    return len(a) == len(b) and all(len(p) == len(q) and all(abs(x - y) <= tol for x, y in zip(p, q)) for p, q in zip(a, b))


def mean(a, b):
    return [tuple((x + y) / 2 for x, y in zip(p, q)) for p, q in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-10s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    # 1. midpoint
    s = dump(w("mid", 'curve "pair" { material m  basis linear  segments 1  count 3\n    %s\n    %s\n}\n' % (G1, G2)))
    g1 = [(0, 0, 0, 0.001), (0, 1, 0, 0.001), (0, 2, 0, 0.001)]
    g2 = [(1, 0, 0, 0.001), (1, 1, 0, 0.001), (1, 2, 0, 0.001)]
    check("midpoint", len(s) == 3 and close(s[0], g1) and close(s[2], g2) and close(s[1], mean(g1, g2)),
          "%d strands; middle = %s" % (len(s), s[1] if len(s) == 3 else "?"))

    # 2. identity, by dump and by render
    s = dump(w("ident", 'curve "pair" { material m  basis linear  segments 1\n    %s\n    %s\n}\n' % (G1, G2)))
    check("identity", len(s) == 2 and close(s[0], g1) and close(s[1], g2))
    from grade_hdr import read_pfm
    import numpy as np
    a = read_pfm(render(os.path.join(OUT, "ident.ftsl"), "ident"))
    b = read_pfm(render(w("ident_flat", 'curve "a" { material m  basis linear  segments 1  %s }\n'
                                          'curve "b" { material m  basis linear  segments 1  %s }\n'
                                          % (G1[6:], G2[6:])), "ident_flat"))
    check("ident_img", np.array_equal(a, b), "rendered byte-identical to the children written out")

    # 3. resample: a 3-point child and a 5-point child (same straight line) -> 5 points, ends exact
    s = dump(w("resamp", 'curve "pair" { material m  basis linear  segments 1  count 3\n'
                         '    curve { point 0 0 0  point 0 1 0  point 0 2 0 }\n'
                         '    curve { point 1 0 0  point 1 0.5 0  point 1 1 0  point 1 1.5 0  point 1 2 0 }\n}\n'))
    check("resample", len(s) == 3 and all(len(st) == 5 for st in s)
          and close([s[1][0]], [(0.5, 0, 0, 0.001)]) and close([s[1][-1]], [(0.5, 2, 0, 0.001)]),
          "points per strand: %s" % [len(st) for st in s])

    # 4. product: level 3
    lvl2a = 'curve { count 3\n        %s\n        %s\n    }' % (G1, G2)
    lvl2b = 'curve { count 3\n        curve { point 0 0 2  point 0 1 2  point 0 2 2 }\n        curve { point 1 0 2  point 1 1 2  point 1 2 2 }\n    }'
    s3 = dump(w("lvl3", 'curve "rows" { material m  basis linear  segments 1  count 2\n    %s\n    %s\n}\n' % (lvl2a, lvl2b)))
    s2 = dump(w("lvl2", 'curve "row" { material m  basis linear  segments 1  count 3\n    %s\n    %s\n}\n' % (G1, G2)))
    check("product", len(s3) == 6 and len(s2) == 3 and all(close(s3[i], s2[i]) for i in range(3)),
          "%d strands; first three == the first child's instances" % len(s3))

    # 5. by name
    s = dump(w("byname", 'curve "g1" { basis linear  segments 1  point 0 0 0  point 0 1 0  point 0 2 0 }\n'
                         'curve "g2" { basis linear  segments 1  point 1 0 0  point 1 1 0  point 1 2 0 }\n'
                         'curve "pair" { material m  basis linear  segments 1  count 3\n    curve "g1"\n    curve "g2"\n}\n'))
    check("byname", len(s) == 3 and close(s[1], mean(g1, g2)))

    # 6. closed
    s = dump(w("closed", 'curve "loop" { material m  basis linear  segments 1  closed  count 4\n    %s\n    %s\n}\n' % (G1, G2)))
    check("closed", len(s) == 4 and close(s[0], g1) and close(s[2], g2), "instance 2 sits on child 1")

    # 7. spline knob on a single strand
    base = 'curve "s" { material m  point 0 0 0  point 0 1 0  point 1 1.2 0  point 1 3 0 }\n'
    su = dump(w("sp_uniform", base.replace('{ material m', '{ material m  spline uniform')))
    s0 = dump(w("sp_none", base))
    sc = dump(w("sp_centri", base.replace('{ material m', '{ material m  spline centripetal')))
    check("spline", close(su[0], s0[0]) and not close(sc[0], s0[0]),
          "uniform == default; centripetal differs")

    print("\n-> curve rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep:
        for fn in os.listdir(OUT):
            os.remove(os.path.join(OUT, fn))
        os.rmdir(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
