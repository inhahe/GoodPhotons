"""The guided-fur rig: prove `fur ... guides` blends authored guides the way the rule says.

    python tools/guide_rig.py [--keep]

Roots are placed DETERMINISTICALLY by growing on a 2-micron quad, so every root sits at one
known point to within 1e-6 and the emitted strand can be compared to a prediction through
`ftrace -dumpcurves` (with `basis linear  segments 1` the polyline is the control polygon).
`jitter 0  clump 0  length_jitter 0  curl 0` so nothing random is on top.

  1. REPRODUCE  -- a strand rooted at a guide's own root, `guide_blend 1`, must BE that guide,
                   point for point (offsets from the root, laid down at the root).
  2. MIDPOINT   -- a root midway between two parallel guides with `guide_blend 2` is at equal
                   distance from both, so the weights are equal and the strand is their average.
  3. NEAREST    -- `guide_blend 1` at a root nearer guide B reproduces B, not the average.
  4. UNGUIDED   -- the same fur block WITHOUT `guides` still builds (the closed-form path is
                   untouched by the feature).

Cited by FTSL.md section 8.7 and the 0.327.0 entries in design.md / TODO.md.
"""
import argparse
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = os.path.join("scraps", "_guide")
EXE = os.path.join(ROOT, "ftrace.exe")

HEAD = """camera "cam" { eye 0.5 1 4  look_at 0.5 1 0  up 0 1 0  fov_y 40  mode R  film { res 64 48 } }
light area { origin -1 3 -1  u 2 0 0  v 0 0 2  normal 0 -1 0  spd preset:d65  power 60 }
material "m" { type diffuse reflect 0.6 }
curve "gA" { basis linear  segments 1  point 0 0 0   point 0 1 0.2   point 0 2 0.5 }
curve "gB" { basis linear  segments 1  point 1 0 0   point 1 1 -0.2  point 1 2 -0.5 }
"""
EPS = 1e-6


def quad(cx, cy, cz):
    # a 2*EPS square in the XZ plane centred on the root, wound so u x v = +y: the UNGUIDED
    # check grows along the normal, and (2e,0,0) x (0,0,2e) points DOWN -- so v comes first.
    return 'quad "pad" { origin %.9g %.9g %.9g  u 0 0 %.9g  v %.9g 0 0  material m }\n' % (cx - EPS, cy, cz - EPS, 2 * EPS, 2 * EPS)


def fur(extra):
    return ('fur "f" { on "pad"  material m  count 4  points 3  basis linear  segments 1\n'
            '    jitter 0  clump 0  length_jitter 0  curl 0  droop 0  %s }\n' % extra)


def dump(name, body):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    p = os.path.join(OUT, name + ".ftsl")
    io.open(p, "w", encoding="utf-8", newline="\n").write(HEAD + body)
    txt = p[:-5] + ".txt"
    r = subprocess.run([EXE, "-in", p, "-dumpcurves", txt], capture_output=True, text=True, check=False)
    if not os.path.exists(txt):
        raise RuntimeError("dump failed for %s:\n%s" % (p, r.stdout + r.stderr))
    strands, cur = [], None
    for line in io.open(txt, encoding="utf-8"):
        if line.startswith("#"):
            continue
        if line.startswith("strand"):
            cur = []; strands.append(cur); continue
        cur.append(tuple(float(x) for x in line.split()))
    return strands, r.stdout + r.stderr


def near(a, b, tol=1e-5):
    return len(a) == len(b) and all(abs(x - y) <= tol for p, q in zip(a, b) for x, y in zip(p[:3], q[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-10s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    gA = [(0, 0, 0), (0, 1, 0.2), (0, 2, 0.5)]
    gB = [(1, 0, 0), (1, 1, -0.2), (1, 2, -0.5)]
    mid = [tuple((x + y) / 2 for x, y in zip(p, q)) for p, q in zip(gA, gB)]

    s, log = dump("repro", quad(0, 0, 0) + fur('guides "gA" "gB"  guide_blend 1'))
    fur_strands = [st for st in s if len(st) == 3][-4:]           # the 4 fur hairs (the 2 guides are definitions)
    check("reproduce", len(fur_strands) == 4 and all(near(st, gA) for st in fur_strands),
          "strand 0 = %s" % (fur_strands[0] if fur_strands else "?"))

    s, _ = dump("mid", quad(0.5, 0, 0) + fur('guides "gA" "gB"  guide_blend 2'))
    fs = [st for st in s if len(st) == 3][-4:]
    check("midpoint", len(fs) == 4 and all(near(st, mid) for st in fs), "strand 0 = %s" % (fs[0] if fs else "?"))

    s, _ = dump("nearest", quad(0.8, 0, 0) + fur('guides "gA" "gB"  guide_blend 1'))
    fs = [st for st in s if len(st) == 3][-4:]
    gB_at = [(0.8 + (q[0] - 1.0), q[1], q[2]) for q in gB]         # B's offsets laid down at the root
    check("nearest", len(fs) == 4 and all(near(st, gB_at) for st in fs))

    s, _ = dump("unguided", quad(0.5, 0, 0) + fur('length 1.0'))
    fs = [st for st in s if len(st) == 3][-4:]
    check("unguided", len(fs) == 4 and all(abs(st[-1][1] - 1.0) < 1e-5 for st in fs),
          "closed-form strand still grows straight up 1.0")

    print("\n-> guide rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep:
        for fn in os.listdir(OUT):
            os.remove(os.path.join(OUT, fn))
        os.rmdir(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
