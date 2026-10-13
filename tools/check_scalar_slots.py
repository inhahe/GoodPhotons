"""Regression check: a slot that takes one number refuses a word that is not one.

    D:/python314/python.exe -u tools/check_scalar_slots.py [--out <dir>] [--exe ftrace.exe]

The loader read a scalar slot with strtod, so a stray word became 0 -- or its leading digits,
`1.5x` as 1.5 -- and 0 is a quiet, plausible value in most slots: `opacity half.spd` made the fur
vanish without a word. Since 0.377.1 such a slot takes a number, or a record / material property
it can read (`g.roughness`), and refuses anything else at load. Loads a few scenes with
-parseonly: each refused case must fail naming its slot, each accepted case must load. Exits 1 on
any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
BASE = [
    'material "g" { type glossy  reflect 0.5  roughness 0.3 }',
    'quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material g }',
    'light area { origin -0.2 1 -0.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0 }',
    'camera "c" { eye 0 1 -2  look_at 0 0 0  up 0 1 0  fov_y 40 }',
]
REFUSED = [   # (slot, material line)
    ("opacity", 'material "fur" { type hair  opacity half.spd }'),
    ("eta", 'material "fur" { type hair  eta 1.5x }'),
    ("beta_m", 'material "fur" { type hair  beta_m wide }'),
]
ACCEPTED = [
    'material "fur" { type hair  opacity 0.4  eta 1.5  beta_m 0.3 }',
    'material "m" { type glossy  reflect 0.4  roughness g.roughness }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_scalar_slots"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)

    def load(line):
        path = os.path.join(d, "s.ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(BASE + [line]) + NL)
        p = subprocess.run([a.exe, "-in", path, "-parseonly"], cwd=d, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout + p.stderr

    ok = True
    for slot, line in REFUSED:
        rc, log = load(line)
        good = rc != 0 and ("'%s' takes a number" % slot) in log
        ok &= good
        print("%s: refused `%s` (exit %d)" % ("ok" if good else "FAIL", line, rc))
    for line in ACCEPTED:
        rc, log = load(line)
        good = rc == 0 and "[parseonly] ok" in log
        ok &= good
        print("%s: accepted `%s` (exit %d)" % ("ok" if good else "FAIL", line, rc))
        if not good:
            print(log[-800:])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
