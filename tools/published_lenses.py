"""Audit the published lens prescriptions behind `preset dgauss | telephoto | wide | fisheye`.

    python tools/published_lenses.py [--fetch]

src/lens.h keeps pbrt's lens files as tables of rows (kLensDGauss, ...), and `-checklens` (12)
locks each one by its focal length and its paraxial and traced f-numbers (the `Native`
constants in src/main.cpp), so that an edited row fails the build's check. This script is
where those constants come from, and what makes them auditable rather than magic numbers:

  * it reads the rows from src/lens.h and traces them itself -- in Python, sharing no code
    with ftrace -- for the focal length (a near-axis parallel ray), the paraxial f-number (the
    stop's pupil magnification) and the traced one (the widest axial pencil every aperture
    passes, by bisection), and compares them with main.cpp's constants;
  * with --fetch it downloads pbrt's own files (github.com/mmp/pbrt-v4-scenes, lenses/) and
    compares every number of every row with lens.h's.

After a deliberate change to a table, run it and paste the printed constants into main.cpp.
Exits 0 when everything agrees, 1 otherwise.
"""
import argparse
import math
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = {'dgauss': ('kLensDGauss', 'dgauss.50mm'), 'telephoto': ('kLensTelephoto', 'telephoto.250mm'),
          'wide': ('kLensWide', 'wide.22mm'), 'fisheye': ('kLensFisheye', 'fisheye.10mm')}
PBRT = 'https://raw.githubusercontent.com/mmp/pbrt-v4-scenes/master/lenses/{}.dat'
LAMBDA_D = 587.6   # the rows give one index per glass; the wavelength only names the design's


def read_tables():
    """name -> rows [(radius, thickness, ior, aperture diameter)] as lens.h has them."""
    src = open(os.path.join(ROOT, 'src', 'lens.h'), encoding='utf-8').read()
    out = {}
    for name, (table, _) in TABLES.items():
        m = re.search(r'inline const double ' + table + r'\[(\d+)\]\[4\] = \{(.*?)\}\};', src, re.S)
        if not m:
            sys.exit(f'{table} not found in src/lens.h')
        nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', m.group(2))]
        rows = [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]
        if len(rows) != int(m.group(1)) or len(nums) % 4:
            sys.exit(f'{table}: {len(nums)} numbers for {m.group(1)} rows')
        out[name] = rows
    return out


def read_locks():
    """name -> (f, nPar, nTr) from main.cpp's `Native{...}` constants."""
    src = open(os.path.join(ROOT, 'src', 'main.cpp'), encoding='utf-8').read()
    return {m.group(1): tuple(float(m.group(k)) for k in (2, 3, 4)) for m in re.finditer(
        r'Native\{"(\w+)",\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\}', src)}


class Lens:
    """The rows in ftrace's frame: +z toward the scene, the front vertex at z = T, a surface's
    centre of curvature at its vertex + radius (the published radius negated), apertures as
    semi-diameters, index 0 read as air, the radius-0 row the stop."""

    def __init__(self, rows):
        self.r = [0.0 if r == 0 else -r for r, _, _, _ in rows]
        self.t = [t for _, t, _, _ in rows]
        self.n = [1.0 if n == 0 else n for _, _, n, _ in rows]   # index on each surface's sensor side
        self.ap = [d / 2 for _, _, _, d in rows]
        self.stop = next(j for j, (r, _, _, _) in enumerate(rows) if r == 0)
        self.T = sum(self.t)
        self.z = [self.T]
        for t in self.t[:-1]:
            self.z.append(self.z[-1] - t)

    def hit(self, j, o, d, clip):
        """Where the ray (o, d) meets surface j, and the normal against it; None if it misses
        or (clip) lands outside the aperture."""
        R = self.r[j]
        if R == 0.0:
            t = (self.z[j] - o[2]) / d[2]
            p = [o[i] + d[i] * t for i in range(3)]
            nrm = [0.0, 0.0, -1.0 if d[2] > 0 else 1.0]
        else:
            c = [0.0, 0.0, self.z[j] + R]
            op = [o[i] - c[i] for i in range(3)]
            b = sum(op[i] * d[i] for i in range(3))
            disc = b * b - (sum(x * x for x in op) - R * R)
            if disc < 0:
                return None
            sq = math.sqrt(disc)
            t0, t1 = -b - sq, -b + sq
            closer = (d[2] > 0) != (R < 0)
            t = min(t0, t1) if closer else max(t0, t1)
            if t < 1e-9:
                t = max(t0, t1) if closer else min(t0, t1)
            if t < 1e-9:
                return None
            p = [o[i] + d[i] * t for i in range(3)]
            nrm = [p[i] - c[i] for i in range(3)]
            ln = math.sqrt(sum(x * x for x in nrm))
            nrm = [x / ln for x in nrm]
            if sum(nrm[i] * d[i] for i in range(3)) > 0:
                nrm = [-x for x in nrm]
        if clip and p[0] * p[0] + p[1] * p[1] > self.ap[j] * self.ap[j]:
            return None
        return p, nrm

    def axial(self, h, clip=True):
        """Trace the ray parallel to the axis at height h from the scene: the heights it meets
        each surface at and its final direction, or None if it is clipped or reflects."""
        o, d, heights = [h, 0.0, self.T + 1.0], [0.0, 0.0, -1.0], []
        for j in range(len(self.r)):
            res = self.hit(j, o, d, clip)
            if res is None:
                return None
            o, nrm = res
            heights.append(abs(o[0]))
            if self.r[j] != 0.0:
                eta = (1.0 if j == 0 else self.n[j - 1]) / self.n[j]
                cosi = -sum(d[i] * nrm[i] for i in range(3))
                k = 1 - eta * eta * (1 - cosi * cosi)
                if k < 0:
                    return None
                d = [d[i] * eta + nrm[i] * (eta * cosi - math.sqrt(k)) for i in range(3)]
                ln = math.sqrt(sum(x * x for x in d))
                d = [x / ln for x in d]
        return heights, d

    def focal(self):
        h = 1e-5 * max(1.0, self.T)
        _, d = self.axial(h, clip=False)
        return abs(h / (-d[0] / d[2]))

    def paraxial_fnumber(self):
        h0 = 1e-3 * max(1e-3, self.ap[0])
        heights, _ = self.axial(h0, clip=False)
        return self.focal() / (2 * self.ap[self.stop] * h0 / heights[self.stop])

    def traced_fnumber(self):
        lo, hi = 0.0, self.ap[0]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.axial(mid) is not None:
                lo = mid
            else:
                hi = mid
        return self.focal() / (2 * lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help="compare lens.h's rows with pbrt's files too")
    a = ap.parse_args()
    tables, locks = read_tables(), read_locks()
    ok = True
    print('the Native constants these rows give (main.cpp, checkLensPresetsFNumber):')
    for name, rows in tables.items():
        L = Lens(rows)
        got = (L.focal(), L.paraxial_fnumber(), L.traced_fnumber())
        lock = locks.get(name)
        same = lock is not None and all(abs(g / w - 1) < 1e-5 for g, w in zip(got, lock))
        ok = ok and same
        print(f'  Native{{"{name}", {got[0]:.6f}, {got[1]:.6f}, {got[2]:.6f}}}   '
              + ('matches main.cpp' if same else f'DIFFERS from main.cpp {lock}'))
    if a.fetch:
        print("lens.h's rows against pbrt's files:")
        for name, (table, fname) in TABLES.items():
            text = urllib.request.urlopen(PBRT.format(fname), timeout=30).read().decode('utf-8')
            theirs = [tuple(float(x) for x in line.split('#')[0].split())
                      for line in text.splitlines() if line.split('#')[0].strip()]
            same = theirs == tables[name]
            ok = ok and same
            print(f'  {table} vs {fname}.dat: {"identical" if same else "DIFFERENT"}')
            if not same:
                for i, (o, t) in enumerate(zip(tables[name], theirs)):
                    if o != t:
                        print(f'     row {i}: lens.h {o}, pbrt {t}')
                if len(theirs) != len(tables[name]):
                    print(f'     {len(tables[name])} rows in lens.h, {len(theirs)} in pbrt')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
