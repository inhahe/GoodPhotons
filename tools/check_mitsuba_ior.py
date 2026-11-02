"""Regression check: tools/mitsuba_to_ftsl.py turns every Mitsuba IOR into a working FTSL `ior`.

    D:/python314/python.exe -u tools/check_mitsuba_ior.py [--out <dir>] [--exe ftrace.exe]
                                                           [--converter <mitsuba_to_ftsl.py>]

Writes a Mitsuba scene with one dielectric per named IOR Mitsuba accepts, plus a numeric
`int_ior`, a dielectric with none, and a name Mitsuba does not know, converts it, and checks:

  - every dielectric comes out as `type dielectric  ior <expr>`;
  - each named IOR resolves, at 589 nm, to within 0.015 of Mitsuba's own value: a `glass:`
    name through the data/glass file it names, a number as written. (BK7, the widest gap, is
    0.012: Mitsuba's table quotes it at 1.5046, the catalogue's n_d is 1.5168);
  - `int_ior 1.33` stays `ior 1.33`, and an absent `int_ior` takes Mitsuba's default, bk7;
  - an unknown name falls back to that default and leaves a `# WARN:` naming it;
  - `ftrace -parseonly` loads the result with no unknown key and no unknown glass.

`--converter` runs the check against another copy of the converter, e.g. an older revision.
Exits 1 on any failure.
"""
import argparse
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mitsuba's table (include/mitsuba/render/ior.h), the reference the import must reproduce.
MITSUBA_IOR = {
    'vacuum': 1.0, 'helium': 1.000036, 'hydrogen': 1.000132, 'air': 1.000277,
    'carbon dioxide': 1.00045, 'water': 1.3330, 'acetone': 1.36, 'ethanol': 1.361,
    'carbon tetrachloride': 1.461, 'glycerol': 1.4729, 'benzene': 1.501,
    'silicone oil': 1.52045, 'bromine': 1.661, 'water ice': 1.31, 'fused quartz': 1.458,
    'pyrex': 1.470, 'acrylic glass': 1.49, 'polypropylene': 1.49, 'bk7': 1.5046,
    'sodium chloride': 1.544, 'amber': 1.55, 'pet': 1.5750, 'diamond': 2.419,
}
TOL = 0.015
UNKNOWN = 'unobtainium'


def glass_files():
    """Map every name data/glass answers to (stem or `# aliases:`, lowercased) to its path,
    the way speclib::index() reads the directory."""
    d = os.path.join(ROOT, 'data', 'glass')
    idx = {}
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        idx[os.path.splitext(fn)[0].lower()] = p
        with open(p, encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                if n > 12:
                    break
                line = line.rstrip('\n')
                if not line or line[0] != '#':
                    if n > 1:
                        break
                    continue
                i = line.find('aliases:')
                if i >= 0:
                    for a in line[i + 8:].split():
                        idx[a.lower()] = p
    return idx


def glass_index(path, nm):
    """Refractive index at `nm` nanometres of the .glass file at `path`."""
    form, k = None, {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith('#'):
                continue
            if t[0] == 'form':
                form = t[1]
            else:
                k[t[0]] = [float(x) for x in t[1:]]
    um2 = (nm * 1e-3) ** 2
    if form == 'sellmeier':
        return math.sqrt(1 + sum(b * um2 / (um2 - c) for b, c in zip(k['B'], k['C'])))
    if form == 'cauchy':
        return k['A'][0] + k['B'][0] / um2
    if form == 'constant':
        return k['n'][0]
    raise ValueError(f"{path}: unknown form {form!r}")


def resolve(expr, glasses):
    """Index at 589 nm of an `ior <expr>` material value, or None if it names nothing."""
    v = expr[len('ior '):].strip()
    if v.startswith('glass:'):
        p = glasses.get(v[len('glass:'):].lower())
        return glass_index(p, 589.0) if p else None
    try:
        return float(v)
    except ValueError:
        return None


def mid(name):
    return 'ior_' + name.replace(' ', '_')


def write_scene(path):
    spheres = []
    cases = [(mid(n), f'<string name="int_ior" value="{n}"/>') for n in MITSUBA_IOR]
    cases += [('ior_numeric', '<float name="int_ior" value="1.33"/>'),
              ('ior_absent', ''),
              (mid(UNKNOWN), f'<string name="int_ior" value="{UNKNOWN}"/>')]
    for i, (bid, prop) in enumerate(cases):
        spheres.append(
            f'  <shape type="sphere"><point name="center" x="{i * 0.3:g}" y="0" z="0"/>'
            f'<float name="radius" value="0.1"/>'
            f'<bsdf type="dielectric" id="{bid}">{prop}</bsdf></shape>')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('<scene version="3.0.0">\n'
                '  <sensor type="perspective"><float name="fov" value="40"/>\n'
                '    <transform name="to_world"><lookat origin="3, 1, 6" target="3, 0, 0" up="0, 1, 0"/></transform>\n'
                '    <film type="hdrfilm"><integer name="width" value="64"/><integer name="height" value="64"/></film>\n'
                '  </sensor>\n'
                '  <emitter type="constant"><rgb name="radiance" value="1, 1, 1"/></emitter>\n'
                + '\n'.join(spheres) + '\n</scene>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'scraps', 'check_mitsuba_ior'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--converter', default=os.path.join(ROOT, 'tools', 'mitsuba_to_ftsl.py'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xml = os.path.join(a.out, 'iors.xml')
    ftsl = os.path.join(a.out, 'iors.ftsl')
    write_scene(xml)

    fails = []
    r = subprocess.run([sys.executable, a.converter, xml, ftsl],
                       capture_output=True, text=True, encoding='utf-8')
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        print('FAIL: the converter exited', r.returncode)
        return 1
    with open(ftsl, encoding='utf-8') as f:
        out = f.read()
    mats = dict(re.findall(r'^material "([^"]+)" \{ type dielectric\s+(.*?)\s*\}$', out, re.M))
    glasses = glass_files()
    default = mats.get('ior_absent')

    for name, want in MITSUBA_IOR.items():
        expr = mats.get(mid(name))
        if expr is None:
            fails.append(f"{name}: no dielectric material in the output")
            continue
        if not expr.startswith('ior '):
            fails.append(f"{name}: material value {expr!r} is not an `ior` expression")
            continue
        n = resolve(expr, glasses)
        if n is None:
            fails.append(f"{name}: {expr!r} names no glass and no number")
        elif abs(n - want) > TOL:
            fails.append(f"{name}: {expr!r} is n {n:.4f} at 589 nm, Mitsuba's is {want}")
        else:
            print(f"  ok  {name:21s} {expr:22s} n {n:.4f}  (Mitsuba {want})")

    if mats.get('ior_numeric') != 'ior 1.33':
        fails.append(f"numeric int_ior 1.33 came out as {mats.get('ior_numeric')!r}")
    if default != 'ior glass:BK7':
        fails.append(f"absent int_ior came out as {default!r}, not Mitsuba's bk7 default")
    if mats.get(mid(UNKNOWN)) != default:
        fails.append(f"unknown name came out as {mats.get(mid(UNKNOWN))!r}, not the default")
    if f"# WARN: unknown named IOR '{UNKNOWN}'" not in out:
        fails.append("unknown name left no # WARN naming it")

    r = subprocess.run([a.exe, ftsl, '-parseonly'], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    log = r.stdout + r.stderr
    if r.returncode != 0:
        fails.append(f"ftrace -parseonly exited {r.returncode}: {log.strip()}")
    for bad in ('unknown key', 'unknown glass'):
        for line in log.splitlines():
            if bad in line:
                fails.append(f"ftrace: {line.strip()}")

    for f_ in fails:
        print('FAIL:', f_)
    print(f"{'FAIL' if fails else 'PASS'}: {len(MITSUBA_IOR)} named IORs, {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
