"""Regression check: tools/mitsuba_to_ftsl.py converts Mitsuba scenes into FTSL that loads right.

    D:/python314/python.exe -u tools/check_mitsuba_import.py [--out <dir>] [--exe ftrace.exe]
                                                              [--converter <mitsuba_to_ftsl.py>]

Writes small Mitsuba scenes, converts each, and checks:

  IOR (a version 3 scene with one dielectric per named IOR Mitsuba accepts):
  - every dielectric comes out as `type dielectric  ior <expr>`;
  - each named IOR resolves, at 589 nm, to within 0.015 of Mitsuba's own value: a `glass:`
    name through the data/glass file it names, a number as written. (BK7, the widest gap, is
    0.012: Mitsuba's table quotes it at 1.5046, the catalogue's n_d is 1.5168);
  - `int_ior 1.33` stays `ior 1.33`, and an absent `int_ior` takes Mitsuba's default, bk7;
  - an unknown name falls back to that default and leaves a `# WARN:` naming it;
  - an `ext_ior` other than air leaves a `# WARN:`; `ext_ior air` does not.

  Version 0.6 (camelCase property names, which Mitsuba's parser renames on load):
  - `intIOR` (named or numeric) is read as `int_ior`, and `toWorld` places the sensor;
  - a diffuse BSDF's `diffuseReflectance` is read as its reflectance;
  - a `<default>` parameter keeps its camelCase name, so `$fovDeg` still substitutes.

  BSDFs and emitters:
  - roughdielectric's alpha becomes the dielectric's `roughness`, with a `# WARN:`;
  - thindielectric becomes a `halfmirror` reflecting a sheet's 2R/(1+R) at normal incidence;
  - a dielectric's specular_transmittance, a conductor's eta/k, and a conductor material FTSL
    has no `metal:` for are each flagged; a known one (Au) is not;
  - a cube area emitter becomes six `light area` faces, each facing out of the cube;
  - a .ply mesh area emitter is loaded as a mesh under an emissive copy of its material, and
    lights a scene alone, without the converter's fallback light;
  - a .serialized mesh is skipped with a `# WARN:` rather than written for ftrace to fail on.

  Every scene: a perspective sensor becomes a `mode R` camera, and `ftrace -parseonly`, run
  from the repo root so it reads the same data/ the check does, loads the result with no
  unknown key and no unknown glass.

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

SENSOR = ('  <sensor type="perspective"><float name="fov" value="40"/>\n'
          '    <transform name="to_world"><lookat origin="3, 1, 6" target="3, 0, 0" up="0, 1, 0"/></transform>\n'
          '    <film type="hdrfilm"><integer name="width" value="64"/><integer name="height" value="64"/></film>\n'
          '  </sensor>\n'
          '  <emitter type="constant"><rgb name="radiance" value="1, 1, 1"/></emitter>\n')


def glass_files():
    """Map every name data/glass answers to (stem or `# aliases:`, lowercased) to its path,
    the way speclib::index() reads the directory."""
    d = os.path.join(ROOT, 'data', 'glass')
    idx = {}
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            continue
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
    """Refractive index at `nm` nanometres of the .glass file at `path`, parsed as
    speclib::loadGlass() does: `#` starts a comment anywhere, a repeated key appends."""
    form, k = None, {'A': [], 'B': [], 'C': [], 'n': []}
    with open(path, encoding='utf-8') as f:
        for line in f:
            t = line.split('#', 1)[0].split()
            if not t:
                continue
            if t[0] == 'form':
                form = t[1] if len(t) > 1 else None
            elif t[0] in k:
                k[t[0]] += [float(x) for x in t[1:]]
    um2 = (nm * 1e-3) ** 2
    if form == 'sellmeier' and len(k['B']) >= 3 and len(k['C']) >= 3:
        return math.sqrt(1 + sum(b * um2 / (um2 - c) for b, c in zip(k['B'][:3], k['C'][:3])))
    if form == 'cauchy' and k['A'] and k['B']:
        return k['A'][0] + k['B'][0] / um2
    if form == 'constant' and k['n']:
        return k['n'][0]
    return None


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


def sphere(i, bsdf):
    return (f'  <shape type="sphere"><point name="center" x="{i * 0.3:g}" y="0" z="0"/>'
            f'<float name="radius" value="0.1"/>{bsdf}</shape>')


def scene_v3():
    cases = [(mid(n), f'<string name="int_ior" value="{n}"/>') for n in MITSUBA_IOR]
    cases += [('ior_numeric', '<float name="int_ior" value="1.33"/>'),
              ('ior_absent', ''),
              (mid(UNKNOWN), f'<string name="int_ior" value="{UNKNOWN}"/>'),
              ('ext_water', '<string name="int_ior" value="air"/><string name="ext_ior" value="water"/>'),
              ('ext_air', '<string name="int_ior" value="water"/><string name="ext_ior" value="air"/>')]
    body = '\n'.join(sphere(i, f'<bsdf type="dielectric" id="{bid}">{p}</bsdf>')
                     for i, (bid, p) in enumerate(cases))
    return '<scene version="3.0.0">\n' + SENSOR + body + '\n</scene>\n'


def scene_v06():
    return ('<scene version="0.6.0">\n'
            '  <default name="fovDeg" value="40"/>\n'
            '  <sensor type="perspective"><float name="fov" value="$fovDeg"/>\n'
            '    <transform name="toWorld"><lookat origin="3, 1, 6" target="3, 0, 0" up="0, 1, 0"/></transform>\n'
            '    <film type="hdrfilm"><integer name="width" value="64"/><integer name="height" value="64"/></film>\n'
            '  </sensor>\n'
            '  <emitter type="constant"><rgb name="radiance" value="1, 1, 1"/></emitter>\n'
            + sphere(0, '<bsdf type="dielectric" id="v06_named"><string name="intIOR" value="water"/></bsdf>') + '\n'
            + sphere(1, '<bsdf type="dielectric" id="v06_numeric"><float name="intIOR" value="2.4"/></bsdf>') + '\n'
            + sphere(2, '<bsdf type="diffuse" id="v06_diffuse"><rgb name="diffuseReflectance" value="0.2, 0.4, 0.6"/></bsdf>')
            + '\n</scene>\n')


PLY = ('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n'
       'property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
       '0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')


def scene_bsdfs(ply):
    parts = [
        sphere(0, '<bsdf type="roughdielectric" id="rough"><float name="alpha" value="0.3"/></bsdf>'),
        sphere(1, '<bsdf type="thindielectric" id="thin"/>'),
        sphere(2, '<bsdf type="dielectric" id="tinted"><rgb name="specular_transmittance" value="0.6, 0.8, 1"/></bsdf>'),
        sphere(3, '<bsdf type="conductor" id="tungsten"><string name="material" value="W"/></bsdf>'),
        sphere(4, '<bsdf type="conductor" id="gold"><string name="material" value="Au"/></bsdf>'),
        sphere(5, '<bsdf type="conductor" id="etak"><rgb name="eta" value="0.2, 0.9, 1.1"/>'
                  '<rgb name="k" value="3.9, 2.4, 2.2"/></bsdf>'),
        '  <shape type="cube"><transform name="to_world"><scale value="0.5"/><translate x="5" y="2" z="1"/></transform>'
        '<emitter type="area"><rgb name="radiance" value="1, 1, 1"/></emitter></shape>',
        f'  <shape type="ply"><string name="filename" value="{ply}"/>'
        '<emitter type="area"><rgb name="radiance" value="1, 0.5, 0.2"/></emitter></shape>',
        '  <shape type="serialized"><string name="filename" value="missing.serialized"/></shape>',
    ]
    return '<scene version="3.0.0">\n' + SENSOR + '\n'.join(parts) + '\n</scene>\n'


def scene_meshlight(ply):
    """Lit by nothing but an emissive mesh."""
    return ('<scene version="3.0.0">\n' + SENSOR.split('  <emitter')[0]
            + sphere(0, '<bsdf type="diffuse" id="floor"/>') + '\n'
            + f'  <shape type="ply"><string name="filename" value="{ply}"/>'
            '<emitter type="area"><rgb name="radiance" value="1, 1, 1"/></emitter></shape>\n</scene>\n')


def check_cube_lights(out, fails):
    """Six `light area` faces whose u x v points away from the cube's centre (5, 2, 1)."""
    faces = re.findall(r'^light area \{ origin (\S+ \S+ \S+)  u (\S+ \S+ \S+)  v (\S+ \S+ \S+)  spd', out, re.M)
    faces = [[[float(x) for x in g.split()] for g in f] for f in faces]
    cube = [f for f in faces if abs(f[0][0] - 5) <= 0.5 + 1e-9]
    if len(cube) != 6:
        fails.append(f"cube emitter: {len(cube)} light area faces, not 6")
        return
    inward = 0
    for o, u, v in cube:
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        mid = [o[i] + 0.5 * (u[i] + v[i]) for i in range(3)]
        if sum(n[i] * (mid[i] - (5, 2, 1)[i]) for i in range(3)) <= 0:
            fails.append(f"cube emitter: face at {mid} faces into the cube")
            inward += 1
    if not inward:
        print("  ok  cube emitter        six light area faces, all facing out")


def warned(out, text):
    return any(line.startswith('# WARN:') and text in line for line in out.splitlines())


def convert(a, name, xml_text, fails):
    """Write `xml_text`, convert it, check it loads; return (FTSL text, ftrace's -parseonly
    log), or (None, '') when the converter fails."""
    xml = os.path.join(a.out, name + '.xml')
    ftsl = os.path.join(a.out, name + '.ftsl')
    with open(xml, 'w', encoding='utf-8', newline='') as f:
        f.write(xml_text)
    r = subprocess.run([sys.executable, a.converter, xml, ftsl],
                       capture_output=True, text=True, encoding='utf-8')
    if r.returncode != 0:
        fails.append(f"{name}: the converter exited {r.returncode}: {(r.stdout + r.stderr).strip()}")
        return None, ''
    with open(ftsl, encoding='utf-8') as f:
        out = f.read()
    r = subprocess.run([a.exe, ftsl, '-parseonly'], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', cwd=ROOT)
    log = r.stdout + r.stderr
    if r.returncode != 0:
        fails.append(f"{name}: ftrace -parseonly exited {r.returncode}: {log.strip()}")
    for line in log.splitlines():
        if 'unknown key' in line or 'unknown glass' in line:
            fails.append(f"{name}: ftrace: {line.strip()}")
    if not re.search(r'^\s*mode R\s*$', out, re.M):
        fails.append(f"{name}: the perspective sensor did not become a mode R camera")
    return out, log


def materials(out):
    return dict(re.findall(r'^material "([^"]+)" \{ type (\w+\s+.*?)\s*\}$', out, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'scraps', 'check_mitsuba_import'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--converter', default=os.path.join(ROOT, 'tools', 'mitsuba_to_ftsl.py'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fails = []
    glasses = glass_files()

    out, _ = convert(a, 'iors', scene_v3(), fails)
    if out is not None:
        mats = materials(out)
        diel = {k: v[len('dielectric'):].strip() for k, v in mats.items() if v.startswith('dielectric')}
        default = diel.get('ior_absent')
        for name, want in MITSUBA_IOR.items():
            expr = diel.get(mid(name))
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
        if diel.get('ior_numeric') != 'ior 1.33':
            fails.append(f"numeric int_ior 1.33 came out as {diel.get('ior_numeric')!r}")
        if default != 'ior glass:BK7':
            fails.append(f"absent int_ior came out as {default!r}, not Mitsuba's bk7 default")
        if diel.get(mid(UNKNOWN)) != default:
            fails.append(f"unknown name came out as {diel.get(mid(UNKNOWN))!r}, not the default")
        if f"# WARN: unknown named IOR '{UNKNOWN}'" not in out:
            fails.append("unknown name left no # WARN naming it")
        if not re.search(r"^# WARN: .*'ext_water': ext_ior 1\.333 dropped", out, re.M):
            fails.append("ext_ior water left no # WARN")
        if "'ext_air': ext_ior" in out:
            fails.append("ext_ior air (Mitsuba's default) was warned about")

    out, _ = convert(a, 'v06', scene_v06(), fails)
    if out is not None:
        mats = materials(out)
        for bid, want in (('v06_named', 'dielectric  ior glass:water'),
                          ('v06_numeric', 'dielectric  ior 2.4'),
                          ('v06_diffuse', 'diffuse  reflect rgb 0.2 0.4 0.6')):
            if mats.get(bid) != want:
                fails.append(f"0.6: {bid} came out as {mats.get(bid)!r}, not {want!r}")
            else:
                print(f"  ok  0.6 {bid:17s} {want}")
        if not re.search(r'^\s*eye 3 1 6\s', out, re.M):
            fails.append("0.6: toWorld did not place the sensor at (3, 1, 6)")
        if not re.search(r'^\s*fov_y 40\s*$', out, re.M):
            fails.append("0.6: $fovDeg did not substitute into the sensor's fov")

    ply = os.path.join(a.out, 'tri.ply').replace(os.sep, '/')
    with open(ply, 'w', encoding='ascii', newline='') as f:
        f.write(PLY)
    out, _ = convert(a, 'bsdfs', scene_bsdfs(ply), fails)
    if out is not None:
        mats = materials(out)
        eta = MITSUBA_IOR['bk7'] / MITSUBA_IOR['air']
        r0 = ((eta - 1) / (eta + 1)) ** 2
        want_thin = 2 * r0 / (1 + r0)
        thin = re.fullmatch(r'halfmirror\s+reflect (\S+)', mats.get('thin', ''))
        expect = [
            (mats.get('rough') == 'dielectric  ior glass:BK7  roughness 0.3' and warned(out, "'rough': alpha 0.3"),
             f"roughdielectric alpha 0.3 came out as {mats.get('rough')!r}"),
            (bool(thin) and abs(float(thin.group(1)) - want_thin) < 1e-5,
             f"thindielectric came out as {mats.get('thin')!r}, not halfmirror reflect {want_thin:.6f}"),
            (warned(out, "'tinted': specular_transmittance dropped"),
             "dielectric specular_transmittance left no # WARN"),
            (mats.get('tungsten') == 'mirror  reflect 0.9' and warned(out, "conductor material 'w'"),
             f"conductor W came out as {mats.get('tungsten')!r} or left no # WARN"),
            (mats.get('gold') == 'mirror  reflect metal:Au' and not warned(out, "material 'au'"),
             f"conductor Au came out as {mats.get('gold')!r} or was warned about"),
            (warned(out, "'etak': eta/k dropped"), "conductor eta/k left no # WARN"),
            (mats.get('_default_emit') == 'diffuse  reflect 0.7  emit rgb 1 0.5 0.2'
             and re.search(rf'^mesh {{ file "{re.escape(ply)}"  material _default_emit\b', out, re.M),
             f"ply emitter: material {mats.get('_default_emit')!r}, or no mesh line bound to it"),
            (warned(out, "missing.serialized") and 'file "missing.serialized"' not in out,
             "a .serialized mesh was written out or left no # WARN"),
        ]
        for ok, msg in expect:
            if not ok:
                fails.append(msg)
        print(f"  ok  bsdfs/emitters      {sum(bool(ok) for ok, _ in expect)} of {len(expect)}")
        check_cube_lights(out, fails)

    out, log = convert(a, 'meshlight', scene_meshlight(ply), fails)
    if out is not None:
        if warned(out, 'no lights found'):
            fails.append("a scene lit only by an emissive mesh got the fallback light")
        m = re.search(r'(\d+) emitters', log)
        if not m or int(m.group(1)) < 1:
            fails.append(f"ftrace loads the emissive-mesh scene with no emitter: {log.strip()}")
        else:
            print(f"  ok  emissive mesh       lights the scene alone ({m.group(1)} emitter)")

    for f_ in fails:
        print('FAIL:', f_)
    print(f"{'FAIL' if fails else 'PASS'}: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
