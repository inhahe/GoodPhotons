"""Regression check: tools/mitsuba_to_ftsl.py converts Mitsuba scenes into FTSL that loads right.

    D:/python314/python.exe -u tools/check_mitsuba_import.py [--out <dir>] [--exe ftrace.exe]
                                                              [--converter <mitsuba_to_ftsl.py>]

Writes small Mitsuba scenes, converts each, and checks:

  IOR (a version 3 scene with one dielectric per named IOR Mitsuba accepts):
  - every dielectric comes out as `type dielectric  ior <expr>`;
  - each named IOR resolves, at 589 nm, to within 0.015 of Mitsuba's own value: a `glass:`
    name through the data/glass file it names, a number as written. (BK7, the widest gap, is
    0.012: Mitsuba's table quotes it at 1.5046, the catalogue's n_d is 1.5168);
  - a constant keeps every digit of Mitsuba's table (helium is `ior 1.000036`);
  - `int_ior 1.33` stays `ior 1.33`, and an absent `int_ior` takes Mitsuba's default, bk7;
  - an unknown name falls back to that default and leaves a `# WARN:` naming it;
  - an `ext_ior` other than air leaves a `# WARN:`; `ext_ior air` does not.

  Version 0.6 (camelCase property names, which Mitsuba's parser renames on load):
  - `intIOR` (named or numeric) is read as `int_ior`, and `toWorld` places the sensor;
  - a diffuse BSDF's `diffuseReflectance` is read as its reflectance;
  - a conductor with no `material` is copper, 0.6's default (3's is a perfect mirror);
  - a `<default>` parameter keeps its camelCase name, so `$fovDeg` still substitutes.

  BSDFs:
  - a microfacet alpha becomes the `roughness` of matching lobe width: 2 alpha for Beckmann,
    1.55 alpha for GGX, capped at 1; an anisotropic alpha_u/alpha_v is averaged and flagged;
  - thindielectric becomes a `halfmirror` reflecting a sheet's 2R/(1+R) at normal incidence;
  - a dielectric's specular_transmittance, a conductor's eta/k, and a conductor material FTSL
    has no `metal:` for are each flagged; a known one (Au) is not.

  Lights:
  - an rgb emitter becomes `rgbillum` at its own magnitude, so lights keep their ratios;
  - a point emitter becomes a sphere light whose radiance is intensity / (pi r^2);
  - a directional emitter becomes a `light sun` whose `dir` points back up its direction;
  - a cube area emitter becomes six `light area` faces, each facing out of the cube;
  - a .ply mesh area emitter is loaded as a mesh under an emissive copy of its material,
    one copy shared by meshes that glow alike, and lights a scene alone, without the
    converter's fallback light;
  - a .serialized mesh is skipped with a `# WARN:` rather than written for ftrace to fail on.

  Mesh placement: a `to_world` of scale, rotations and translation places the mesh's points
  where Mitsuba does, through FTSL's scale / Euler `rotate` / translate; a shear and a mirror
  are each flagged.

  Cameras: a perspective sensor becomes a `mode R` camera; a thinlens one becomes mode A and,
  beside glass, is flagged.

  Every scene: `ftrace -parseonly`, run from the repo root so it reads the same data/ the
  check does, loads the result with no unknown key and no unknown glass.

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
          '  </sensor>\n')
ENV = '  <emitter type="constant"><rgb name="radiance" value="1, 1, 1"/></emitter>\n'

PLY = ('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n'
       'property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
       '0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')

# Mesh placements: (name, Mitsuba <transform> children as (op, args), what it should warn).
PLACEMENTS = [
    ('placed', [('scale', (2, 1, 0.5)), ('rotate', (1, 0, 0, -90)), ('rotate', (0, 1, 0, 30)),
                ('rotate', (0, 0, 1, 20)), ('translate', (1, 2, 3))], None),
    ('upright', [('rotate', (1, 0, 0, -90))], None),
    ('gimbal', [('rotate', (1, 0, 0, 25)), ('rotate', (0, 1, 0, 90)), ('translate', (0, 1, 0))], None),
    ('sheared', [('rotate', (0, 0, 1, 45)), ('scale', (2, 1, 1))], 'shears'),
    ('mirrored', [('scale', (-1, 1, 1))], 'mirrors'),
]


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


# --- placing points, independently of the converter -------------------------------------
def mitsuba_place(ops, p):
    """Where Mitsuba's <transform> children `ops`, applied in document order, put point p."""
    x = list(p)
    for op, a in ops:
        if op == 'scale':
            x = [x[i] * a[i] for i in range(3)]
        elif op == 'translate':
            x = [x[i] + a[i] for i in range(3)]
        elif op == 'rotate':   # Rodrigues about the unit axis
            n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
            k = [a[0] / n, a[1] / n, a[2] / n]
            t = math.radians(a[3]); c, s = math.cos(t), math.sin(t)
            kx = [k[1] * x[2] - k[2] * x[1], k[2] * x[0] - k[0] * x[2], k[0] * x[1] - k[1] * x[0]]
            kd = sum(k[i] * x[i] for i in range(3))
            x = [x[i] * c + kx[i] * s + k[i] * kd * (1 - c) for i in range(3)]
    return x


def ftsl_place(line, p):
    """Where a converted `mesh { … translate … scale … rotate … }` line puts point p, by the
    formula of mesh.h's MeshXform: scale, then Rx, Ry, Rz (degrees), then translate."""
    def vec(key, d):
        m = re.search(rf'\b{key} (\S+)(?: (\S+) (\S+))?(?=  | }})', line)
        if not m:
            return d
        v = [float(g) for g in m.groups() if g is not None]
        return v * 3 if len(v) == 1 else v
    t, s, r = vec('translate', [0, 0, 0]), vec('scale', [1, 1, 1]), vec('rotate', [0, 0, 0])
    x = [p[i] * s[i] for i in range(3)]
    cx, sx = math.cos(math.radians(r[0])), math.sin(math.radians(r[0]))
    cy, sy = math.cos(math.radians(r[1])), math.sin(math.radians(r[1]))
    cz, sz = math.cos(math.radians(r[2])), math.sin(math.radians(r[2]))
    x = [x[0], cx * x[1] - sx * x[2], sx * x[1] + cx * x[2]]
    x = [cy * x[0] + sy * x[2], x[1], -sy * x[0] + cy * x[2]]
    x = [cz * x[0] - sz * x[1], sz * x[0] + cz * x[1], x[2]]
    return [x[i] + t[i] for i in range(3)]


def xml_ops(ops):
    out = []
    for op, a in ops:
        if op == 'rotate':
            out.append(f'<rotate x="{a[0]}" y="{a[1]}" z="{a[2]}" angle="{a[3]}"/>')
        else:
            out.append(f'<{op} x="{a[0]}" y="{a[1]}" z="{a[2]}"/>')
    return ''.join(out)


# --- scenes --------------------------------------------------------------------------------
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
    return '<scene version="3.0.0">\n' + SENSOR + ENV + body + '\n</scene>\n'


def scene_v06():
    return ('<scene version="0.6.0">\n'
            '  <default name="fovDeg" value="40"/>\n'
            '  <sensor type="perspective"><float name="fov" value="$fovDeg"/>\n'
            '    <transform name="toWorld"><lookat origin="3, 1, 6" target="3, 0, 0" up="0, 1, 0"/></transform>\n'
            '    <film type="hdrfilm"><integer name="width" value="64"/><integer name="height" value="64"/></film>\n'
            '  </sensor>\n' + ENV
            + sphere(0, '<bsdf type="dielectric" id="v06_named"><string name="intIOR" value="water"/></bsdf>') + '\n'
            + sphere(1, '<bsdf type="dielectric" id="v06_numeric"><float name="intIOR" value="2.4"/></bsdf>') + '\n'
            + sphere(2, '<bsdf type="diffuse" id="v06_diffuse"><rgb name="diffuseReflectance" value="0.2, 0.4, 0.6"/></bsdf>') + '\n'
            + sphere(3, '<bsdf type="conductor" id="v06_conductor"/>')
            + '\n</scene>\n')


def scene_bsdfs(ply):
    parts = [
        sphere(0, '<bsdf type="roughdielectric" id="rough"><float name="alpha" value="0.3"/></bsdf>'),
        sphere(1, '<bsdf type="roughdielectric" id="rough_ggx"><float name="alpha" value="0.3"/>'
                  '<string name="distribution" value="ggx"/></bsdf>'),
        sphere(2, '<bsdf type="roughconductor" id="aniso"><string name="material" value="Au"/>'
                  '<float name="alpha_u" value="0.2"/><float name="alpha_v" value="0.1"/></bsdf>'),
        sphere(3, '<bsdf type="roughplastic" id="capped"><float name="alpha" value="0.7"/></bsdf>'),
        sphere(4, '<bsdf type="thindielectric" id="thin"/>'),
        sphere(5, '<bsdf type="dielectric" id="tinted"><rgb name="specular_transmittance" value="0.6, 0.8, 1"/></bsdf>'),
        sphere(6, '<bsdf type="conductor" id="tungsten"><string name="material" value="W"/></bsdf>'),
        sphere(7, '<bsdf type="conductor" id="gold"><string name="material" value="Au"/></bsdf>'),
        sphere(8, '<bsdf type="conductor" id="etak"><rgb name="eta" value="0.2, 0.9, 1.1"/>'
                  '<rgb name="k" value="3.9, 2.4, 2.2"/></bsdf>'),
        '  <shape type="cube"><transform name="to_world"><scale value="0.5"/><translate x="5" y="2" z="1"/></transform>'
        '<emitter type="area"><rgb name="radiance" value="50, 25, 10"/></emitter></shape>',
        f'  <shape type="ply"><string name="filename" value="{ply}"/>'
        '<emitter type="area"><rgb name="radiance" value="1, 0.5, 0.2"/></emitter></shape>',
        '  <shape type="serialized"><string name="filename" value="missing.serialized"/></shape>',
        '  <emitter type="point"><point name="position" x="1" y="3" z="1"/><rgb name="intensity" value="1, 2, 3"/></emitter>',
        '  <emitter type="directional"><vector name="direction" x="0" y="-1" z="0"/>'
        '<rgb name="irradiance" value="2, 2, 2"/></emitter>',
    ]
    return '<scene version="3.0.0">\n' + SENSOR + ENV + '\n'.join(parts) + '\n</scene>\n'


def scene_meshes(ply):
    """Lit by nothing but two emissive meshes that glow alike, plus the placement cases."""
    parts = [sphere(0, '<bsdf type="diffuse" id="floor"/>')]
    for i in range(2):
        parts.append(f'  <shape type="ply"><string name="filename" value="{ply}"/>'
                     f'<transform name="to_world"><translate x="{i}" y="0" z="0"/></transform>'
                     '<emitter type="area"><rgb name="radiance" value="1, 1, 1"/></emitter></shape>')
    for name, ops, _ in PLACEMENTS:
        parts.append(f'  <shape type="ply" id="{name}"><string name="filename" value="{ply}"/>'
                     f'<transform name="to_world">{xml_ops(ops)}</transform>'
                     f'<bsdf type="diffuse" id="m_{name}"/></shape>')
    return '<scene version="3.0.0">\n' + SENSOR + '\n'.join(parts) + '\n</scene>\n'


def scene_thinlens():
    return ('<scene version="3.0.0">\n'
            '  <sensor type="thinlens"><float name="fov" value="40"/><float name="aperture_radius" value="0.05"/>'
            '<float name="focus_distance" value="3"/>\n'
            '    <transform name="to_world"><lookat origin="0, 1, 6" target="0, 0, 0" up="0, 1, 0"/></transform>\n'
            '    <film type="hdrfilm"><integer name="width" value="64"/><integer name="height" value="64"/></film>\n'
            '  </sensor>\n' + ENV
            + sphere(0, '<bsdf type="dielectric" id="glass"/>') + '\n</scene>\n')


# --- checking ------------------------------------------------------------------------------
def warned(out, text):
    return any(line.startswith('# WARN:') and text in line for line in out.splitlines())


def convert(a, name, xml_text, fails, mode='R'):
    """Write `xml_text`, convert it, check it loads and its camera is in `mode`; return
    (FTSL text, ftrace's -parseonly log), or (None, '') when the converter fails."""
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
    if not re.search(rf'^\s*(aperture .*  )?mode {mode}\s*$', out, re.M):
        fails.append(f"{name}: the sensor did not become a mode {mode} camera")
    return out, log


def materials(out):
    return dict(re.findall(r'^material "([^"]+)" \{ type (\w+\s+.*?)\s*\}$', out, re.M))


def expect_all(label, expect, fails):
    """Record each failed (ok, message) pair; report `label` ok only if none failed."""
    bad = [msg for ok, msg in expect if not ok]
    fails.extend(bad)
    if not bad:
        print(f"  ok  {label:20s} {len(expect)} of {len(expect)}")


def close(x, want, rel=1e-5):
    return x is not None and abs(x - want) <= rel * max(1.0, abs(want))


def floats(s):
    return [float(v) for v in s.split()] if s else None


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
        c = [o[i] + 0.5 * (u[i] + v[i]) for i in range(3)]
        if sum(n[i] * (c[i] - (5, 2, 1)[i]) for i in range(3)) <= 0:
            fails.append(f"cube emitter: face at {c} faces into the cube")
            inward += 1
    if not inward:
        print("  ok  cube emitter        six light area faces, all facing out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'scraps', 'check_mitsuba_import'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--converter', default=os.path.join(ROOT, 'tools', 'mitsuba_to_ftsl.py'))
    a = ap.parse_args()
    # ftrace runs from ROOT, so every path handed to it must not depend on the caller's cwd.
    a.out, a.exe, a.converter = (os.path.abspath(p) for p in (a.out, a.exe, a.converter))
    os.makedirs(a.out, exist_ok=True)
    fails = []
    glasses = glass_files()
    ply = os.path.join(a.out, 'tri.ply').replace(os.sep, '/')
    with open(ply, 'w', encoding='ascii', newline='') as f:
        f.write(PLY)

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
                print(f"  ok  {name:21s} {expr:22s} n {n:.6f}  (Mitsuba {want})")
        expect_all('ior cases', [
            (diel.get('ior_helium') == 'ior 1.000036',
             f"helium came out as {diel.get('ior_helium')!r}, not every digit of 1.000036"),
            (diel.get('ior_numeric') == 'ior 1.33',
             f"numeric int_ior 1.33 came out as {diel.get('ior_numeric')!r}"),
            (default == 'ior glass:BK7', f"absent int_ior came out as {default!r}, not bk7's"),
            (diel.get(mid(UNKNOWN)) == default,
             f"unknown name came out as {diel.get(mid(UNKNOWN))!r}, not the default"),
            (warned(out, f"unknown named IOR '{UNKNOWN}'"), "unknown name left no # WARN naming it"),
            (bool(re.search(r"^# WARN: .*'ext_water': ext_ior 1\.333 dropped", out, re.M)),
             "ext_ior water left no # WARN"),
            ("'ext_air': ext_ior" not in out, "ext_ior air (Mitsuba's default) was warned about"),
        ], fails)

    out, _ = convert(a, 'v06', scene_v06(), fails)
    if out is not None:
        mats = materials(out)
        expect_all('0.6 scene', [
            (mats.get(b) == w, f"0.6: {b} came out as {mats.get(b)!r}, not {w!r}")
            for b, w in (('v06_named', 'dielectric  ior glass:water'),
                         ('v06_numeric', 'dielectric  ior 2.4'),
                         ('v06_diffuse', 'diffuse  reflect rgb 0.2 0.4 0.6'),
                         ('v06_conductor', 'mirror  reflect metal:Cu'))
        ] + [
            (bool(re.search(r'^\s*eye 3 1 6\s', out, re.M)), "0.6: toWorld did not place the sensor at (3, 1, 6)"),
            (bool(re.search(r'^\s*fov_y 40\s*$', out, re.M)), "0.6: $fovDeg did not substitute into the sensor's fov"),
        ], fails)

    out, _ = convert(a, 'bsdfs', scene_bsdfs(ply), fails)
    if out is not None:
        mats = materials(out)
        eta = MITSUBA_IOR['bk7'] / MITSUBA_IOR['air']
        r0 = ((eta - 1) / (eta + 1)) ** 2
        want_thin = 2 * r0 / (1 + r0)
        rough = lambda bid: (lambda m: float(m.group(1)) if m else None)(
            re.search(r'roughness (\S+)$', mats.get(bid, '')))
        thin = re.fullmatch(r'halfmirror\s+reflect (\S+)', mats.get('thin', ''))
        expect_all('bsdfs', [
            (mats.get('rough', '').startswith('dielectric  ior glass:BK7') and close(rough('rough'), 0.6)
             and warned(out, "'rough': alpha 0.3 -> roughness 0.6"),
             f"roughdielectric Beckmann alpha 0.3 came out as {mats.get('rough')!r}, not roughness 0.6"),
            (close(rough('rough_ggx'), 0.465), f"GGX alpha 0.3 came out as {mats.get('rough_ggx')!r}, not roughness 0.465"),
            (close(rough('aniso'), 0.3) and warned(out, "'aniso': anisotropic alpha_u 0.2 / alpha_v 0.1"),
             f"alpha_u 0.2 / alpha_v 0.1 came out as {mats.get('aniso')!r} or left no # WARN"),
            (close(rough('capped'), 1.0), f"alpha 0.7 came out as {mats.get('capped')!r}, not capped at 1"),
            (bool(thin) and close(float(thin.group(1)), want_thin),
             f"thindielectric came out as {mats.get('thin')!r}, not halfmirror reflect {want_thin:.6f}"),
            (warned(out, "'tinted': specular_transmittance dropped"), "dielectric specular_transmittance left no # WARN"),
            (mats.get('tungsten') == 'mirror  reflect 0.9' and warned(out, "conductor material 'w'"),
             f"conductor W came out as {mats.get('tungsten')!r} or left no # WARN"),
            (mats.get('gold') == 'mirror  reflect metal:Au' and not warned(out, "material 'au'"),
             f"conductor Au came out as {mats.get('gold')!r} or was warned about"),
            (warned(out, "'etak': eta/k dropped"), "conductor eta/k left no # WARN"),
        ], fails)

        cube_spd = set(re.findall(r'^light area \{ origin 4\.5 .*spd (.*) \}$', out, re.M)) | \
            set(re.findall(r'^light area \{ origin 5\.5 .*spd (.*) \}$', out, re.M))
        pt = re.search(r'^light sphere \{ center 1 3 1  radius (\S+)  spd rgbillum (\S+ \S+ \S+) \}$', out, re.M)
        k = 1 / (math.pi * float(pt.group(1)) ** 2) if pt else None
        sun = re.search(r'^light sun \{ dir (\S+ \S+ \S+)  spd (.*) \}$', out, re.M)
        emit = re.search(r'^mesh \{ file "' + re.escape(ply) + r'"  material (\S+)\b', out, re.M)
        expect_all('lights', [
            (cube_spd == {'rgbillum 50 25 10'}, f"cube emitter radiance 50 25 10 came out as {cube_spd}"),
            (mats.get('_default_emit') == 'diffuse  reflect 0.7  emit rgbillum 1 0.5 0.2'
             and emit is not None and emit.group(1) == '_default_emit',
             f"ply emitter: material {mats.get('_default_emit')!r}, or no mesh line bound to it"),
            (pt is not None and all(close(x, w * k, 1e-4) for x, w in zip(floats(pt.group(2)), (1, 2, 3))),
             f"point intensity 1 2 3 came out as {pt.group(0) if pt else None!r}, not radiance intensity/(pi r^2)"),
            (sun is not None and floats(sun.group(1)) == [0, 1, 0] and sun.group(2) == 'rgbillum 2 2 2',
             f"directional (0,-1,0), irradiance 2 came out as {sun.group(0) if sun else None!r}"),
            (warned(out, "missing.serialized") and 'file "missing.serialized"' not in out,
             "a .serialized mesh was written out or left no # WARN"),
        ], fails)
        check_cube_lights(out, fails)

    out, log = convert(a, 'meshes', scene_meshes(ply), fails)
    if out is not None:
        m = re.search(r'(\d+) emitters', log)
        emits = [n for n, b in materials(out).items() if '  emit ' in b]
        expect = [
            (not warned(out, 'no lights found'), "a scene lit only by emissive meshes got the fallback light"),
            (bool(m) and int(m.group(1)) >= 1, f"ftrace loads the emissive-mesh scene with no emitter: {log.strip()}"),
            (len(emits) == 1, f"two meshes glowing alike made {len(emits)} emissive materials, not 1: {emits}"),
        ]
        probes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.3, -0.7, 0.2)]
        for name, ops, flag in PLACEMENTS:
            line = re.search(rf'^mesh \{{ file "[^"]*"  material m_{name}\b.*\}}$', out, re.M)
            if flag:
                expect.append((warned(out, f"(material m_{name}): to_world {flag} it"),
                               f"placement {name}: no # WARN that it {flag}"))
                continue
            if not line:
                expect.append((False, f"placement {name}: no mesh line"))
                continue
            err = max(abs(u - v) for p in probes
                      for u, v in zip(ftsl_place(line.group(0), p), mitsuba_place(ops, p)))
            expect.append((err < 1e-4, f"placement {name}: {line.group(0)!r} puts points {err:.3g} from Mitsuba's"))
            expect.append((not warned(out, f"(material m_{name}): to_world"),
                           f"placement {name}: warned though exact"))
        expect_all('meshes', expect, fails)

    out, _ = convert(a, 'thinlens', scene_thinlens(), fails, mode='A')
    if out is not None:
        expect_all('thinlens', [
            (warned(out, 'thinlens sensor -> mode A'), "a thinlens camera beside glass left no # WARN"),
        ], fails)

    for f_ in fails:
        print('FAIL:', f_)
    print(f"{'FAIL' if fails else 'PASS'}: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
