#!/usr/bin/env python3
"""Mesh-format import matrix: does COLOUR and SEE-THROUGH survive each loader?

ftrace reads six mesh formats and they do not carry material data the same way, so
"the viewer's Color/See-through buttons do nothing" can mean either a broken toggle or
a loader that never imported a material to toggle. This builds ONE geometry -- two
opaque spheres and two of coloured glass -- in every format that can express it, renders
each with colour on/off and see-through on/off, and measures whether the toggles moved
anything. Answering it by rendering rather than by reading code is the point.

    python tools/mesh_format_matrix.py [--keep]

Expected, as of 0.225.0:

    obj     YES / YES     .mtl companion (Kd, d/Tr, Ni, Tf, illum)
    glb     YES / YES     glTF core + KHR ior/transmission/volume/dispersion
    fbx     YES / YES     ufbx pbr + legacy FBX property sets  (needs your own .fbx)
    ply     no  / no      vertex colours are not read; format has no transparency
    stl     no  / no      the format carries no material data at all
    ftmesh  no  / no      deliberately geometry-only (see mesh.h)

A `no` in the last three rows is the expected answer, not a regression. A `no` in the
first three is a bug.
"""
import json, math, os, struct, subprocess, sys, tempfile, shutil

import numpy as np
from PIL import Image


OUT = os.path.join(tempfile.gettempdir(), 'ftrace_fmt')
os.makedirs(OUT, exist_ok=True)

def sphere(cx, cy, cz, r, nu=32, nv=16):
    """UV sphere -> (verts, faces) with faces indexing into verts."""
    V, F = [], []
    for j in range(nv + 1):
        th = math.pi * j / nv
        for i in range(nu):
            ph = 2 * math.pi * i / nu
            V.append((cx + r * math.sin(th) * math.cos(ph),
                      cy + r * math.cos(th),
                      cz + r * math.sin(th) * math.sin(ph)))
    for j in range(nv):
        for i in range(nu):
            a = j * nu + i
            b = j * nu + (i + 1) % nu
            c = (j + 1) * nu + (i + 1) % nu
            d = (j + 1) * nu + i
            if j != 0:      F.append((a, b, c))
            if j != nv - 1: F.append((a, c, d))
    return V, F

# name, centre, radius, Kd (diffuse rgb), alpha (d), Ni (ior), Tf (transmission filter)
GROUPS = [
    ('mat_red_opaque',   (-0.75, 0, 0), 0.26, (0.85, 0.10, 0.10), 1.0, 1.0, None),
    ('mat_green_opaque', (-0.25, 0, 0), 0.26, (0.10, 0.75, 0.15), 1.0, 1.0, None),
    ('mat_glass_amber',  ( 0.25, 0, 0), 0.26, (1.00, 1.00, 1.00), 0.15, 1.52, (0.90, 0.55, 0.08)),
    ('mat_glass_cyan',   ( 0.75, 0, 0), 0.26, (1.00, 1.00, 1.00), 0.15, 1.52, (0.10, 0.75, 0.85)),
]

verts, faces, groups = [], [], []
for name, c, r, kd, alpha, ni, tf in GROUPS:
    V, F = sphere(*c, r)
    base = len(verts)
    start = len(faces)
    verts += V
    faces += [(a + base, b + base, cc + base) for a, b, cc in F]
    groups.append((name, start, len(faces) - start, kd, alpha, ni, tf))

# ---------------------------------------------------------------- OBJ + MTL
with open(f'{OUT}/spheres.obj', 'w') as f:
    f.write('# ftrace format-matrix test: 2 opaque + 2 glass spheres\n')
    f.write('mtllib spheres.mtl\n')
    for v in verts:
        f.write(f'v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n')
    for name, start, count, *_ in groups:
        f.write(f'g {name}\nusemtl {name}\n')
        for a, b, c in faces[start:start + count]:
            f.write(f'f {a+1} {b+1} {c+1}\n')

with open(f'{OUT}/spheres.mtl', 'w') as f:
    f.write('# companion material library\n')
    for name, _s, _c, kd, alpha, ni, tf in groups:
        f.write(f'\nnewmtl {name}\n')
        f.write(f'Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n')
        f.write('Ks 0.1 0.1 0.1\nNs 32\n')
        f.write(f'Ni {ni:.4f}\n')
        f.write(f'd {alpha:.4f}\n')
        if tf:
            f.write(f'Tf {tf[0]:.4f} {tf[1]:.4f} {tf[2]:.4f}\n')
        f.write('illum %d\n' % (7 if alpha < 1.0 else 2))

# ---------------------------------------------------------------- PLY (ascii, vertex colour)
vcol = [(200, 200, 200)] * len(verts)
for name, start, count, kd, *_ in groups:
    for a, b, c in faces[start:start + count]:
        for idx in (a, b, c):
            vcol[idx] = tuple(int(255 * x) for x in kd)
with open(f'{OUT}/spheres.ply', 'w') as f:
    f.write('ply\nformat ascii 1.0\n')
    f.write(f'element vertex {len(verts)}\n')
    f.write('property float x\nproperty float y\nproperty float z\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    for v, c in zip(verts, vcol):
        f.write(f'{v[0]:.6g} {v[1]:.6g} {v[2]:.6g} {c[0]} {c[1]} {c[2]}\n')
    for a, b, c in faces:
        f.write(f'3 {a} {b} {c}\n')

# ---------------------------------------------------------------- STL (binary)
with open(f'{OUT}/spheres.stl', 'wb') as f:
    f.write(b'\0' * 80)
    f.write(struct.pack('<I', len(faces)))
    for a, b, c in faces:
        A, B, C = verts[a], verts[b], verts[c]
        u = [B[i] - A[i] for i in range(3)]
        v = [C[i] - A[i] for i in range(3)]
        n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        L = math.sqrt(sum(x*x for x in n)) or 1.0
        f.write(struct.pack('<3f', *[x / L for x in n]))
        for P in (A, B, C):
            f.write(struct.pack('<3f', *P))
        f.write(struct.pack('<H', 0))

# ---------------------------------------------------------------- GLB (KHR glass)
_buf = bytearray(); _views = []; _accs = []
def _add(data, target):
    while len(_buf) % 4: _buf.append(0)
    off = len(_buf); _buf.extend(data)
    _views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data), 'target': target})
    return len(_views) - 1
_prims, _mats = [], []
for gi, (name, c, r, kd, alpha, ni, tf) in enumerate(GROUPS):
    V, F = sphere(*c, r)
    vv = _add(b''.join(struct.pack('<3f', *v) for v in V), 34962)
    iv = _add(b''.join(struct.pack('<H', i) for tri in F for i in tri), 34963)
    _accs.append({'bufferView': vv, 'componentType': 5126, 'count': len(V), 'type': 'VEC3',
                  'min': [min(v[i] for v in V) for i in range(3)],
                  'max': [max(v[i] for v in V) for i in range(3)]})
    _accs.append({'bufferView': iv, 'componentType': 5123, 'count': len(F) * 3, 'type': 'SCALAR'})
    m = {'name': name, 'pbrMetallicRoughness': {
            'baseColorFactor': list(kd) + [1.0], 'metallicFactor': 0.0, 'roughnessFactor': 0.2}}
    if alpha < 1.0:
        # The colour of glTF glass lives in the extensions, never in baseColorFactor.
        m['pbrMetallicRoughness']['baseColorFactor'] = [1, 1, 1, 1]
        m['extensions'] = {
            'KHR_materials_transmission': {'transmissionFactor': 1.0},
            'KHR_materials_ior': {'ior': ni},
            'KHR_materials_volume': {'attenuationColor': [max(1e-3, x) for x in tf],
                                     'attenuationDistance': 0.02, 'thicknessFactor': 0.02}}
    _mats.append(m)
    _prims.append({'attributes': {'POSITION': len(_accs) - 2}, 'indices': len(_accs) - 1, 'material': gi})
_js = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
       'meshes': [{'primitives': _prims}], 'materials': _mats, 'accessors': _accs,
       'bufferViews': _views, 'buffers': [{'byteLength': len(_buf)}],
       'extensionsUsed': ['KHR_materials_transmission', 'KHR_materials_ior', 'KHR_materials_volume']}
_jb = json.dumps(_js).encode()
while len(_jb) % 4: _jb += b' '
while len(_buf) % 4: _buf.append(0)
with open(f'{OUT}/spheres.glb', 'wb') as f:
    f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(_jb) + 8 + len(_buf)))
    f.write(struct.pack('<II', len(_jb), 0x4E4F534A)); f.write(_jb)
    f.write(struct.pack('<II', len(_buf), 0x004E4942)); f.write(bytes(_buf))

# ---------------------------------------------------------------- .ftmesh (via ftrace)
subprocess.run(['./ftrace.exe', f'{OUT}/spheres.obj', '-nd', '4',
                '-nd-export', f'{OUT}/spheres.ftmesh'], capture_output=True, timeout=600)

print(f'{len(verts)} verts, {len(faces)} tris')
for e in ('obj', 'mtl', 'ply', 'stl'):
    p = f'{OUT}/spheres.{e}'
    print(f'  {p}  {os.path.getsize(p):,} bytes')



FMT = OUT
OUTD = os.path.join(OUT, 'render')
os.makedirs(OUTD, exist_ok=True)
FILES = ['spheres.obj', 'spheres.ply', 'spheres.stl', 'spheres.ftmesh', 'spheres.glb']
VIEW = ['-view', '0', '0.15', '2.2', '0', '0', '0', '0', '1', '0', '40']

def render(src, out, extra):
    cmd = ['./ftrace.exe', src, '-raster', '-r', '360', '-o', out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return os.path.exists(out), r.stdout + r.stderr

def stats(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(int)
    sat = (a.max(axis=2) - a.min(axis=2))
    return a, float(sat.mean()), int(sat.max())

print(f'{"format":10s} {"loads":6s} {"sat(color on)":>14s} {"sat(color off)":>15s} '
      f'{"colour works":>13s} {"see-through delta":>18s} {"works":>6s}')
print('-' * 92)
for fn in FILES:
    src = f'{FMT}/{fn}'
    ext = fn.rsplit('.', 1)[1]
    if not os.path.exists(src):
        print(f'{ext:10s} (absent)')
        continue
    ok1, log = render(src, f'{OUTD}/{ext}_col.png', [])
    if not ok1:
        err = [l for l in log.splitlines() if 'rror' in l or 'annot' in l]
        print(f'{ext:10s} FAIL   {err[:1]}')
        continue
    render(src, f'{OUTD}/{ext}_flat.png', ['-flat'])
    render(src, f'{OUTD}/{ext}_st.png',   ['-see-through'])
    a, s1, m1 = stats(f'{OUTD}/{ext}_col.png')
    b, s2, m2 = stats(f'{OUTD}/{ext}_flat.png')
    c, _, _   = stats(f'{OUTD}/{ext}_st.png')
    colour_works = 'YES' if (m1 - m2) > 25 else 'no'
    st_delta = float(np.abs(a - c).mean())
    st_works = 'YES' if st_delta > 1.0 else 'no'
    print(f'{ext:10s} ok     {s1:6.1f}/{m1:<7d} {s2:6.1f}/{m2:<8d} {colour_works:>13s} '
          f'{st_delta:18.2f} {st_works:>6s}')

if '--keep' in sys.argv:
    print()
    print('assets and renders kept in ' + OUT)
else:
    shutil.rmtree(OUT, ignore_errors=True)
