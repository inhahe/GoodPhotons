#!/usr/bin/env python3
"""Per-vertex colour: does it survive every importer, every backend, and instancing?

A vertex colour has several independent ways to get lost, so this checks all of them by
RENDERING rather than by reading code:

  * the importer         — PLY red/green/blue, OBJ `v x y z r g b`, glTF COLOR_0
  * the CPU rasterizer    — its shade pass is deferred, so it needs a G-buffer channel
  * the GPU rasterizer    — a separate DPTri + kShade port
  * the spectral tracer   — the RGB has to become reflect(lambda) per hit, and the CPU
                            and GPU tracers are two more separate ports
  * instancing (BLAS)     — `mesh_asset` copies triangles into its own array, and the
                            raster path bakes those in a SECOND loop, which is exactly
                            where an attribute added to the first one gets forgotten

    python tools/vertex_color_check.py [--keep]

The model is a sphere with a hue ramp around it: a stage that works shows a smooth
rainbow, a stage that drops the colour shows flat grey. `sat` is per-pixel saturation
(max channel − min channel), so grey reads near zero and the ramp reads ~50/240.
"""
import json, math, os, struct, subprocess, sys, tempfile, shutil
import numpy as np
from PIL import Image

OUT = os.path.join(tempfile.gettempdir(), 'ftrace_vcol')
REND = os.path.join(OUT, 'render')
os.makedirs(REND, exist_ok=True)


def sphere(cx, cy, cz, r, nu=24, nv=12):
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
            if j:            F.append((a, b, c))
            if j != nv - 1:  F.append((a, c, d))
    return V, F


V, F = sphere(0, 0, 0, 0.9)


def hue(v):
    a = (math.atan2(v[2], v[0]) + math.pi) / (2 * math.pi)
    i = int(a * 6) % 6
    f = a * 6 - int(a * 6)
    return [(1, f, 0), (1 - f, 1, 0), (0, 1, f), (0, 1 - f, 1), (f, 0, 1), (1, 0, 1 - f)][i]


C = [hue(v) for v in V]

# ---- PLY (uchar red/green/blue, display-space) ---------------------------------
with open(f'{OUT}/s.ply', 'w') as f:
    f.write(f'ply\nformat ascii 1.0\nelement vertex {len(V)}\n')
    f.write('property float x\nproperty float y\nproperty float z\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    f.write(f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
    for v, c in zip(V, C):
        f.write(f'{v[0]:.6g} {v[1]:.6g} {v[2]:.6g} '
                f'{int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n')
    for a, b, c in F:
        f.write(f'3 {a} {b} {c}\n')

# ---- OBJ, the extended `v x y z r g b` form ------------------------------------
with open(f'{OUT}/s.obj', 'w') as f:
    for v, c in zip(V, C):
        f.write(f'v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n')
    for a, b, c in F:
        f.write(f'f {a+1} {b+1} {c+1}\n')

# ---- GLB with COLOR_0 (already linear, per spec) -------------------------------
buf = bytearray(); views = []; accs = []


def add(data, target):
    while len(buf) % 4: buf.append(0)
    off = len(buf); buf.extend(data)
    views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data), 'target': target})
    return len(views) - 1


pv = add(b''.join(struct.pack('<3f', *v) for v in V), 34962)
cv = add(b''.join(struct.pack('<3f', *c) for c in C), 34962)
iv = add(b''.join(struct.pack('<H', i) for t in F for i in t), 34963)
accs.append({'bufferView': pv, 'componentType': 5126, 'count': len(V), 'type': 'VEC3',
             'min': [min(v[i] for v in V) for i in range(3)],
             'max': [max(v[i] for v in V) for i in range(3)]})
accs.append({'bufferView': cv, 'componentType': 5126, 'count': len(V), 'type': 'VEC3'})
accs.append({'bufferView': iv, 'componentType': 5123, 'count': len(F) * 3, 'type': 'SCALAR'})
js = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
      'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'COLOR_0': 1},
                                  'indices': 2, 'material': 0}]}],
      'materials': [{'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1],
                                              'metallicFactor': 0.0, 'roughnessFactor': 0.6}}],
      'accessors': accs, 'bufferViews': views, 'buffers': [{'byteLength': len(buf)}]}
jb = json.dumps(js).encode()
while len(jb) % 4: jb += b' '
while len(buf) % 4: buf.append(0)
with open(f'{OUT}/s.glb', 'wb') as f:
    f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(buf)))
    f.write(struct.pack('<II', len(jb), 0x4E4F534A)); f.write(jb)
    f.write(struct.pack('<II', len(buf), 0x004E4942)); f.write(bytes(buf))


def measure(out):
    if not os.path.exists(out): return None
    a = np.asarray(Image.open(out).convert('RGB')).astype(int)
    s = a.max(axis=2) - a.min(axis=2)
    return float(s.mean()), int(s.max())


def run(args, out):
    subprocess.run(['./ftrace.exe'] + args + ['-o', out], capture_output=True, timeout=2400)
    return measure(out)


def fmt(r):
    return 'FAIL' if r is None else f'{r[0]:.1f}/{r[1]}'


print(f'{"format":8s} {"raster cpu":>12s} {"raster gpu":>12s} {"trace cpu":>12s} {"trace gpu":>12s}')
print('-' * 62)
for ext in ('ply', 'obj', 'glb'):
    src = f'{OUT}/s.{ext}'
    cols = [
        run([src, '-raster', '-device', 'cpu', '-r', '300'], f'{REND}/{ext}_rc.png'),
        run([src, '-raster', '-device', 'gpu', '-r', '300'], f'{REND}/{ext}_rg.png'),
        run([src, '-mode', 'R', '-spp', '48', '-device', 'cpu', '-r', '300'], f'{REND}/{ext}_tc.png'),
        run([src, '-mode', 'R', '-spp', '48', '-device', 'gpu', '-r', '300'], f'{REND}/{ext}_tg.png'),
    ]
    print(f'{ext:8s} ' + ' '.join(f'{fmt(c):>12s}' for c in cols))

# ---- instancing: the same asset through mesh_asset / mesh_instance -------------
scene = os.path.join(OUT, 'inst.ftsl')
with open(scene, 'w') as f:
    f.write(f'''scene {{ units meters spectral 360 830 1 }}
material "white" {{ type diffuse reflect whitewall 0.9 }}
mesh_asset "ball" {{ file "{OUT}/s.ply"  material white }}
mesh_instance {{ of "ball"  translate -1.1 0 0  scale 0.5 0.5 0.5 }}
mesh_instance {{ of "ball"  translate  0.0 0 0  scale 0.5 0.5 0.5 }}
mesh_instance {{ of "ball"  translate  1.1 0 0  scale 0.5 0.5 0.5 }}
light env {{ spd 0.7 }}
camera "cam" {{ eye 0 0.3 4.2  look_at 0 0 0  up 0 1 0  fov_y 40  film {{ res 360 200 }} }}
''')

print()
print('instanced (mesh_asset + mesh_instance), same PLY asset:')
for label, extra in (('raster cpu', ['-raster', '-device', 'cpu']),
                     ('raster gpu', ['-raster', '-device', 'gpu']),
                     ('trace  cpu', ['-mode', 'R', '-spp', '48', '-device', 'cpu']),
                     ('trace  gpu', ['-mode', 'R', '-spp', '48', '-device', 'gpu'])):
    tag = label.replace(' ', '_')
    r = run(['-in', scene, '-r', '360', '200'] + extra, f'{REND}/inst_{tag}.png')
    print(f'  {label}: {fmt(r)}')

if '--keep' in sys.argv:
    print()
    print('assets and renders kept in ' + OUT)
else:
    shutil.rmtree(OUT, ignore_errors=True)
