"""settle_scene.py — settle several named objects in one ftrace scene at once.

Where tools/settle.py rests a *single* object on a surface, this tool takes a whole
.ftsl scene, runs ONE rigid-body simulation containing many of its objects at once,
and lets them fall onto the floor **and onto each other** (mutual object-on-object
collisions). It then writes a new .ftsl in which each settled object is wrapped in a
`group { translate … rotate … <original block> }` carrying the pose it came to rest
in. Nothing about the original blocks is edited — the group just applies the extra
rigid motion on top of whatever transform the block already had, so re-running is safe
and the field/mesh definitions stay untouched.

Which objects move:
  --all              settle every named `mesh`/`isosurface` object in the scene
  --settle a,b,c     settle only these (by their ftsl block name)
Objects that are NOT selected still take part in the sim as *static* colliders, so a
selected object can come to rest leaning on or stacked atop a fixed one (`--onto
object`), as well as on the floor.

The floor is a horizontal plane. By default it is y=0 (the platform top in the sample
scenes); override with `--floor plane:<y>`.

Object geometry:
  * `mesh { file "…" … }`  — the OBJ is loaded and the block's authored
    translate/rotate/scale (world = translate + Rz·Ry·Rx·(scale⊙local), Euler XYZ in
    DEGREES, per src/mesh.h) is applied to get its start-of-sim world pose.
  * `isosurface "name" { … }` — polygonised to a world-space mesh by shelling out to
    `ftrace -export-mesh` (whose OBJ groups are named after the block, so we can match
    each group back to its object). Needs the built ftrace binary.

Requirements: numpy, trimesh, and pybullet (for the physics). VHACD (bundled with
pybullet) convex-decomposes concave dynamic objects for a faithful collision shape.

Usage:
  python tools/settle_scene.py --scene gallery.ftsl --all --out gallery_settled.ftsl
  python tools/settle_scene.py --scene s.ftsl --settle blobA,klein --floor plane:0.0 --out s2.ftsl
"""
import argparse, math, os, re, sys, tempfile, subprocess, glob
import numpy as np
import trimesh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from settle import euler_xyz_deg  # R -> (rx,ry,rz) deg such that R = Rz·Ry·Rx

# Cap on the triangle count of a dynamic object's VHACD collision proxy. Convex
# decomposition time scales with tri count and only the gross shape matters for
# resting a rigid body, so meshes above this are quadric-decimated first (needs the
# `fast_simplification` package; falls back to the full mesh if unavailable).
COLLISION_TRI_CAP = 40000


# ---------------------------------------------------------------- ftsl parsing
def strip_comments(text):
    """Blank out `#` line comments (keeping length + newlines) so brace/offset math
    over the original string stays valid while comments can't hide braces."""
    out = []
    for line in text.splitlines(keepends=True):
        h = line.find('#')
        if h >= 0:
            nl = '\n' if line.endswith('\n') else ''
            out.append(line[:h] + ' ' * (len(line) - h - len(nl)) + nl)
        else:
            out.append(line)
    return ''.join(out)


def find_top_blocks(text):
    """Yield top-level blocks as dicts: {keyword, name, start, body_start, body_end,
    end}. `start`/`end` bracket the whole `keyword [\"name\"] { … }` in the ORIGINAL
    text; body_* bracket the inside of the braces."""
    scan = strip_comments(text)
    blocks = []
    i, n = 0, len(scan)
    # match a block header:  keyword  optional "name"  {
    hdr = re.compile(r'([A-Za-z_][\w]*)\s*(?:"([^"]*)")?\s*\{')
    while i < n:
        m = hdr.match(scan, i)
        if not m:
            i += 1
            continue
        # only accept if we're at top level (previous non-space char is not inside a block —
        # guaranteed here because we advance i past whole top-level blocks below)
        body_start = m.end()
        depth = 1
        j = body_start
        while j < n and depth:
            c = scan[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            j += 1
        blocks.append({
            'keyword': m.group(1), 'name': m.group(2) or '',
            'start': m.start(), 'body_start': body_start,
            'body_end': j - 1, 'end': j,
        })
        i = j
    return blocks


def resolve_path(raw, scene_dir):
    """Resolve an ftsl `file` path. ftrace resolves relative paths against the repo
    root (its CWD when rendering), but authors may also mean them relative to the
    scene file. Try both (and absolute); return the first that exists, else None."""
    if os.path.isabs(raw):
        return raw if os.path.exists(raw) else None
    repo_root = os.path.abspath(os.path.join(_HERE, '..'))
    for cand in (os.path.join(repo_root, raw),
                 os.path.join(scene_dir, raw),
                 os.path.join(os.getcwd(), raw)):
        if os.path.exists(cand):
            return cand
    return None


def _floats(s):
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]


def parse_mesh_xform(body):
    """Pull top-level translate/rotate/scale/file from a mesh block body."""
    def line_after(key):
        m = re.search(r'\b' + key + r'\b([^\n{}]*)', body)
        return m.group(1) if m else None
    file_m = re.search(r'\bfile\s+"([^"]+)"', body)
    tr = _floats(line_after('translate') or '')
    ro = _floats(line_after('rotate') or '')
    scw = line_after('scale')
    sc = _floats(scw) if scw else []
    translate = np.array((tr + [0, 0, 0])[:3], float)
    rotate = np.array((ro + [0, 0, 0])[:3], float)
    if len(sc) == 0:
        scale = np.array([1.0, 1.0, 1.0])
    elif len(sc) == 1:
        scale = np.array([sc[0]] * 3, float)
    else:
        scale = np.array(sc[:3], float)
    return (file_m.group(1) if file_m else None), translate, rotate, scale


# ---------------------------------------------------------------- transforms
def rot_xyz(rx, ry, rz):
    """R = Rz·Ry·Rx from Euler degrees, matching src/mesh.h."""
    ax, ay, az = map(math.radians, (rx, ry, rz))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_mesh_xform(mesh, translate, rotate, scale):
    """Return a copy in world space per world = translate + Rz·Ry·Rx·(scale⊙local)."""
    w = mesh.copy()
    V = w.vertices * scale                       # component-wise scale
    V = V @ rot_xyz(*rotate).T                   # rotate
    V = V + translate                            # translate
    w.vertices = V
    return w


# ---------------------------------------------------------------- isosurface meshing
def find_ftrace():
    for c in (os.path.join(_HERE, '..', 'build_cuda', 'bin', 'ftrace.exe'),
              os.path.join(_HERE, '..', 'build', 'bin', 'ftrace.exe'),
              os.path.join(_HERE, '..', 'build_cuda', 'bin', 'ftrace')):
        if os.path.exists(c):
            return os.path.abspath(c)
    hits = glob.glob(os.path.join(_HERE, '..', '**', 'ftrace*'), recursive=True)
    hits = [h for h in hits if os.path.isfile(h) and os.access(h, os.X_OK)]
    return os.path.abspath(hits[0]) if hits else None


def export_isosurface_meshes(scene_path, res):
    """Run `ftrace -export-mesh` and return {group_name: trimesh} in world space."""
    exe = find_ftrace()
    if not exe:
        sys.exit('[settle_scene] could not find the ftrace binary to polygonise isosurfaces')
    tmp = os.path.join(tempfile.mkdtemp(prefix='settlescene_'), 'iso.obj')
    cmd = [exe, '-in', scene_path, '-export-mesh', tmp, '-mesh-res', str(res)]
    print('[settle_scene] polygonising isosurfaces:', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(tmp):
        sys.stderr.write(r.stdout + '\n' + r.stderr + '\n')
        sys.exit('[settle_scene] -export-mesh produced no file')
    # `-export-mesh` writes one `o <block-name>` object per isosurface, with GLOBAL
    # cumulative 1-based vertex indices in `f a//na …` faces. Parse the `o` groups
    # DIRECTLY rather than via `trimesh.load(force='scene')`: without `usemtl` lines
    # (marching-cubes output has none), some trimesh versions silently merge every `o`
    # group into a single Trimesh, collapsing all names to the first — which breaks the
    # match back to the ftsl blocks (only the first object survives). A manual parse is
    # robust regardless of trimesh's grouping heuristics.
    return parse_obj_groups(tmp)


def parse_obj_groups(path):
    """Parse a Wavefront OBJ into {object-name: trimesh} by its `o` groups. Vertices are
    global/cumulative; each group's faces (0-based into the shared vertex pool) are
    remapped onto just the vertices that group references, so bounds/centroid are exact."""
    verts = []            # all `v` positions (global, 0-based after read)
    groups = []           # list of (name, [ (i,j,k) 0-based tris ])
    cur = None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('v '):
                verts.append([float(t) for t in ln.split()[1:4]])
            elif ln.startswith('o '):
                name = ln.split(None, 1)[1].strip() if len(ln.split(None, 1)) > 1 else f'obj{len(groups)}'
                cur = (name, [])
                groups.append(cur)
            elif ln.startswith('f '):
                if cur is None:
                    cur = ('obj0', [])
                    groups.append(cur)
                # each token is `v`, `v/vt`, or `v//vn`; take the vertex index (1-based)
                idx = [int(tok.split('/', 1)[0]) - 1 for tok in ln.split()[1:]]
                # fan-triangulate any n-gon (marching cubes emits tris, but be safe)
                for t in range(1, len(idx) - 1):
                    cur[1].append((idx[0], idx[t], idx[t + 1]))
    V = np.asarray(verts, float)
    out = {}
    for name, tris in groups:
        if not tris:
            continue
        F = np.asarray(tris, np.int64)
        used = np.unique(F)
        remap = {g: i for i, g in enumerate(used)}
        localF = np.vectorize(remap.__getitem__)(F)
        out[name] = trimesh.Trimesh(vertices=V[used], faces=localF, process=False)
    return out


# ---------------------------------------------------------------- physics
def settle_bodies(worlds, selected, floor_y, max_steps, friction):
    """worlds: {name: trimesh in world space}. selected: list of names to make dynamic.
    Returns {name: (translate xyz, R 3x3)} — the extra rigid transform each selected
    object must be wrapped in (applied ON TOP of its authored world pose)."""
    try:
        import pybullet as p
    except ImportError:
        sys.exit('settle_scene needs pybullet:  python -m pip install pybullet')

    tmpdir = tempfile.mkdtemp(prefix='settlescene_sim_')
    p.connect(p.DIRECT)
    p.setGravity(0, -9.81, 0)
    p.setPhysicsEngineParameter(numSolverIterations=80)

    # floor plane
    floor_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0])
    fb = p.createMultiBody(0, floor_col, basePosition=[0, floor_y, 0])
    p.changeDynamics(fb, -1, lateralFriction=friction)

    dyn = {}   # name -> (body_id, com c)
    for name, mesh in worlds.items():
        path = os.path.join(tmpdir, re.sub(r'\W+', '_', name) + '.obj')
        if name in selected:
            # dynamic: center on COM, convex-decompose, spawn at COM so v_work=v_world-c.
            # center_mass needs a clean watertight volume; a repaired art mesh can have
            # degenerate faces / inconsistent winding that make it NaN, so fall back to
            # the vertex centroid and finally the bbox centre — any finite interior-ish
            # point works (it's only the spawn reference for the rigid delta).
            c = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid, float)
            if not np.all(np.isfinite(c)):
                c = np.asarray(mesh.centroid, float)
            if not np.all(np.isfinite(c)):
                c = mesh.bounds.mean(axis=0)
            cen = mesh.copy(); cen.apply_translation(-c)
            # VHACD's cost scales with triangle count and it only needs the gross shape,
            # so cap the collision proxy (a fine art mesh can be 100s of k tris, which
            # makes convex decomposition take many minutes). The visual mesh in the scene
            # is untouched — this proxy is thrown away after the pose delta is computed.
            if len(cen.faces) > COLLISION_TRI_CAP:
                try:
                    cen = cen.simplify_quadric_decimation(face_count=COLLISION_TRI_CAP)
                    print(f'[settle_scene] decimated "{name}" collision proxy to '
                          f'{len(cen.faces)} tris (from {len(mesh.faces)})')
                except Exception as e:
                    print(f'[settle_scene] proxy decimation of "{name}" failed ({e}); '
                          'using full-resolution mesh (VHACD may be slow)')
            cen.export(path)
            vh = path[:-4] + '_vhacd.obj'
            try:
                p.vhacd(path, vh, os.path.join(tmpdir, 'vhacd.log'))
                col_file = vh if os.path.exists(vh) else path
            except Exception:
                col_file = path
            col = p.createCollisionShape(p.GEOM_MESH, fileName=col_file)
            body = p.createMultiBody(1.0, col, basePosition=[float(c[0]), float(c[1]), float(c[2])])
            p.changeDynamics(body, -1, lateralFriction=friction, spinningFriction=0.02,
                             rollingFriction=0.02, restitution=0.0)
            dyn[name] = (body, c)
        else:
            # static concave collider at its authored world pose (mesh already world-space)
            mesh.export(path)
            col = p.createCollisionShape(p.GEOM_MESH, fileName=path,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            sb = p.createMultiBody(0, col)
            p.changeDynamics(sb, -1, lateralFriction=friction)

    p.setTimeStep(1.0 / 240.0)
    still = 0
    for _ in range(max_steps):
        p.stepSimulation()
        moving = False
        for body, _c in dyn.values():
            lv, av = p.getBaseVelocity(body)
            if np.linalg.norm(lv) > 1e-3 or np.linalg.norm(av) > 1e-3:
                moving = True
                break
        still = 0 if moving else still + 1
        if still > 90:                            # ~0.4s at rest -> settled
            break

    result = {}
    for name, (body, c) in dyn.items():
        pos, quat = p.getBasePositionAndOrientation(body)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # final = pos + R·(v_world - c) = (pos - R·c) + R·v_world  -> delta on authored pose
        translate = np.asarray(pos, float) - R @ c
        result[name] = (translate, R)
    p.disconnect()
    return result


# ---------------------------------------------------------------- rewrite
def fmt(v):
    return ' '.join(f'{x:.6g}' for x in v)


def wrap_blocks(text, blocks_by_name, deltas):
    """Wrap each settled block in a group{} carrying its delta transform. Process from
    the last block to the first so earlier character offsets stay valid."""
    items = [(blocks_by_name[n], n) for n in deltas]
    items.sort(key=lambda it: it[0]['start'], reverse=True)
    for blk, name in items:
        t, R = deltas[name]
        rot = euler_xyz_deg(R)
        orig = text[blk['start']:blk['end']]
        indented = '\n'.join('    ' + ln if ln.strip() else ln for ln in orig.splitlines())
        wrapped = (f'group {{  # settled by tools/settle_scene.py\n'
                   f'    translate {fmt(t)}\n'
                   f'    rotate    {fmt(rot)}\n'
                   f'{indented}\n'
                   f'}}')
        text = text[:blk['start']] + wrapped + text[blk['end']:]
    return text


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Settle several named objects in one ftrace scene at once.')
    ap.add_argument('--scene', required=True, help='input .ftsl scene')
    ap.add_argument('--out', required=True, help='output .ftsl to write')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--all', action='store_true', help='settle every named mesh/isosurface object')
    g.add_argument('--settle', help='comma-separated object names to settle')
    ap.add_argument('--floor', default='plane:0.0', help='floor surface: plane:<y> (default plane:0.0)')
    ap.add_argument('--mesh-res', type=int, default=160, help='isosurface polygonisation resolution')
    ap.add_argument('--friction', type=float, default=0.8, help='lateral friction for all bodies')
    ap.add_argument('--max-steps', type=int, default=8000, help='max simulation steps')
    a = ap.parse_args()

    if not a.floor.lower().startswith('plane:'):
        sys.exit('[settle_scene] --floor must be plane:<y>')
    floor_y = float(a.floor.split(':', 1)[1])

    text = open(a.scene, encoding='utf-8').read()
    blocks = find_top_blocks(text)
    objs = [b for b in blocks if b['keyword'] in ('mesh', 'isosurface') and b['name']]
    if not objs:
        sys.exit('[settle_scene] no named mesh/isosurface objects found in the scene')
    by_name = {b['name']: b for b in objs}

    if a.all:
        selected = [b['name'] for b in objs]
    else:
        selected = [s.strip() for s in a.settle.split(',') if s.strip()]
        missing = [s for s in selected if s not in by_name]
        if missing:
            sys.exit(f'[settle_scene] unknown object name(s): {", ".join(missing)}\n'
                     f'  available: {", ".join(by_name)}')

    scene_dir = os.path.dirname(os.path.abspath(a.scene))

    # build world-space meshes for every named object (settled + static colliders)
    worlds = {}
    need_iso = any(by_name[n]['keyword'] == 'isosurface' for n in by_name)
    iso_meshes = export_isosurface_meshes(a.scene, a.mesh_res) if need_iso else {}
    for name, blk in by_name.items():
        if blk['keyword'] == 'mesh':
            body = text[blk['body_start']:blk['body_end']]
            fpath, translate, rotate, scale = parse_mesh_xform(body)
            if not fpath:
                sys.exit(f'[settle_scene] mesh "{name}" has no file')
            fpath = resolve_path(fpath, scene_dir)
            if not fpath:
                sys.exit(f'[settle_scene] mesh "{name}": cannot find file')
            m = trimesh.load(fpath, force='mesh')
            worlds[name] = apply_mesh_xform(m, translate, rotate, scale)
        else:
            if name not in iso_meshes:
                sys.exit(f'[settle_scene] isosurface "{name}" not found in -export-mesh output '
                         f'(groups: {", ".join(iso_meshes) or "none"})')
            worlds[name] = iso_meshes[name]

    print(f'[settle_scene] objects: {", ".join(by_name)}')
    print(f'[settle_scene] settling: {", ".join(selected)}  (others are static colliders)')
    print(f'[settle_scene] floor: y={floor_y}')

    deltas = settle_bodies(worlds, set(selected), floor_y, a.max_steps, a.friction)

    for name in selected:
        t, R = deltas[name]
        print(f'  {name}:  translate {fmt(t)}   rotate {fmt(euler_xyz_deg(R))}')

    out_text = wrap_blocks(text, by_name, deltas)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write(out_text)
    print(f'[settle_scene] wrote {a.out}')


if __name__ == '__main__':
    main()
