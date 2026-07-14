"""settle.py — rest an object on a surface for an ftrace scene.

ftrace is a renderer, not a physics engine: an isosurface/mesh sits wherever its
transform puts it, even if that pose is gravitationally impossible (a blob balanced
on a point, an object floating above its stand). This tool computes a physically
plausible resting transform *offline* and prints the exact `translate` / `rotate`
to bake onto the object's block in the .ftsl.

Two modes:

  drop    (default, no physics — needs only numpy + trimesh + scipy)
          Lower the object straight down along -Y, keeping its authored orientation,
          until it first touches the surface. Use this when the object is already
          oriented the way you want and just needs to kiss the surface. It will NOT
          fix an impossible pose — a blob balanced on a point stays balanced, only
          lower.

  settle  (full rigid-body — needs pybullet)
          Drop the object under gravity onto the surface and let it tip over and come
          to rest in a stable pose. This DOES reorient the object. Concave objects are
          convex-decomposed (VHACD) for the collision shape.

The object and surface are OBJ meshes (the surface may instead be a flat floor via
`plane:<y>`). Output is the transform ftrace applies as
    world = translate + Rz(rz)·Ry(ry)·Rx(rx) · vertex          (src/mesh.h)
i.e. Euler XYZ in DEGREES. Bake it onto the object's `mesh { ... }` /
`isosurface { function { ... } }` block, or wrap the block in a `group { }`.

Usage:
  python tools/settle.py --object blob.obj --surface stand.obj
  python tools/settle.py --object blob.obj --surface plane:0.0 --mode drop --gap 0.002
  python tools/settle.py --object blob.obj --surface stand.obj --mode settle --out frag.ftsl

Notes:
  * Straight-down drop assumes ftrace's +Y is up (the convention in the sample scenes).
  * `--yaw D` pre-rotates the object about Y by D degrees before dropping/settling.
"""
import argparse, math, os, sys, tempfile
import numpy as np
import trimesh


# ---------------------------------------------------------------- mesh loading
def load_mesh(path):
    m = trimesh.load(path, force='mesh')
    if m is None or m.is_empty or len(m.vertices) == 0:
        sys.exit(f'could not load a mesh from {path}')
    return m


def parse_surface(spec):
    """Return ('plane', y) or ('mesh', trimesh)."""
    if spec.lower().startswith('plane:'):
        return ('plane', float(spec.split(':', 1)[1]))
    if spec.lower() == 'plane':
        return ('plane', 0.0)
    return ('mesh', load_mesh(spec))


def yaw_matrix(deg):
    """3x3 rotation about +Y (matches ftrace's Ry)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def euler_xyz_deg(R):
    """Rotation matrix -> (rx, ry, rz) degrees such that R = Rz·Ry·Rx.

    scipy's extrinsic 'xyz' Euler decomposition is exactly R = Rz·Ry·Rx, and it
    handles the gimbal-lock branch, so use it when available; fall back to a direct
    closed form otherwise."""
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    except Exception:
        sy = -R[2, 0]
        sy = max(-1.0, min(1.0, sy))
        if abs(sy) < 0.999999:
            rx = math.atan2(R[2, 1], R[2, 2])
            ry = math.asin(sy)
            rz = math.atan2(R[1, 0], R[0, 0])
        else:  # gimbal lock: pitch = ±90°, fold roll into yaw
            ry = math.copysign(math.pi / 2, sy)
            rx = math.atan2(-R[1, 2], R[1, 1])
            rz = 0.0
        return np.degrees([rx, ry, rz])


# ---------------------------------------------------------------- drop mode
def drop(obj, surface, gap):
    """Lower obj (already yawed) straight down until it touches surface.
    Returns (translate xyz, rot 3x3=I)."""
    verts = obj.vertices
    kind, data = surface
    if kind == 'plane':
        y0 = data
        dy = (y0 + gap) - verts[:, 1].min()          # lift bottom to the plane
        return np.array([0.0, dy, 0.0]), np.eye(3)

    S = data
    # Cast a ray straight down from every object vertex; the object can descend by
    # the smallest per-vertex clearance before the tightest vertex meets the surface.
    origins = verts.copy()
    dirs = np.tile([0.0, -1.0, 0.0], (len(origins), 1))
    locs, iray, _ = S.ray.intersects_location(origins, dirs, multiple_hits=True)
    best = {}
    eps = 1e-9
    for loc, ir in zip(locs, iray):
        vy = origins[ir][1]
        if loc[1] <= vy + eps:                        # surface point below this vertex
            best[ir] = max(best.get(ir, -math.inf), loc[1])
    if not best:
        top = S.bounds[1][1]
        sys.stderr.write('[settle] warning: object footprint misses the surface mesh; '
                         'resting on the surface top plane instead.\n')
        dy = (top + gap) - verts[:, 1].min()
        return np.array([0.0, dy, 0.0]), np.eye(3)
    clearance = min(origins[ir][1] - by for ir, by in best.items())
    dy = -clearance + gap
    return np.array([0.0, dy, 0.0]), np.eye(3)


# ---------------------------------------------------------------- settle mode
def settle(obj_path, obj, surface, gap, yaw_deg, max_steps):
    """Full rigid-body drop with PyBullet. Returns (translate xyz, rot 3x3).

    The mesh is centred on its centroid before simulation so PyBullet's reported base
    frame coincides with the centre of mass (uniform density); the pre-centring offset
    is folded back out so the transform still applies to the ORIGINAL mesh vertices."""
    try:
        import pybullet as p
    except ImportError:
        sys.exit("settle mode needs pybullet:  python -m pip install pybullet\n"
                 "(or use --mode drop, which needs no physics engine)")

    c = np.asarray(obj.center_mass if obj.is_watertight else obj.centroid, float)
    centered = obj.copy()
    centered.apply_translation(-c)
    if yaw_deg:
        Ry = np.eye(4); Ry[:3, :3] = yaw_matrix(yaw_deg)
        centered.apply_transform(Ry)

    tmpdir = tempfile.mkdtemp(prefix='settle_')
    cen_path = os.path.join(tmpdir, 'obj_centered.obj')
    vh_path = os.path.join(tmpdir, 'obj_vhacd.obj')
    log_path = os.path.join(tmpdir, 'vhacd.log')
    centered.export(cen_path)

    p.connect(p.DIRECT)
    p.setGravity(0, -9.81, 0)                          # +Y up, matching ftrace scenes

    # --- surface (static) ---
    kind, data = surface
    if kind == 'plane':
        surf_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0])
        p.createMultiBody(0, surf_col, basePosition=[0, data, 0])
        surf_top = data
    else:
        S = data
        surf_path = os.path.join(tmpdir, 'surface.obj')
        S.export(surf_path)
        surf_col = p.createCollisionShape(p.GEOM_MESH, fileName=surf_path,
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        sb = p.createMultiBody(0, surf_col)
        p.changeDynamics(sb, -1, lateralFriction=0.9)
        surf_top = S.bounds[1][1]

    # --- object (dynamic, convex-decomposed) ---
    try:
        p.vhacd(cen_path, vh_path, log_path)
        col_file = vh_path if os.path.exists(vh_path) else cen_path
    except Exception:
        col_file = cen_path                            # fall back to convex hull
    obj_col = p.createCollisionShape(p.GEOM_MESH, fileName=col_file)
    bottom = centered.vertices[:, 1].min()
    start_y = surf_top + 0.05 - bottom                 # start just above the surface
    start = [float(c[0]), float(start_y), float(c[2])]
    body = p.createMultiBody(1.0, obj_col, basePosition=start)
    p.changeDynamics(body, -1, lateralFriction=0.9, spinningFriction=0.02,
                     rollingFriction=0.02, restitution=0.0)

    p.setTimeStep(1.0 / 240.0)
    still = 0
    for _ in range(max_steps):
        p.stepSimulation()
        lv, av = p.getBaseVelocity(body)
        if np.linalg.norm(lv) < 1e-3 and np.linalg.norm(av) < 1e-3:
            still += 1
            if still > 60:                             # ~0.25s at rest -> settled
                break
        else:
            still = 0

    pos, quat = p.getBasePositionAndOrientation(body)   # base frame == centred origin == COM
    R_sim = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    p.disconnect()

    # The collision mesh was  v_work = Ryaw · (v_original - c),  so at rest
    #   world = pos + R_sim · v_work = pos + (R_sim·Ryaw)·(v_original - c).
    # Let R_total = R_sim·Ryaw. Then world = (pos - R_total·c) + R_total·v_original,
    # i.e. ftrace translate = pos - R_total·c, rotate = euler(R_total).
    R_total = R_sim @ yaw_matrix(yaw_deg)
    translate = np.asarray(pos, float) - R_total @ c + gap * np.array([0.0, 1.0, 0.0])
    return translate, R_total


# ---------------------------------------------------------------- output
def fmt(v):
    return ' '.join(f'{x:.5g}' for x in v)


def main():
    ap = argparse.ArgumentParser(description='Rest an object on a surface for an ftrace scene.')
    ap.add_argument('--object', required=True, help='OBJ mesh to drop')
    ap.add_argument('--surface', required=True, help='OBJ mesh, or plane:<y> for a flat floor')
    ap.add_argument('--mode', choices=['drop', 'settle'], default='drop')
    ap.add_argument('--gap', type=float, default=0.0, help='resting clearance above contact (world units)')
    ap.add_argument('--yaw', type=float, default=0.0, help='pre-rotate object about +Y (degrees)')
    ap.add_argument('--max-steps', type=int, default=6000, help='settle: max simulation steps')
    ap.add_argument('--out', help='also write an .ftsl transform fragment to this path')
    a = ap.parse_args()

    obj = load_mesh(a.object)
    surface = parse_surface(a.surface)

    if a.mode == 'drop':
        work = obj.copy()
        if a.yaw:
            Ry = np.eye(4); Ry[:3, :3] = yaw_matrix(a.yaw)
            work.apply_transform(Ry)
        t, R = drop(work, surface, a.gap)
        rot = np.array([0.0, a.yaw, 0.0])
    else:
        t, R = settle(a.object, obj, surface, a.gap, a.yaw, a.max_steps)
        rot = euler_xyz_deg(R)

    print(f'[settle] mode={a.mode}  object={os.path.basename(a.object)}  '
          f'surface={a.surface}')
    print(f'  translate {fmt(t)}')
    print(f'  rotate    {fmt(rot)}')
    frag = ('# settled by tools/settle.py (mode=%s) — bake onto the object block,\n'
            '# or wrap the block in:  group { translate %s  rotate %s  <block> }\n'
            'translate %s\nrotate %s\n' % (a.mode, fmt(t), fmt(rot), fmt(t), fmt(rot)))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(frag)
        print(f'  wrote {a.out}')


if __name__ == '__main__':
    main()
