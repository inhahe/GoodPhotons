"""Generate `meshes/collar_klein.obj` — the shaped mount that holds the Klein bottle upright.

WHY A SHAPED COLLAR, when every simpler mount was tried first

The decisive measurement is about the piece, not the mount. Take the convex hull of the placed
mesh and keep the faces whose supporting plane has the centre of mass over it: those are exactly
the orientations in which the piece can rest on a flat surface. There are 44, and THE MOST
UPRIGHT OF THEM LEANS 73 DEGREES. The Klein bottle has no near-upright equilibrium, so nothing
it merely RESTS on can hold it up. Measured, in order:

  * a circular seat/bore rim -> exactly 2 load-bearing contacts 180 deg apart, a knife edge,
    because the piece's horizontal section swings from 25 mm to 130 mm radius about the pedestal
    axis, so a circle can only touch two lobes;
  * a spherical dish -> a sphere is the one surface on which rolling is free, and it rolled off;
  * a sleeve/cage above a flat cap -> the flank NARROWS downward, so leaning always OPENS
    clearance; the lean runs away 2 -> 6 -> 11 -> 19 -> 35 -> 65 deg and it pivots over the rim;
  * a conforming cradle under the dome -> the underside is a paraboloid rho^2/244 mm, and a
    sphere rolls freely inside its own negative, so tilt is not resisted at all; worse, colliding
    a mesh against a surface coincident with it is numerically pathological (1400-1700 N of
    penetration recovery on a 9.81 N piece, and a lean that wanders and grows);
  * discrete museum-style posts -> a support point needs a near-horizontal surface normal, and
    the underside is only shallow within rho ~ 70 mm, barely past the COM's own 67 mm offset.

What is left is a mount that GRIPS rather than supports. The flank widens upward, so a collar
whose bore is cut to the piece's own outline captures it: the piece cannot sink (the taper jams)
and its weight is carried all round the perimeter instead of on two lobes. Wedging was never the
bug — it is the mechanism; the bug was that a CIRCULAR bore wedges against only two points.

The piece is NOT captive: the bore widens monotonically upward (enforced below), so the bottle
still lifts straight out, which is the rule for a display mount.

MEASURED RESULT (scraps/shaped_collar.py, at settle_scene's own rolling/spinning friction of
5e-4 — earlier sweeps hard-coded 2e-3/2e-2, i.e. 4x and 40x too much, and that alone made every
seat look stable in the harness and fail in the bake):

    band y            clearance   rest drift   lean    worst poke   survivors
    1.030 .. 1.090      0.5 mm       2.1 mm    0.4 deg     72.0 mm   24/36
    1.030 .. 1.090      1.5 mm       4.6 mm    0.9 deg    135.0 mm   24/36
    1.030 .. 1.110      0.5 mm       2.0 mm    0.3 deg     19.4 mm   36/36   <-- chosen

The band has to reach 1.110, not 1.090: the piece escapes by RISING (rising frees radial
clearance at the taper's ~1:1 slope), so the collar needs enough height that rising far enough
to lean costs more than the poke can pay.

The bore is cut to the VHACD PROXY's outline, not to the true mesh's. The proxy's convex hulls
contain the true surface, so it is the wider of the two; a bore cut to the true mesh would bury
the body the simulation actually collides. The cost is a ~5 mm gap between bottle and collar in
the render, which is the proxy's own error — see known-issues.md.

THE BORE MUST BE THE EXACT OUTLINE, NOT A RADIUS PER AZIMUTH

The first version of this generator lofted the bore through r[level, azimuth] samples — the
farthest the outline reached along each of 48 azimuths. That is safe (it can never cut into the
piece) but it fills in every radial concavity, and the section here is strongly non-star-shaped
about the pedestal axis. Measured, the polar bore is bigger than the true 0.5 mm offset by

    y = 1.030   1.08x      y = 1.070   1.36x      y = 1.110   1.20x
    y = 1.050   1.08x      y = 1.090   1.49x

— 114 cm^2 of void at y = 1.090. That is the entire grip: the collar built that way rested at
20.1 deg lean, 95 mm off, and held only 12 of 36 pokes (scraps/validate_collar.py), against
0.3 deg / 2.0 mm / 36 of 36 for the swept slab stack. So the bore is cut from the outline
POLYGON, exactly as the sweep cut it, and the collar is a stack of slabs like the sweep's.

The slabs are unioned into one watertight solid with manifold3d, because settle_scene feeds a
single static mesh per named object. Each slab's bore is the outline at the slab's MID height,
accumulated as a running union up the stack so the bore can never re-narrow — that is what keeps
the mount non-captive, and it means the binding contact is the upward-facing ledge at the top of
whichever slab the piece jams in, a ledge whose plan shape is the piece's own section. That is
where the keying against yaw and sway comes from.

The output is kept under settle_scene's STATIC_TRI_CAP (4000) on purpose: a static collider
above that is quadric-decimated, which would destroy a bore cut to fractions of a millimetre.
The two knobs that buy tris without costing grip are SIMPLIFY_TOL (the outline is offset by
CLEARANCE + tol and then simplified by tol, so the bore still provably contains the CLEARANCE
offset — checked per level, with a union fallback) and OUTER_SEGS (the outer wall is a plain
cylinder, so a 32-gon's 0.8 mm sagitta at 340 mm diameter is invisible).

The three remaining numbers were then swept against the tri cap (scraps/collar_configs.py),
scored on settle_scene's OWN gate — poke_drift > POKE_TOL (10 mm) prints TOPPLES — rather than
on the sweep's looser 30 mm:

    clr mm  tol mm  dy mm  segs   tris   rest dx   lean   worst poke   <30mm   <10mm
      0.50    0.50    4.0    32   3902     5.3    1.11 deg    15.2 mm   36/36   24/36
      0.50    0.25    5.0    32   3534     2.9    0.38 deg    39.7 mm   24/36   24/36
      0.25    0.50    4.0    32   3888     3.5    0.59 deg     1.9 mm   36/36   36/36   <--
      0.25    0.25    4.0    24   4018   over the cap, would be decimated

CLEARANCE dominates, exactly as the sweep predicted, and it beats tol by a wide margin: 0.25 mm
of clearance with a 0.5 mm simplification holds every poke inside 1.9 mm, while 0.5 mm of
clearance with a 0.25 mm simplification wanders 15 mm. Do not go below 0.25 mm, though — at
0.1 mm the contact set gets deep enough that pybullet grinds to a halt resolving it.
"""
import os, sys, argparse, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, trimesh, pybullet as p
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from settle_scene import apply_mesh_xform, cache_dir

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# The gallery's stand_klein pedestal axis and cap top, and the klein block's authored placement.
# Kept here as constants rather than parsed out of the .ftsl because the collar IS part of that
# placement: if the bottle moves, the collar is wrong and must be regenerated anyway.
AX, AZ = 5.9, 2.6
CAP_TOP = 1.00
KLEIN_OBJ = os.path.join(ROOT, 'meshes', 'klein_hunyuan.obj')
KLEIN_XFORM = (np.array([5.8601, 1.3163, 2.6377]), (0.0, 40.0, 0.0), 0.30)

Y_LO, Y_HI = 1.030, 1.110       # the gripping band, swept in scraps/shaped_collar.py
CLEARANCE = 0.00025             # radial gap between bore and the collided proxy
SLAB_DY = 0.004                 # slab thickness, as swept
SIMPLIFY_TOL = 0.0005           # outline simplification; bore still contains the CLEARANCE offset
R_OUT = 0.170                   # collar outer radius
OUTER_SEGS = 32                 # outer-wall segments (0.8 mm sagitta at 340 mm dia: not visible)


def placed_mesh():
    return apply_mesh_xform(trimesh.load(KLEIN_OBJ, force='mesh'), *KLEIN_XFORM)


def proxy_mesh(W, verbose=True):
    """The VHACD proxy in world space — the body the settle actually collides."""
    com = np.asarray(W.center_mass, float)
    out = os.path.join(cache_dir(), 'collar_klein_proxy.obj')
    if not os.path.exists(out):
        tmp = tempfile.mkdtemp(prefix='collar_')
        cen = W.copy(); cen.apply_translation(-com)
        if len(cen.faces) > 40000:
            cen = cen.simplify_quadric_decimation(face_count=40000)
        src = os.path.join(tmp, 'k.obj'); cen.export(src)
        if verbose:
            print('[make_klein_collar] running VHACD (once; cached afterwards)')
        p.connect(p.DIRECT); p.vhacd(src, out, os.path.join(tmp, 'v.log')); p.disconnect()
    m = trimesh.load(out, force='mesh')
    m.apply_translation(com)
    return m


def footprint(mesh, y):
    """The piece's outer footprint at height y as a shapely polygon in world (x, z).

    Built from the section's closed loops, not from Path3D.to_2D(), whose plane frame has an
    arbitrary origin (it once reported every radius ~9 m out). Every loop is FILLED and unioned:
    a section here can be several disjoint loops (the body wall plus the tube passing through
    it), and the collar is cut as disc-minus-this, so missing a loop would leave collar material
    standing inside the bottle."""
    sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if sec is None:
        return None
    polys = []
    for line in sec.discrete:
        q = np.asarray(line)[:, [0, 2]]
        if len(q) >= 4:
            g = Polygon(q).buffer(0)
            if not g.is_empty:
                polys.append(g)
    if not polys:
        return None
    u = unary_union([Polygon(g.exterior) for g in polys])
    return u if u.geom_type == 'Polygon' else unary_union([Polygon(g.exterior) for g in u.geoms])


def bores(mesh, ys, dy, clearance, tol):
    """The bore polygon for each slab: the piece's own outline at the slab's mid height.

    Two properties are guaranteed, in this order of importance:

    1. Each bore CONTAINS the outline offset by `clearance`. The offset is taken at
       clearance + tol and then simplified by tol, which is normally enough on its own (a
       Douglas-Peucker chord deviates by at most tol from the curve it replaces), but the
       simplifier can still shave a sliver off a tight concavity — one 0.12 mm^2 nick at
       y = 1.044 here — so the containment is CHECKED and the offset unioned back in when it
       fails, rather than assumed.
    2. The bores are non-decreasing going up, because each is unioned with everything below it.
       That is what keeps the mount non-captive: the bore never re-narrows above any height, so
       the piece still lifts straight out.

    The running union is SEEDED with every section from the piece's own base up to the first
    slab, because build() drops that first slab all the way to the cap (see there). It happens
    to change nothing here — the outline at the first slab's mid height already contains all
    0.0 mm^2 of what is below it — but seeding makes the skirt safe by construction instead of
    by luck, so the skirt cannot start clipping the piece if the placement is ever nudged."""
    out, acc, patched = [], None, 0
    for y in np.arange(mesh.bounds[0][1] + 1e-4, ys[0] + 0.5 * dy, 0.001):
        fp = footprint(mesh, y)
        if fp is not None:
            acc = fp if acc is None else unary_union([acc, fp])
    for y in ys:
        fp = footprint(mesh, y + 0.5 * dy)
        if fp is None:
            raise SystemExit(f'[make_klein_collar] no section at y={y + 0.5 * dy:.4f}')
        acc = fp if acc is None else unary_union([acc, fp])
        core = acc.buffer(clearance)
        b = acc.buffer(clearance + tol, join_style='mitre', mitre_limit=6).simplify(tol)
        if not b.contains(core):
            b = unary_union([b, core])
            patched += 1
        out.append(b)
    return out, patched


def build(bore_polys, ys, dy):
    """Stack the slabs — an R_OUT disc with that slab's bore punched out — into one solid.

    They are unioned rather than left separate (the sweep collided them as separate static
    bodies) because settle_scene takes ONE static mesh per named scene object. manifold3d does
    the union; without it trimesh has no boolean engine and this raises.

    The bottom slab is dropped all the way to CAP_TOP so the collar SITS on the pedestal cap
    instead of floating 30 mm above it. That is free — it is the same prism, just taller, so it
    costs no triangles — and it is safe because bores() seeds its running union with every
    section below the first slab, so this bore already contains the piece everywhere it now
    spans. It is also physically inert: the piece's base is at y=1.017 and the bore down here
    is 95 cm^2 around a section that is a fraction of that, so the skirt never touches it."""
    disc = Point(AX, AZ).buffer(R_OUT, quad_segs=OUTER_SEGS // 4)
    slabs = []
    for i, (y, b) in enumerate(zip(ys, bore_polys)):
        y0 = CAP_TOP if i == 0 else y
        g = trimesh.creation.extrude_polygon(disc.difference(b), y + dy - y0)
        # extrude_polygon builds the polygon in x-y and extrudes along +z, so stand it up with
        # +90 deg about x: (x, y, z) -> (x, -z, y), i.e. polygon-y becomes world z and the
        # extrusion becomes -dy..0 in world y. Rotating the OTHER way maps polygon-y to -world z,
        # which silently mirrors the collar to z = -2.6 and the piece falls straight through.
        g.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        g.apply_translation([0.0, y + dy, 0.0])
        slabs.append(g)
    g = trimesh.boolean.union(slabs)
    c = 0.5 * (g.bounds[0] + g.bounds[1])
    assert abs(c[0] - AX) < 0.02 and abs(c[2] - AZ) < 0.02, \
        f'collar is not on the pedestal axis: centre {c} vs ({AX}, -, {AZ})'
    return g


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-o', default=os.path.join(ROOT, 'meshes', 'collar_klein.obj'))
    ap.add_argument('--clearance', type=float, default=CLEARANCE)
    ap.add_argument('--dy', type=float, default=SLAB_DY)
    ap.add_argument('--tol', type=float, default=SIMPLIFY_TOL)
    a = ap.parse_args()
    W = placed_mesh()
    P = proxy_mesh(W)
    ys = np.arange(Y_LO, Y_HI - 1e-9, a.dy)
    bp, patched = bores(P, ys, a.dy, a.clearance, a.tol)
    g = build(bp, ys, a.dy)
    g.export(a.o)
    area = [b.area * 1e4 for b in bp]
    print(f'[make_klein_collar] {len(ys)} slabs of {a.dy*1000:.1f} mm gripping over y '
          f'{Y_LO:.3f}..{Y_HI:.3f} (the bottom one skirted down to the cap at y {CAP_TOP:.3f})'
          f'; bore {area[0]:.1f}..{area[-1]:.1f} cm2, clearance {a.clearance*1000:.2f} mm, '
          f'simplified by {a.tol*1000:.2f} mm ({patched} levels needed the union fallback)')
    print(f'[make_klein_collar] wrote {a.o}: {len(g.faces)} tris, watertight={g.is_watertight}, '
          f'volume {g.volume*1e3:.2f} L, y {g.bounds[0][1]:.4f}..{g.bounds[1][1]:.4f}')
    if len(g.faces) > 4000:
        print('[make_klein_collar] WARNING: over settle_scene STATIC_TRI_CAP; it will be '
              'decimated and the bore will be destroyed')
    # Overlap check, at the AUTHORED pose, against both bodies.
    #
    # The rendered mesh must come out clean: the bore is cut to the (wider) proxy, so the true
    # surface should sit inside the bore everywhere and the render must never show slate cutting
    # through glass.
    #
    # The proxy is EXPECTED to overlap, and by up to about dy/2 of rise. Each slab's bore is the
    # outline at the slab's MID height, so the proxy's own material in the upper half of a slab
    # is inside the collar until the piece sinks — which is the whole mechanism, and settle_scene
    # resolves it in the first few steps. What matters is that the overlap stays SHALLOW: a deep
    # one is a violent spawn, so print the depth rather than just a count.
    for name, m in (('rendered mesh', W), ('collision proxy', P)):
        # from CAP_TOP, not Y_LO: build() drops the bottom slab to the cap, so the skirt is
        # part of what can clip the piece and has to be checked too
        band = m.slice_plane([0, CAP_TOP, 0], [0, 1, 0]).slice_plane([0, Y_HI, 0], [0, -1, 0])
        q = np.asarray(band.vertices)
        inside = g.contains(q) if len(q) else np.zeros(0, bool)
        depth = (g.nearest.on_surface(q[inside])[1].max() * 1000.0) if inside.any() else 0.0
        print(f'[make_klein_collar] {name}: {len(q)} verts in the band, {int(inside.sum())} '
              f'inside the collar, deepest {depth:.2f} mm')

