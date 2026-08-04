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

Every settled piece is checked THREE ways before the pose is written, because equilibrium is
not stability and stability is not correctness:
  * its COM must project inside the convex hull of its load-bearing contacts (else PERCHED),
  * it must survive a poke — a small random shove + spin — without moving (else TOPPLES),
  * and it must end up resting ON TOP OF something the author placed it over (else FELL).
The third one is not redundant: the first two are LOCAL tests, and a piece that slid off its
cap, dropped a metre and wedged between two pedestal shafts passes both of them with healthy
numbers, because it really is immovably at rest down there. Only a test against the AUTHORED
scene knows the author didn't put it on the floor. The tool prints a per-piece verdict of
OK / FELL / PERCHED / TOPPLES.
A piece that reports TOPPLES on every retry has no stable rest pose at all and its GEOMETRY
is what needs changing — no amount of simulation can invent a rest that doesn't exist. One
that reports FELL on every retry needs either a different authored pose or a MOUNT that grips
it — a collar whose bore is cut to the piece's own cross-section. A mount is the LAST resort,
not the first: the gallery's Klein bottle needed one only for as long as the mesh was a shape
with no upright equilibrium, and swapping in a bottle with a real punted foot retired both the
mount and its generator (see design.md).

Keeping pieces over their pedestals — two strategies for the same failure:
  A faithful free settle drops each piece onto NARROW pedestals, so anything wider than
  its column (or authored slightly off-centre over it) tends to tip and roll OFF onto the
  floor. Two ways to prevent that:

  --tether [k]   (during-sim, PHYSICAL)  A horizontal restoring spring applied AT each
    body's COM every step, pulling it back toward its authored XZ. Because it acts at the
    COM it exerts no torque, so the piece is free to tip/rotate onto its cap while being
    stopped from walking off it sideways. The final pose is a genuine physics rest pose —
    just one that stayed home. Bare `--tether` = k 150 N/m; raise k for stiffer holding,
    lower it for pieces that should be free to drift a little. This is the preferred fix
    (settles the gallery correctly in one pass — all pieces land on their stands).

  --jitter/--seed (during-sim, PHYSICAL)  A small random spawn tilt, so a symmetric piece
    cannot come to rest in an unstable equilibrium the solver has no reason to leave. Any
    piece that is still not stably resting is automatically re-thrown with a fresh draw.

  --seat         (post-hoc, GEOMETRIC)  Runs a free settle, then keeps the ORIENTATION
    each piece came to rest in but returns it to the exact spot the author placed it and
    lowers it straight down onto its stand. Faster but the pose can look stiff (it's the
    orientation from wherever the piece ended up, not one settled in place). Pass
    `--seat auto` to pair each piece with the nearest other named object (its pedestal),
    or explicit `piece:stand,…` pairs. Overhang past the rim prints a warning.

Usage:
  python tools/settle_scene.py --scene gallery.ftsl --all --out gallery_settled.ftsl
  python tools/settle_scene.py --scene s.ftsl --settle blobA,klein --floor plane:0.0 --out s2.ftsl
  python tools/settle_scene.py --scene g.ftsl --settle klein,heart --tether --out g_tethered.ftsl
  python tools/settle_scene.py --scene g.ftsl --settle klein,heart --seat auto --out g_seated.ftsl
"""
import argparse, hashlib, math, os, re, shutil, sys, tempfile, subprocess, glob, time
import numpy as np
import trimesh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from settle import euler_xyz_deg, drop  # euler decomp; drop = vertical rest-on-surface

# Cap on the triangle count of a dynamic object's VHACD collision proxy. Convex
# decomposition time scales with tri count and only the gross shape matters for
# resting a rigid body, so meshes above this are quadric-decimated first (needs the
# `fast_simplification` package; falls back to the full mesh if unavailable).
COLLISION_TRI_CAP = 40000

# Cap on a STATIC concave collider's triangle count. Unlike the dynamic proxies these
# aren't convex-decomposed (they stay concave via GEOM_FORCE_CONCAVE_TRIMESH), but their
# tri count still sets the per-step collision cost against every settling body — an
# un-capped res-160 isosurface stand can make each sim step take ~100 ms. Only the gross
# resting surface matters, so decimate above this. Museum stands are simple box-unions, so
# a few thousand tris capture the flat resting top exactly while keeping steps cheap.
STATIC_TRI_CAP = 4000

# Horizontal slabs used by slab_sections() / slab_hulls() for a static collider that will NOT
# decimate to STATIC_TRI_CAP (the box-union pedestals). Each slab is a stair-step of constant
# cross-section, so a horizontal FEATURE is only resolved if it is thicker than one slab —
# which is why the slab count is derived from a target THICKNESS rather than fixed. A fixed 32
# slabs is 32 mm on a 1 m pedestal, and that quantised stand_klein's cradle collar into a
# 32 mm dimple whose floor sat 3 mm above the real cap, leaving the bottle nothing to seat
# against. 8 mm resolves every feature the gallery stands have (the thinnest is a 30 mm cap
# plate) at ~10k tris per stand; the cap keeps a tall stand from exploding.
STATIC_SLAB_MAX_T = 0.008
STATIC_SLABS_MAX = 192


def slab_count(mesh, max_t=STATIC_SLAB_MAX_T, cap=STATIC_SLABS_MAX):
    """Slabs to cut `mesh` into so no slab is thicker than `max_t` (bounded by `cap`)."""
    h = float(mesh.bounds[1][1] - mesh.bounds[0][1])
    return int(max(4, min(cap, math.ceil(h / max(1e-9, max_t)))))

# Tolerance (metres) used to simplify each slab's cross-section outline in slab_sections().
# An outline traced around marching-cubes output carries thousands of near-collinear
# vertices; extruding them verbatim costs ~83k tris per stand, versus ~2400 at 0.5 mm. Well
# below any feature that can matter to a resting contact.
SECTION_SIMPLIFY = 0.0005

# Radius (metres) of the --tether spring's dead zone around each piece's authored XZ
# anchor. Inside it the spring is OFF so the piece settles naturally; outside it the spring
# engages on the excess to stop walk-off.
TETHER_DEADBAND = 0.03

# Per-step motion (metres moved + orientation change) below which a body counts as still
# for the displacement-based settled test.
SETTLE_STILL_EPS = 2.0e-4

# Steps over which the tether spring is ramped linearly to zero before the pose is read
# back. The tether is a FICTITIOUS body force, so a pose recorded while it is still active
# can be one gravity alone would not hold (a piece resting on the corner of its cap with
# its COM out over empty air — see known-issues.md). Releasing it and re-settling makes the
# recorded rest purely gravitational. 480 steps = 2 s at the 240 Hz timestep: slow enough
# that the release doesn't kick the piece, fast enough to be free before the relax budget.
RELAX_RAMP_STEPS = 480

# Fraction of a body's TOTAL normal impulse a single contact must carry to count as
# load-bearing for the support-polygon test. This has to be relative, not an absolute force:
# a VHACD-decomposed piece resting on a concave trimesh generates one manifold per
# convex-child/triangle pair, so the body's weight is split across dozens of points and each
# one's share shrinks as the proxy gets finer. An absolute threshold therefore rejects every
# genuine resting contact on a well-decomposed body while accepting them on a coarse one —
# which is exactly how `oiljack` came to report "contacts 0, resting on nothing" while
# sitting squarely on its stand.
CONTACT_FORCE_FRAC = 0.01

# Rolling / spinning friction for the settling bodies. Bullet's rollingFriction is a
# resistance ARM IN METRES: it caps the resistive torque at mu_r * normalForce, so a body of
# radius R cannot tip past asin(mu_r / R). The old value of 0.02 (= 2 cm!) pinned
# `brass_dumbbell` — a wheel of world radius 0.117 — upright on its rim below 9.8 deg, so the
# ring balance the user reported was being held by a fictitious torque rather than by
# geometry. Real rolling resistance for metal on stone is a fraction of a millimetre.
ROLLING_FRICTION = 5e-4
SPINNING_FRICTION = 5e-4

# How far OUTSIDE its support polygon a settled piece's COM may sit before the pose is
# rejected outright. This is a tolerance, not a required inset: a smooth body genuinely
# touches at a point (a ball on a table) or along a line (a rolling pin), so demanding the
# COM sit strictly *inside* a polygon would fail every curved piece in the scene. What it
# still catches is the gross failure — `oiljack` resting on the corner of its cap with the
# COM 119 mm out over empty air.
STABILITY_MARGIN = -0.010

# The support-polygon test only proves the piece is in EQUILIBRIUM, not that the equilibrium
# is STABLE: a wheel balanced on its rim has its COM exactly over the contact and passes.
# So each settled piece is also poked — given a small random shove and spin — and re-settled.
# A stable rest absorbs the poke and returns to essentially the same place; an unstable one
# topples. This is the criterion that actually matches "does it look settled in the render".
POKE_SPEED = 0.03      # m/s linear kick
POKE_SPIN = 0.30       # rad/s angular kick
POKE_TOL = 0.010       # a stable piece moves less than this (m) in response

# NEITHER of the two tests above notices that a piece landed somewhere else entirely. They
# are both LOCAL: "is the COM over the contacts it actually has" and "does that rest survive
# a shove". A piece that slides off its cap, drops a metre and wedges between two pedestal
# shafts passes both — it is genuinely, immovably at rest down there. `heart` did exactly
# that and was reported `OK ... on stand_dumbbell, stand_heart`, which reads like a success
# and is a total failure: the author put it 30 mm above stand_heart's cap and the bake buried
# it on the floor, where it happened to touch two shafts on the way down.
#
# So the bake also asks the only question the sim cannot answer by itself — did the piece end
# up on the thing the AUTHOR put it over? That intent lives in the authored scene (see
# intended_supports()), not in the physics. Note the test is deliberately NOT "how far did
# the COM move": a piece that honestly tips from its authored tilt onto a stable face of its
# own cap moves its COM by 100+ mm and is completely correct (`heart` under --tether does
# exactly this), while a piece can slide clean off a narrow cap having moved much less.

# How many times a piece that came to rest PERCHED (COM outside its support polygon) is
# re-thrown with a fresh random perturbation before we give up on it. A symmetric body can
# land in an unstable equilibrium the solver has no reason to leave — `brass_dumbbell` is a
# wheel whose ring rim is the only part that reaches its cap, and it will happily balance
# on that rim forever if the perturbation happened to be about its own axis of symmetry
# (a rotation that maps the body onto itself, so it breaks nothing). Re-throwing only the
# perched pieces, in the already-built world, is far cheaper than rebuilding VHACD proxies.
SETTLE_ATTEMPTS = 4


# ------------------------------------------------------ static-stability (support polygon)
def convex_hull_2d(pts):
    """Andrew's monotone chain. Returns the hull as a CCW list of (x, z) points."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return list(pts)
    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out
    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def support_margin(hull, pt):
    """Signed distance from `pt` to the boundary of convex polygon `hull` (CCW):
    positive inside, negative outside. A degenerate hull (a point or a line — which is
    exactly what a piece balanced on a rim or a single corner produces) is never
    stable, so it reports 0 or the negative distance to that feature."""
    x, z = pt
    if len(hull) < 3:
        if not hull:
            return -float('inf')
        if len(hull) == 1:
            return -math.hypot(x - hull[0][0], z - hull[0][1])
        (ax, az), (bx, bz) = hull                       # a segment: no width to stand on
        ex, ez = bx - ax, bz - az
        L2 = ex * ex + ez * ez
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * ex + (z - az) * ez) / L2))
        return -math.hypot(x - (ax + t * ex), z - (az + t * ez))
    inside = True
    best = float('inf')
    n = len(hull)
    for i in range(n):
        ax, az = hull[i]
        bx, bz = hull[(i + 1) % n]
        ex, ez = bx - ax, bz - az
        if ex * (z - az) - ez * (x - ax) < 0.0:         # right of a CCW edge -> outside
            inside = False
        L = math.hypot(ex, ez)
        if L > 0:
            best = min(best, abs(ex * (z - az) - ez * (x - ax)) / L)
    if inside:
        return best
    # outside: distance to the nearest edge segment
    d = float('inf')
    for i in range(n):
        ax, az = hull[i]
        bx, bz = hull[(i + 1) % n]
        ex, ez = bx - ax, bz - az
        L2 = ex * ex + ez * ez
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * ex + (z - az) * ez) / L2))
        d = min(d, math.hypot(x - (ax + t * ex), z - (az + t * ez)))
    return -d


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
    """Locate the ftrace binary, PREFERRING the repo-root copy that build.bat installs.

    Order matters: stale CMake build dirs from earlier configurations hang around (this
    repo has both `build_cuda` and the current `build_cuda2`), and picking one of those
    first silently polygonises with a binary that can be weeks out of date — so a scene
    using any newer feature meshes wrongly, or not at all, with no error. `build.bat`
    always copies the freshly built exe to the repo root, so that is the authoritative
    one; the build dirs are only fallbacks, newest first."""
    root = os.path.join(_HERE, '..')
    for c in (os.path.join(root, 'ftrace.exe'), os.path.join(root, 'ftrace')):
        if os.path.exists(c):
            return os.path.abspath(c)
    hits = [h for h in glob.glob(os.path.join(root, '**', 'ftrace*'), recursive=True)
            if os.path.isfile(h) and os.access(h, os.X_OK)]
    hits.sort(key=os.path.getmtime, reverse=True)      # freshest build dir wins
    return os.path.abspath(hits[0]) if hits else None


def cache_dir():
    """Where polygonised/decomposed collision geometry is memoised between runs.

    Polygonising the gallery at -mesh-res 64 takes ~40 s and VHACD-decomposing the hero
    pieces takes minutes, yet BOTH are pure functions of (scene text, res) / (mesh) — so
    a settle run that only changes a physics knob re-pays that cost for nothing. Keyed by
    content hash, so an edited scene or a rebuilt ftrace invalidates itself automatically.
    Lives under scraps/ (git-ignored) per the repo's file conventions."""
    d = os.path.join(_HERE, '..', 'scraps', '.settle_cache')
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def export_isosurface_meshes(scene_path, res, use_cache=True):
    """Run `ftrace -export-mesh` and return {group_name: trimesh} in world space."""
    exe = find_ftrace()
    if not exe:
        sys.exit('[settle_scene] could not find the ftrace binary to polygonise isosurfaces')
    key = hashlib.sha1()
    with open(scene_path, 'rb') as fh:
        key.update(fh.read())
    # the binary itself is part of the key: a rebuilt polygoniser can emit different tris
    key.update(f'|{res}|{os.path.getmtime(exe):.0f}'.encode())
    tmp = os.path.join(cache_dir(), 'iso_' + key.hexdigest()[:16] + '.obj')
    if use_cache and os.path.exists(tmp):
        print(f'[settle_scene] reusing cached polygonisation {os.path.basename(tmp)}')
        return parse_obj_groups(tmp)
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
def slab_hulls(mesh, slabs=None):
    """Fallback for slab_sections(): a stack of convex hulls, one per horizontal slab.

    Cheap and robust (nothing but a hull per slab), and exact in cap height and XZ extent,
    but it CONVEXIFIES each slab — so any bore or notch is silently filled in. That is fine
    for a plain pedestal and wrong for anything hollow, which is why slab_sections() is
    tried first and this only runs if the sectioning fails.

    Slabs OVERLAP by `eps` (one polygonisation cell) so consecutive hulls interpenetrate;
    without it a slab boundary landing on a vertical wall can leave a hairline seam."""
    slabs = slab_count(mesh) if slabs is None else slabs
    v = np.asarray(mesh.vertices)
    lo, hi = float(v[:, 1].min()), float(v[:, 1].max())
    if hi - lo < 1e-9:
        return [mesh.convex_hull]
    eps = max((hi - lo) / slabs * 0.25,
              float(np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                   mesh.vertices[mesh.edges[:, 1]], axis=1).mean()))
    cuts = np.linspace(lo, hi, slabs + 1)
    out = []
    for i in range(slabs):
        pts = v[(v[:, 1] >= cuts[i] - eps) & (v[:, 1] <= cuts[i + 1] + eps)]
        if len(pts) < 4:
            continue
        try:
            h = trimesh.Trimesh(vertices=pts).convex_hull
        except Exception:
            continue
        if h.volume > 1e-12:          # skip degenerate (coplanar) slabs
            out.append(h)
    return out or [mesh.convex_hull]


def slab_sections(mesh, slabs=None, tol=SECTION_SIMPLIFY):
    """Reduce a static collider to a stack of prisms extruded from its true cross-section.

    A cheap collider for a *static* body only has to reproduce the surface a piece can come
    to rest on, so the marching-cubes tessellation is enormously over-detailed for it (see
    the STATIC_TRI_CAP comment). The reduction has to preserve three things exactly: the
    height of every resting surface, the XZ outline, and — critically — any HOLE.

    Convexifying cannot do that. A single whole-mesh convex hull fills the taper between a
    wide base and a narrow column, inventing a shoulder a piece could rest on; hulling each
    horizontal slab separately fixes that, but still fills any bore in a slab, which would
    quietly turn a cradle collar into a flat disc and make the settle meaningless.

    So instead each slab is rebuilt from the mesh's actual cross-section at its mid-height:
    `section_multiplane` gives closed 2D outlines (with interior holes as holes), which are
    extruded back to the slab's full thickness. Measured on the gallery stands this is exact
    in cap height and XZ extent, holds volume to within a few percent (versus +46% to +111%
    for slab hulls), and costs ~2400 tris per stand — no worse than hulling.

    The outlines are simplified to `tol` first: an outline traced around marching-cubes
    output has thousands of collinear vertices, and extruding those directly gives ~83k tris
    per stand, which would defeat the whole exercise. Half a millimetre is far below any
    feature that matters here.

    Each slab is a stair-step of constant cross-section, so a horizontal feature only
    survives if it is thicker than a slab: the count comes from slab_count()'s target
    thickness, not a fixed number (see STATIC_SLAB_MAX_T).

    The prisms are CONCAVE (that is the point), so callers must load them as concave
    trimeshes rather than convex shapes."""
    slabs = slab_count(mesh) if slabs is None else slabs
    v = np.asarray(mesh.vertices)
    lo, hi = float(v[:, 1].min()), float(v[:, 1].max())
    if hi - lo < 1e-9:
        return None
    cuts = np.linspace(lo, hi, slabs + 1)
    mids = (cuts[:-1] + cuts[1:]) / 2.0
    try:
        secs = mesh.section_multiplane(plane_origin=[0.0, lo, 0.0],
                                       plane_normal=[0.0, 1.0, 0.0],
                                       heights=(mids - lo))
    except Exception:
        return None
    out = []
    for i, sec in enumerate(secs):
        if sec is None:
            continue
        thick = float(cuts[i + 1] - cuts[i])
        to_3d = sec.metadata.get('to_3D')
        if to_3d is None:
            return None       # without the section's own frame we'd guess the axes wrong
        for poly in sec.polygons_full:
            q = poly.simplify(tol) if tol else poly
            if q.is_empty or q.area <= 0.0:
                q = poly
            try:
                pr = trimesh.creation.extrude_polygon(q, thick)
            except Exception:
                continue
            # extrude_polygon builds in the section's own 2D frame extruding +z; to_3D maps
            # that frame back to world (its +z becomes the plane normal, i.e. world +y) and
            # puts z=0 at the section height — which is the slab's MIDDLE, so drop by half a
            # slab to make the prism span exactly [cuts[i], cuts[i+1]].
            pr.apply_transform(to_3d)
            pr.apply_translation([0.0, -0.5 * thick, 0.0])
            out.append(pr)
    return out or None


def intended_supports(worlds, selected, floor_y):
    """{piece: {support name: its top y}} — what the AUTHOR placed each settled piece over.

    This is the reference the FELL verdict is checked against, and it has to be read off the
    authored scene because the simulation has no notion of intent: to pybullet, "wedged on
    the floor between two pedestals" and "sitting on its cap" are both just rest.

    A support is any other named object whose plan (XZ) footprint overlaps the piece's and
    whose top is below the piece's mid height. The mid-height test (rather than "below the
    piece's underside") is what lets a MOUNT count: a collar's top is above the piece's lowest
    point whenever the piece hangs down inside the bore — but the collar is still the thing
    holding it up. A piece over nothing is meant to rest on the floor."""
    out = {}
    for pc in selected:
        plo, phi = worlds[pc].bounds
        mid = 0.5 * (plo[1] + phi[1])
        sup = {}
        for st, m in worlds.items():
            if st == pc:
                continue
            slo, shi = m.bounds
            if shi[0] < plo[0] or slo[0] > phi[0] or shi[2] < plo[2] or slo[2] > phi[2]:
                continue                      # no plan overlap: it is not underneath
            if shi[1] > mid:
                continue                      # towers past the piece: a neighbour, not a support
            sup[st] = float(shi[1])
        out[pc] = sup or {'floor': float(floor_y)}
    return out


def settle_bodies(worlds, selected, floor_y, max_steps, friction, tether=0.0,
                  jitter_deg=0.0, seed=0, use_cache=True):
    """worlds: {name: trimesh in world space}. selected: list of names to make dynamic.
    Returns (deltas, report):
      deltas — {name: (translate xyz, R 3x3)}, the extra rigid transform each selected
               object must be wrapped in (applied ON TOP of its authored world pose);
      report — {name: {...}} static-stability diagnostics, see below.

    `tether` (>0) enables a during-sim horizontal restoring spring: each step a force
    `k·(anchor_xz - com_xz)` (critically damped) pulls every dynamic body's COM back
    toward its authored XZ anchor. The force acts AT the COM, so it applies no torque —
    the piece is free to tip/rotate onto its narrow cap while being kept from walking off
    it sideways. This is the clean physical fix for the free-settle failure where a piece
    wider than its pedestal tips, rolls past the rim, and tumbles onto the floor. `k` is
    the spring stiffness in N/m (bodies have unit mass).

    The tether is then RELEASED (ramped to zero over RELAX_RAMP_STEPS) and the sim run to
    rest again before the pose is read back, so what gets baked is a pose gravity alone
    holds. Without that release the spring can prop a piece up in a physically impossible
    overhang — the 2026-08-03 gallery bug where `oiljack` and `heart` baked 0.45 m off
    their caps yet passed a COM-height check.

    `jitter_deg` (>0) spawns each body with a small random tilt (and lifts it just clear
    of whatever is under it) so SYMMETRIC pieces cannot come to rest in an unstable
    equilibrium the solver has no reason to leave — e.g. `brass_dumbbell`, a wheel whose
    ring rim is the only thing that reaches its cap, standing balanced on that rim
    forever. `seed` makes the perturbation reproducible. Any piece that still comes to rest
    PERCHED is automatically re-thrown (up to SETTLE_ATTEMPTS times) with a fresh draw,
    since a single unlucky draw — e.g. a tilt about the piece's own symmetry axis, which
    maps the body onto itself and therefore breaks nothing — perturbs nothing at all."""
    try:
        import pybullet as p
    except ImportError:
        sys.exit('settle_scene needs pybullet:  python -m pip install pybullet')

    # Read the author's intent off the scene BEFORE anything moves — once the sim runs, the
    # authored pose is gone and there is nothing left to check the result against.
    supports = intended_supports(worlds, selected, floor_y)
    for _pc, _sup in supports.items():
        print(f'[settle_scene] "{_pc}" is authored over: '
              + ', '.join(f'{s} (top {y:.3f})' for s, y in sorted(_sup.items())))

    tmpdir = tempfile.mkdtemp(prefix='settlescene_sim_')
    p.connect(p.DIRECT)
    p.setGravity(0, -9.81, 0)
    p.setPhysicsEngineParameter(numSolverIterations=80)

    # floor plane
    floor_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0])
    fb = p.createMultiBody(0, floor_col, basePosition=[0, floor_y, 0])
    p.changeDynamics(fb, -1, lateralFriction=friction)

    rng = np.random.default_rng(seed)
    dyn = {}      # name -> (body_id, com c)
    statics = {}  # body_id -> name, for naming what each piece ended up resting on
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
            # VHACD is the single most expensive setup step (tens of seconds to minutes per
            # hero piece) and is a PURE FUNCTION of the proxy mesh, so memoise it on the
            # proxy's content hash. Re-running the bake with a different tether/jitter/seed
            # — the normal iteration loop — then costs nothing at all here.
            with open(path, 'rb') as fh:
                vkey = hashlib.sha1(fh.read()).hexdigest()[:16]
            vh = os.path.join(cache_dir(), 'vhacd_' + vkey + '.obj')
            if use_cache and os.path.exists(vh):
                print(f'[settle_scene] reusing cached VHACD proxy for "{name}"', flush=True)
                col_file = vh
            else:
                try:
                    p.vhacd(path, vh, os.path.join(tmpdir, 'vhacd.log'))
                    col_file = vh if os.path.exists(vh) else path
                except Exception:
                    col_file = path
            col = p.createCollisionShape(p.GEOM_MESH, fileName=col_file)
            body = p.createMultiBody(1.0, col, basePosition=[float(c[0]), float(c[1]), float(c[2])])
            p.changeDynamics(body, -1, lateralFriction=friction,
                             spinningFriction=SPINNING_FRICTION,
                             rollingFriction=ROLLING_FRICTION, restitution=0.0,
                             # Keep the body in the solver even once it stops moving. A
                             # deactivated (sleeping) body is skipped by the constraint
                             # solver, so its contact manifolds stop carrying an applied
                             # impulse and the support-polygon test below sees it resting
                             # on NOTHING. It also must be awake to react to the tether
                             # being released.
                             activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
            dyn[name] = (body, c, float(np.linalg.norm(cen.vertices, axis=1).max()))
            print(f'[settle_scene] spawn "{name}" COM = ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})')
        else:
            # Static collider at its authored world pose (mesh already world-space).
            #
            # Isosurface stands come out of `ftrace -export-mesh` at the polygonisation res
            # (100s of k tris). That is not merely wasteful, it dominates the whole run:
            # contact-manifold generation is proportional to the number of static triangles
            # UNDERNEATH a resting piece, and a marching-cubes pedestal cap is thousands of
            # slivers where two triangles would do. Measured on the gallery, un-reduced
            # stands cost 60 ms/step — a settle is ~30 000 steps, so ~30 min of pure
            # collision detection. Reduced, the same run steps in 1.5 ms.
            #
            # Two reduction paths, in order of fidelity:
            #  1. Quadric decimation, when it actually reaches the cap (gyroid, the lamps,
            #     chrome_ring). This keeps the concave shape, so it is used whenever it works.
            #  2. Slab-section decomposition, for the meshes where decimation stalls. The
            #     stands do: they are unions of boxes whose marching-cubes tessellation
            #     resists edge-collapse, bottoming out at 25-47% of the input no matter how
            #     many passes or how much aggression (and losing 29% of the volume on the
            #     way, so pushing harder is not an option either). See slab_sections().
            parts = None
            if len(mesh.faces) > STATIC_TRI_CAP:
                try:
                    d = mesh.simplify_quadric_decimation(face_count=STATIC_TRI_CAP)
                    if len(d.faces) <= STATIC_TRI_CAP:
                        parts = [d]
                        print(f'[settle_scene] decimated static collider "{name}" to '
                              f'{len(d.faces)} tris (from {len(mesh.faces)})')
                except Exception as e:
                    print(f'[settle_scene] static-collider decimation of "{name}" failed ({e})')
                if parts is None:
                    parts = slab_sections(mesh)
                    how = 'slab sections'
                    if parts is None:                     # shapely/section machinery missing
                        parts = slab_hulls(mesh)
                        how = 'slab HULLS (holes will be filled!)'
                    print(f'[settle_scene] static collider "{name}" would not decimate; '
                          f'{len(parts)} {how}, '
                          f'{sum(len(h.faces) for h in parts)} tris (from {len(mesh.faces)})')
            else:
                parts = [mesh]
            for j, hpart in enumerate(parts):
                ppath = path if len(parts) == 1 else path[:-4] + f'_{j}.obj'
                hpart.export(ppath)
                col = p.createCollisionShape(p.GEOM_MESH, fileName=ppath,
                                             flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
                sb = p.createMultiBody(0, col)
                p.changeDynamics(sb, -1, lateralFriction=friction)
                statics[sb] = name        # every slab reports as the stand it came from

    p.setTimeStep(1.0 / 240.0)
    # Settled test is DISPLACEMENT-based (how far each body actually moved this step), not
    # instantaneous velocity: a tethered piece held in slight tension against friction has
    # a tiny non-zero velocity forever (never tripping a velocity threshold) yet does not
    # actually move, so a per-step position/orientation delta detects rest correctly and
    # lets the sim stop early instead of always grinding through max_steps.
    prev = {}

    def throw(names, rng):
        """(Re)place `names` at their authored COM with a fresh symmetry-breaking tilt.

        A tilt about a random horizontal axis would drive a corner of the piece into
        whatever it is standing on, so the spawn is also lifted by the sagitta that tilt
        sweeps (rmax·(1-cos θ), plus a hair) — the piece then falls that tiny distance and
        lands under gravity. The tilt AXIS matters: a body that is a solid of revolution
        (brass_dumbbell is a ring + coaxial spheres) is unchanged by a tilt about its own
        symmetry axis, so such a draw breaks nothing and the piece stays balanced on its
        rim. Re-throwing with a new draw eventually picks an axis that does tip it."""
        for name in names:
            body, c, rmax = dyn[name]
            spawn = [float(c[0]), float(c[1]), float(c[2])]
            quat = [0.0, 0.0, 0.0, 1.0]
            if jitter_deg > 0.0:
                axis = rng.normal(size=3); axis[1] = 0.0
                nrm = float(np.linalg.norm(axis))
                axis = np.array([1.0, 0.0, 0.0]) if nrm < 1e-9 else axis / nrm
                th = math.radians(jitter_deg) * float(rng.uniform(0.5, 1.0))
                s = math.sin(th * 0.5)
                quat = [float(axis[0] * s), 0.0, float(axis[2] * s), math.cos(th * 0.5)]
                spawn[1] += rmax * (1.0 - math.cos(th)) + 1e-4
            p.resetBasePositionAndOrientation(body, spawn, quat)
            p.resetBaseVelocity(body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for body, _c, _r in dyn.values():
            pos, quat = p.getBasePositionAndOrientation(body)
            prev[body] = (np.asarray(pos, float), np.asarray(quat, float))

    def apply_tether(k):
        """Horizontal restoring spring per body — applied at the COM (no torque), so
        rotation onto the cap stays free while lateral walk-off is cancelled. A DEADBAND
        (TETHER_DEADBAND) leaves the spring OFF while the piece is within a small radius of
        its anchor: this lets it settle NATURALLY there (its COM comes to rest a little off
        the anchor after tipping, so a spring that was always on would hold it in permanent
        tension — a residual micro-jitter that never trips the settled test and forces the
        full max_steps). The spring engages only on the EXCESS past the deadband, so it
        still yanks a piece back the moment it actually starts rolling off."""
        tdamp = 2.0 * math.sqrt(k)                    # critical damping (unit mass)
        for body, c, _r in dyn.values():
            (px, py, pz), _ = p.getBasePositionAndOrientation(body)
            dx, dz = float(c[0]) - px, float(c[2]) - pz
            dist = math.hypot(dx, dz)
            if dist <= TETHER_DEADBAND:
                continue                              # inside tolerance -> settle freely
            (vx, _vy, vz), _ = p.getBaseVelocity(body)
            excess = dist - TETHER_DEADBAND
            ux, uz = dx / dist, dz / dist             # unit vector toward the anchor
            p.applyExternalForce(body, -1,
                                 [k * excess * ux - tdamp * vx, 0.0,
                                  k * excess * uz - tdamp * vz],
                                 [px, py, pz], p.WORLD_FRAME)

    def run(steps, k_of_step, min_steps=0, label='run'):
        """Step until every dynamic body has been still for ~0.5 s, or `steps` elapse.
        `k_of_step(i)` gives the tether stiffness for step i (0 = spring off). `min_steps`
        forbids the early exit before that many steps have run — essential for the tether
        RELEASE phase, which begins with every body already at rest: without it the "still"
        counter trips after ~120 steps and the sim stops while the ramp is still at 75% of
        full stiffness, so the pose read back is not gravity-only after all."""
        still = 0
        t0 = time.perf_counter()
        for i in range(steps):
            k = k_of_step(i)
            if k > 0.0:
                apply_tether(k)
            p.stepSimulation()
            moved = 0.0
            for body, _c, _r in dyn.values():
                pos, quat = p.getBasePositionAndOrientation(body)
                pp, pq = prev[body]
                dp = float(np.linalg.norm(np.asarray(pos) - pp))      # metres moved
                dq = 1.0 - abs(float(np.dot(quat, pq)))               # 0 = same orientation
                moved = max(moved, dp + dq)
                prev[body] = (np.asarray(pos), np.asarray(quat))
            still = 0 if moved > SETTLE_STILL_EPS else still + 1
            if still > 120 and i >= min_steps:        # ~0.5 s of no motion -> settled
                _phase(label, i + 1, steps, t0)
                return i + 1
        _phase(label, steps, steps, t0)
        return steps

    def _phase(label, done, steps, t0):
        dt = time.perf_counter() - t0
        print(f'[settle_scene]   {label}: {done}/{steps} steps in {dt:.1f}s '
              f'({1000.0 * dt / max(1, done):.2f} ms/step)'
              f'{"" if done < steps else "  <-- hit the cap, did not settle"}', flush=True)

    def poses():
        return {n: np.asarray(p.getBasePositionAndOrientation(b)[0], float)
                for n, (b, _c, _r) in dyn.items()}

    def measure(name, held, free):
        """Pose delta + static-stability verdict for one body, read from the manifolds the
        last solver step left behind. Do NOT call performCollisionDetection() first: it
        rebuilds the manifolds, and a freshly created contact point carries no applied
        impulse yet, so every genuine resting contact reads as zero normal force and the
        piece appears to be resting on nothing."""
        body, c, _r = dyn[name]
        pos, quat = p.getBasePositionAndOrientation(body)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # final = pos + R·(v_world - c) = (pos - R·c) + R·v_world  -> delta on authored pose
        delta = (np.asarray(pos, float) - R @ c, R)

        # The real acceptance test: a body at rest is stable iff its COM projects INSIDE
        # the convex hull of its load-bearing contact points. Checking COM *height* instead
        # (the old scraps/check_settle_heights.py heuristic) silently passes a piece that
        # drifted off its cap sideways but happens to be at the right altitude.
        cps = p.getContactPoints(bodyA=body)
        total = sum(cp[9] for cp in cps)              # cp[9] = normalForce (solver impulse)
        cut = CONTACT_FORCE_FRAC * total
        pts, on = [], set()
        for cp in cps:
            if cp[9] <= cut:
                continue
            pts.append((round(cp[5][0], 6), round(cp[5][2], 6)))   # cp[5] = positionOnA
            on.add(statics.get(cp[2]) or next((n for n, (b, _, _) in dyn.items()
                                               if b == cp[2]), 'floor'))
        hull = convex_hull_2d(pts)
        # Did it land on what the author put it over? Two ways to fail: it is not touching
        # any intended support at all, or it is touching one but has sunk BELOW that
        # support's top — i.e. it is against the side of the pedestal, not on the cap. The
        # second half is what catches the wedged-on-the-floor case, which still grazes the
        # shafts it fell between and so passes a bare "is it touching its stand" test.
        sup = supports[name]
        top = max(sup.values())
        # The body was spawned exactly at its authored COM `c` and is centred on it, so
        # `pos - c` is the piece's displacement — reported for diagnosis, not gated on.
        return delta, {
            'margin':  support_margin(hull, (float(pos[0]), float(pos[2]))),
            'contacts': len(pts),
            'resting_on': sorted(on),
            'relax_drift': float(np.linalg.norm(free[name] - held[name])),
            'supports': sorted(sup),
            'fell': not (on & set(sup)) or float(pos[1]) <= top,
            'displaced': float(np.linalg.norm(np.asarray(pos, float) - c)),
            'dropped':   float(c[1] - pos[1]),
        }

    # Throw the pieces, settle them, and re-throw any that ended up PERCHED (balanced in an
    # unstable equilibrium) with a fresh perturbation. Pieces that already came to rest
    # stably are left where they are — re-running the sim around them is harmless, they are
    # at rest — so each attempt only has to get the remaining stragglers right.
    result, report = {}, {}
    pending = list(dyn)
    for attempt in range(SETTLE_ATTEMPTS if jitter_deg > 0.0 else 1):
        if attempt:
            print(f'[settle_scene] re-throwing {", ".join(pending)} '
                  f'(attempt {attempt + 1}/{SETTLE_ATTEMPTS}): perched in an unstable rest')
        throw(pending, rng)

        # --- phase 1: settle, with the tether holding pieces over their authored XZ -----
        run(max_steps, (lambda _i: tether) if tether > 0 else (lambda _i: 0.0),
            label=f'attempt {attempt + 1} phase 1 (tethered settle)')
        held = poses()

        # --- phase 2: RELEASE the tether and re-settle, so the pose is gravity-only ------
        # A pose recorded under the spring can be one gravity cannot hold. Ramping k to
        # zero rather than cutting it lets a piece that was being propped up topple or
        # slide off on its own, which is exactly the information the bake needs.
        if tether > 0:
            run(max_steps, lambda i: tether * max(0.0, 1.0 - i / RELAX_RAMP_STEPS),
                min_steps=RELAX_RAMP_STEPS,
                label=f'attempt {attempt + 1} phase 2 (tether release)')
        free = poses()

        # --- phase 3: POKE every piece and re-settle, untethered -------------------------
        # Equilibrium is not stability. A wheel balanced on its rim has its COM exactly over
        # its contact point, so it satisfies the support-polygon test perfectly — and falls
        # over the instant anything touches it. Kicking each piece and letting it re-settle
        # is a direct test of the thing we actually care about, and the pose that SURVIVES
        # the kick is the one worth baking, so the poked pose is what `measure` reads.
        for body, _c, _r in dyn.values():
            d = rng.normal(size=3); d[1] = 0.0
            nrm = float(np.linalg.norm(d))
            d = np.array([1.0, 0.0, 0.0]) if nrm < 1e-9 else d / nrm
            spin = rng.normal(size=3)
            spin *= POKE_SPIN / max(1e-9, float(np.linalg.norm(spin)))
            p.resetBaseVelocity(body, [float(d[0]) * POKE_SPEED, 0.0, float(d[2]) * POKE_SPEED],
                                [float(spin[0]), float(spin[1]), float(spin[2])])
        run(max_steps, lambda _i: 0.0, min_steps=240,
            label=f'attempt {attempt + 1} phase 3 (poke)')
        poked = poses()

        for name in dyn:
            result[name], report[name] = measure(name, held, free)
            report[name]['poke_drift'] = float(np.linalg.norm(poked[name] - free[name]))
        pending = [n for n in dyn if report[n]['margin'] < STABILITY_MARGIN
                   or report[n]['poke_drift'] > POKE_TOL or report[n]['fell']]
        if not pending:
            break

    p.disconnect()
    return result, report


# ---------------------------------------------------------------- seat-on-stand
def seat_on_stand(piece_world, R, stand_world, gap):
    """Re-seat a settled piece on its stand by lowering it straight down in place.

    A faithful free physics settle drops each hero onto NARROW museum pedestals, so a
    piece wider than its column top (or slightly overhanging it) tends to tip and roll
    OFF onto the floor rather than resting on display. This takes the ORIENTATION physics
    gave it (the natural way it came to rest) but does NOT relocate the piece laterally:
    it keeps the piece's left-right / forward-back position exactly where the author put
    it BEFORE the sim, then lowers it straight down until it just touches the stand —
    exactly the "drop it directly downward until it stops" move, applied whether the piece
    stayed on the stand or tumbled off. The result is a rigid transform `(translate, R)`
    in the same form settle_bodies returns, so it drops straight into the group wrapper.

    Note the rotation `R` is applied about the object's origin, which slides its bounding
    centroid; we cancel that XZ drift so the piece's authored footprint centre is restored
    (a pure spin-in-place at the original spot) before dropping.

    `piece_world` / `stand_world` are the objects' authored world-space meshes; `R` is the
    settled rotation about the piece's origin (world' = translate + R·world)."""
    rot = piece_world.copy()
    rot.vertices = piece_world.vertices @ R.T          # orient as it settled (about origin)
    auth_c = piece_world.bounds.mean(axis=0)           # authored (pre-sim) footprint centre
    rot_c  = rot.bounds.mean(axis=0)
    tx = float(auth_c[0] - rot_c[0])                   # cancel the rotation's XZ drift only
    tz = float(auth_c[2] - rot_c[2])                   # -> piece stays at its original XZ
    placed = rot.copy(); placed.apply_translation([tx, 0.0, tz])   # spin in place, no move
    dt, _ = drop(placed, ('mesh', stand_world), gap)   # vertical rest onto the stand top
    # Overhang check: because we keep the authored XZ, a piece the author placed OFF the
    # centre of its stand rests on the stand's edge with the rest of it hanging past the
    # rim. drop() never lets it sink INTO the stand, but the overhanging part can dangle
    # below the stand-top level. Flag it so the user knows to nudge the authored position.
    final_bot = float(placed.vertices[:, 1].min() + dt[1])
    stand_top = float(stand_world.bounds[1][1])
    if final_bot < stand_top - 0.01:
        print(f'[settle_scene] WARNING: seated piece overhangs its stand — {stand_top - final_bot:.3f} m '
              f'of it hangs below the stand top (authored off-centre from the stand). '
              f'Re-centre the piece over the stand if you want it to sit squarely.')
    return np.array([tx, float(dt[1]), tz], float), R


def parse_seat_pairs(spec, selected, by_name, worlds):
    """Resolve a `--seat piece:stand,…` spec (or `auto`) to a {piece: stand} dict. Auto
    pairs each settled piece to the nearest OTHER named object by authored XZ centre — in
    the sample scenes that's the pedestal each hero was posed above."""
    if spec.strip().lower() == 'auto':
        stands = [n for n in by_name if n not in selected]
        if not stands:
            sys.exit('[settle_scene] --seat auto: no non-settled objects to use as stands')
        pairs = {}
        for pc in selected:
            pcc = worlds[pc].bounds.mean(axis=0)
            best, bd = None, math.inf
            for st in stands:
                sc = worlds[st].bounds.mean(axis=0)
                d = (pcc[0] - sc[0]) ** 2 + (pcc[2] - sc[2]) ** 2   # XZ distance
                if d < bd:
                    bd, best = d, st
            pairs[pc] = best
        return pairs
    pairs = {}
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' not in tok:
            sys.exit(f'[settle_scene] --seat needs piece:stand pairs (got "{tok}")')
        pc, st = (x.strip() for x in tok.split(':', 1))
        if pc not in selected:
            sys.exit(f'[settle_scene] --seat piece "{pc}" is not among the settled objects')
        if st not in by_name:
            sys.exit(f'[settle_scene] --seat stand "{st}" is not a named object in the scene')
        pairs[pc] = st
    return pairs


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
    ap.add_argument('--tether', nargs='?', type=float, const=150.0, default=0.0,
                    help='during-sim horizontal restoring spring (N/m, unit-mass bodies) that '
                         'keeps each settling piece over its authored XZ so it tips onto its '
                         'narrow pedestal in place instead of rolling off onto the floor. Acts '
                         'at the COM (no torque -> rotation stays free). Bare "--tether" = 150; '
                         'higher = stiffer. Default off.')
    ap.add_argument('--jitter', type=float, default=2.0,
                    help='degrees of random spawn tilt applied to each settling piece to '
                         'break symmetry, so a piece cannot come to rest in an unstable '
                         'equilibrium (a ring balanced on its rim) that the solver has no '
                         'reason to leave. Default 2; 0 disables.')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for --jitter (default 0)')
    ap.add_argument('--no-cache', action='store_true',
                    help='ignore scraps/.settle_cache and re-polygonise / re-run VHACD')
    ap.add_argument('--max-steps', type=int, default=8000, help='max simulation steps')
    ap.add_argument('--seat', help='after settling, re-seat pieces squarely on their stands: '
                                   '"auto" (nearest non-settled object per piece) or explicit '
                                   'piece:stand,… pairs. Keeps each piece\'s settled orientation '
                                   'but centres it over the stand top and drops it straight down.')
    ap.add_argument('--seat-gap', type=float, default=0.001,
                    help='clearance left between a seated piece and its stand top (default 0.001)')
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
    iso_meshes = export_isosurface_meshes(a.scene, a.mesh_res, not a.no_cache) if need_iso else {}
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

    if a.tether > 0:
        print(f'[settle_scene] tether spring: k={a.tether:g} N/m (keeps pieces over their XZ), '
              f'released over the last {RELAX_RAMP_STEPS} steps so the baked pose is gravity-only')
    if a.jitter > 0:
        print(f'[settle_scene] spawn jitter: up to {a.jitter:g} deg (seed {a.seed})')
    deltas, report = settle_bodies(worlds, set(selected), floor_y, a.max_steps, a.friction,
                                   a.tether, a.jitter, a.seed, not a.no_cache)

    for name in selected:
        t, R = deltas[name]
        print(f'  {name}:  translate {fmt(t)}   rotate {fmt(euler_xyz_deg(R))}')

    # Static-stability verdict. A piece whose COM does not project inside the convex hull
    # of its load-bearing contacts is not resting — it is perched, and will read as
    # "floating off its pedestal" in the render even though the sim reported it settled.
    print('[settle_scene] stability (landed on its own stand, COM over support polygon, '
          'then poked to see if it holds):')
    unstable = []
    for name in selected:
        r = report[name]
        perched = r['margin'] < STABILITY_MARGIN
        wobbly = r['poke_drift'] > POKE_TOL
        if r['fell'] or perched or wobbly:
            unstable.append(name)
        # FELL first: a piece on the floor is stably at rest down there, so its margin and
        # poke numbers look fine and would otherwise print as OK.
        verdict = ('FELL   ' if r['fell'] else 'PERCHED' if perched else
                   'TOPPLES' if wobbly else 'OK     ')
        print(f'  {verdict} {name}:  margin {r["margin"]*1000:+7.1f} mm  '
              f'contacts {r["contacts"]:3d}  on {", ".join(r["resting_on"]) or "nothing"}  '
              f'(wanted {", ".join(r["supports"])}; tether release {r["relax_drift"]*1000:6.1f} mm, '
              f'poke {r["poke_drift"]*1000:6.1f} mm, moved {r["displaced"]*1000:6.1f} mm, '
              f'dropped {r["dropped"]*1000:+7.1f} mm)')
    if unstable:
        print(f'[settle_scene] WARNING: {len(unstable)} piece(s) did NOT settle where they '
              f'were authored: {", ".join(unstable)}\n'
              f'[settle_scene]   FELL = the piece is not resting on top of anything the author '
              f'placed it over — it slid or toppled off its stand and came to rest somewhere '
              f'else (the floor, or wedged against a pedestal shaft). It IS at rest there, so '
              f'margin/poke look healthy; those are local tests and cannot see this. Try '
              f'--tether, or a MOUNT that grips it — a collar whose bore is cut to the '
              f'piece\'s own cross-section. Consider first whether the MESH is the problem: '
              f'a shape with no upright equilibrium is better replaced than propped up.\n'
              f'[settle_scene]   PERCHED = the COM lies outside the hull of its load-bearing '
              f'contacts, so the piece overhangs whatever it is touching — check the authored '
              f'position is really over the stand.\n'
              f'[settle_scene]   TOPPLES = the piece was in equilibrium but fell over when poked, '
              f'i.e. it was balancing. If it does that on every retry the SHAPE has no stable '
              f'rest pose (e.g. a wheel whose rim is its lowest feature) and the geometry, not '
              f'the sim, needs changing.')

    if a.seat:
        pairs = parse_seat_pairs(a.seat, selected, by_name, worlds)
        print(f'[settle_scene] seating: '
              + ', '.join(f'{pc}->{st}' for pc, st in pairs.items()))
        for pc, st in pairs.items():
            deltas[pc] = seat_on_stand(worlds[pc], deltas[pc][1], worlds[st], a.seat_gap)
            t, R = deltas[pc]
            print(f'  seated {pc} on {st}:  translate {fmt(t)}   rotate {fmt(euler_xyz_deg(R))}')

    out_text = wrap_blocks(text, by_name, deltas)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write(out_text)
    print(f'[settle_scene] wrote {a.out}')


if __name__ == '__main__':
    main()
