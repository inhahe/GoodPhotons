"""Build a shootable ftrace rig out of each camera GLB under cameras/.

The idea: these assets are MODELS OF CAMERAS, so the interesting thing to do with
one is not to photograph it, it is to LOOK THROUGH it. This tool solves each GLB's
own optical axis and front-element vertex, places the mesh so that vertex lands on
the world origin pointing down +Z, and then puts a real ftrace `camera` at exactly
that point with the sensor / focal length / f-number the real camera would have.
The render is then, literally, the photograph that camera would take of the scene
it is sitting in -- including its own shadow, cast by its own body, which is behind
the eye and therefore never in frame.

Two rigs are emitted per model:

  scenes/camrig_<name>.ftsl      mode A -- the finite-lens forward physical camera.
                                 An IDEAL thin lens: the focal length and f-number
                                 are honoured exactly, so field of view and depth of
                                 field are right, but the lens has no aberrations.
  scenes/camrig_<name>_sim.ftsl  mode R + a physical `lens { }` prescription. Every
                                 camera ray is refracted through real glass at its
                                 own wavelength, so on top of the same DoF you also
                                 get spherical and chromatic aberration, field
                                 curvature, focus shift and cat's-eye vignetting.

All ten rigs share ONE subject, deliberately: the same bench through five different
cameras is a direct read-out of each one's format, focal length and aperture. It is
a low checkered bench carrying

  * three upright resolution charts at 1.00 / 1.45 / 1.95 m. UPRIGHT because the eye
    is at bench height -- anything LYING on the bench is seen nearly edge-on and its
    depth ordering is unreadable, while a standing card occupies real image height at
    every distance. Their 10 mm checker is what visibly mushes as DoF runs out.
  * a STAR ROW of pinpoint emitters at the focus plane, fanned right across the frame.
    In focus, a point source images to the lens's point spread function, so this row
    is a star test: clean points under the ideal element, coma and colour fringes
    under traced glass. Count the ones in frame and you have read the field of view.
  * a BOKEH ROW of the same pinpoints 1.32 m behind focus -- the aperture gauge. Blur
    disc diameter is A*f*(u-uf)/(u*(uf-f)), which comes out ~5 px on the f/3.5
    point-and-shoot and ~24 px on the f/1.7 rangefinder.

Every element is placed in its own angular lane so nothing occludes anything, and the
outer members of both rows sit outside the narrow cameras' frames on purpose, so the
rows double as a field-of-view ruler across the set.

  python tools/camera_rig.py            # solve + write all ten scenes
  python tools/camera_rig.py --report   # just print the solves

--- how the placement is solved -------------------------------------------------

The lens of these assets is always a STACK of coaxial surfaces of revolution (barrel
rings, filter ring, element rims), so:

  1. Keep every node-geometry that really is a surface of revolution (circular
     cross-section, near-full angular coverage).
  2. Group them by shared axis direction (parallel up to sign).
  3. Within a direction group, keep those whose centres are COLLINEAR along that
     axis -- that is one barrel.
  4. Score each barrel by sum(radius^2 * thickness): the taking lens is the largest
     cylindrical assembly on the camera. Viewfinder / flash / dials lose.
  5. Orient the axis to point AWAY from the body centroid (the way it looks) and take
     the front as the extreme vertex of the barrel group along it.

Each of those steps earned a guard the hard way; see the comments in `solve()`.
"""
import sys, os, numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. the solver
# ---------------------------------------------------------------------------

def revolution(v):
    """Best surface-of-revolution fit for a vertex cloud: the covariance eigenvector
    whose two ORTHOGONAL spreads are most nearly equal (a circle projects
    isotropically). Returns (ratio, axis, angular_coverage, radius, thickness,
    centroid) or None."""
    c = v.mean(0); X = v - c
    _, V = np.linalg.eigh(np.cov(X.T))
    best = None
    for i in range(3):
        ax = V[:, i]; o = [V[:, j] for j in range(3) if j != i]
        a0, a1 = X @ o[0], X @ o[1]
        s0, s1 = np.std(a0), np.std(a1)
        if min(s0, s1) <= 1e-12: continue
        ratio = max(s0, s1) / min(s0, s1)
        if best is None or ratio < best[0]:
            ang = np.arctan2(a1, a0)
            cover = float((np.histogram(ang, 36, (-np.pi, np.pi))[0] > 0).mean())
            rad = float(np.linalg.norm(np.stack([a0, a1], 1), axis=1).max())
            best = (ratio, ax, cover, rad, float(np.ptp(X @ ax)), c)
    return best


def solve(path, ratio_max=1.15, cover_min=0.85, verbose=False):
    sc = trimesh.load(path, process=False)
    if not isinstance(sc, trimesh.Scene): sc = trimesh.Scene(sc)

    parts, allv = [], []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        v = np.asarray(g.vertices, np.float64) @ T[:3, :3].T + T[:3, 3]
        mat = (getattr(getattr(g.visual, "material", None), "name", "") or "").lower()
        if "ground" in mat or "floor" in mat or "backdrop" in mat:
            continue                              # bundled backdrop, not the camera
        allv.append(v)
        if len(v) < 12: continue
        b = revolution(v)
        if b is None: continue
        ratio, ax, cover, rad, thick, c = b
        if ratio <= ratio_max and cover >= cover_min:
            parts.append(dict(name=gname, mat=mat, ax=ax, rad=rad,
                              thick=thick, c=c, v=v))

    A = np.vstack(allv)
    lo, hi = A.min(0), A.max(0); bc = 0.5 * (lo + hi)

    # --- group by axis direction, then by radial anchor (collinear centres) ---
    barrels = []
    used = [False] * len(parts)
    for i, p in enumerate(parts):
        if used[i]: continue
        ax = p["ax"]
        grp = []
        for j, q in enumerate(parts):
            if used[j]: continue
            if abs(float(np.dot(ax, q["ax"]))) < 0.985: continue
            # Radial offset between the two axes' anchors, measured across `ax`.
            # Gate on the SMALLER radius: keyed on the larger, a huge body shell
            # that happens to be axis-parallel swallows the little lens next to it
            # (that is what put the pocket camera's whole chassis in its barrel).
            d = q["c"] - p["c"]
            perp = np.linalg.norm(d - np.dot(d, ax) * ax)
            if perp > 0.45 * min(p["rad"], q["rad"]) + 1e-6: continue
            grp.append(j)
        if len(grp) < 2: continue
        for j in grp: used[j] = True
        # A camera strap is also a long coaxial tube and outscores the lens on
        # radius^2 * length, so require a LENS aspect: a photographic barrel is
        # roughly as long as it is wide, never 4x. Measured: portable 1.4,
        # vintage 1.4, cinema 2.5 -- the SLR's "Body Belt" strap is 4.1.
        V = np.vstack([parts[j]["v"] for j in grp])
        span = float(np.ptp(V @ ax))
        rmax = max(parts[j]["rad"] for j in grp)
        if span > 3.0 * rmax: continue
        score = sum(parts[j]["rad"] ** 2 * (parts[j]["thick"] + 1e-3) for j in grp)
        barrels.append((score, ax, grp))

    if not barrels:
        return None
    barrels.sort(key=lambda b: -b[0])
    score, ax, grp = barrels[0]

    # A lens stack has comparable element radii. A body shell that happens to sit
    # on the same axis does not -- the pocket camera's chassis (r=20.3) joined its
    # own lens (r=1.7-2.6) this way and dragged the "front" onto the body face.
    rads = sorted(parts[j]["rad"] for j in grp)
    rmed = rads[len(rads) // 2]
    grp = [j for j in grp if rmed / 2.5 <= parts[j]["rad"] <= 2.5 * rmed]

    # Consensus axis: the individual part axes disagree when one element is a
    # TAPERED cone (its covariance axis tilts), which is what threw the SLR off by
    # 8 deg. The element centres, though, are collinear along the true axis by
    # construction -- so gather candidate directions and vote.
    C = np.array([parts[j]["c"] for j in grp])
    W = np.array([parts[j]["rad"] ** 2 * (parts[j]["thick"] + 1e-3) for j in grp])
    cand = [parts[j]["ax"] for j in grp]
    m = np.zeros(3)
    for k, j in enumerate(grp):
        a = parts[j]["ax"]
        m += (a if np.dot(a, ax) >= 0 else -a) * W[k]
    if np.linalg.norm(m) > 1e-9: cand.append(m / np.linalg.norm(m))
    if len(C) >= 3:                      # total-least-squares line through the centres
        d = C - np.average(C, axis=0, weights=W)
        _, Vv = np.linalg.eigh(np.cov(d.T))
        cand.append(Vv[:, -1])
    # Prefer the direction the MOST elements independently agree on. Scatter-of-
    # centroids alone is not enough: a partial ring (a hood cut, a bayonet notch)
    # has an off-axis centroid, and a tilt that "explains" it can win by a hair --
    # that is how the SLR kept an 8-deg tilt that only its one tapered cone voted
    # for, against four rings that all said +Z. Count votes, break ties by weight.
    bestax, bestvote, bestw = ax, -1, -1.0
    for a in cand:
        a = a / np.linalg.norm(a)
        agree = [j for j in grp if abs(float(np.dot(a, parts[j]["ax"]))) > 0.996]  # ~5 deg
        wsum = float(sum(parts[j]["rad"] ** 2 * (parts[j]["thick"] + 1e-3) for j in agree))
        if (len(agree), wsum) > (bestvote, bestw):
            bestvote, bestw, bestax = len(agree), wsum, a
    ax = bestax if np.dot(bestax, ax) >= 0 else -bestax

    # Radial anchor from the elements that actually share this axis (a tapered or
    # partial part would bias it), using the median to shrug off any survivor.
    agree = [j for j in grp if abs(float(np.dot(ax, parts[j]["ax"]))) > 0.996] or grp
    P = np.array([parts[j]["c"] - np.dot(parts[j]["c"], ax) * ax for j in agree])
    anchor_perp = np.median(P, axis=0)

    G = np.vstack([parts[j]["v"] for j in grp])
    if np.dot(ax, G.mean(0) - bc) < 0: ax = -ax     # point out of the body
    t = G @ ax
    front = float(t.max())
    ring = G[t > front - 0.05 * (np.ptp(t) + 1e-9)]
    pupil = anchor_perp + front * ax
    semi = float(np.linalg.norm(ring - (anchor_perp + (ring @ ax)[:, None] * ax),
                                axis=1).max())

    if verbose:
        print(f"  barrel parts ({len(grp)}), score {score:.3f}:")
        for j in grp:
            q = parts[j]
            print(f"    {q['name'][:34]:<35} r={q['rad']:.3f} th={q['thick']:.3f} "
                  f"c={np.round(q['c'],3)}")

    return dict(lo=lo, hi=hi, size=hi - lo, bc=bc, axis=ax, pupil=pupil,
                front=front, semi=semi, nbarrel=len(grp), tmin=float(t.min()))


# ---------------------------------------------------------------------------
# 2. what each asset actually IS
# ---------------------------------------------------------------------------
# `body_mm` is the real camera's body WIDTH, which sets the mesh scale (the GLBs are
# all authored at an arbitrary unit scale). Where the asset labels itself, that label
# is the anchor and the body width is the cross-check -- see the notes.

CAMERAS = [
    dict(
        key="cinema", glb="cinema_camera.glb",
        title="Blackmagic-style 4K cinema camera + 35 mm T2.1 cine prime",
        body_mm=162.5, sensor=(21.12, 11.88), res=(1280, 720),
        focal=35.0, fstop=2.0, skip=["ground"],
        ident=(
            "The asset labels its own lens: '35  T2.1' and 'FILTER 88MM' are painted on\n"
            "# the barrel, and the body reads '4K'. That is a Blackmagic Production Camera 4K\n"
            "# with a 35 mm cine prime. The two labels give two INDEPENDENT scale anchors and\n"
            "# they agree to 1%: the solved barrel rim is 0.5416 model units in radius, so an\n"
            "# 88 mm filter thread means 81.24 mm per unit, and at that scale the 2.0-unit body\n"
            "# comes out 162.5 mm against the real camera's 161 mm.\n"
            "#\n"
            "# T2.1 is a TRANSMISSION stop -- the geometric aperture behind it is about f/2.0\n"
            "# once the ~10% loss through a cine prime's elements is taken out, and f/2.0 is\n"
            "# what the ray tracer needs. Sensor is the BMPC4K's 21.12 x 11.88 mm Super-35\n"
            "# (native 16:9, hence the 1280x720 film).\n"
            "#\n"
            "# This GLB ships a 17.5-unit ground plane bundled in with the camera, on material\n"
            "# 'ground'. `skip_material ground` drops it at load; without that it would cut\n"
            "# straight through the bench."),
    ),
    dict(
        key="pocket", glb="pocket_camera.glb",
        title="90s 35 mm compact point-and-shoot, fixed 30 mm f/3.5",
        body_mm=118.0, sensor=(36.0, 24.0), res=(1200, 800),
        focal=30.0, fstop=3.5, skip=[],
        ident=(
            "A clamshell 35 mm film compact in the Ricoh R1 / Olympus mju mould -- fixed\n"
            "# moderate-wide lens, no interchangeable mount, a flash window and an optical\n"
            "# finder above the lens. Full-frame 36 x 24 film (it IS 35 mm film), 30 mm f/3.5,\n"
            "# which is the canonical spec for the class.\n"
            "#\n"
            "# Scale anchored on the front element: the solved rim is 2.540 units, and a\n"
            "# compact's front element is about 19 mm across, giving 3.72 mm per unit -- at\n"
            "# which the 31.73-unit body is 118 mm wide, right for a 90s compact.\n"
            "#\n"
            "# This is the SLOW camera of the set and it should look like it: f/3.5 at 30 mm\n"
            "# is an 8.6 mm entrance pupil, so the depth of field is enormous and the\n"
            "# out-of-focus lights at the back collapse to ~5 px specks instead of discs."),
    ),
    dict(
        key="portable", glb="portable_camera.glb",
        title="APS-C mirrorless + fast 35 mm f/1.4 normal prime",
        body_mm=120.0, sensor=(23.4, 15.6), res=(1200, 800),
        focal=35.0, fstop=1.4, skip=[],
        ident=(
            "A white-bodied APS-C mirrorless with a short, fat, deeply-coated prime -- the\n"
            "# purple bloom on the front element in the asset's own preview is a multicoating\n"
            "# reflection. Body 120 mm wide (typical rangefinder-style mirrorless), which puts\n"
            "# 1.689 mm on a unit; the barrel then measures 58.7 mm across the front and 42 mm\n"
            "# long, i.e. a 35 mm f/1.4 with an oversized front group.\n"
            "#\n"
            "# APS-C is quoted as 23.4 x 15.6 mm (exactly 3:2; the usual 23.6 figure is not).\n"
            "# 35 mm on APS-C is a 52 mm-equivalent normal -- and at f/1.4 it is the shallowest\n"
            "# lens in the set, so this is the rig where the simulated variant differs most\n"
            "# from the analytic one: a fast lens is where real glass stops behaving ideally."),
    ),
    dict(
        key="vintage", glb="vintage_camera.glb",
        title="Soviet rangefinder (FED / Zorki lineage) + collapsible 50 mm f/3.5",
        body_mm=135.8, sensor=(36.0, 24.0), res=(1200, 800),
        focal=50.0, fstop=3.5, skip=[],
        ident=(
            "The purple-leatherette folding rangefinder -- a FED/Zorki with a collapsible\n"
            "# chrome lens tube, i.e. an Industar-22 50 mm f/3.5, the Soviet Elmar copy. 35 mm\n"
            "# film, 36 x 24. Body 135.8 mm wide (a Zorki-1 is 133, a FED-2 is 140), which is\n"
            "# the same 0.0285 scale the mirror-sphere scene solved independently.\n"
            "#\n"
            "# A 50 mm on full frame is the reference normal, so this rig is the set's\n"
            "# baseline: whatever the bench looks like here is what 'normal' looks like."),
    ),
    dict(
        key="vintage_slr", glb="vintage_slr_camera.glb",
        title="Chrome fixed-lens rangefinder + 45 mm f/1.7",
        body_mm=150.0, sensor=(36.0, 24.0), res=(1200, 800),
        focal=45.0, fstop=1.7, skip=[],
        ident=(
            "NOTE THE ASSET IS MISNAMED. 'vintage_slr_camera.glb' has no pentaprism hump, no\n"
            "# mirror box and no lens mount -- it has a rangefinder window, a parallax finder\n"
            "# and a permanently-fitted lens. It is a fixed-lens 35 mm rangefinder in the\n"
            "# Yashica Electro 35 / Canonet QL17 mould. The name is kept because `preset\n"
            "# vintage-slr` is shipped API, but the OPTICS specced here are the rangefinder's.\n"
            "#\n"
            "# Two anchors agree: the body's 8.446 x 5.137 unit proportion is 1.64, matching an\n"
            "# Electro 35's 150 x 89 mm (1.69); at 150 mm wide the solved front rim comes out\n"
            "# 57.7 mm, and the Electro 35 takes 55 mm filters. So: 45 mm f/1.7 on 36 x 24,\n"
            "# which is exactly the Electro 35 GSN's Yashinon-DX."),
    ),
]


# ---------------------------------------------------------------------------
# 3. the shared subject
# ---------------------------------------------------------------------------
# Everything below is authored in the RIG frame: the camera's entrance pupil is the
# world origin, its optical axis is +Z, its up is +Y. The bench top is at y = -REST,
# where REST is however far that particular camera's base sits below its own axis --
# so every camera is resting ON the bench rather than floating over it, and the eye
# height above the bench is whatever the real camera's is.

FOCUS = 1.00            # metres; the plane the middle chart stands on

# --- resolution charts: (axial z, lateral x, r, g, b) ------------------------------
# Upright 120 x 140 mm cards carrying a fine 10 mm checker, standing on the bench and
# facing the lens. UPRIGHT is the whole point: the camera sits at bench height, so
# anything LYING on the bench is seen nearly edge-on and its depth ordering collapses
# onto the horizon, while a standing card occupies real image height at every distance.
# The near card is the focus target; the other two are a calibrated 0.45 m and 0.95 m
# behind it, and their checker is what visibly mushes as the depth of field runs out.
#
# The three lateral positions are not decorative -- they are solved so that, seen from
# the origin, the three cards' ANGULAR spans are disjoint from each other AND from the
# star row's lanes, so nothing occludes anything on any camera in the set:
#
#      card      z       x        horizontal span      star lanes it must miss
#      focus   1.00   +0.100     +2.3 deg .. +9.1     0, +10.8
#      mid     1.45   -0.127     -2.7     .. -7.4     0, -10.8
#      far     1.95   +0.460    +11.6     .. +14.9    +10.8, +20.8
#
# The far card's outer edge lands at 14.9 deg against the NARROWEST camera's 16.8 deg
# half-width, so all three are fully in frame even on the 35 mm Super-35 cinema rig.
# (An earlier layout put the mid card out at -17.9 deg and the cinema rig sliced it in
# half at the frame edge -- hence the explicit budget above.)
CHARTS = [(1.00,  0.100, 0.86, 0.84, 0.80),    # the focus target
          (1.45, -0.127, 0.30, 0.50, 0.62),    # +0.45 m behind focus
          (1.95,  0.460, 0.62, 0.44, 0.26)]    # +0.95 m behind focus
CARD_W, CARD_H = 0.120, 0.140

# --- star row: point sources AT the focus plane ------------------------------------
# A row of 3 mm pinpoints floating 100 mm over the bench at exactly the focus distance,
# fanned right across the frame. In focus, a point source images to the lens's POINT
# SPREAD FUNCTION, so this row is a direct star test: the analytic rig's ideal element
# collapses every one of them to a few pixels no matter how far off axis, while traced
# glass flares the outer ones into coma and fringes them with lateral colour.
#
# 3 mm is chosen so the geometric image is 3-7 px on every camera in the set -- small
# enough that the PSF, not the source, sets the shape. Shrinking a sphere light does
# not dim it (its radiance is what the SPD fixes, so a resolved disc keeps the same
# brightness per pixel); it only costs photons, i.e. noise.
#
# The row floats at y = 0 -- ON the optical axis plane, not resting on the bench. That
# is deliberate and it took a render to learn: at any other height EVERY star is off
# axis, so there is no axial reference to compare the flared ones against. At y = 0 the
# centre star is the true on-axis PSF and the rest are displaced purely LATERALLY, so
# the row is a clean one-dimensional sweep of field angle -- exactly what a star test
# wants. y = 0 also happens to sit 25-63 mm above the bench (the camera's own eye
# height) and therefore above the bench/backdrop horizon, so each star is read against
# flat grey rather than against the bench's grazing glare.
#
# The outer pairs only enter frame on the wider cameras, so the row doubles as a
# field-of-view ruler you can literally count: 3 stars visible = 16.8 deg half-width.
STAR_X = [0.0, 0.19, -0.19, 0.38, -0.38, 0.57, -0.57, 0.78, -0.78]
STAR_Y, STAR_R = 0.0, 0.0015

# --- bokeh row: the same pinpoints, 1.32 m BEHIND focus ----------------------------
# Each renders as a blur disc of sensor diameter A*f*(u-uf)/(u*(uf-f)) -- a pure
# entrance-pupil readout, ~5 px on the 30 mm f/3.5 compact and ~22 px on the 35 mm
# f/2.0 cinema rig. Under the simulated rig the outer ones also go cat's-eye, because
# an off-axis cone gets clipped by the barrel at both ends.
#
# 6 mm (geometric image 2.6-5.6 px) so the blur disc, not the source, sets the size.
# y = 0.32 is a clearance solve: the sightline to it crosses z = 1.00 at y = 0.138,
# clear of the tallest card top in the set (0.115 on the pocket camera), so no card
# ever eclipses a bokeh light -- while still landing inside the narrowest camera's
# 0.393 m half-height at that distance, near the top edge where cat's-eye shows best.
BOKEH_X = [0.0, 0.19, -0.19, 0.40, -0.40, 0.62, -0.62, 0.86, -0.86]
BOKEH_Z, BOKEH_Y, BOKEH_R = 2.32, 0.32, 0.003


def bench(rest):
    """The subject, as FTSL text. `rest` is the bench-top depth below the axis."""
    y = -rest
    L = []
    A = L.append
    A("# --- the bench ------------------------------------------------------------------")
    A("# A checkered top, 45 mm squares, running away from the camera and meeting the")
    A("# backdrop at z = 2.60. It is the continuous depth reference: the square size is")
    A("# fixed, so how fast the squares shrink IS the focal length, and where they stop")
    A("# resolving IS the depth of field. The camera is RESTING on this bench -- its own")
    A(f"# base sits {rest*1000:.1f} mm below its own optical axis, which is why the top is at")
    A(f"# y = {y:.4f} and not some round number.")
    A('pattern  "chk"      { type checker  size 0.045 }')
    A('material "chk_pale" { type diffuse reflect rgb 0.74 0.71 0.66 }')
    A('material "chk_dark" { type diffuse reflect rgb 0.11 0.11 0.13 }')
    A('material "bench"    { type mix  layer "chk_pale" 0.5  layer "chk_dark" 0.5'
      '  weight_map pattern:chk }')
    A('material "backdrop" { type diffuse reflect rgb 0.44 0.45 0.48 }')
    A(f"quad {{ origin -2.20 {y:.4f} -1.00   u 4.40 0 0   v 0 0 3.60   material bench }}")
    A(f"quad {{ origin -2.60 {y:.4f} 2.60    u 5.20 0 0   v 0 2.40 0   material backdrop }}")
    A("")
    A("# --- the resolution charts -------------------------------------------------------")
    A(f"# Three upright {CARD_W*1000:.0f} x {CARD_H*1000:.0f} mm cards on a fine 10 mm checker,"
      " standing on the bench and")
    A("# facing the lens. The pale one at z = 1.00 m is the focus target; the other two are")
    A("# 0.45 m and 0.95 m behind it, so how fast their checker mushes IS the depth of field.")
    A("# Their lateral positions are solved so the three angular spans are disjoint from each")
    A("# other and from the star row, hence the odd x values -- with the eye at bench height,")
    A("# an object LYING on the bench reads as a smear on the horizon, so the subject stands.")
    A('pattern "chkfine" { type checker  size 0.010 }')
    A('material "card_dark" { type diffuse reflect rgb 0.05 0.05 0.06 }')
    for i, (z, x, r, g, b) in enumerate(CHARTS):
        A(f'material "card_pale{i}" {{ type diffuse reflect rgb {r} {g} {b} }}')
        A(f'material "card{i}" {{ type mix  layer "card_pale{i}" 0.5  layer "card_dark" 0.5'
          f'  weight_map pattern:chkfine }}')
    for i, (z, x, _, _, _) in enumerate(CHARTS):
        # u along +Y, v along +X  ->  u x v = -Z, i.e. the face turned toward the lens
        A(f"quad {{ origin {x - CARD_W/2:.4f} {y:.4f} {z:.3f}   u 0 {CARD_H:.3f} 0"
          f"   v {CARD_W:.3f} 0 0   material card{i} }}")
    A("")
    A("# --- the star row (in focus): the point-spread-function readout ------------------")
    A(f"# {STAR_R*2000:.0f} mm pinpoints floating AT the focus distance and exactly ON the"
      " optical axis plane,")
    A(f"# {rest*1000:.0f} mm over the bench, fanned sideways across the whole frame. An in-focus")
    A("# point source images to the lens's PSF, so this row is a star test: the analytic rig's")
    A("# ideal element collapses every one of them to a few pixels no matter how far off axis,")
    A("# while traced glass flares the outer ones into coma and fringes them with lateral")
    A("# colour. y = 0 makes the centre star the true ON-AXIS reference and displaces the")
    A("# rest purely laterally, so the row is a clean sweep of field angle; it also puts them")
    A("# above the bench/backdrop horizon, read against flat grey instead of bench glare.")
    A("# Count the ones in frame and you have read the field of view directly.")
    for x in STAR_X:
        A(f"light sphere {{ center {x:.3f} {STAR_Y:.3f} {FOCUS:.3f}"
          f"  radius {STAR_R:.4f}  spd blackbody 5500 }}")
    A("")
    A("# --- the bokeh row (1.32 m behind focus): the aperture readout -------------------")
    A(f"# The same pinpoints, warm and {BOKEH_R*2000:.0f} mm across, far enough back to be"
      " thoroughly defocused.")
    A("# Blur-disc diameter on the sensor is A*f*(u-uf)/(u*(uf-f)), pure entrance pupil:")
    A("# ~5 px at 30 mm f/3.5, ~22 px at 35 mm f/2.0. Under the simulated rig the outer")
    A("# ones additionally go cat's-eye, clipped at both ends of the barrel. They sit high")
    A("# enough (y = 0.32) that no resolution card can eclipse one on any camera in the set.")
    for x in BOKEH_X:
        A(f"light sphere {{ center {x:.3f} {BOKEH_Y:.3f} {BOKEH_Z:.3f}"
          f"  radius {BOKEH_R:.4f}  spd blackbody 2700 }}")
    A("")
    A("# --- lighting -------------------------------------------------------------------")
    A("# BOTH lights sit BEHIND the eye (z < 0), which is forced by two independent things:")
    A("#")
    A("#   * the resolution cards face -Z, i.e. they face the lens. A light in front of them")
    A("#     lights their BACKS, and all three would render as flat silhouettes lit only by")
    A("#     bounce. Anything meant to light the subject has to be behind the camera.")
    A("#   * the camera's OWN SHADOW is the whole point of the rig -- the body is behind the")
    A("#     eye and can never be in frame, but it is still solid. A key in FRONT throws that")
    A("#     shadow backwards, behind the camera, where no one can see it. Only a key behind")
    A("#     and above throws it FORWARD, across the near bench and into the frame.")
    A("#")
    A("# The key is placed by solving that shadow: the ray from the key over the body's top")
    A("# edge meets the bench about 0.5 m in front of the lens and ~13 deg off axis, inside")
    A("# even the narrowest frame here. Being behind the eye, neither light can be stared at")
    A("# directly by the taking camera, and both are outside the witness camera's frame too.")
    A("light sphere { center -0.25 0.35 -1.10  radius 0.130  spd blackbody 5200 }")
    A("light sphere { center  0.85 0.55 -0.20  radius 0.100  spd blackbody 4200 }")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 4. mesh placement
# ---------------------------------------------------------------------------

def place(axis, pupil, s):
    """ftrace composes a mesh transform as scale, Rx, Ry, Rz, translate (src/mesh.h),
    so we need Euler XYZ angles taking the solved axis onto +Z, and the translation
    that then lands the front-element vertex on the origin.

    Setting rz = 0, Rx must kill the y-component and Ry the x-component:
        rx = atan2(ay, az)            -> (ax, 0, hypot(ay,az))
        ry = atan2(-ax, hypot(ay,az)) -> (0, 0, 1)
    Returns (rot_deg[3], translate[3], residual_deg)."""
    a = axis / np.linalg.norm(axis)
    h = float(np.hypot(a[1], a[2]))
    rx = float(np.arctan2(a[1], a[2]))
    ry = float(np.arctan2(-a[0], h))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    M = Ry @ Rx
    t = -(M @ (s * pupil))
    resid = float(np.degrees(np.arccos(np.clip((M @ a)[2], -1, 1))))
    return np.degrees([rx, ry, 0.0]), t, resid


# ---------------------------------------------------------------------------
# 5. emit
# ---------------------------------------------------------------------------

HEAD = """\
# {title}
#
# THE CAMERA IN THIS SCENE IS THE CAMERA YOU ARE LOOKING THROUGH. The GLB's own
# optical axis and front-element vertex were solved from its geometry (tools/
# camera_rig.py), and the mesh is placed so that vertex sits exactly on the render
# camera's entrance pupil, pointing down +Z. So the body, the barrel and the finder
# are all BEHIND the eye and can never appear in frame -- but they are still solid,
# so the camera occludes the key light and casts its own shadow into its own
# photograph. That shadow across the near bench is the proof the rig is real and not
# a free-floating viewpoint.
#
# {ident}
#
# --- solved placement (do not hand-edit; regenerate with tools/camera_rig.py) ----
#   model bbox    {size}  ({body_mm:.1f} mm wide -> scale {scale:.6g})
#   optical axis  {axis}   (residual after the rotation below: {resid:.4f} deg)
#   front vertex  {pupil}  -> world origin
#   front rim     {rim_mm:.1f} mm across
#   the body sits {rest_mm:.1f} mm below its own axis, so the bench top is at y = {benchy:.4f}
#     and the camera is RESTING on it (eye height above the bench = its real one)
#
# --- what this variant is --------------------------------------------------------
{variant}
#
#   ftrace -in {scenefile} -camera taking {cli} \\
#          -window -keepwindow -interval 20{ckpt} -o png/camrigs/{key}{suffix}.png
#
# The scene carries a second camera, "witness", an outside 3/4 view of the whole rig
# so you can see the camera model that this photograph is being taken THROUGH:
#
#   ftrace -in {scenefile} -camera witness -mode R -time 120 \\
#          -window -keepwindow -o png/camrigs/{key}{suffix}_witness.png

scene {{ units meters  spectral 360 830 1 }}

{bench}

# --- the camera itself -----------------------------------------------------------
# The GLB carries its own pbrMetallicRoughness materials, so the FTSL `material` line
# is only a FALLBACK for primitives that have none (it would take over entirely under
# `import_materials no`).
material "camfallback" {{ type diffuse  reflect rgb 0.38 0.38 0.42 }}
mesh "body" {{
    file "cameras/{glb}"
    material camfallback{skipline}
    scale {scale:.8g}
    rotate {rot[0]:.5f} {rot[1]:.5f} {rot[2]:.5f}
    translate {tr[0]:.6f} {tr[1]:.6f} {tr[2]:.6f}
}}

{camera}
# An outside view of the rig: the camera model on the bench, with the still life it
# is photographing behind it. Pinhole, so nothing here is a claim about the optics.
camera "witness" {{
    eye -0.80 0.40 -0.62   look_at 0.02 {witness_y:.4f} 0.58   up 0 1 0
    fov_y 42   mode R
    film {{ res 1200 800 }}
}}
"""

ANALYTIC_VARIANT = """\
# ANALYTIC rig -- mode A, the finite-lens forward physical camera. The aperture is a
# real disc of radius focal/2N ({apr:.2f} mm here) and photons are splatted through it,
# so the field of view and the depth of field are exactly right. What it does NOT have
# is glass: the lens is an ideal thin element, so there is no spherical or chromatic
# aberration, no distortion, no field curvature, and the blur discs are perfectly
# uniform right into the corners.
#
# So this is the CONTROL. On this render the star row should be a row of clean points
# all the way to the frame edge, and the bokeh row should be a row of identical round
# discs including the outermost one. Compare against camrig_{key}_sim.ftsl, which is
# the same camera -- same focal length, same f-number, same sensor, same subject --
# with the ideal element replaced by traced glass; every difference between the two
# images is exactly what the glass did."""

SIM_VARIANT = """\
# SIMULATED rig -- mode R through a physical `lens {{ }}` prescription. Every camera ray
# is refracted through actual glass interfaces at its own wavelength before it enters
# the scene, so on top of the same {focal:.0f} mm f/{fstop} geometry you now get spherical
# aberration, lateral colour, field curvature, focus shift and vignetting for free --
# nothing here is a post-effect, it is all just where the rays went.
#
# What to look at, against the analytic control:
#   * the STAR ROW. In focus, a point source images to the lens's point spread function.
#     The control puts every one of them on a pixel; here the outer ones smear into
#     coma and fringe blue-one-side / red-the-other from lateral colour.
#   * the BOKEH ROW. The control's discs are uniform circles everywhere; here the ones
#     near the frame edge go CAT'S-EYE, clipped by the barrel at both ends of the cone.
#   * the CORNERS, which darken, because rays the glass clips are simply lost.
#   * the FOCUS PLANE, which is not quite where you asked -- a doublet has focus shift,
#     and the chart at z = 1.00 may be slightly softer than the analytic one.
#
# Honest caveat: `preset achromat` is a cemented BK7/SF10 doublet scaled to this focal
# length and f-number. It is a REAL lens and it aberrates like one, but it is not this
# camera's actual prescription -- no published prescription exists for any of these
# assets. What is genuinely per-camera here is the focal length, the f-number, the
# sensor format and the fact that the rays are traced. At f/{fstop} a bare doublet
# aberrates rather more than a real {focal:.0f} mm would, so read this as "what glass does",
# not "what this exact lens does"."""


def emit(cam, r, sim):
    s = (cam["body_mm"] / 1000.0) / float(r["size"][0])
    rot, tr, resid = place(r["axis"], r["pupil"], s)
    # how far the body's lowest point sits below the optical axis, after scaling
    rest = float(r["pupil"][1] - r["lo"][1]) * s
    # (the rotation is sub-degree for every asset here, so the axis-aligned bbox
    #  extent is the right measure of "how low does it hang"; assert that)
    assert resid < 1e-6 and abs(rot[0]) < 1.0 and abs(rot[1]) < 1.0, (cam["key"], rot, resid)
    rim_mm = 2000.0 * r["semi"] * s
    apr = cam["focal"] / (2.0 * cam["fstop"])
    W, H = cam["sensor"]; RX, RY = cam["res"]
    key = cam["key"]
    suffix = "_sim" if sim else ""
    scenefile = f"scenes/camrig_{key}{suffix}.ftsl"

    if sim:
        campart = f"""camera "taking" {{
    eye 0 0 0   look_at 0 0 1   up 0 1 0
    mode R
    focus {FOCUS:.3f}
    film {{ res {RX} {RY}  size {W} {H} }}
    lens {{ preset achromat  focal {cam['focal']:g}  fstop {cam['fstop']:g}  glass BK7 }}
}}
"""
        variant = SIM_VARIANT.format(focal=cam["focal"], fstop=cam["fstop"])
        cli, ckpt = "-time 900", ""
    else:
        campart = f"""camera "taking" {{
    eye 0 0 0   look_at 0 0 1   up 0 1 0
    mode A
    lens {cam['focal']:g}   fstop {cam['fstop']:g}   focus {FOCUS:.3f}
    film {{ res {RX} {RY}  size {W} {H} }}
}}
"""
        variant = ANALYTIC_VARIANT.format(apr=apr, key=key)
        cli, ckpt = "-time 900", "  -checkpoint"

    skipline = ""
    if cam["skip"]:
        skipline = "\n    skip_material " + " ".join(cam["skip"]) + \
                   "        # this asset bundles a backdrop plane"

    txt = HEAD.format(
        title=cam["title"], ident=cam["ident"],
        size=np.round(r["size"], 4), body_mm=cam["body_mm"], scale=s,
        axis=np.round(r["axis"], 6), resid=resid,
        pupil=np.round(r["pupil"], 5), rim_mm=rim_mm,
        rest_mm=rest * 1000.0, benchy=-rest,
        variant=variant, scenefile=scenefile, cli=cli, ckpt=ckpt,
        key=key, suffix=suffix, bench=bench(rest), glb=cam["glb"],
        skipline=skipline, rot=rot, tr=tr, camera=campart,
        witness_y=-rest + 0.04)
    path = os.path.join(ROOT, scenefile)
    with open(path, "w") as f:
        f.write(txt)
    return scenefile, s, rest, rim_mm


def main():
    report_only = "--report" in sys.argv
    for cam in CAMERAS:
        r = solve(os.path.join(ROOT, "cameras", cam["glb"]))
        if r is None:
            print(f"!! {cam['key']}: no barrel found"); continue
        s = (cam["body_mm"] / 1000.0) / float(r["size"][0])
        print(f"{cam['key']:<14} scale={s:<12.6g} axis={np.round(r['axis'],5)} "
              f"pupil={np.round(r['pupil'],4)} rim={2000*r['semi']*s:.1f}mm")
        if report_only: continue
        for sim in (False, True):
            f, s, rest, rim = emit(cam, r, sim)
            print(f"    wrote {f}   (bench y = {-rest:+.4f} m)")


if __name__ == "__main__":
    main()
