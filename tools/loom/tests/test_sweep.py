"""M4 tests: rotation-minimizing frames + profile sweeping/skinning geometry.

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom.sweep import (  # noqa: E402
    tangents, rmf_frames, sweep_rings, skin_rings, circle_profile, line_profile,
    ring_normals, _dot, _cross, _norm, _sub,
)


def _circle_spine(n=48, r=1.0):
    return [(r * math.cos(2 * math.pi * i / n), 0.0, r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def test_rmf_is_orthonormal():
    pts = _circle_spine(40)
    T, R, S = rmf_frames(pts, closed=True)
    for i in range(len(pts)):
        assert abs(_dot(T[i], T[i]) - 1) < 1e-6
        assert abs(_dot(R[i], R[i]) - 1) < 1e-6
        assert abs(_dot(S[i], S[i]) - 1) < 1e-6
        assert abs(_dot(T[i], R[i])) < 1e-6
        assert abs(_dot(T[i], S[i])) < 1e-6
        assert abs(_dot(R[i], S[i])) < 1e-6


def test_rmf_has_no_flips():
    # adjacent reference normals should vary smoothly (no sudden sign flip)
    pts = _circle_spine(60)
    _, R, _ = rmf_frames(pts, closed=True)
    n = len(R)
    maxjump = max(math.dist(R[i], R[(i + 1) % n]) for i in range(n))
    assert maxjump < 0.5, f"frame flipped, max jump {maxjump}"


def test_closed_frame_seam_closes():
    # After twist redistribution, transporting the last frame onto point 0 should
    # land close to R[0] (the loop closes with minimal residual roll).
    pts = _circle_spine(50)
    _, R, S = rmf_frames(pts, closed=True)
    n = len(pts)
    # continuity across the wrap seam
    seam = math.dist(R[n - 1], R[0])
    step = max(math.dist(R[i], R[i + 1]) for i in range(n - 1))
    assert seam < step * 3.0 + 1e-6, (seam, step)


def test_tube_mesh_counts_and_radius():
    pts = _circle_spine(32, r=1.0)
    sides = 12
    prof = circle_profile(sides, radius=1.0)
    rings = sweep_rings(pts, prof, scales=[0.2] * len(pts),
                        twists=[0.0] * len(pts), closed_spine=True)
    verts, faces = skin_rings(rings, closed_spine=True, closed_profile=True)
    assert len(verts) == len(pts) * sides
    assert len(faces) == 2 * len(pts) * sides
    # every tube vertex sits ~0.2 from its spine center
    for i, c in enumerate(pts):
        for j in range(sides):
            v = verts[i * sides + j]
            assert abs(math.dist(v, c) - 0.2) < 1e-6


def test_ribbon_mesh_open_profile():
    pts = _circle_spine(20)
    prof = line_profile(0.5)             # 2-point profile
    rings = sweep_rings(pts, prof, scales=[0.3] * len(pts),
                        twists=[0.0] * len(pts), closed_spine=True)
    verts, faces = skin_rings(rings, closed_spine=True, closed_profile=False)
    assert len(verts) == len(pts) * 2
    # open profile (1 edge) x closed spine (n spans) x 2 tris
    assert len(faces) == 2 * len(pts) * 1


def _swept_scene(tmp, preset="tube"):
    import loom as L
    # anchors wobble seamlessly over the loop so the swept geometry animates
    anchors = [L.vec(math.cos(2 * math.pi * i / 6) + L.LoopNoise(cells=3, seed=i, amp=0.2),
                     0.3 * math.sin(i) + L.LoopNoise(cells=3, seed=10 + i, amp=0.2),
                     math.sin(2 * math.pi * i / 6)) for i in range(6)]
    spine = L.LoopCurve(L.PointPath(anchors, closed=True), L.Const(0.0))
    s = L.Scene(L.Camera(eye=(0, 1, 3), look_at=(0, 0, 0), res=(32, 32)))
    s.add(L.Material("m", "diffuse", reflect=0.7))
    if preset == "tube":
        s.add(L.tube(spine, radius=0.12, sides=10, count=40, material="m"))
    elif preset == "ribbon":
        s.add(L.ribbon(spine, width=0.25, count=40, turns=1.0, material="m"))
    elif preset == "blob":
        s.add(L.blob(spine, radius=0.15, count=48, material="m"))
    return s


def test_sweptmesh_emits_mesh_and_writes_obj():
    import tempfile
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        s = _swept_scene(tmp, "tube")
        txt = s.emit(L.Clock.at_frame(3, 24), L.Cache(),
                     assets_dir=tmp, tag="003")
        assert "mesh {" in txt and 'material "m"' in txt
        objs = [p for p in os.listdir(tmp) if p.endswith(".obj")]
        assert objs, "SweptMesh should have written an OBJ asset"
        # the OBJ has vertices and faces
        with open(os.path.join(tmp, objs[0]), encoding="utf-8") as f:
            body = f.read()
        assert body.count("\nv ") + body.startswith("v ") >= 1
        assert "f " in body


def test_sweptmesh_animates_and_is_seamless():
    import tempfile
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        s = _swept_scene(tmp, "ribbon")
        a0 = s.emit(L.Clock.at_frame(0, 24), L.Cache(), assets_dir=tmp, tag="a0")
        amid = s.emit(L.Clock.at_frame(12, 24), L.Cache(), assets_dir=tmp, tag="amid")
        awrap = s.emit(L.Clock.at_frame(24, 24), L.Cache(), assets_dir=tmp, tag="a0")
        # the mesh reference line is identical (same filename); compare OBJ contents
        import glob
        f0 = open(glob.glob(os.path.join(tmp, "*a0.obj"))[0], encoding="utf-8").read()
        fm = open(glob.glob(os.path.join(tmp, "*amid.obj"))[0], encoding="utf-8").read()
        assert f0 != fm, "swept mesh should differ across the loop"
        # frame 24 (tag a0) overwrote frame 0's OBJ with identical content -> seamless
        f0b = open(glob.glob(os.path.join(tmp, "*a0.obj"))[0], encoding="utf-8").read()
        assert f0b == f0, "loop wrap must reproduce frame 0's geometry exactly"


def test_sweptmesh_check_cycles_ok():
    import tempfile
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        s = _swept_scene(tmp, "blob")
        s.check_cycles()  # must not raise; LoopCurve spine is acyclic


# --- closed-spine seam: `turns` must be a whole number, `twist` need not be -------
#
# Regression for the 2026-08-06 bug where a closed sweep's ridges did not line up at
# the seam. The RMF closes correctly on its own (test_closed_frame_seam_closes above);
# the tear came from `turns`, which ACCUMULATES along the spine. A closed spine joins
# ring[n-1] back to ring[0] vertex-for-vertex, so a fractional number of profile
# revolutions cannot close -- and profile symmetry does not rescue it (the silhouette
# maps onto itself, the vertex *indices* do not). Uniform `twist` adds the same angle
# to every ring, so it is seamless at any value; that is the channel an animation must
# drive.

def _lobed_profile(sides=18, lobes=3, bulge=0.34):
    out = []
    for j in range(sides):
        a = 2.0 * math.pi * j / sides
        r = 1.0 + bulge * math.cos(lobes * a)
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def _seam_error(turns, base_tw=0.0, n=120, radius=1.05):
    """Max vertex gap between ring[0] and the continuation of ring[n-1].

    Sweeps the closed ring, then sweeps one step PAST the end (same spine point as
    ring[0], carrying the twist it had accumulated) and compares. If the sweep closes,
    those two rings coincide to within the spine's discretization.
    """
    pts = _circle_spine(n, r=radius)
    prof = _lobed_profile()
    scales = [1.0] * n
    twists = [base_tw + turns * 2.0 * math.pi * (k / n) for k in range(n)]
    rings = sweep_rings(pts, prof, scales, twists, True)

    cont = sweep_rings(pts + [pts[0]], prof, scales + [1.0],
                       twists + [base_tw + turns * 2.0 * math.pi], False)[-1]
    return max(math.dist(a, b) for a, b in zip(cont, rings[0]))


# The floor an integer `turns` reaches (~0.030) is spine discretization, not
# misregistration; a fractional `turns` misses by whole lobes (~2.2-2.7).
_SEAM_FLOOR = 0.05


def test_closed_sweep_seams_on_integer_turns():
    for t in (0.0, 1.0, 2.0, 3.0, -3.0):
        err = _seam_error(t)
        assert err < _SEAM_FLOOR, f"turns={t} should close, got seam error {err:.4f}"


def test_closed_sweep_tears_on_fractional_turns():
    for t in (1.0 / 3.0, 0.5, 1.5, 2.7):
        err = _seam_error(t)
        assert err > 10 * _SEAM_FLOOR, \
            f"turns={t} cannot close; expected a large seam error, got {err:.4f}"


def test_uniform_twist_is_seamless_at_any_value():
    # `twist` is the channel an animation may drive continuously.
    for bt in (0.0, 0.5, 1.0, 2.0943, 3.7, 5.0):
        err = _seam_error(0.0, base_tw=bt)
        assert err < _SEAM_FLOOR, f"uniform twist {bt} should be seamless, got {err:.4f}"
    # ...and it composes with an integer `turns`, which is the supported combination.
    for bt in (0.0, 1.3, 2.9, 4.4):
        err = _seam_error(3.0, base_tw=bt)
        assert err < _SEAM_FLOOR, f"turns=3 twist={bt} should be seamless, got {err:.4f}"


def test_sweptmesh_warns_once_on_fractional_turns_closed_spine():
    import tempfile
    import warnings
    import loom as L

    spine = L.LoopCurve(L.PointPath(
        [L.vec(math.cos(2 * math.pi * i / 6), 0.0, math.sin(2 * math.pi * i / 6))
         for i in range(6)], closed=True), L.Const(0.0))

    def emit_twice(mesh, tmp):
        s = L.Scene(L.Camera(eye=(0, 1, 3), look_at=(0, 0, 0), res=(32, 32)))
        s.add(L.Material("m", "diffuse", reflect=0.7))
        s.add(mesh)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.emit(L.Clock.at_frame(0, 4), L.Cache(), assets_dir=tmp, tag="w0")
            s.emit(L.Clock.at_frame(1, 4), L.Cache(), assets_dir=tmp, tag="w1")
            return [x for x in w if "turns" in str(x.message)]

    kw = dict(count=40, scale=0.2, closed_spine=True, closed_profile=True,
              material="m")
    with tempfile.TemporaryDirectory() as tmp:
        bad = emit_twice(L.SweptMesh(spine, circle_profile(10), turns=0.5,
                                     name="bad", **kw), tmp)
        # exactly once: this runs every frame, so it must not spam an animation
        assert len(bad) == 1, f"expected one warning, got {len(bad)}"
        assert "closed_spine" in str(bad[0].message)

        ok = emit_twice(L.SweptMesh(spine, circle_profile(10), turns=3,
                                    twist=1.234, name="ok", **kw), tmp)
        assert ok == [], f"integer turns + uniform twist must be silent, got {ok}"


# ---- the `smooth` flag is a CREASE ANGLE downstream ------------------------
# Regression: loom emitted its 0/1 flag straight into ftrace's `mesh { smooth <deg> }`,
# which is an angle in DEGREES.  `smooth 1` therefore asked for a 1-degree crease
# threshold, and since ftrace merges two faces only when their normals differ by LESS
# than that, nothing on a real sweep was ever smoothed — a tube's facets step ~30 deg
# around the profile.  Every swept mesh rendered fully faceted at any tessellation
# density.  Guard both halves: the emitted angle, and the geometric fact that makes a
# small angle useless.

def test_smooth_flag_emits_a_usable_crease_angle():
    from loom.scene import _smooth_clause, FTRACE_DEFAULT_CREASE_DEG

    # The 0/1 flag every loom preset authors must become ftrace's default angle,
    # NOT the literal "1".
    on = _smooth_clause(1)
    assert "smooth 1 " not in on, f"1 leaked through as a 1-degree crease: {on!r}"
    assert f"smooth {FTRACE_DEFAULT_CREASE_DEG:g}" in on, on
    assert _smooth_clause(True) == on

    # Off omits the directive entirely rather than emitting `smooth 0`, which ftrace
    # reads as "smoothing on, 0-degree threshold" — a no-op that still costs the weld.
    for off in (0, False, None):
        assert _smooth_clause(off) == "", f"{off!r} should emit nothing"

    # An explicit angle passes through.
    assert "smooth 25" in _smooth_clause(25.0)
    assert "smooth 12.5" in _smooth_clause(12.5)


def test_default_tube_facets_exceed_a_one_degree_crease():
    """Why `smooth 1` was a no-op: the facets are nowhere near coplanar."""
    from collections import defaultdict

    n, sides = 64, 12                      # loom's tube() defaults
    pts = _circle_spine(n, 1.0)
    rings = sweep_rings(pts, circle_profile(sides, 1.0), [0.1] * n, [0.0] * n, True)
    verts, faces = skin_rings(rings, True, True)

    def face_normal(f):
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        return _norm(_cross(_sub(b, a), _sub(c, a)))

    edges = defaultdict(list)
    for fi, f in enumerate(faces):
        for k in range(3):
            edges[tuple(sorted((f[k], f[(k + 1) % 3])))].append(fi)

    angles = []
    for fs in edges.values():
        if len(fs) == 2:
            d = max(-1.0, min(1.0, _dot(face_normal(faces[fs[0]]), face_normal(faces[fs[1]]))))
            angles.append(math.degrees(math.acos(d)))

    # The profile seam steps a full 360/sides between adjacent facets.
    assert max(angles) > 0.9 * (360.0 / sides), f"max dihedral {max(angles):.1f} deg"

    def merged_at(thr):
        return sum(1 for a in angles if a < thr)

    # At 1 deg only the exactly-coplanar in-quad diagonals merge; at 40 deg, everything.
    assert merged_at(1.0) < 0.5 * len(angles), "a 1-degree crease should smooth almost nothing"
    assert merged_at(40.0) == len(angles), "40 deg should smooth the whole tube"


# ---- analytic normals: what the crease heuristic cannot know ----------------
# A crease angle is a GUESS made from triangles alone, and from triangles alone a
# coarsely-sampled smooth curve and a genuine sharp fold are the same thing.  The
# generator doesn't have to guess: the ring lattice IS a parameterisation P(u,v) of
# the swept surface, so dP/du x dP/dv is the real normal.  `ring_normals` computes
# it, and `SweptMesh` ships it whenever `smooth=True` (the default).

def _arc_profile(k, sweep_deg=150.0):
    """An OPEN profile — a circular arc, so the ends need one-sided differences."""
    return [(math.cos(math.radians(-sweep_deg / 2 + sweep_deg * i / (k - 1))),
             math.sin(math.radians(-sweep_deg / 2 + sweep_deg * i / (k - 1))))
            for i in range(k)]


def _straight_spine(n=41, step=0.1):
    return [(0.0, 0.0, i * step) for i in range(n)]


def _angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))


def test_ring_normals_are_exact_on_a_torus():
    """A torus is the case where the discrete answer happens to be the exact one.

    Central differences of a circle land exactly parallel to its tangent
    (cos(t+h) - cos(t-h) = -2 sin t sin h — a scaled tangent, no truncation term),
    and a torus is circular in BOTH parameters.  So the normal here must agree with
    the closed form to floating point, not merely to some tolerance: any deviation
    is an indexing or wrap-around bug, not discretisation error.
    """
    n, sides, r = 48, 12, 0.25
    pts = _circle_spine(n, 1.0)
    rings = sweep_rings(pts, circle_profile(sides, 1.0), [r] * n, [0.0] * n, True)
    verts, _ = skin_rings(rings, True, True)
    nrm = ring_normals(rings, True, True)

    assert len(nrm) == len(verts) == n * sides
    for v in nrm:
        assert abs(math.sqrt(_dot(v, v)) - 1.0) < 1e-12, "normals must be unit length"

    # On a tube of circular cross-section the normal is the direction from the spine
    # point straight out to the vertex.  Sign matters too: outward, not inward.
    for i in range(n):
        for j in range(sides):
            vi = i * sides + j
            expect = _norm(_sub(verts[vi], pts[i]))
            assert _dot(expect, nrm[vi]) > 1 - 1e-9, \
                f"ring {i} slot {j}: {nrm[vi]} vs outward radial {expect}"


def test_ring_normals_agree_with_the_winding_of_skin_rings():
    """The `dv x du` (not `du x dv`) guard.

    `skin_rings` winds a quad (i,j) (i,j+1) (i+1,j+1), whose geometric normal is
    cross(dv, du).  Flip the cross product and every normal points into the solid —
    which a two-sided `abs(n.z)` shader in the viewer would happily hide, and a
    renderer doing one-sided lighting would not.  Cover an open spine too, so the
    one-sided end differences are exercised.
    """
    cases = [
        ("closed torus", _circle_spine(24, 1.0), circle_profile(10, 1.0), True, True),
        ("open lobed tube", _straight_spine(21), _lobed_profile(sides=18), False, True),
        ("open arc sheet", _straight_spine(21), _arc_profile(9), False, False),
    ]
    for label, pts, prof, closed_spine, closed_profile in cases:
        m = len(pts)
        rings = sweep_rings(pts, prof, [0.3] * m, [0.0] * m, closed_spine)
        verts, faces = skin_rings(rings, closed_spine, closed_profile)
        nrm = ring_normals(rings, closed_spine, closed_profile)
        assert len(nrm) == len(verts), label
        worst = 1.0
        for f in faces:
            a, b, c = (verts[k] for k in f)
            fn = _norm(_cross(_sub(b, a), _sub(c, a)))
            for k in f:
                worst = min(worst, _dot(fn, nrm[k]))
        assert worst > 0.0, f"{label}: a vertex normal opposes its own face ({worst:.3f})"


def test_ring_normals_converge_second_order_under_refinement():
    """They are the *true* surface normal, approached at the rate a central
    difference should approach it — quartering the error each time the profile is
    doubled.  Pinning the rate is what distinguishes "converges to the surface" from
    "converges to something else"; a wrong-but-consistent formula would also settle
    down, just not on the right answer and not at this rate.
    """
    spine = _straight_spine(41)
    m = len(spine)

    def normals_at(k):
        rings = sweep_rings(spine, _lobed_profile(sides=k), [0.3] * m, [0.0] * m, False)
        return ring_normals(rings, False, True)

    fine = 18 * 32
    ref = normals_at(fine)
    row = 20                                  # an interior ring, away from the ends
    errs = {}
    for k in (18, 36, 72, 144):
        nr = normals_at(k)
        step = fine // k
        errs[k] = max(_angle_between(nr[row * k + j], ref[row * fine + j * step])
                      for j in range(k))

    assert errs[18] < 12.0, errs                     # even the coarse one is close
    for k in (18, 36, 72):
        assert errs[2 * k] < errs[k] / 3.0, \
            f"doubling {k}->{2 * k} only cut the error {errs[k]:.2f}->{errs[2 * k]:.2f} deg"


def test_analytic_normals_beat_a_crease_angle_where_the_heuristic_must_guess():
    """The motivating measurement, kept as a test so the tradeoff stays visible.

    `r(a) = 1 + 0.34*cos(3a)` at 18 samples has no creases at all, yet its valleys
    turn far more than any usable crease threshold per step.  ftrace (and the loom
    viewer's mesh pane) therefore REFUSE to merge those edges and the surface renders
    faceted — correctly, by the rule, and wrongly, by the surface.  The analytic
    normal has no such failure mode because it never looks at the triangles.
    """
    from collections import defaultdict

    from loom.scene import FTRACE_DEFAULT_CREASE_DEG

    spine = _straight_spine(41)
    m = len(spine)
    k = 18
    rings = sweep_rings(spine, _lobed_profile(sides=k), [0.3] * m, [0.0] * m, False)
    verts, faces = skin_rings(rings, False, True)
    nrm = ring_normals(rings, False, True)

    def face_normal(f):
        a, b, c = (verts[i] for i in f)
        return _norm(_cross(_sub(b, a), _sub(c, a)))

    edges = defaultdict(list)
    for fi, f in enumerate(faces):
        for i in range(3):
            edges[tuple(sorted((f[i], f[(i + 1) % 3])))].append(fi)
    dihedrals = [_angle_between(face_normal(faces[fs[0]]), face_normal(faces[fs[1]]))
                 for fs in edges.values() if len(fs) == 2]

    # Half the story: the heuristic genuinely cannot smooth this mesh.  A large share of
    # its interior edges fold past any threshold you'd dare raise it to (raising it far
    # enough to catch these would also erase real corners elsewhere).
    creased = [a for a in dihedrals if a >= FTRACE_DEFAULT_CREASE_DEG]
    assert max(dihedrals) > 60.0, f"max dihedral only {max(dihedrals):.1f} deg"
    assert len(creased) > 0.10 * len(dihedrals), \
        f"only {len(creased)}/{len(dihedrals)} edges defeat the crease angle"

    # The other half: those same vertices carry normals that are close to the TRUE
    # surface normal.  Measure against a 32x-refined lattice rather than against the
    # facets — comparing to the facets is circular, since a vertex normal necessarily
    # bisects the faces around it and so lands at exactly half the dihedral by
    # construction, whatever formula produced it.
    fine = k * 32
    ref = ring_normals(sweep_rings(spine, _lobed_profile(sides=fine),
                                   [0.3] * m, [0.0] * m, False), False, True)
    row = 20
    err = max(_angle_between(nrm[row * k + j], ref[row * fine + j * 32]) for j in range(k))
    assert err < FTRACE_DEFAULT_CREASE_DEG / 3.0, \
        f"analytic normals are {err:.1f} deg off the true surface"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
