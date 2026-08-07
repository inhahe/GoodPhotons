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
    _dot, _cross, _norm, _sub,
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
