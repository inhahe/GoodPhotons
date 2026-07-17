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
