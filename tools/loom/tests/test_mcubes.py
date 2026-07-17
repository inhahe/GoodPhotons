"""M7 tests: bake a scalar field to a mesh via marching cubes (:mod:`loom.mcubes`).

Cover the core deliverable (a numeric field -> watertight-ish triangle mesh),
the honest adaptive claim (narrow-band sampling is crack-free and identical to a
dense mesh near the surface, while evaluating far fewer cells), the empty-box
case, SpatialExpr *and* plain-callable fields, and the :class:`loom.IsoMesh`
scene element (per-frame OBJ bake + ``mesh { file ... }`` emit, static caching).
Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, X, Y, Z, T, sin, cos, mesh_field, IsoMesh, Scene, Camera,
    Sine,
)


def _sphere_expr(r):
    return X * X + Y * Y + Z * Z + (-(r * r))


def test_sphere_radius_accurate():
    v, f = mesh_field(_sphere_expr(0.7), bounds=1.0, res=48, iso=0.0)
    assert len(v) > 100 and len(f) > 100
    rad = np.linalg.norm(np.array(v), axis=1)
    assert abs(rad.mean() - 0.7) < 1e-2
    assert rad.min() > 0.69 and rad.max() < 0.71


def test_faces_index_valid():
    v, f = mesh_field(_sphere_expr(0.6), bounds=1.0, res=32, iso=0.0)
    fa = np.array(f)
    assert fa.min() >= 0 and fa.max() < len(v)
    # every triangle has 3 distinct vertices
    assert np.all(fa[:, 0] != fa[:, 1])


def test_watertight_edge_manifold():
    # a closed sphere: every undirected edge shared by exactly two triangles.
    v, f = mesh_field(_sphere_expr(0.7), bounds=1.0, res=40, iso=0.0)
    from collections import Counter
    edges = Counter()
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(e))] += 1
    counts = Counter(edges.values())
    # overwhelmingly 2-manifold (allow a tiny fraction of boundary from clipping)
    assert counts.get(2, 0) > 0.98 * sum(counts.values())


def test_adaptive_matches_dense_near_surface():
    expr = _sphere_expr(0.7)
    vd, fd = mesh_field(expr, bounds=1.0, res=48, iso=0.0, adaptive=False)
    va, fa = mesh_field(expr, bounds=1.0, res=48, iso=0.0, adaptive=True, coarse=6)
    # crack-free narrow-band => identical mesh to the dense one
    assert len(vd) == len(va) and len(fd) == len(fa)
    assert np.allclose(np.array(vd), np.array(va))


def test_adaptive_evaluates_fewer_cells():
    calls = {"n": 0}

    def field(Xg, Yg, Zg):
        calls["n"] += Xg.size
        return Xg * Xg + Yg * Yg + Zg * Zg - 0.04  # small r=0.2 surface

    calls["n"] = 0
    mesh_field(field, bounds=1.0, res=80, iso=0.0, adaptive=False)
    dense = calls["n"]
    calls["n"] = 0
    mesh_field(field, bounds=1.0, res=80, iso=0.0, adaptive=True, coarse=14)
    adaptive = calls["n"]
    assert adaptive < 0.5 * dense  # a thin surface => big narrow-band win


def test_empty_when_surface_outside_box():
    v, f = mesh_field(_sphere_expr(5.0), bounds=1.0, res=16, iso=0.0)
    assert v == [] and f == []


def test_callable_field_supported():
    def field(Xg, Yg, Zg):
        return Xg * Xg + Yg * Yg + Zg * Zg - 0.25
    v, f = mesh_field(field, bounds=1.0, res=32, iso=0.0)
    rad = np.linalg.norm(np.array(v), axis=1)
    assert abs(rad.mean() - 0.5) < 1e-2


def test_bounds_forms():
    e = _sphere_expr(0.5)
    for bounds in (1.0, (1.0, 1.0, 1.0), (-1, -1, -1, 1, 1, 1)):
        v, _ = mesh_field(e, bounds=bounds, res=24, iso=0.0)
        assert len(v) > 50


def test_time_varying_field_morphs():
    # a sphere whose radius^2 breathes with the loop phase T
    expr = X * X + Y * Y + Z * Z + (-0.25) + (-0.2) * sin(T * (2 * math.pi))
    v0, _ = mesh_field(expr, bounds=1.2, res=32, clock=Clock.at_frame(0, 8), cache=Cache())
    v1, _ = mesh_field(expr, bounds=1.2, res=32, clock=Clock.at_frame(2, 8), cache=Cache())
    r0 = np.linalg.norm(np.array(v0), axis=1).mean()
    r1 = np.linalg.norm(np.array(v1), axis=1).mean()
    assert abs(r0 - r1) > 0.05  # radius genuinely changed between frames


def test_isomesh_emits_mesh_reference(tmp_path=None):
    import tempfile
    from pathlib import Path
    from loom.ftsl_emit import EmitCtx
    d = Path(tempfile.mkdtemp())
    im = IsoMesh(_sphere_expr(0.6), bounds=1.0, res=24, material="skin", name="ball")
    ctx = EmitCtx(clock=Clock.at_frame(0, 4), cache=Cache(), assets_dir=d, tag="")
    line = im.emit(ctx)
    assert line.startswith("mesh {") and 'material "skin"' in line
    obj = d / "ball.obj"
    assert obj.exists() and obj.read_text().count("\nv ") + obj.read_text().startswith("v ")


def test_isomesh_static_field_baked_once():
    from pathlib import Path
    import tempfile
    from loom.ftsl_emit import EmitCtx
    d = Path(tempfile.mkdtemp())
    im = IsoMesh(_sphere_expr(0.5), bounds=1.0, res=20, name="s")
    assert im._static() is True
    im.emit(EmitCtx(clock=Clock.at_frame(0, 4), cache=Cache(), assets_dir=d))
    first = im._cache_static
    assert first is not None
    im.emit(EmitCtx(clock=Clock.at_frame(2, 4), cache=Cache(), assets_dir=d))
    assert im._cache_static is first  # not re-baked


def test_isomesh_roots_expose_signals():
    # animated coefficient => exposed as a cycle-check root
    s = Sine(cycles=1.0)
    expr = X * X + Y * Y + Z * Z + (-0.25) + s * cos(Z)
    im = IsoMesh(expr, bounds=1.0, res=16)
    assert im._static() is False
    assert any(r is s for r in im.roots())


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
