"""Dataset placement Transform: a Grid/Scatter carries a local->world Transform,
and a world-space sampling curve is inverse-mapped into the dataset's local frame.

This is the "decoupled curve" feature: moving / resizing / skewing the data object
changes what values the (independent, world-space) curve reads back -- while the
curve itself stays put.  Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Clock, Const, vec, VecSignal, PointPath, Grid, Scatter,
    GridField, VecGridField, ScatterField, FieldCurve, Transform, Sine,
    detect_signal_cycle, walk,
)


def _clk(t: float = 0.0) -> Clock:
    return Clock(t=t, frame=0, frames=1000, fps=30.0)


def _linear_grid():
    # scalar field f(x, y) = x + 10*y sampled at the unit-square corners
    return Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1), values=[0.0, 10.0, 1.0, 11.0])


# ---- Transform.inverse_apply round-trips the forward map ------------------

def _fwd3(tr, rot, scl, sk, local):
    """Reference forward transform matching ftrace (mesh.h MeshXform + shear)."""
    x, y, z = local
    a, b, c = sk
    x, y, z = x + a * y + b * z, y + c * z, z            # shear
    x, y, z = x * scl[0], y * scl[1], z * scl[2]         # scale
    d2r = math.pi / 180.0
    cx, sx = math.cos(rot[0] * d2r), math.sin(rot[0] * d2r)
    cy, sy = math.cos(rot[1] * d2r), math.sin(rot[1] * d2r)
    cz, sz = math.cos(rot[2] * d2r), math.sin(rot[2] * d2r)
    x, y, z = x, cx * y - sx * z, sx * y + cx * z        # Rx
    x, y, z = cy * x + sy * z, y, -sy * x + cy * z       # Ry
    x, y, z = cz * x - sz * y, sz * x + cz * y, z        # Rz
    return (x + tr[0], y + tr[1], z + tr[2])


def test_inverse_apply_roundtrip_3d():
    tr, rot, scl, sk = (0.3, -0.7, 1.1), (20, -35, 50), (1.4, 0.6, 2.1), (0.5, 0.2, -0.3)
    local = (0.4, -0.2, 0.9)
    world = _fwd3(tr, rot, scl, sk, local)
    T = Transform(translate=tr, rotate=rot, scale=scl, skew=sk)
    rec = T.inverse_apply(vec(*world)).at(_clk())
    assert max(abs(a - b) for a, b in zip(local, rec)) < 1e-9


def test_inverse_apply_roundtrip_2d():
    # 2-D uses in-plane params only: translate/scale XY, rotate Z, skew X-along-Y
    a = 0.6
    local = (0.3, -0.4)
    x, y = local
    x, y = x + a * y, y
    x, y = x * 1.7, y * 0.8
    d2r = math.pi / 180.0
    c, s = math.cos(33 * d2r), math.sin(33 * d2r)
    world = (c * x - s * y + 0.5, s * x + c * y - 0.2)
    T = Transform(translate=(0.5, -0.2, 0), rotate=(0, 0, 33), scale=(1.7, 0.8, 1), skew=(a, 0, 0))
    rec = T.inverse_apply(vec(*world)).at(_clk())
    assert max(abs(p - q) for p, q in zip(local, rec)) < 1e-9


def test_inverse_apply_rejects_bad_dim():
    T = Transform(translate=(1, 0, 0))
    with pytest.raises(ValueError):
        T.inverse_apply(vec(1.0, 2.0, 3.0, 4.0))   # dim 4 unsupported


# ---- the decoupling behaviour on real fields ------------------------------

def test_grid_translate_changes_sampled_value():
    q = VecSignal([Const(0.5), Const(0.5)])
    base = GridField(_linear_grid(), q).at(_clk())
    assert abs(base - 5.5) < 1e-12                      # 0.5 + 10*0.5
    # translate the data object +0.3 in x: world (0.5,0.5) -> local (0.2,0.5)
    g = _linear_grid().transformed(translate=(0.3, 0, 0))
    assert abs(GridField(g, q).at(_clk()) - 5.2) < 1e-12


def test_grid_scale_changes_sampled_value():
    q = VecSignal([Const(0.5), Const(0.5)])
    g = _linear_grid().transformed(scale=2.0)           # world (0.5,0.5) -> local (0.25,0.25)
    assert abs(GridField(g, q).at(_clk()) - 2.75) < 1e-12


def test_grid_skew_changes_sampled_value():
    q = VecSignal([Const(0.5), Const(0.5)])
    # skew x-along-y by a=0.4: forward x'=x+0.4y ; inverse local x = 0.5 - 0.4*0.5 = 0.3
    g = _linear_grid().transformed(skew=(0.4, 0, 0))
    assert abs(GridField(g, q).at(_clk()) - (0.3 + 5.0)) < 1e-12


def test_identity_transform_is_a_noop():
    q = VecSignal([Const(0.3), Const(0.7)])
    plain = GridField(_linear_grid(), q).at(_clk())
    ident = GridField(_linear_grid().transformed(Transform()), q).at(_clk())
    assert plain == ident


def test_scatter_translate_changes_sampled_value():
    sc = Scatter([(vec(0.0, 0.0), 0.0), (vec(1.0, 0.0), 1.0),
                  (vec(1.0, 1.0), 2.0), (vec(0.0, 1.0), 1.0)])
    q = VecSignal([Const(0.5), Const(0.5)])
    base = ScatterField(sc, q).at(_clk())
    moved = ScatterField(
        Scatter([(vec(0.0, 0.0), 0.0), (vec(1.0, 0.0), 1.0),
                 (vec(1.0, 1.0), 2.0), (vec(0.0, 1.0), 1.0)]).transformed(translate=(0.4, 0, 0)),
        q).at(_clk())
    assert abs(base - moved) > 1e-6                     # the curve now reads elsewhere


def test_fieldcurve_over_transformed_grid_threads_params_into_dag():
    # an *animated* transform param must be cycle-checkable and reachable in the DAG
    path = PointPath([vec(0.0, 0.0), vec(1.0, 0.0), vec(1.0, 1.0), vec(0.0, 1.0)], closed=True)
    g = _linear_grid().transformed(translate=(Sine(cycles=1, amp=0.3), 0, 0))
    fc = FieldCurve(path, lambda qq: GridField(g, qq), u=Const(0.3))
    detect_signal_cycle(fc.value)                        # must not raise
    ids = {n.id for n in walk(fc.value)}
    assert g.id in ids and path.id in ids


def test_animated_transform_value_moves_over_time():
    q = VecSignal([Const(0.5), Const(0.5)])
    g = _linear_grid().transformed(translate=(Sine(cycles=1, amp=0.3), 0, 0))
    f = GridField(g, q)
    v0 = f.at(_clk(0.0))
    v1 = f.at(_clk(0.25))                                # sine at quarter phase != 0
    assert abs(v0 - v1) > 1e-6


def test_vecgrid_transform_applies():
    q = VecSignal([Const(0.5), Const(0.5)])
    mk = lambda: Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
                      values=[vec(0.0, 0.0), vec(0.0, 10.0), vec(1.0, 0.0), vec(1.0, 10.0)],
                      channels=("cx", "cy"))
    base = VecGridField(mk(), q).at(_clk())
    moved = VecGridField(mk().transformed(translate=(0.3, 0, 0)), q).at(_clk())
    assert base != moved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
