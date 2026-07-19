"""H4 tests: FieldCurve — a curve routed through a field.

Runnable directly (``python tests/test_fieldcurve.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Clock, Const, vec, VecSignal, PointPath, Grid, Scatter,
    GridField, VecGridField, ScatterField, LoopCurve, FieldCurve,
    detect_signal_cycle, walk,
)


def _clk(t: float = 0.0) -> Clock:
    return Clock(t=t, frame=0, frames=1000, fps=30.0)


def _square_path():
    # a closed loop over the unit square corners
    return PointPath([vec(0.0, 0.0), vec(1.0, 0.0), vec(1.0, 1.0), vec(0.0, 1.0)],
                     closed=True)


def _scalar_grid():
    # values = x + 10*y at the 4 corners (index order: (i0,i1) C-order over (2,2))
    return Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
                values=[0.0, 10.0, 1.0, 11.0])


# ---------------------------------------------------------------------------

def test_fieldcurve_coords_match_curve():
    path = _square_path()
    u = Const(0.3)
    grid = _scalar_grid()
    fc = FieldCurve(path, lambda q: GridField(grid, q), u=u)
    ref = LoopCurve(path, u)
    assert fc.position.at(_clk()) == ref.at(_clk())


def test_fieldcurve_value_matches_field_at_coords():
    path = _square_path()
    grid = _scalar_grid()
    for uval in (0.0, 0.2, 0.55, 0.9):
        fc = FieldCurve(path, lambda q: GridField(grid, q), u=Const(uval))
        coords, vals = fc.sample(uval, _clk())
        # value should equal the field evaluated directly at those coords
        direct = GridField(grid, vec(*coords)).at(_clk())
        assert abs(vals[0] - direct) < 1e-12, (uval, vals, direct)


def test_fieldcurve_vector_channels_named():
    path = _square_path()
    g = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
             values=[vec(0.0, 0.0), vec(0.0, 5.0), vec(2.0, 0.0), vec(2.0, 5.0)],
             channels=("temp", "flow"))
    fc = FieldCurve(path, lambda q: VecGridField(g, q), u=Const(0.4))
    assert fc.channels() == ("temp", "flow")
    coords, vals = fc.sample(0.4, _clk())
    assert set(vals.keys()) == {"temp", "flow"}
    # named-channel DAG view matches the sampled map
    clk = _clk()
    assert abs(fc.channel("temp").at(clk) - vals["temp"]) < 1e-9
    # ... but note the DAG channel uses the *bound* u (also 0.4 here)


def test_fieldcurve_channel_drives_downstream():
    path = _square_path()
    g = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
             values=[vec(0.0, 0.0), vec(0.0, 5.0), vec(2.0, 0.0), vec(2.0, 5.0)],
             channels=("temp", "flow"))
    fc = FieldCurve(path, lambda q: VecGridField(g, q), u=Const(0.4))
    downstream = fc.channel("flow") * 2.0 + fc.position.components[0]
    detect_signal_cycle(downstream)          # must not raise
    ids = {n.id for n in walk(downstream)}
    assert g.id in ids and path.id in ids    # dataset + path threaded into the DAG


def test_fieldcurve_scalar_value_map_keyed_zero():
    path = _square_path()
    grid = _scalar_grid()
    fc = FieldCurve(path, lambda q: GridField(grid, q), u=Const(0.1))
    _, vals = fc.sample(0.1, _clk())
    assert list(vals.keys()) == [0]


def test_fieldcurve_scatter_field():
    path = _square_path()
    sc = Scatter([(vec(0.0, 0.0), 0.0), (vec(1.0, 0.0), 1.0),
                  (vec(1.0, 1.0), 2.0), (vec(0.0, 1.0), 1.0)])
    fc = FieldCurve(path, lambda q: ScatterField(sc, q), u=Const(0.25))
    coords, vals = fc.sample(0.25, _clk())
    direct = ScatterField(sc, vec(*coords)).at(_clk())
    assert abs(vals[0] - direct) < 1e-12


def test_fieldcurve_over_ready_position():
    # a bare VecSignal position (no PointPath): DAG use works, explicit sample() does not.
    grid = _scalar_grid()
    pos = VecSignal([Const(0.5), Const(0.5)])
    fc = FieldCurve(pos, lambda q: GridField(grid, q))
    assert abs(fc.value.at(_clk()) - GridField(grid, pos).at(_clk())) < 1e-12
    with pytest.raises(TypeError):
        fc.sample(0.3, _clk())


def test_fieldcurve_bad_args():
    path = _square_path()
    grid = _scalar_grid()
    with pytest.raises(ValueError):
        FieldCurve(path, lambda q: GridField(grid, q))   # PointPath needs u
    with pytest.raises(TypeError):
        FieldCurve(path, GridField(grid, vec(0.0, 0.0)), u=Const(0.1))  # not a builder
    with pytest.raises(TypeError):
        FieldCurve(42, lambda q: GridField(grid, q), u=Const(0.1))      # bad curve


def test_fieldcurve_dim_mismatch_message():
    # a 2-D curve routed through a 3-D grid: the field builder's dim check fires,
    # and FieldCurve re-raises it with its own context so the author knows where.
    path = _square_path()                      # dim 2
    grid3 = Grid(shape=(2, 2, 2), lo=(0, 0, 0), hi=(1, 1, 1),
                 values=[float(i) for i in range(8)])
    with pytest.raises(ValueError) as ei:
        FieldCurve(path, lambda q: GridField(grid3, q), u=Const(0.1))
    assert "FieldCurve" in str(ei.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
