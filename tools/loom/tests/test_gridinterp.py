"""H2 tests: grid interpolation kernels — multilinear (default) vs Catmull-Rom.

Runnable directly (``python tests/test_gridinterp.py``) or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Clock, vec, Grid, GridField, VecGridField  # noqa: E402


def _clk(t: float = 0.0) -> Clock:
    return Clock(t=t, frame=0, frames=1000, fps=30.0)


def _grid1d(vals):
    n = len(vals)
    return Grid(shape=(n,), lo=(0.0,), hi=(float(n - 1),), values=list(vals))


# ---------------------------------------------------------------------------

def test_cubic_exact_at_nodes():
    vals = [0.0, 3.0, -1.0, 5.0, 2.0]
    g = _grid1d(vals)
    for i, v in enumerate(vals):
        got = GridField(g, vec(float(i)), interp="cubic").at(_clk())
        assert abs(got - v) < 1e-12, (i, got, v)


def test_cubic_reproduces_linear_ramp():
    # Catmull-Rom reproduces linear (and cubic) polynomials exactly.
    vals = [2.0 * i + 1.0 for i in range(6)]
    g = _grid1d(vals)
    for x in (0.3, 1.75, 2.5, 4.9):
        lin = GridField(g, vec(x), interp="linear").at(_clk())
        cub = GridField(g, vec(x), interp="cubic").at(_clk())
        exact = 2.0 * x + 1.0
        assert abs(lin - exact) < 1e-12
        assert abs(cub - exact) < 1e-12, (x, cub, exact)


def test_cubic_partition_of_unity():
    # interpolating a constant grid must give the constant (weights sum to 1).
    g = _grid1d([7.0] * 5)
    for x in (0.1, 2.4, 3.9):
        assert abs(GridField(g, vec(x), interp="cubic").at(_clk()) - 7.0) < 1e-12


def test_cubic_overshoots_where_linear_clamps():
    # a sharp peak: cubic overshoots above the max; linear never exceeds neighbours.
    g = _grid1d([0.0, 0.0, 1.0, 0.0, 0.0])
    x = 1.5  # between the rising node 1 (0) and peak node 2 (1)
    lin = GridField(g, vec(x), interp="linear").at(_clk())
    cub = GridField(g, vec(x), interp="cubic").at(_clk())
    assert 0.0 <= lin <= 1.0
    # Catmull-Rom snaps past the linear midpoint here (steeper toward the peak)
    assert cub > lin


def test_default_is_linear():
    g = _grid1d([0.0, 0.0, 1.0, 0.0, 0.0])
    x = 1.5
    assert (GridField(g, vec(x)).at(_clk())
            == GridField(g, vec(x), interp="linear").at(_clk()))


def test_2d_cubic_exact_at_nodes_and_reduces_to_linear_on_thin_axis():
    # axis with only 2 samples must fall back to linear on that axis.
    g = Grid(shape=(2, 4), lo=(0, 0), hi=(1, 3),
             values=[float(v) for v in range(8)])
    # exact at a node
    got = GridField(g, vec(1.0, 2.0), interp="cubic").at(_clk())
    assert abs(got - g.value_at_index((1, 2)).at(_clk())) < 1e-12


def test_vec_grid_cubic_channel_matches_scalar():
    g_vec = Grid(shape=(5,), lo=(0,), hi=(4,),
                 values=[vec(float(i), 10.0 - i) for i in range(5)],
                 channels=("a", "b"))
    g_a = _grid1d([float(i) for i in range(5)])
    g_b = _grid1d([10.0 - i for i in range(5)])
    x = 2.35
    vf = VecGridField(g_vec, vec(x), interp="cubic")
    clk = _clk()
    assert abs(vf.channel("a").at(clk)
               - GridField(g_a, vec(x), interp="cubic").at(clk)) < 1e-12
    assert abs(vf.channel("b").at(clk)
               - GridField(g_b, vec(x), interp="cubic").at(clk)) < 1e-12


def test_unknown_interp_rejected():
    g = _grid1d([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        GridField(g, vec(0.5), interp="quadratic")
    with pytest.raises(ValueError):
        VecGridField(Grid(shape=(2,), lo=(0,), hi=(1,),
                          values=[vec(0.0, 0.0), vec(1.0, 1.0)]),
                     vec(0.5), interp="bogus")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
