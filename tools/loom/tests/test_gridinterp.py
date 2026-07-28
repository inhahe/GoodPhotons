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


# --- J1: out-of-domain policy (clamp / raise / wrap) -----------------------

def test_on_outside_default_is_clamp():
    g = _grid1d([0.0, 1.0, 2.0, 3.0])
    for x in (-2.0, -0.5, 3.5, 9.0):
        default = GridField(g, vec(x)).at(_clk())
        clamp = GridField(g, vec(x), on_outside="clamp").at(_clk())
        assert default == clamp
    # clamp edge-extends to the boundary sample
    assert GridField(g, vec(-5.0), on_outside="clamp").at(_clk()) == 0.0
    assert GridField(g, vec(99.0), on_outside="clamp").at(_clk()) == 3.0


def test_on_outside_raise():
    g = _grid1d([0.0, 1.0, 2.0, 3.0])          # domain [0, 3]
    # inside (and exactly on the boundary) must NOT raise
    for x in (0.0, 1.5, 3.0):
        GridField(g, vec(x), on_outside="raise").at(_clk())
    # outside must raise, on either side
    for x in (-0.5, 3.5):
        with pytest.raises(ValueError):
            GridField(g, vec(x), on_outside="raise").at(_clk())


def test_on_outside_wrap_periodic():
    # value[0] == value[n-1] makes this a genuine period-4 field.
    g = _grid1d([0.0, 1.0, 2.0, 3.0, 0.0])     # domain [0, 4], period 4
    def w(x):
        return GridField(g, vec(x), on_outside="wrap").at(_clk())
    # folding by one period returns the same value
    assert abs(w(4.5) - w(0.5)) < 1e-12
    assert abs(w(-0.5) - w(3.5)) < 1e-12
    # explicit values: 0.5 -> lerp(v0=0, v1=1) = 0.5
    assert abs(w(0.5) - 0.5) < 1e-12
    # -0.5 wraps to 3.5 -> lerp(v3=3, v4->v0=0) = 1.5
    assert abs(w(-0.5) - 1.5) < 1e-12
    # at the seam coord == hi folds to sample 0
    assert abs(w(4.0) - 0.0) < 1e-12


def test_on_outside_wrap_cubic_constant_and_node():
    g = _grid1d([5.0, 5.0, 5.0, 5.0, 5.0])
    # partition of unity holds under wrap, even for a query a full period out
    assert abs(GridField(g, vec(6.3), interp="cubic",
                         on_outside="wrap").at(_clk()) - 5.0) < 1e-12
    # exact at an interior node under cubic+wrap
    g2 = _grid1d([0.0, 2.0, -1.0, 4.0, 0.0])
    assert abs(GridField(g2, vec(2.0), interp="cubic",
                         on_outside="wrap").at(_clk()) - (-1.0)) < 1e-12


def test_on_outside_vec_field_wrap():
    g = Grid(shape=(5,), lo=(0,), hi=(4,),
             values=[vec(float(i % 4), 0.0) for i in range(5)],  # a: 0,1,2,3,0
             channels=("a", "b"))
    vf = VecGridField(g, vec(4.5), on_outside="wrap")
    assert abs(vf.channel("a").at(_clk()) - 0.5) < 1e-12


def test_on_outside_extrapolate_linear():
    # a linear ramp continues past both boundary cells with slope preserved.
    g = _grid1d([0.0, 1.0, 2.0, 3.0])          # value == coord on [0,3], slope 1
    def e(x):
        return GridField(g, vec(x), on_outside="extrapolate").at(_clk())
    assert abs(e(-1.0) - (-1.0)) < 1e-12       # extend below: 0 - 1
    assert abs(e(-2.5) - (-2.5)) < 1e-12
    assert abs(e(4.0) - 4.0) < 1e-12           # extend above: 3 + 1
    assert abs(e(6.25) - 6.25) < 1e-12
    # inside the domain it matches clamp/linear exactly
    assert abs(e(1.5) - 1.5) < 1e-12


def test_on_outside_extrapolate_differs_from_clamp():
    g = _grid1d([0.0, 1.0, 2.0, 3.0])
    clamp = GridField(g, vec(9.0), on_outside="clamp").at(_clk())
    extrap = GridField(g, vec(9.0), on_outside="extrapolate").at(_clk())
    assert clamp == 3.0                        # edge-extend pins to boundary sample
    assert extrap > 3.0                        # extrapolation keeps rising


def test_on_outside_extrapolate_cubic_ramp():
    # Catmull-Rom reproduces a linear ramp, and extrapolate keeps it linear off-edge.
    vals = [2.0 * i + 1.0 for i in range(5)]
    g = _grid1d(vals)
    for x in (-1.5, 5.5):
        got = GridField(g, vec(x), interp="cubic",
                        on_outside="extrapolate").at(_clk())
        assert abs(got - (2.0 * x + 1.0)) < 1e-9, (x, got)


def test_on_outside_unknown_rejected():
    g = _grid1d([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        GridField(g, vec(0.5), on_outside="reflect")


# ---------------------------------------------------------------------------
# lo/hi defaults + broadcast + scalar-hi uniform-lattice derivation
# ---------------------------------------------------------------------------

def test_lo_default_is_zero_hi_default_is_index_lattice():
    # lo omitted -> all-zeros; hi omitted -> unit-spacing index lattice (hi=shape-1),
    # so a query coordinate equals a sample index.
    g = Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5])
    assert g.lo == (0.0, 0.0)
    assert g.hi == (1.0, 2.0)                       # shape-1 per axis
    # coordinate == index: (1,2) is the last corner -> value 5
    assert g.sample(1.0, 2.0) == 5.0
    assert g.sample(0.0, 1.0) == 1.0               # exactly sample [0,1]


def test_lo_scalar_broadcasts():
    g = Grid(shape=(2, 2), values=[0, 1, 2, 3], lo=0.5, hi=(1.5, 1.5))
    assert g.lo == (0.5, 0.5)


def test_hi_scalar_derives_uniform_lattice():
    # scalar hi pins axis 0 and derives the rest as ONE spacing h on every axis.
    g = Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], lo=0.0, hi=1.0)
    # h = (1-0)/(2-1) = 1 ; hi[1] = 0 + 1*(3-1) = 2  -> box [0,1]x[0,2]
    assert g.hi == (1.0, 2.0)
    # uniform spacing => sample coords are the same on both axes
    assert g.axis_coords(0) == [0.0, 1.0]
    assert g.axis_coords(1) == [0.0, 1.0, 2.0]


def test_hi_length1_sequence_same_as_scalar():
    a = Grid(shape=(2, 4), values=list(range(8)), lo=0.0, hi=3.0)
    b = Grid(shape=(2, 4), values=list(range(8)), lo=0.0, hi=(3.0,))
    assert a.hi == b.hi


def test_hi_full_tuple_is_verbatim():
    g = Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], lo=(0, 0), hi=(1, 1))
    assert g.hi == (1.0, 1.0)                       # deliberately anisotropic cells


def test_bad_lo_hi_length_rejected():
    with pytest.raises(ValueError):
        Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], lo=(0, 0, 0))
    with pytest.raises(ValueError):
        Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], hi=(1, 1, 1))


# ---------------------------------------------------------------------------
# shape inference from nested values (shape= is optional)
# ---------------------------------------------------------------------------

def test_shape_inferred_from_nested_2d():
    # nesting carries the lattice shape; no shape= needed.
    g = Grid([[0, 1, 2], [3, 4, 5]])
    assert g.shape == (2, 3)
    assert g.sample(1.0, 2.0) == 5.0          # last corner (index lattice default)
    # identical to the explicit-shape flat form
    assert g.shape == Grid([0, 1, 2, 3, 4, 5], shape=(2, 3)).shape


def test_flat_values_are_1d_by_default():
    g = Grid([0.0, 1.0, 2.0, 3.0])
    assert g.shape == (4,)
    assert g.sample(2.0) == 2.0


def test_explicit_shape_folds_flat_list():
    g = Grid([0, 1, 2, 3, 4, 5], shape=(2, 3))
    assert g.shape == (2, 3)
    assert g.sample(0.0, 1.0) == 1.0


def test_shape_inferred_3d_nested():
    g = Grid([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    assert g.shape == (2, 2, 2)
    assert g.sample(1.0, 1.0, 1.0) == 7.0


def test_shape_inferred_vector_grid_nested():
    # bare lists are axes; vec(...) is a stored value → a 2x2 grid of 2-vectors.
    g = Grid([[vec(0.0, 10.0), vec(1.0, 11.0)],
              [vec(2.0, 12.0), vec(3.0, 13.0)]])
    assert g.shape == (2, 2)
    assert g.is_vector and g.value_dim == 2
    assert g.sample(1.0, 1.0) == (3.0, 13.0)


def test_ragged_nested_values_rejected():
    with pytest.raises(ValueError):
        Grid([[0, 1, 2], [3, 4]])             # rows of differing length


def test_explicit_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        Grid([0, 1, 2, 3, 4], shape=(2, 3))   # 5 values, 6 slots


# ---------------------------------------------------------------------------
# __call__ / sample() sugar (ftsl-style n(x,y))
# ---------------------------------------------------------------------------

def test_call_matches_explicit_gridfield():
    g = Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], lo=(0, 0), hi=(1, 1))
    # scalar-args form and vec form both equal the explicit GridField
    want = GridField(g, vec(0.5, 0.5)).at(_clk())
    assert g(0.5, 0.5).at(_clk()) == want
    assert g(vec(0.5, 0.5)).at(_clk()) == want
    assert want == 2.5


def test_call_passes_interp_and_outside_kwargs():
    g = _grid1d([0.0, 1.0, 4.0, 9.0])
    assert g(1.5, interp="cubic").at(_clk()) == GridField(
        g, vec(1.5), interp="cubic").at(_clk())
    assert g(99.0, on_outside="clamp").at(_clk()) == 9.0


def test_sample_eager_scalar_and_vector():
    g = Grid(shape=(2, 3), values=[0, 1, 2, 3, 4, 5], lo=(0, 0), hi=(1, 1))
    assert g.sample(0.5, 0.5) == 2.5               # no clock needed (static grid)
    gv = Grid(shape=(2,), values=[vec(0.0, 10.0), vec(2.0, 20.0)], lo=(0,), hi=(1,))
    assert gv.sample(0.5) == (1.0, 15.0)


def test_call_dispatches_vector_grid():
    from loom import VecGridField
    gv = Grid(shape=(2,), values=[vec(0.0, 10.0), vec(2.0, 20.0)],
              lo=(0,), hi=(1,), channels=("a", "b"))
    f = gv(0.5)
    assert isinstance(f, VecGridField)
    assert f.channel("b").at(_clk()) == 15.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
