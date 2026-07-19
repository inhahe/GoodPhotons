"""H1 tests: vector-valued Grid/Scatter fields (shared domain weights).

Runnable directly (``python tests/test_vecfields.py``) or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, Sine, vec, VecSignal, detect_signal_cycle, walk,
    Grid, Scatter, GridField, ScatterField, VecGridField, VecScatterField,
)


def _clk(t: float, frame: int = 0) -> Clock:
    return Clock(t=t, frame=frame, frames=1000, fps=30.0)


# ---------------------------------------------------------------------------
# dataset value-model inference
# ---------------------------------------------------------------------------

def test_grid_infers_scalar_vs_vector():
    g_scalar = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1), values=[0.0, 1.0, 2.0, 3.0])
    assert g_scalar.value_dim == 1 and g_scalar.is_vector is False
    g_vec = Grid(shape=(2,), lo=(0,), hi=(1,),
                 values=[vec(0.0, 10.0), vec(1.0, 20.0)])
    assert g_vec.value_dim == 2 and g_vec.is_vector is True


def test_grid_rejects_mixed_values():
    with pytest.raises(ValueError):
        Grid(shape=(2,), lo=(0,), hi=(1,), values=[0.0, vec(1.0, 2.0)])
    with pytest.raises(ValueError):
        Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(1.0, 2.0, 3.0)])


def test_channels_validation():
    with pytest.raises(ValueError):
        Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(1.0, 1.0)],
             channels=("r", "g", "b"))          # 3 names, dim 2
    with pytest.raises(ValueError):
        Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(1.0, 1.0)],
             channels=("r", "r"))               # duplicate
    g = Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(1.0, 1.0)],
             channels=("u", "v"))
    assert g.channel_index("v") == 1
    assert g.channel_index(-1) == 1
    with pytest.raises(KeyError):
        g.channel_index("w")
    with pytest.raises(IndexError):
        g.channel_index(5)


# ---------------------------------------------------------------------------
# VecGridField
# ---------------------------------------------------------------------------

def test_vec_grid_corners_and_center():
    # 2x2 grid of 2-vectors; channel 0 = x-ish, channel 1 = 10*y-ish
    g = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
             values=[vec(0.0, 0.0), vec(0.0, 10.0),
                     vec(2.0, 0.0), vec(2.0, 10.0)],
             channels=("a", "b"))
    f = VecGridField(g, vec(0.0, 0.0))
    assert f.at(_clk(0.0)) == (0.0, 0.0)
    assert VecGridField(g, vec(1.0, 1.0)).at(_clk(0.0)) == (2.0, 10.0)
    # center = mean of the four corners
    c = VecGridField(g, vec(0.5, 0.5)).at(_clk(0.0))
    assert abs(c[0] - 1.0) < 1e-12 and abs(c[1] - 5.0) < 1e-12


def test_vec_grid_channel_matches_scalar_grid():
    # a vector field's channel must equal the scalar field over the same channel.
    g_vec = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
                 values=[vec(0.0, 5.0), vec(1.0, 6.0),
                         vec(2.0, 7.0), vec(3.0, 8.0)],
                 channels=("p", "q"))
    g_p = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1), values=[0.0, 1.0, 2.0, 3.0])
    g_q = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1), values=[5.0, 6.0, 7.0, 8.0])
    q = vec(0.37, 0.62)
    vf = VecGridField(g_vec, q)
    clk = _clk(0.0)
    assert abs(vf.channel("p").at(clk) - GridField(g_p, q).at(clk)) < 1e-12
    assert abs(vf.channel("q").at(clk) - GridField(g_q, q).at(clk)) < 1e-12
    # tuple output agrees with the per-channel views
    out = vf.at(clk)
    assert abs(out[0] - vf.channel("p").at(clk)) < 1e-12
    assert abs(out[1] - vf.channel(1).at(clk)) < 1e-12


def test_vec_grid_is_a_dag_node():
    g = Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(4.0, 8.0)])
    q = vec(Sine(cycles=1.0, amp=0.5, bias=0.5))  # driven query
    f = VecGridField(g, q)
    down = f.channel(0) * 2.0 + f.channel(1)
    detect_signal_cycle(down)          # must not raise
    ids = {n.id for n in walk(down)}
    assert g.id in ids                 # dataset threaded into the DAG


def test_scalar_gridfield_rejects_vector_grid():
    g = Grid(shape=(2,), lo=(0,), hi=(1,), values=[vec(0.0, 0.0), vec(1.0, 1.0)])
    with pytest.raises(TypeError):
        GridField(g, vec(0.5))
    g2 = Grid(shape=(2,), lo=(0,), hi=(1,), values=[0.0, 1.0])
    with pytest.raises(TypeError):
        VecGridField(g2, vec(0.5))


# ---------------------------------------------------------------------------
# VecScatterField
# ---------------------------------------------------------------------------

def test_vec_scatter_exact_at_samples():
    sc = Scatter([
        (vec(0.0, 0.0), vec(1.0, 10.0)),
        (vec(1.0, 0.0), vec(2.0, 20.0)),
        (vec(0.0, 1.0), vec(3.0, 30.0)),
    ], channels=("m", "n"))
    f = VecScatterField(sc, vec(0.0, 0.0))
    assert f.at(_clk(0.0)) == (1.0, 10.0)
    assert VecScatterField(sc, vec(1.0, 0.0)).at(_clk(0.0)) == (2.0, 20.0)


def test_vec_scatter_channel_matches_scalar():
    samples_pos = [vec(0.0, 0.0), vec(1.0, 0.0), vec(0.0, 1.0), vec(1.0, 1.0)]
    m = [1.0, 2.0, 3.0, 4.0]
    n = [10.0, 20.0, 30.0, 40.0]
    sc = Scatter([(p, vec(a, b)) for p, a, b in zip(samples_pos, m, n)],
                 channels=("m", "n"))
    sc_m = Scatter([(p, a) for p, a in zip(samples_pos, m)])
    sc_n = Scatter([(p, b) for p, b in zip(samples_pos, n)])
    q = vec(0.4, 0.7)
    vf = VecScatterField(sc, q)
    clk = _clk(0.0)
    assert abs(vf.channel("m").at(clk) - ScatterField(sc_m, q).at(clk)) < 1e-12
    assert abs(vf.channel("n").at(clk) - ScatterField(sc_n, q).at(clk)) < 1e-12
    out = vf.at(clk)
    # blended value stays within each channel's sample range
    assert min(m) <= out[0] <= max(m)
    assert min(n) <= out[1] <= max(n)


def test_scalar_scatterfield_rejects_vector_scatter():
    sc = Scatter([(vec(0.0,), vec(1.0, 2.0)), (vec(1.0,), vec(3.0, 4.0))])
    with pytest.raises(TypeError):
        ScatterField(sc, vec(0.5))
    sc2 = Scatter([(vec(0.0,), 1.0), (vec(1.0,), 3.0)])
    with pytest.raises(TypeError):
        VecScatterField(sc2, vec(0.5))


def test_vec_field_caches_weights_once():
    # sanity: caching path returns the same value as uncached.
    g = Grid(shape=(2, 2), lo=(0, 0), hi=(1, 1),
             values=[vec(0.0, 0.0), vec(0.0, 10.0), vec(2.0, 0.0), vec(2.0, 10.0)])
    f = VecGridField(g, vec(0.5, 0.5))
    cache = Cache()
    a = f.at(_clk(0.0), cache)
    b = f.at(_clk(0.0), cache)   # served from cache
    assert a == b == f.at(_clk(0.0))


if __name__ == "__main__":
    import pytest as _pt
    raise SystemExit(_pt.main([__file__, "-q"]))
