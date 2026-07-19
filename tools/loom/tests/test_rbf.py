"""H3 tests: RBF scatter interpolation (scipy-backed).

Skipped cleanly if scipy is unavailable.  Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

scipy = pytest.importorskip("scipy")

from loom import (  # noqa: E402
    Clock, Cache, vec, Scatter, Sine,
    RbfScatterField, VecRbfScatterField, ScatterField, detect_signal_cycle, walk,
)


def _clk(t: float = 0.0, frame: int = 0) -> Clock:
    return Clock(t=t, frame=frame, frames=1000, fps=30.0)


def _square():
    # unit square corners + centre, scalar values.
    return Scatter([
        (vec(0.0, 0.0), 0.0),
        (vec(1.0, 0.0), 1.0),
        (vec(0.0, 1.0), 1.0),
        (vec(1.0, 1.0), 2.0),
        (vec(0.5, 0.5), 1.0),
    ])


# ---------------------------------------------------------------------------

def test_rbf_exact_at_samples():
    sc = _square()
    samples = [((0.0, 0.0), 0.0), ((1.0, 0.0), 1.0),
               ((0.0, 1.0), 1.0), ((1.0, 1.0), 2.0), ((0.5, 0.5), 1.0)]
    for (x, y), v in samples:
        got = RbfScatterField(sc, vec(x, y)).at(_clk())
        assert abs(got - v) < 1e-7, ((x, y), got, v)


def test_rbf_reproduces_linear_field():
    # thin-plate spline reproduces linear (has the linear polynomial tail).
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.3, 0.7), (0.8, 0.2)]
    f = lambda x, y: 3.0 * x - 2.0 * y + 1.0
    sc = Scatter([(vec(x, y), f(x, y)) for x, y in pts])
    field = RbfScatterField(sc, vec(0.42, 0.61))
    assert abs(field.at(_clk()) - f(0.42, 0.61)) < 1e-6


def test_rbf_clamp_outside_hull():
    sc = _square()   # data range [0, 2]
    # far outside the hull: clamp keeps it within the data range.
    clamped = RbfScatterField(sc, vec(10.0, 10.0), on_outside="clamp").at(_clk())
    assert 0.0 <= clamped <= 2.0
    # extrapolate is allowed to leave the range (and generally does here).
    raw = RbfScatterField(sc, vec(10.0, 10.0), on_outside="extrapolate").at(_clk())
    assert raw != clamped or raw > 2.0 or raw < 0.0

    with pytest.raises(ValueError):
        RbfScatterField(sc, vec(10.0, 10.0), on_outside="raise").at(_clk())


def test_rbf_vector_channels_match_scalar():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.4, 0.6)]
    a = [0.0, 1.0, 2.0, 3.0, 1.5]
    b = [10.0, 8.0, 6.0, 4.0, 7.0]
    sc_vec = Scatter([(vec(x, y), vec(av, bv))
                      for (x, y), av, bv in zip(pts, a, b)], channels=("a", "b"))
    sc_a = Scatter([(vec(x, y), av) for (x, y), av in zip(pts, a)])
    sc_b = Scatter([(vec(x, y), bv) for (x, y), bv in zip(pts, b)])
    q = vec(0.55, 0.35)
    vf = VecRbfScatterField(sc_vec, q)
    clk = _clk()
    assert abs(vf.channel("a").at(clk) - RbfScatterField(sc_a, q).at(clk)) < 1e-7
    assert abs(vf.channel("b").at(clk) - RbfScatterField(sc_b, q).at(clk)) < 1e-7
    out = vf.at(clk)
    assert abs(out[0] - vf.channel(0).at(clk)) < 1e-9
    assert abs(out[1] - vf.channel(1).at(clk)) < 1e-9


def test_rbf_1d_hull_and_exact():
    sc = Scatter([(vec(0.0,), 0.0), (vec(1.0,), 1.0), (vec(2.0,), 4.0)])
    f = RbfScatterField(sc, vec(1.0))
    assert abs(f.at(_clk()) - 1.0) < 1e-7
    # outside the 1-D hull clamps to [0, 4]
    assert 0.0 <= RbfScatterField(sc, vec(-5.0)).at(_clk()) <= 4.0


def test_rbf_is_a_dag_node():
    sc = _square()
    q = vec(Sine(cycles=1.0, amp=0.3, bias=0.5), Sine(cycles=1.0, amp=0.3, bias=0.5))
    f = RbfScatterField(sc, q)
    down = f * 2.0
    detect_signal_cycle(down)
    ids = {n.id for n in walk(down)}
    assert sc.id in ids


def test_rbf_bad_kernel_and_type_guards():
    sc = _square()
    with pytest.raises(ValueError):
        RbfScatterField(sc, vec(0.5, 0.5), kernel="not_a_kernel")
    with pytest.raises(ValueError):
        RbfScatterField(sc, vec(0.5, 0.5), on_outside="bogus")
    # scalar scatter -> Vec* rejects, vector scatter -> scalar rejects
    with pytest.raises(TypeError):
        VecRbfScatterField(sc, vec(0.5, 0.5))
    sc_vec = Scatter([(vec(0.0, 0.0), vec(1.0, 2.0)), (vec(1.0, 0.0), vec(3.0, 4.0)),
                      (vec(0.0, 1.0), vec(5.0, 6.0))])
    with pytest.raises(TypeError):
        RbfScatterField(sc_vec, vec(0.5, 0.5))


def test_rbf_cache_consistent():
    sc = _square()
    f = RbfScatterField(sc, vec(0.3, 0.4))
    cache = Cache()
    a = f.at(_clk(), cache)
    b = f.at(_clk(), cache)
    assert a == b == f.at(_clk())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
