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


# --- change-detection caching (rebuild only when the samples actually move) ---

def test_rbf_static_field_builds_once_across_frames():
    # Static positions AND values, but the query moves every frame.  The
    # interpolator must be built exactly once and reused for every frame.
    sc = _square()
    q = vec(Sine(cycles=1.0, amp=0.3, bias=0.5), Sine(cycles=1.0, amp=0.3, bias=0.5))
    f = RbfScatterField(sc, q)
    for frame in range(20):
        f.at(_clk(t=frame / 20.0, frame=frame))
    assert f._eng._builds == 1, f._eng._builds


def test_rbf_static_reuse_is_bit_identical():
    # The value produced from the reused cache must be bit-for-bit identical to a
    # freshly rebuilt interpolator at the same query point.
    sc = _square()
    q = vec(Sine(cycles=1.0, amp=0.3, bias=0.5), Sine(cycles=1.0, amp=0.3, bias=0.5))
    cached = RbfScatterField(sc, q)
    for frame in range(1, 15):
        clk = _clk(t=frame / 15.0, frame=frame)
        fresh = RbfScatterField(sc, q)          # its own engine -> always rebuilds
        assert cached.at(clk) == fresh.at(clk)  # bit-identical, not just close
    assert cached._eng._builds == 1


def test_rbf_animated_values_rebuild_each_change_and_stay_correct():
    # Values change every frame -> the interpolator must rebuild, and the result
    # must match a freshly built interpolator bit-for-bit.
    drivers = [Sine(cycles=1.0, amp=1.0, bias=1.0, phase=k * 0.1) for k in range(5)]
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
    sc = Scatter([(vec(x, y), d) for (x, y), d in zip(pts, drivers)])
    q = vec(0.35, 0.55)
    cached = RbfScatterField(sc, q)
    for frame in range(1, 8):
        clk = _clk(t=frame / 8.0, frame=frame)
        fresh = RbfScatterField(sc, q)
        assert cached.at(clk) == fresh.at(clk)
    # one build per distinct frame that was evaluated (frames 1..7, plus none reused)
    assert cached._eng._builds == 7, cached._eng._builds


def test_rbf_vector_static_field_builds_once():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.4, 0.6)]
    sc = Scatter([(vec(x, y), vec(av, bv))
                  for (x, y), av, bv in zip(pts, [0, 1, 2, 3, 1.5], [10, 8, 6, 4, 7])],
                 channels=("a", "b"))
    q = vec(Sine(cycles=1.0, amp=0.3, bias=0.5), Sine(cycles=1.0, amp=0.3, bias=0.5))
    vf = VecRbfScatterField(sc, q)
    for frame in range(12):
        vf.at(_clk(t=frame / 12.0, frame=frame))
    assert vf._eng._builds == 1, vf._eng._builds


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
