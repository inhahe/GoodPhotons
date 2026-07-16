"""M1 foundation tests: cycle detection, seamless closed curve, interpolators.

Runnable directly (``python tests/test_foundation.py``) or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, TimeFn, Sine, LoopNoise, vec, VecSignal, lerp,
    RefSignal, detect_signal_cycle, SignalCycleError, walk,
    PointPath, Grid, Scatter, LoopCurve, GridField, ScatterField,
)


def _clk(t: float, frame: int = 0) -> Clock:
    return Clock(t=t, frame=frame, frames=1000, fps=30.0)


# ---------------------------------------------------------------------------

def test_cycle_detection_fires():
    ref = RefSignal("loop")
    expr = ref + 1.0          # expr depends on ref ...
    ref.bind(expr)            # ... and ref now points back to expr -> cycle
    try:
        detect_signal_cycle(expr)
    except SignalCycleError:
        return
    raise AssertionError("expected SignalCycleError on a self-referential graph")


def test_acyclic_passes_and_walks():
    v = Sine(cycles=2) * 3.0 + Const(1.0)
    detect_signal_cycle(v)                     # must not raise
    ids = {n.id for n in walk(v)}
    assert len(ids) >= 4, ids


def test_vector_math():
    a = vec(3.0, 4.0)
    assert abs(a.length().at(_clk(0.0)) - 5.0) < 1e-12
    assert abs(a.dot(vec(1.0, 0.0)).at(_clk(0.0)) - 3.0) < 1e-12
    n = a.normalized().at(_clk(0.0))
    assert abs(math.hypot(*n) - 1.0) < 1e-9
    m = lerp(vec(0.0, 0.0), vec(10.0, 20.0), 0.25).at(_clk(0.0))
    assert m == (2.5, 5.0), m


def test_closed_curve_is_seamless():
    # An irregular N-D (here 3-D) closed control polygon.
    path = PointPath([
        vec(0.0, 0.0, 0.0),
        vec(2.0, 1.0, 0.5),
        vec(1.0, 3.0, -1.0),
        vec(-1.0, 2.0, 0.2),
        vec(-2.0, -1.0, 1.0),
    ], closed=True)
    u = TimeFn(lambda t: t)              # travel around the loop as t advances
    curve = LoopCurve(path, u)
    detect_signal_cycle(curve)          # curve graph must be acyclic

    # C0 seamlessness: value at t=0 equals the limit as t->1.
    p0 = curve.at(_clk(0.0))
    p1 = curve.at(_clk(1.0 - 1e-7))
    gap = math.dist(p0, p1)
    assert gap < 1e-4, f"seam gap too large: {gap}"

    # No jumps anywhere around the loop (continuity).
    prev = curve.at(_clk(0.0))
    maxstep = 0.0
    N = 400
    for k in range(1, N + 1):
        cur = curve.at(_clk(k / N))
        maxstep = max(maxstep, math.dist(prev, cur))
        prev = cur
    # Wrap step (last -> first) too.
    maxstep = max(maxstep, math.dist(prev, curve.at(_clk(0.0))))
    assert maxstep < 0.2, f"discontinuity detected, max step {maxstep}"


def test_loopnoise_is_periodic():
    n = LoopNoise(cells=6, seed=42, freq=1)
    assert abs(n.at(_clk(0.0)) - n.at(_clk(1.0 - 1e-9))) < 1e-3
    # deterministic
    assert n.at(_clk(0.37)) == LoopNoise(cells=6, seed=42, freq=1).at(_clk(0.37))


def test_gridfield_interpolates():
    # 2x2 grid on the unit square; corner values 0,1,2,3 (C-order: (i0,i1)).
    grid = Grid(shape=(2, 2), lo=(0.0, 0.0), hi=(1.0, 1.0),
                values=[0.0, 1.0, 2.0, 3.0])
    # exact at corners
    assert GridField(grid, vec(0.0, 0.0)).at(_clk(0.0)) == 0.0
    assert GridField(grid, vec(0.0, 1.0)).at(_clk(0.0)) == 1.0
    assert GridField(grid, vec(1.0, 0.0)).at(_clk(0.0)) == 2.0
    assert GridField(grid, vec(1.0, 1.0)).at(_clk(0.0)) == 3.0
    # center = average
    c = GridField(grid, vec(0.5, 0.5)).at(_clk(0.0))
    assert abs(c - 1.5) < 1e-12, c
    # edge-extend outside domain
    assert GridField(grid, vec(-5.0, -5.0)).at(_clk(0.0)) == 0.0


def test_gridfield_is_a_signal():
    grid = Grid(shape=(2, 2), lo=(0.0, 0.0), hi=(1.0, 1.0),
                values=[0.0, 10.0, 0.0, 10.0])
    # a modulator drives the query, and the field feeds another modulator.
    # values vary along the *last* axis (i1), so drive that component.
    q = vec(Const(0.0), Sine(cycles=1, amp=0.5, bias=0.5))
    field = GridField(grid, q)
    downstream = field * 2.0 + 1.0
    detect_signal_cycle(downstream)
    # at t=0, sin=0 -> y=0.5 -> value 5 -> downstream 11
    val = downstream.at(_clk(0.0))
    assert abs(val - 11.0) < 1e-9, val


def test_scatterfield_reproduces_samples():
    sc = Scatter([
        (vec(0.0, 0.0), 1.0),
        (vec(1.0, 0.0), 2.0),
        (vec(0.0, 1.0), 3.0),
    ])
    f = ScatterField(sc, vec(0.0, 0.0))
    assert abs(f.at(_clk(0.0)) - 1.0) < 1e-9
    f2 = ScatterField(sc, vec(1.0, 0.0))
    assert abs(f2.at(_clk(0.0)) - 2.0) < 1e-9
    # a point in between is a weighted blend within value range
    mid = ScatterField(sc, vec(0.4, 0.4)).at(_clk(0.0))
    assert 1.0 <= mid <= 3.0, mid


def test_cache_shares_evaluation():
    calls = {"n": 0}

    def f(t):
        calls["n"] += 1
        return t

    shared = TimeFn(f)
    a = shared + 1.0
    b = shared * 2.0
    cache = Cache()
    clk = _clk(0.3, frame=7)
    a.at(clk, cache)
    b.at(clk, cache)
    assert calls["n"] == 1, f"shared node evaluated {calls['n']} times, expected 1"


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
