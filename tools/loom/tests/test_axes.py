"""E5 tests: loom.axes — axis-typed signals (one influence model).

Runnable directly (``python tests/test_axes.py``) or under pytest.  Covers the
four E5 pillars: axis-set inference + broadcast/pointwise, the sample/select
grammar, the explicit :class:`Reduce` cross-axis node, and the pin/mod edge
model with target-declared neutrals.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.axes import (  # noqa: E402
    Ax, AConst, Lift, AFn, Sample, select, Reduce, Binding, Target, combine,
    ADDITIVE, GAIN, BIPOLAR, AXIS_T, AXIS_S,
)
from loom.signals.core import Const, TimeFn, detect_signal_cycle, walk  # noqa: E402


# ---- axis-set inference ----------------------------------------------------

def test_leaf_axes():
    assert Ax("t").axes == {"t"}
    assert Ax("s").axes == {"s"}
    assert AConst(3.0).axes == frozenset()


def test_composition_unions_axes():
    s, t = Ax("s"), Ax("t")
    node = s * t + 1.0
    assert node.axes == {"s", "t"}
    assert (s + s).axes == {"s"}          # same axis stays pointwise
    assert (s * 2.0).axes == {"s"}         # const contributes no axis


def test_eval_pointwise_shared_axis():
    t = Ax("t")
    node = t * t
    assert node.eval(t=0.5) == 0.25       # same t everywhere (lockstep)
    assert node.eval({"t": 3.0}) == 9.0


# ---- broadcast -------------------------------------------------------------

def test_broadcast_ignores_extra_axes():
    # a {t} node evaluated at a {s,t} point ignores s → same value for every s
    tnode = Ax("t") * 10.0
    v0 = tnode.eval(t=0.4, s=0.0)
    v1 = tnode.eval(t=0.4, s=0.9)
    assert v0 == v1 == 4.0


def test_broadcast_shifts_whole_spatial_curve():
    # B.y(s,t) = s + timecurve(t): the {t} term lifts the whole {s} elevation.
    s, t = Ax("s"), Ax("t")
    elevation = s + t
    assert elevation.axes == {"s", "t"}
    base = [elevation.eval(s=x, t=0.0) for x in (0.0, 0.5, 1.0)]
    lifted = [elevation.eval(s=x, t=0.3) for x in (0.0, 0.5, 1.0)]
    # every point rose by exactly the same 0.3 (broadcast, not per-point)
    assert all(abs((b + 0.3) - l) < 1e-12 for b, l in zip(base, lifted))


def test_missing_axis_errors():
    node = Ax("s") + Ax("t")
    with pytest.raises(ValueError):
        node.eval(t=0.5)                   # 's' missing


# ---- Lift: bridge the legacy Signal DAG -----------------------------------

def test_lift_legacy_signal_is_t_typed():
    sig = TimeFn(lambda t: 2.0 * t, periodic=False)
    lifted = Lift(sig)
    assert lifted.axes == {"t"}
    assert abs(lifted.eval(t=0.25) - 0.5) < 1e-12
    # composes with spatial axes and broadcasts
    node = Ax("s") + lifted
    assert node.axes == {"s", "t"}
    assert abs(node.eval(s=1.0, t=0.5) - 2.0) < 1e-12


# ---- sample/select grammar -------------------------------------------------

def test_sample_binds_param_axis():
    # curve(t): sampling a curve at the current t yields a {t} value.
    curve = lambda p: math.sin(2 * math.pi * p)
    node = Sample(curve, Ax("t"))
    assert node.axes == {"t"}
    assert abs(node.eval(t=0.25) - 1.0) < 1e-9


def test_sample_component_pick():
    # curve(t).y — a vector-valued sample, then component 1.
    curve = lambda p: (p, 2 * p, 3 * p)
    y = Sample(curve, Ax("s")).comp(1)
    assert y.axes == {"s"}
    assert abs(y.eval(s=0.5) - 1.0) < 1e-12


def test_select_is_discrete_constant():
    items = [Ax("s"), Ax("t"), AConst(9.0)]
    assert select(items, 2).eval() == 9.0
    assert select(items, 0).axes == {"s"}
    with pytest.raises(IndexError):
        select(items, 5)


# ---- Reduce: the only cross-axis node -------------------------------------

def test_reduce_consumes_axis():
    body = Ax("s") + Ax("t")
    red = Reduce(body, "s", samples=3, op="sum", lo=0.0, hi=1.0)
    assert red.axes == {"t"}               # 's' consumed
    # sum over s in {0, .5, 1} of (s + t) = (0+.5+1) + 3t
    assert abs(red.eval(t=0.0) - 1.5) < 1e-12
    assert abs(red.eval(t=1.0) - 4.5) < 1e-12


def test_reduce_integral_trapezoid():
    # ∫_0^1 s ds = 0.5
    red = Reduce(Ax("s"), "s", samples=51, op="integral")
    assert abs(red.eval() - 0.5) < 1e-3


def test_reduce_mean_min_max():
    body = Ax("s")
    assert abs(Reduce(body, "s", 5, "mean").eval() - 0.5) < 1e-12
    assert Reduce(body, "s", 5, "min").eval() == 0.0
    assert Reduce(body, "s", 5, "max").eval() == 1.0


def test_reduce_requires_present_axis():
    with pytest.raises(ValueError):
        Reduce(Ax("t"), "s", 3)            # body has no 's'


# ---- pin / mod edges + target neutrals ------------------------------------

def test_additive_mod_accumulates_from_zero():
    tgt = combine(ADDITIVE, [Binding(AConst(2.0), "mod", 1.0),
                             Binding(AConst(3.0), "mod", 0.5)])
    assert tgt.eval() == 2.0 + 0.5 * 3.0   # neutral 0 + 2 + 1.5


def test_additive_base_then_mod():
    tgt = Target(ADDITIVE, [Binding(AConst(1.0), "mod", 1.0)], base=10.0)
    assert tgt.eval() == 11.0


def test_gain_mod_multiplies_from_one():
    tgt = combine(GAIN, [Binding(AConst(2.0), "mod", 1.0),
                         Binding(AConst(3.0), "mod", 1.0)])
    assert abs(tgt.eval() - 6.0) < 1e-12   # neutral 1 * 2 * 3
    # gain as exponent: 4 ** 0.5 = 2
    g = combine(GAIN, [Binding(AConst(4.0), "mod", 0.5)])
    assert abs(g.eval() - 2.0) < 1e-12


def test_bipolar_mod_is_half_centred_and_clamped():
    # neutral ½; two +0.5 pushes clamp at 1.
    tgt = combine(BIPOLAR, [Binding(AConst(1.0), "mod", 1.0),
                            Binding(AConst(1.0), "mod", 1.0)])
    assert tgt.eval() == 1.0
    # a single centred value returns itself
    one = combine(BIPOLAR, [Binding(AConst(0.7), "mod", 1.0)])
    assert abs(one.eval() - 0.7) < 1e-12


def test_pin_is_last_write_wins():
    tgt = combine(ADDITIVE, [Binding(AConst(5.0), "mod", 1.0),
                             Binding(AConst(2.0), "pin", 1.0)])
    assert tgt.eval() == 2.0               # pin replaces the accumulated 5
    # gain<1 on pin blends
    blend = combine(ADDITIVE, [Binding(AConst(10.0), "mod", 1.0),
                               Binding(AConst(0.0), "pin", 0.5)])
    assert blend.eval() == 5.0             # 10*(1-.5) + 0*.5


def test_target_axes_and_broadcast():
    # base is {s}, driver is {t} → target is {s,t}; the {t} driver broadcasts.
    tgt = Target(ADDITIVE, [Binding(Ax("t"), "mod", 1.0)], base=Ax("s"))
    assert tgt.axes == {"s", "t"}
    assert abs(tgt.eval(s=2.0, t=0.3) - 2.3) < 1e-12
    assert abs(tgt.eval(s=5.0, t=0.3) - 5.3) < 1e-12


# ---- reuse: cycle detection / walk over axial nodes ------------------------

def test_cycle_detector_and_walk_work_on_axial_nodes():
    node = (Ax("s") + Ax("t")) * Sample(lambda p: p, Ax("t"))
    detect_signal_cycle(node)              # no raise
    ids = {n.id for n in walk(node)}
    assert node.id in ids and len(ids) >= 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all axes tests passed")
