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
    CurveSample, RecordSample, sample, mod, pin, as_ax,
    Lower, LowerVec, lower,
    ADDITIVE, GAIN, BIPOLAR, AXIS_T, AXIS_S,
)
from loom.signals.core import (  # noqa: E402
    Const, TimeFn, Clock, Cache, Signal, as_signal, detect_signal_cycle, walk,
)
from loom.signals.vector import VecSignal  # noqa: E402
from loom.ftsl_emit import num, vec3, value_token, site_node  # noqa: E402
from loom import PointPath, LoopCurve, Sine, vec, Record  # noqa: E402


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


# ---- fold: real loom curves / records bound into the sample grammar --------

def _square_path():
    return PointPath([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], closed=True)


def test_curve_sample_static_is_st_typed_but_broadcasts_t():
    # a static spatial LoopCurve: sampling at param s is {s, t}, but the value
    # doesn't actually move with t (control points are constant → broadcast).
    curve = LoopCurve(_square_path(), 0.0)
    node = CurveSample(curve, Ax("s"))
    assert node.axes == {"s", "t"}
    v0, v1 = node.eval(s=0.3, t=0.0), node.eval(s=0.3, t=0.7)
    assert v0 == v1                                    # static ⇒ same for every t
    assert v0 == curve.sample(0.3, Clock(t=0.0))       # matches the curve itself


def test_curve_sample_animated_is_genuinely_st():
    # animate one control point over t → the sampled shape moves with time.
    pp = PointPath([vec(Sine(), 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                   closed=True)
    curve = LoopCurve(pp, 0.0)
    node = CurveSample(curve, Ax("s"))
    assert node.eval(s=0.0, t=0.0) != node.eval(s=0.0, t=0.25)


def test_curve_sample_component_pick():
    curve = LoopCurve(_square_path(), 0.0)
    y = CurveSample(curve, Ax("s")).comp(1)
    assert y.axes == {"s", "t"}
    assert abs(y.eval(s=0.6, t=0.0) - curve.sample(0.6, Clock(t=0.0))[1]) < 1e-12


def test_curve_sample_custom_clock_axis():
    # the clock axis is configurable (e.g. bind the shape-clock to 'u').
    curve = LoopCurve(_square_path(), 0.0)
    node = CurveSample(curve, Ax("s"), clock_axis="u")
    assert node.axes == {"s", "u"}
    with pytest.raises(ValueError):
        node.eval(s=0.5)                               # 'u' now required


def test_record_sample_is_driver_typed_and_static():
    rec = Record.from_channels("R", 0.0, 1.0, [("h", ["0.0", "1.0"])])
    node = RecordSample(rec, "h", Ax("s"))
    assert node.axes == {"s"}                           # static LUT, no clock
    assert abs(node.eval(s=0.0) - 0.0) < 1e-12
    assert abs(node.eval(s=0.5) - 0.5) < 1e-12
    assert abs(node.eval(s=1.0) - 1.0) < 1e-12


def test_record_sample_vector_component():
    rec = Record.from_channels("R", 0.0, 1.0,
                               [("p", [["0.0", "0.0"], ["2.0", "4.0"]])])
    node = RecordSample(rec, "p", Ax("s"))
    assert node.comp(1).eval(s=0.5) == 2.0              # midpoint of 0..4


def test_sample_dispatch_picks_the_right_node():
    curve = LoopCurve(_square_path(), 0.0)
    rec = Record.from_channels("R", 0.0, 1.0, [("h", ["0.0", "1.0"])])
    assert isinstance(sample(curve, Ax("s")), CurveSample)
    assert isinstance(sample(rec, Ax("s"), channel="h"), RecordSample)
    assert isinstance(sample(lambda p: p * 2, Ax("t")), Sample)
    with pytest.raises(ValueError):
        sample(rec, Ax("s"))                            # Record needs channel=
    with pytest.raises(TypeError):
        sample(123, Ax("s"))                            # not sampleable


def test_curve_sample_composes_with_reduce_over_s():
    # a mean over s of a curve's x component is a cross-index op → {t} scalar.
    curve = LoopCurve(_square_path(), 0.0)
    x = CurveSample(curve, Ax("s")).comp(0)
    red = Reduce(x, "s", samples=8, op="mean")
    assert red.axes == {"t"}
    expect = sum(curve.sample(i / 7.0, Clock(t=0.0))[0] for i in range(8)) / 8.0
    assert red.eval(t=0.0) == pytest.approx(expect)


def test_curve_sample_walk_reaches_control_points():
    # the loom curve node threads into the axis-layer walk (like Lift), so a
    # cycle through a control point is still catchable.
    curve = LoopCurve(_square_path(), 0.0)
    node = CurveSample(curve, Ax("s"))
    ids = {n.id for n in walk(node)}
    assert curve.id in ids and curve.path.id in ids


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


# ---- edge sugar: mod / pin / as_ax -----------------------------------------

def test_mod_and_pin_build_bindings():
    m = mod(2.0, 0.5)
    assert isinstance(m, Binding) and m.mode == "mod" and m.gain == 0.5
    p = pin(Ax("t"))
    assert p.mode == "pin" and p.gain == 1.0


def test_binding_coerces_a_raw_source_to_an_axis_node():
    b = Binding(3.0, "mod", 1.0)
    assert isinstance(b.source, AConst) and b.source.eval() == 3.0


def test_binding_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown binding mode"):
        Binding(AConst(1.0), "blend", 1.0)


def test_as_ax_lifts_a_legacy_signal():
    node = as_ax(Sine())
    assert isinstance(node, Lift) and node.axes == {"t"}
    assert as_ax(Ax("s")) is not None and as_ax(2.0).eval() == 2.0


def test_gain_target_rejects_a_negative_source():
    # x ** gain would silently go complex; the target says so instead.
    tgt = combine(GAIN, [mod(AConst(-1.0), 0.5)])
    with pytest.raises(ValueError, match="non-negative source"):
        tgt.eval()


# ---- Lower: an axis node at a scalar value-site ----------------------------

def test_lower_evaluates_a_target_against_the_clock():
    # base 2, doubled by a {t}-typed mod driver: 2 * (1+t)
    tgt = Target(GAIN, [mod(1.0 + Ax("t"))], base=2.0)
    low = Lower(tgt)
    assert isinstance(low, Signal)
    assert abs(low.at(Clock(t=0.0)) - 2.0) < 1e-12
    assert abs(low.at(Clock(t=0.5, frame=1)) - 3.0) < 1e-12


def test_lower_reports_an_unbound_axis_at_construction():
    with pytest.raises(ValueError, match=r"'s'.* unbound|\['s'\]"):
        Lower(Ax("s") * 2.0)


def test_lower_binds_another_axis_to_a_constant():
    low = Lower(Ax("s") * 10.0, bind={"s": 0.25})
    assert abs(low.at(Clock(t=0.9)) - 2.5) < 1e-12


def test_lower_binds_another_axis_to_a_signal():
    # sweep s over the loop by pinning it to a clock-driven Signal
    low = Lower(Ax("s"), bind={"s": TimeFn(lambda t: t)})
    assert abs(low.at(Clock(t=0.3)) - 0.3) < 1e-12
    assert abs(low.at(Clock(t=0.8, frame=1)) - 0.8) < 1e-12
    # the bound Signal is a child, so the walk reaches it
    assert len(list(walk(low))) >= 3


def test_lower_rejects_a_vector_valued_node():
    path = PointPath([(0, 0, 0), (1, 0, 0)], closed=False)
    cs = CurveSample(LoopCurve(path, Const(0.0)), AConst(0.5))
    with pytest.raises(ValueError, match="use LowerVec"):
        Lower(cs).at(Clock(t=0.0))


# ---- LowerVec: an axis node at a vector value-site -------------------------

def _square_path():
    return PointPath([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], closed=True)


def test_lower_vec_probes_its_dim_and_evaluates_whole():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    lv = LowerVec(cs)
    assert isinstance(lv, VecSignal) and lv.dim == 3
    v = lv.at(Clock(t=0.0))
    assert len(v) == 3 and all(isinstance(c, float) for c in v)


def test_lower_vec_rejects_a_scalar_node():
    with pytest.raises(ValueError, match="needs a vector-valued node"):
        LowerVec(Ax("t"))


def test_lower_vec_dim_mismatch_is_reported():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    lv = LowerVec(cs, dim=2)          # forced: no probe
    with pytest.raises(ValueError, match="expected a 2-vector"):
        lv.at(Clock(t=0.0))


def test_lower_vec_caches_per_frame():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    lv = LowerVec(cs)
    cache = Cache()
    c = Clock(t=0.0, frame=7)
    a = lv.at(c, cache)
    assert lv.at(c, cache) is a       # second read is the cached tuple


def test_lower_vec_components_are_lower_nodes():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    lv = LowerVec(cs)
    assert all(isinstance(c, Lower) for c in lv.components)
    whole = lv.at(Clock(t=0.0))
    per = tuple(c.at(Clock(t=0.0)) for c in lv.components)
    assert all(abs(a - b) < 1e-12 for a, b in zip(whole, per))


def test_lower_dispatches_scalar_vs_vector():
    assert isinstance(lower(Target(ADDITIVE, [mod(Ax("t"))], base=1.0)), Lower)
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    assert isinstance(lower(cs), LowerVec)


def test_lower_sweeps_a_curve_axis_with_a_bound_signal():
    # s bound to a clock ramp: the sample walks the loop over the animation
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), Ax("s"))
    lv = lower(cs, dim=3, bind={"s": TimeFn(lambda t: t)})
    a = lv.at(Clock(t=0.0))
    b = lv.at(Clock(t=0.5, frame=1))
    assert a != b                     # genuinely swept, not frozen


# ---- coercion: an AxSignal handed straight to a value-site -----------------

def test_as_signal_lowers_an_axis_node():
    s = as_signal(Target(ADDITIVE, [mod(Ax("t"))], base=1.0))
    assert isinstance(s, Lower)
    assert abs(s.at(Clock(t=0.25)) - 1.25) < 1e-12


def test_as_signal_rejects_a_vector_axis_node_at_a_scalar_site():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    with pytest.raises(TypeError, match="pick a component"):
        as_signal(cs)


def test_vecsignal_of_lowers_a_vector_axis_node():
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.25))
    v = VecSignal.of(cs)
    assert isinstance(v, LowerVec) and v.dim == 3


def test_vecsignal_of_rejects_a_scalar_axis_node():
    with pytest.raises(TypeError, match="got a scalar axis node"):
        VecSignal.of(Ax("t"))


def test_site_node_is_memoised_so_identity_is_stable():
    # node identity is the per-frame Cache key AND what roots() must hand the
    # cycle detector, so the same axis node must lower to the SAME Signal.
    tgt = Target(ADDITIVE, [mod(Ax("t"))], base=1.0)
    assert site_node(tgt) is site_node(tgt)
    assert as_signal(tgt) is site_node(tgt)


def test_site_node_passes_plain_values_through():
    assert site_node(3.0) is None and site_node((1, 2, 3)) is None
    sine = Sine()
    assert site_node(sine) is sine


# ---- the payoff: a Target driving a real scene value-site ------------------

def test_target_drives_a_scene_value_site_end_to_end():
    from loom.scene import Sphere, element_roots

    radius = Target(GAIN, [mod(as_ax(0.6 + 0.4 * Sine()))], base=0.3)
    center = CurveSample(LoopCurve(_square_path(), Const(0.0)), Ax("s"))
    sp = Sphere(center=lower(center, dim=3, bind={"s": TimeFn(lambda t: t)}),
                radius=radius, material="gold")

    clock = Clock(t=0.0, frame=0)
    cache = Cache()
    assert abs(num(sp.radius, clock, cache) - 0.3 * 0.6) < 1e-9
    assert len(vec3(sp.center, clock, cache)) == 3

    # every lowered node is reachable, acyclic and walkable from roots()
    roots = element_roots(sp)
    assert any(isinstance(r, (Lower, LowerVec)) for r in roots)
    for r in roots:
        detect_signal_cycle(r)
        assert len(list(walk(r))) >= 2


def test_target_emits_a_scene_token():
    tgt = Target(ADDITIVE, [mod(Ax("t"))], base=1.0)
    assert value_token(tgt, Clock(t=0.5)) == "1.5"
    cs = CurveSample(LoopCurve(_square_path(), Const(0.0)), AConst(0.0))
    assert len(value_token(cs, Clock(t=0.0)).split()) == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all axes tests passed")
