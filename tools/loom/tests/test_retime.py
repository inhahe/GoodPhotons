"""Retime tests: sampling a sub-graph at another time (TODO §J "loom retime /
4D time-shear node").

Covers the scalar/vector :class:`loom.Retime` nodes, the ``freeze``/``delay``/
``warp`` conveniences, the two things the roadmap flagged as easy to get wrong
(the frame-keyed :class:`loom.Cache` must not be poisoned by a retimed read, and
the structural cycle detector must still own *both* edges), and the spatial half
:class:`loom.SigAt` — a signal sampled at a spatially varying phase (the 4-D
time shear).

Runnable directly (``python tests/test_retime.py``) or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Cache, Clock, Const, Phase, Ramp, Retime, SigAt, Sine, Cosine, VecRetime,
    VecSignal, X, T, delay, detect_signal_cycle, freeze, retime, retimed_clock,
    vec, walk, warp,
)

np = pytest.importorskip("numpy")

FRAMES = 12


def _loop(frame: int) -> Clock:
    return Clock.at_frame(frame, FRAMES)


def _open(frame: int, frames: int = FRAMES) -> Clock:
    return Clock.at_frame(frame, frames, loop=False)


class Counter(Sine):
    """A Sine that records how many times it was actually evaluated."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.calls = 0

    def _eval(self, clock, cache):
        self.calls += 1
        return super()._eval(clock, cache)


# ---------------------------------------------------------------------------
# Phase — time as a value
# ---------------------------------------------------------------------------

def test_phase_is_the_clock_phase():
    p = Phase()
    for f in range(FRAMES):
        assert p.at(_loop(f)) == pytest.approx(f / FRAMES)
    # open clock spans [0, 1] inclusive
    assert p.at(_open(0)) == pytest.approx(0.0)
    assert p.at(_open(FRAMES - 1)) == pytest.approx(1.0)
    # arithmetic on it is ordinary signal algebra
    assert (Phase() * 2.0 - 0.5).at(_loop(3)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# retimed_clock
# ---------------------------------------------------------------------------

def test_retimed_clock_maps_phase_and_frame():
    c = _loop(3)
    rc = retimed_clock(c, 0.5)
    assert rc.t == pytest.approx(0.5)
    assert rc.frame == 6                      # 0.5 * 12
    assert (rc.frames, rc.fps, rc.loop) == (c.frames, c.fps, c.loop)
    # a closed clock wraps by default, so the frame index stays in range
    assert retimed_clock(c, 1.25).t == pytest.approx(0.25)
    assert retimed_clock(c, -0.25).t == pytest.approx(0.75)
    assert retimed_clock(c, 0.999).frame == 0            # rounds to 12 -> wraps
    # an open clock does NOT wrap by default: off the end stays off the end
    o = _open(3)
    assert retimed_clock(o, 1.25).t == pytest.approx(1.25)
    assert retimed_clock(o, 1.25).frame == FRAMES - 1    # frame index clamped
    assert retimed_clock(o, -2.0).frame == 0
    # explicit wrap overrides in both directions
    assert retimed_clock(o, 1.25, wrap=True).t == pytest.approx(0.25)
    assert retimed_clock(c, 1.25, wrap=False).t == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# freeze / delay / warp
# ---------------------------------------------------------------------------

def test_freeze_holds_one_phase():
    s = Sine()
    held = freeze(s, 0.25)
    vals = [held.at(_loop(f)) for f in range(FRAMES)]
    assert vals == pytest.approx([1.0] * FRAMES)
    # ... and the un-frozen signal genuinely moves (anti-vacuity)
    moving = [s.at(_loop(f)) for f in range(FRAMES)]
    assert max(moving) - min(moving) > 1.9


def test_freeze_point_may_itself_be_animated():
    # hold at phase 0 for the first half, then track the clock: a scrubbed freeze
    held = freeze(Sine(), Phase() * 2.0 - 1.0, wrap=True)
    assert held.at(_loop(0)) == pytest.approx(Sine().at(_loop(0)))
    # frame 9 -> t=0.75 -> sample phase 0.5
    assert held.at(_loop(9)) == pytest.approx(Sine().at(_loop(6)))


def test_delay_reads_the_past_and_wraps_on_a_loop():
    s = Sine()
    d = delay(s, 3.0 / FRAMES)
    for f in range(FRAMES):
        assert d.at(_loop(f)) == pytest.approx(s.at(_loop(f - 3)))
    # wrapping is what keeps a delayed seamless loop seamless
    assert d.at(_loop(0)) == pytest.approx(d.at(_loop(FRAMES)))
    # a negative delay looks ahead — well-defined for a pure function of time
    ahead = delay(s, -2.0 / FRAMES)
    assert ahead.at(_loop(1)) == pytest.approx(s.at(_loop(3)))


def test_delay_on_an_open_clock_runs_off_the_end():
    # Ramp is the open-timeline leaf: value == t, so it reports the phase directly.
    d = delay(Ramp(), 0.5)
    assert d.at(_open(FRAMES - 1)) == pytest.approx(0.5)
    assert d.at(_open(0)) == pytest.approx(-0.5)   # no wrap: honestly before the start


def test_warp_accepts_a_callable_or_a_signal():
    s = Sine()
    by_fn = warp(s, lambda t: t * 2.0)
    by_sig = warp(s, Phase() * 2.0)
    for f in range(FRAMES):
        c = _loop(f)
        expect = s.at(retimed_clock(c, (f / FRAMES) * 2.0))
        assert by_fn.at(c) == pytest.approx(expect)
        assert by_sig.at(c) == pytest.approx(expect)
    # a warp whose span is a whole number of loops keeps the loop closed
    assert by_fn.at(_loop(0)) == pytest.approx(by_fn.at(_loop(FRAMES)))


def test_non_finite_sample_phase_is_rejected():
    # the driver is an ordinary Signal, so the ordinary guard owns it — and the
    # error names the offending *node*, not the retime that happened to read it.
    bad = Retime(Sine(), Const(float("inf")))
    with pytest.raises(ValueError, match="non-finite"):
        bad.at(_loop(0))
    # ... and a warp built from a misbehaving callable is caught the same way
    with pytest.raises(ValueError, match="non-finite"):
        warp(Sine(), lambda t: float("nan")).at(_loop(3))


# ---------------------------------------------------------------------------
# the cache — the thing the roadmap said not to get wrong
# ---------------------------------------------------------------------------

def test_retimed_read_does_not_poison_the_frame_cache():
    shared = Sine()
    d = delay(shared, 0.25)
    c = _loop(3)                       # t = 0.25 -> sin = 1.0; delayed -> 0.0
    for order in ("normal-first", "retimed-first"):
        cache = Cache()
        if order == "normal-first":
            a = shared.at(c, cache)
            b = d.at(c, cache)
        else:
            b = d.at(c, cache)
            a = shared.at(c, cache)
        assert a == pytest.approx(1.0), order
        assert b == pytest.approx(0.0), order


def test_a_sub_frame_retime_does_not_poison_the_frame_cache():
    """The case a *frame*-keyed memo cannot survive on its own.

    A delay of less than half a frame lands on the **same** ``clock.frame`` at a
    different phase, so ``(node id, frame)`` names both values at once.  Only the
    nested scope keeps them apart — without it whichever read happens first wins
    and silently answers for the other.
    """
    dt = 0.25 / FRAMES                 # a quarter frame: same frame index, new phase
    shared = Sine()
    d = delay(shared, dt)
    c = _loop(1)
    now = shared.at(c)
    then = shared.at(retimed_clock(c, c.t - dt))
    assert retimed_clock(c, c.t - dt).frame == c.frame   # the keys really collide
    assert abs(now - then) > 1e-3, "anti-vacuity: the two phases must differ"
    for order in ("normal-first", "retimed-first"):
        cache = Cache()
        if order == "normal-first":
            a = shared.at(c, cache)
            b = d.at(c, cache)
        else:
            b = d.at(c, cache)
            a = shared.at(c, cache)
        assert a == pytest.approx(now), order
        assert b == pytest.approx(then), order


def test_retime_scope_shares_work_within_one_sample_point():
    shared = Counter()
    both = Retime(shared + shared * 2.0, Const(0.25))
    cache = Cache()
    assert both.at(_loop(0), cache) == pytest.approx(3.0)
    assert shared.calls == 1, "the scoped cache should memoise inside one sample"
    # a *second* sample point must not reuse the first one's value
    shared2 = Counter()
    two = Retime(shared2, Phase())
    cache2 = Cache()
    v0 = two.at(_loop(0), cache2)
    v3 = two.at(_loop(3), cache2)
    assert shared2.calls == 2
    assert v0 == pytest.approx(0.0) and v3 == pytest.approx(1.0)


def test_two_delays_of_the_same_signal_stay_independent():
    s = Sine()
    a, b = delay(s, 3.0 / FRAMES), delay(s, 6.0 / FRAMES)
    cache = Cache()
    c = _loop(3)
    assert a.at(c, cache) == pytest.approx(s.at(_loop(0)))
    assert b.at(c, cache) == pytest.approx(s.at(_loop(-3)))
    assert s.at(c, cache) == pytest.approx(s.at(_loop(3)))


def test_cached_and_uncached_agree_everywhere():
    g = delay(Sine() * 0.5 + Cosine(cycles=2), 0.2) + Sine(cycles=3)
    for f in range(FRAMES):
        c = _loop(f)
        assert g.at(c, Cache()) == pytest.approx(g.at(c, None))


# ---------------------------------------------------------------------------
# graph protocol: both edges are structural
# ---------------------------------------------------------------------------

def test_both_edges_are_reported_to_the_walker_and_cycle_check():
    body, driver = Sine(), Phase() * 0.5
    r = Retime(body, driver)
    kids = r.children()
    assert body in kids and r.when is kids[1]
    ids = {n.id for n in walk(r)}
    assert body.id in ids and driver.id in ids
    detect_signal_cycle(r)             # a retime is not a recurrence


def test_retime_rejects_a_vector_and_dispatches_through_retime():
    v = vec(Sine(), Cosine())
    with pytest.raises(TypeError, match="VecRetime"):
        Retime(v, 0.0)                 # type: ignore[arg-type]
    assert isinstance(retime(Sine(), 0.0), Retime)
    assert isinstance(retime(v, 0.0), VecRetime)


# ---------------------------------------------------------------------------
# vector retime
# ---------------------------------------------------------------------------

def test_vec_retime_moves_every_component_together():
    v = vec(Sine(), Cosine(), Const(7.0))
    r = retime(v, 0.25)
    assert r.dim == 3
    got = r.at(_loop(9), Cache())
    assert got == pytest.approx(v.at(_loop(3)))
    # ... and at a different clock it is still the same frozen sample
    assert r.at(_loop(1), Cache()) == pytest.approx(got)


def test_vec_retime_as_vec_matches_and_is_a_plain_vecsignal():
    v = vec(Sine(), Cosine())
    r = VecRetime(v, Phase() - 0.25)
    flat = r.as_vec()
    assert isinstance(flat, VecSignal) and flat.dim == 2
    for f in range(FRAMES):
        c = _loop(f)
        assert flat.at(c, Cache()) == pytest.approx(r.at(c, Cache()))


def test_vec_retime_shares_one_sample_across_components():
    shared = Counter()
    r = VecRetime(VecSignal([shared, shared * 2.0]), Const(0.25))
    assert r.at(_loop(0), Cache()) == pytest.approx((1.0, 2.0))
    assert shared.calls == 1


def test_vec_retime_walks_and_cycle_checks():
    v = vec(Sine(), Cosine())
    r = VecRetime(v, Phase())
    ids = {n.id for n in walk(r)}
    assert {c.id for c in v.components} <= ids and r.when.id in ids
    detect_signal_cycle(r)


# ---------------------------------------------------------------------------
# SigAt — the 4-D time shear
# ---------------------------------------------------------------------------

def _coords(xs):
    z = np.zeros_like(xs)
    return (xs, z, z)


def test_sigat_shears_the_phase_across_space():
    xs = np.linspace(0.0, 1.0, 5)
    f = SigAt(Sine(), T - X / 4.0)
    c = _loop(3)                                     # t = 0.25
    got = f.eval_np(_coords(xs), c, Cache())
    want = np.sin(2.0 * np.pi * (0.25 - xs / 4.0))
    assert got == pytest.approx(want)
    # anti-vacuity: the un-sheared coefficient is constant over space
    flat = (Sine() * (X * 0.0 + 1.0)).eval_np(_coords(xs), c, Cache())
    assert np.ptp(np.asarray(flat)) == pytest.approx(0.0)
    assert np.ptp(got) > 0.5


def test_sigat_with_a_constant_phase_matches_a_plain_coefficient():
    xs = np.linspace(0.0, 1.0, 4)
    c = _loop(5)
    sheared = SigAt(Sine(), T).eval_np(_coords(xs), c, Cache())
    plain = (Sine() * (X * 0.0 + 1.0)).eval_np(_coords(xs), c, Cache())
    assert sheared == pytest.approx(np.asarray(plain))


def test_sigat_does_not_poison_the_frame_cache():
    """A sheared read and a plain read of the *same* signal, one cache, one frame.

    ``_Sig`` bakes ``shared`` once per frame under ``(node id, frame)``; ``SigAt``
    reads it at other phases in that same frame.  Both must come out right
    whichever runs first.  The shear here is deliberately *sub-frame* — every
    sample lands on ``clock.frame``, so the frame-keyed memo names all of them at
    once and only the nested scope can tell them apart.
    """
    xs = np.linspace(0.0, 1.0, 8)
    shared = Sine()
    c = _loop(1)
    now = shared.at(c)
    phases = c.t - xs * 0.03
    assert all(retimed_clock(c, float(t)).frame == c.frame for t in phases)
    sheared_want = np.array([Sine().at(retimed_clock(c, float(t))) for t in phases])
    assert np.ptp(sheared_want) > 0.1                  # anti-vacuity
    for order in ("plain-first", "sheared-first"):
        cache = Cache()
        sh = SigAt(shared, T - X * 0.03)
        flat = shared * (X * 0.0 + 1.0)
        if order == "plain-first":
            a = flat.eval_np(_coords(xs), c, cache)
            b = sh.eval_np(_coords(xs), c, cache)
        else:
            b = sh.eval_np(_coords(xs), c, cache)
            a = flat.eval_np(_coords(xs), c, cache)
        assert np.asarray(a) == pytest.approx(np.full(8, now)), order
        assert b == pytest.approx(np.asarray(sheared_want)), order


def test_sigat_quantize_collapses_the_number_of_evaluations():
    xs = np.linspace(0.0, 1.0, 64)
    c = _loop(0)
    counter = Counter()
    SigAt(counter, T + X, quantize=4).eval_np(_coords(xs), c, Cache())
    # phases snap to {0, .25, .5, .75, 1} -- five values, but on a closed clock
    # 1 wraps onto 0, so they name only four distinct sample *clocks*, and the
    # per-sample cache scope (keyed on the wrapped phase) shares that one.
    assert counter.calls == 4
    exact = Counter()
    SigAt(exact, T + X).eval_np(_coords(xs), c, Cache())
    assert exact.calls == 63           # 64 points, minus the same 1 -> 0 wrap
    with pytest.raises(ValueError, match="quantize"):
        SigAt(Sine(), T, quantize=0)


def test_sigat_wraps_on_a_loop_so_a_sheared_loop_stays_seamless():
    xs = np.linspace(0.0, 1.0, 9)
    f = SigAt(Sine(), T - X)
    first = f.eval_np(_coords(xs), _loop(0), Cache())
    last = f.eval_np(_coords(xs), _loop(FRAMES), Cache())
    assert first == pytest.approx(last)


def test_sigat_refuses_to_emit_ftsl():
    f = SigAt(Sine(), T - X)
    with pytest.raises(TypeError, match="no ftsl spelling"):
        f.emit(("x", "y", "z"), object())


def test_sigat_reports_time_and_rebuilds():
    s = Sine()
    f = SigAt(s, T - X / 2.0, wrap=False, quantize=8)
    assert f.uses_time() and s in f.time_signals()
    # substitute rewrites the phase field in place, preserving the knobs
    g = (f * 1.0).substitute({"u": X})
    rebuilt = f._rebuild([T - X])
    assert isinstance(rebuilt, SigAt)
    assert rebuilt.wrap is False and rebuilt.quantize == 8 and rebuilt.sig is s
    assert isinstance(g, type(f * 1.0))


def test_sigat_rejects_a_non_finite_phase_field():
    xs = np.array([0.0, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        SigAt(Sine(), T + X).eval_np(_coords(xs), _loop(0), Cache())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all retime tests passed")
