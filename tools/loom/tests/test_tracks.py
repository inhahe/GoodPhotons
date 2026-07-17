"""Tests for multi-track N-D curves — a sequence carrying Y side-curves keyed at
the same control points (a camera_curve's position + speed + orientation model).

Covers TrackedPath (dataset), TrackedCurve (samples the point + every track at one
shared parameter) and Reparam (retime traversal by a per-waypoint speed/density
track).  Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Const, Sine, vec, VecSignal,
    PointPath, TrackedPath, LoopCurve, TrackedCurve, Reparam, eval_curve,
    detect_signal_cycle,
)


def _clk(t: float = 0.0, frame: int = 0) -> Clock:
    return Clock(t=t, frame=frame, frames=1, fps=30.0)


# --- a reusable tracked sequence -------------------------------------------

def _make() -> TrackedPath:
    pts = [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 3.0, 1.0), (-1.0, 1.0, 2.0)]
    orient = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    speed = [1.0, 2.0, 0.5, 1.5]
    return TrackedPath(pts, tracks={"orient": orient, "speed": speed}, closed=True)


# --- TrackedPath dataset ----------------------------------------------------

def test_trackedpath_keys_tracks_together():
    tp = _make()
    assert tp.npoints == 4 and tp.dim == 3
    assert set(tp.tracks) == {"orient", "speed"}
    assert tp.is_scalar("speed") and not tp.is_scalar("orient")
    # scalar track stored 1-D, vector track keeps its dim
    assert tp.tracks["speed"][0].dim == 1
    assert tp.tracks["orient"][0].dim == 3


def test_trackedpath_length_mismatch_raises():
    try:
        TrackedPath([(0, 0), (1, 1)], tracks={"bad": [1.0]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-length track")


def test_weights_of_rejects_vector_track():
    tp = _make()
    try:
        tp.weights_of("orient")
    except ValueError:
        return
    raise AssertionError("weights_of should reject a vector track")


# --- TrackedCurve: point + every track share the parameter ------------------

def test_position_matches_plain_loopcurve():
    tp = _make()
    tc = TrackedCurve(tp, u=0.37)
    clk = _clk()
    main_pts = [p.at(clk) for p in tp.path.points]
    expect = eval_curve(main_pts, 0.37, True)
    got = tc.position.at(clk)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, expect)), (got, expect)


def test_tracks_sampled_on_same_parameter():
    tp = _make()
    for u in (0.0, 0.2, 0.5, 0.81):
        tc = TrackedCurve(tp, u=u)
        clk = _clk()
        # vector track
        opts = [p.at(clk) for p in tp.tracks["orient"]]
        oexp = eval_curve(opts, u, True)
        ogot = tc.track("orient").at(clk)
        assert all(abs(a - b) < 1e-12 for a, b in zip(ogot, oexp)), (u, ogot, oexp)
        # scalar track, via item access
        spts = [p.at(clk) for p in tp.tracks["speed"]]
        sexp = eval_curve(spts, u, True)[0]
        sgot = tc["speed"].at(clk)
        assert abs(sgot - sexp) < 1e-12, (u, sgot, sexp)


def test_scalar_track_is_a_scalar_signal():
    tc = TrackedCurve(_make(), u=0.1)
    from loom import Signal  # noqa: WPS433
    assert isinstance(tc.track("speed"), Signal)
    assert isinstance(tc.track("orient"), VecSignal)


def test_unknown_track_raises():
    tc = TrackedCurve(_make(), u=0.1)
    try:
        tc.track("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown track")


# --- Reparam: retime traversal by a speed/density track ---------------------

def test_uniform_weights_are_identity():
    r = Reparam([1.0, 1.0, 1.0, 1.0], s=0.0, closed=True)
    # rebuild per-s since s is a leaf here; drive by t via a fresh Reparam each time
    for s in (0.0, 0.13, 0.5, 0.777, 0.99):
        rr = Reparam([1.0, 1.0, 1.0, 1.0], s=s, closed=True)
        assert abs(rr.at(_clk()) - s) < 1e-12, (s, rr.at(_clk()))


def test_dwell_fraction_follows_weights():
    # bin 0 gets 3x the dwell of bin 1: 75% of travel maps into u < 0.5.
    weights = [3.0, 1.0]
    lo = sum(1 for k in range(1000)
             if Reparam(weights, s=k / 1000.0, closed=True).at(_clk()) < 0.5)
    frac = lo / 1000.0
    assert abs(frac - 0.75) < 0.01, frac


def test_reparam_monotonic_and_bounded():
    weights = [0.5, 3.0, 1.0, 2.0]
    prev = -1.0
    for k in range(200):
        s = k / 200.0
        u = Reparam(weights, s=s, closed=True).at(_clk())
        assert 0.0 <= u < 1.0, (s, u)
        assert u >= prev - 1e-12, (s, u, prev)
        prev = u


def test_nonpositive_weight_does_not_explode():
    # a zero weight is floored to eps, so the map stays finite & monotonic.
    u0 = Reparam([0.0, 1.0, 1.0], s=0.0, closed=True).at(_clk())
    u1 = Reparam([0.0, 1.0, 1.0], s=0.999, closed=True).at(_clk())
    assert 0.0 <= u0 < 1.0 and 0.0 <= u1 < 1.0, (u0, u1)


def test_traveling_dwells_where_density_high():
    # one waypoint has huge density: near s spent there, u barely advances.
    tp = TrackedPath([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
                     tracks={"density": [1.0, 20.0, 1.0, 1.0]}, closed=True)
    tc = TrackedCurve.traveling(tp, s=0.0, density="density")
    # sample u over uniform s; the heavy bin (index 1 -> u in [0.25,0.5)) should
    # capture far more than its 1/4 share of travel.
    heavy = sum(1 for k in range(1000)
                if 0.25 <= TrackedCurve.traveling(
                    tp, s=k / 1000.0, density="density").u.at(_clk()) < 0.5)
    assert heavy / 1000.0 > 0.6, heavy / 1000.0


# --- graph integrity --------------------------------------------------------

def test_tracked_graph_is_acyclic():
    tc = TrackedCurve(_make(), u=Sine(1.0))
    detect_signal_cycle(tc.position)
    detect_signal_cycle(tc.track("orient"))
    detect_signal_cycle(tc.track("speed"))


def test_animated_track_value_changes_over_time():
    # a per-point speed driven by a modulator animates the sampled track.
    tp = TrackedPath([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
                     tracks={"speed": [Sine(1.0), Const(1.0), Const(1.0)]},
                     closed=True)
    tc = TrackedCurve(tp, u=0.0)
    a = tc.track("speed").at(Clock.at_frame(0, 8))
    b = tc.track("speed").at(Clock.at_frame(2, 8))
    assert abs(a - b) > 1e-6, (a, b)


def _run_all() -> int:
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
