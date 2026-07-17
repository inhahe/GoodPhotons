"""M6.5 tests: opt-in looping.

Seamless looping is a *choice* (DESIGN.md §11.6): a **closed** clock wraps
(frame N == frame 0) and periodic leaves / closed curves loop; an **open** clock
has distinct endpoints and the non-periodic leaves (Ramp/Ease) / open paths are
one-shot.  Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Sine, Ramp, Ease, PointPath, LoopCurve,
)


def _at(sig, clock):
    return sig.at(clock, Cache())


# --- Clock open/closed mapping ---------------------------------------------

def test_closed_clock_wraps_to_frame0():
    N = 24
    assert Clock.at_frame(N, N, loop=True).t == 0.0
    assert Clock.at_frame(0, N, loop=True).t == 0.0
    # frame N is a phantom duplicate of frame 0 in closed mode
    assert Clock.at_frame(N, N, loop=True).t == Clock.at_frame(0, N, loop=True).t


def test_open_clock_endpoints_distinct():
    N = 24
    a = Clock.at_frame(0, N, loop=False)
    b = Clock.at_frame(N - 1, N, loop=False)
    assert a.t == 0.0
    assert abs(b.t - 1.0) < 1e-12, "last open frame must reach t=1 (no wrap)"
    assert a.t != b.t, "open timeline endpoints must be distinct (no phantom frame)"


def test_open_clock_is_evenly_spaced_inclusive():
    N = 5  # t = 0, .25, .5, .75, 1.0
    ts = [Clock.at_frame(k, N, loop=False).t for k in range(N)]
    assert ts == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_closed_default_preserves_m1_m6_mapping():
    # default loop=True must be byte-identical to the historical mapping
    N = 48
    for k in (0, 1, 12, 47, 48):
        assert Clock.at_frame(k, N).t == (k % N) / N


# --- non-periodic leaves ----------------------------------------------------

def test_ramp_is_open_not_seamless():
    N = 10
    r = Ramp(0.0, 1.0)
    first = _at(r, Clock.at_frame(0, N, loop=False))
    last = _at(r, Clock.at_frame(N - 1, N, loop=False))
    assert first == 0.0 and abs(last - 1.0) < 1e-12
    assert first != last, "a ramp on an open timeline is one-shot, not a loop"


def test_ease_in_out_smoothstep():
    N = 100
    e = Ease(0.0, 1.0, mode="in-out")
    assert abs(_at(e, Clock.at_frame(0, N, loop=False)) - 0.0) < 1e-12
    assert abs(_at(e, Clock.at_frame(N - 1, N, loop=False)) - 1.0) < 1e-12
    mid = _at(e, Clock(t=0.5))
    assert abs(mid - 0.5) < 1e-12, "smoothstep(0.5) == 0.5"


def test_ease_in_and_out_shapes():
    ein = Ease(0.0, 1.0, mode="in")
    eout = Ease(0.0, 1.0, mode="out")
    assert abs(_at(ein, Clock(t=0.5)) - 0.25) < 1e-12   # t^2
    assert abs(_at(eout, Clock(t=0.5)) - 0.75) < 1e-12  # t(2-t)


def test_ease_rejects_bad_mode():
    try:
        Ease(mode="bounce")
    except ValueError:
        return
    raise AssertionError("Ease must reject an unknown mode")


# --- periodic content still loops on a closed clock -------------------------

def test_periodic_leaf_still_loops_closed():
    N = 24
    s = Sine(cycles=3.0)
    assert _at(s, Clock.at_frame(N, N, loop=True)) == _at(s, Clock.at_frame(0, N, loop=True))


# --- curve open/closed fork -------------------------------------------------

def test_closed_curve_is_seamless():
    # probe the seam as the limit u->1- (u=1.0 wraps to 0.0), which must
    # approach the u=0 point for a closed curve.
    path = PointPath([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
    c = LoopCurve(path, 0.0)
    a = c.sample(0.0, Clock(t=0.0))
    b = c.sample(1.0 - 1e-6, Clock(t=0.0))
    assert all(abs(x - y) < 1e-3 for x, y in zip(a, b)), "closed curve wraps seamlessly"


def test_open_path_is_not_seamless():
    path = PointPath([(0, 0), (1, 0), (2, 0), (3, 1)], closed=False)
    c = LoopCurve(path, 0.0)  # inherits closed=False from the path
    assert c.closed is False
    a = c.sample(0.0, Clock(t=0.0))
    b = c.sample(1.0 - 1e-6, Clock(t=0.0))
    assert any(abs(x - y) > 0.1 for x, y in zip(a, b)), \
        "an open path must not join its endpoints"


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
