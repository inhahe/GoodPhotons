"""Tests for ``examples/pastel_jack.py`` — the worked driven-channel example.

``pastel_jack`` is the scene behind the project README's demo clip, and it is the one
example that exercises loom's whole stack end to end: modulator signals of the loop
phase, dependent geometry *solved* from those signals, and the authored quantities
published as named :class:`~loom.anim.Slot` channels a ``CurveDrive`` can push at
runtime.  What that buys — and what these tests guard — is that the scene stays
*correct under drive*, not merely correct at its authored values:

1. every channel the docstring advertises is discoverable by name, and a name that is
   read from several places resolves to a single shared ``Slot``, so setting it once
   sets it everywhere;
2. pushing a value through a ``SceneDriver`` really does change the emitted ``.ftsl``,
   ``mod`` bindings accumulate onto the authored default, and a reset restores the
   byte-identical default frame;
3. the two **contact invariants** hold at every combination of lean and ring tilt in
   range and at every phase of the loop — the jack's lowest ball on the floor, and the
   ring's lowest point on the floor.  These are the ones that a naive keyframed version
   would break the moment a channel moved off its authored value;
4. the clamps bite, and the scene is still grounded when they do;
5. emission is a pure function of the clock (same frame ⇒ byte-identical text), and the
   loop is seamless by construction (every angle is periodic in the phase).

Runnable directly (``python tests/test_example_pastel_jack.py``) or under pytest.
"""

from __future__ import annotations

import math
import os
import re
import sys

import pytest

_LOOM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _LOOM)
sys.path.insert(0, os.path.join(_LOOM, "examples"))

import jumping_jack as jj                                              # noqa: E402
import pastel_jack as pj                                               # noqa: E402
from loom import Cache, Clock                                          # noqa: E402
from loom.anim import (ChannelBinding, CurveDrive, SceneDriver,        # noqa: E402
                       collect_slots)

CHANNELS = {"ring_spin", "ring_tilt", "jack_spin", "jack_precess", "jack_lean"}

# `fmt` writes the scene at %.6g, so a value read back OUT of the emitted text carries up
# to ~1e-5 of formatting error at these magnitudes.  That is the resolution of the test,
# not slop in the geometry: the solves themselves are exact.
TOL = 2e-5

_TORUS = re.compile(r"torus \{ major (\S+)\s+minor (\S+)\s+rotate (\S+) (\S+) 0\s+"
                    r"translate (\S+) (\S+) (\S+) \}")
_SPHERE = re.compile(r"sphere \{ center (\S+) (\S+) (\S+)\s+radius (\S+) \}")


def emit(scene, frame=0, frames=None):
    frames = jj.FRAMES if frames is None else frames
    return scene.emit(Clock.at_frame(frame, frames), Cache())


def lowest_points(txt):
    """``(lowest ball bottom, lowest point of the ring)`` read back out of the text."""
    balls = [float(m.group(2)) - float(m.group(4)) for m in _SPHERE.finditer(txt)
             if float(m.group(4)) == jj.BALL]
    m = _TORUS.search(txt)
    major, minor, tilt, _az, _cx, cy, _cz = (float(g) for g in m.groups())
    # Extreme of a torus of the given radii tipped `tilt` about x, measured from its
    # centre: major*sin(tilt) out along the tipped plane, plus the tube radius.
    return min(balls), cy - (major * math.sin(math.radians(tilt)) + minor)


# ---------------------------------------------------------------------------
# 1. the channels exist and are shared
# ---------------------------------------------------------------------------
def test_all_channels_are_discoverable():
    assert set(collect_slots(pj.build_scene())) == CHANNELS


def test_a_channel_reached_by_several_roots_is_one_shared_slot():
    slots = collect_slots(pj.build_scene())
    # `jack_lean` is deliberately read from four places — the jack's pose, the jack's
    # centre height, the ring's centre height and the ring's reach — because the lean is
    # what the standing height and the ring's span are solved FROM.  Every hit must be
    # the same object or a push would only reach some of them.
    assert len(slots["jack_lean"]) > 1
    for name, hits in slots.items():
        assert len({id(s) for s in hits}) == 1, f"{name} resolved to distinct Slots"


def test_channel_defaults_are_the_authored_values():
    slots = collect_slots(pj.build_scene())
    assert slots["jack_lean"][0].default == jj.TILT
    assert slots["ring_tilt"][0].default == pj.RING_TILT
    for name in ("ring_spin", "jack_spin", "jack_precess"):
        assert slots[name][0].default == 0.0


def test_the_proposed_drive_targets_exactly_those_channels():
    assert {b.target for b in pj.DRIVE.bindings} == CHANNELS
    # All `mod`, never `pin`: a pin would replace the authored value outright, so a flat
    # driving curve would silently discard the scene's own animation.  With `mod` a flat
    # curve is a no-op and the authored motion survives.
    assert all(b.mode == "mod" for b in pj.DRIVE.bindings)


# ---------------------------------------------------------------------------
# 2. driving the scene
# ---------------------------------------------------------------------------
def test_pushed_values_reach_the_emitted_scene_and_reset_cleanly():
    scene = pj.build_scene()
    base = emit(scene)
    drive = CurveDrive(3, [(0.0,) * 3, (1.0,) * 3],
                       [ChannelBinding(0, "ring_spin", mode="mod", gain=90.0),
                        ChannelBinding(1, "ring_tilt", mode="mod", gain=25.0),
                        ChannelBinding(2, "jack_lean", mode="mod", gain=-10.0)])
    sd = SceneDriver(scene, drive, strict=True)

    assert sd.emit_frame([1.0, 1.0, 1.0], Clock.at_frame(0, jj.FRAMES)) != base

    resolved = sd.set_values([1.0, 1.0, 1.0])
    assert resolved["ring_spin"] == pytest.approx(90.0)          # 0 default + 90
    assert resolved["ring_tilt"] == pytest.approx(pj.RING_TILT + 25.0)
    assert resolved["jack_lean"] == pytest.approx(jj.TILT - 10.0)

    sd.set_values([0.0, 0.0, 0.0])
    assert emit(scene) == base


# ---------------------------------------------------------------------------
# 3. the contact invariants survive being driven
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lean", [jj.LEAN_MIN, 20.0, jj.TILT, 40.0, jj.LEAN_MAX])
@pytest.mark.parametrize("ring_tilt", [10.0, pj.RING_TILT, 90.0])
def test_both_bodies_stay_on_the_floor_at_any_lean_and_tilt(lean, ring_tilt):
    scene = pj.build_scene()
    slots = collect_slots(scene)
    slots["jack_lean"][0].set(lean)
    slots["ring_tilt"][0].set(ring_tilt)
    for frame in range(0, jj.FRAMES, 37):
        ball, ring = lowest_points(emit(scene, frame))
        assert ball == pytest.approx(jj.Y0, abs=TOL)
        assert ring == pytest.approx(jj.Y0, abs=TOL)


# ---------------------------------------------------------------------------
# 4. clamps
# ---------------------------------------------------------------------------
def test_out_of_range_pushes_clamp_and_stay_grounded():
    scene = pj.build_scene()
    slots = collect_slots(scene)
    slots["jack_lean"][0].set(500.0)
    slots["ring_tilt"][0].set(-90.0)
    txt = emit(scene)

    assert float(_TORUS.search(txt).group(3)) == pytest.approx(pj.RING_TILT_MIN)
    ball, ring = lowest_points(txt)
    assert ball == pytest.approx(jj.Y0, abs=TOL)
    assert ring == pytest.approx(jj.Y0, abs=TOL)


# ---------------------------------------------------------------------------
# 5. purity and periodicity
# ---------------------------------------------------------------------------
def test_emission_is_a_pure_function_of_the_clock():
    a, b = pj.build_scene(), pj.build_scene()
    for frame in (0, 97, jj.FRAMES - 1):
        assert emit(a, frame) == emit(b, frame)
    assert emit(a, 97) == emit(a, 97)


def test_every_angle_is_periodic_in_the_phase():
    """The loop is seamless *by construction*, not by matching the ends up.

    Every animated angle is ``k * 360 * phase`` for an integral ``k``, so evaluating it
    one whole period on gives back the same angle mod a full turn.  ``Clock.at_frame``
    wraps ``t`` itself, which would make a frame-based check pass trivially — so this
    reads the signals at a hand-built ``t = 1.0`` instead.
    """
    scene = pj.build_scene()
    ring = next(e for e in scene.elements if isinstance(e, pj.Ring))
    jack = next(e for e in scene.elements if isinstance(e, jj.Jack))
    c0, c1 = Clock(t=0.0), Clock(t=1.0)
    for signal, period in ((ring.azimuth, 360.0),
                           (jack.spin_angle, 2.0 * math.pi),
                           (jack.precess_angle, 2.0 * math.pi)):
        a = signal.at(c0, Cache()) % period
        b = signal.at(c1, Cache()) % period
        assert a == pytest.approx(b, abs=1e-9)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
