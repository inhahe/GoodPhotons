"""Motion keys (ftrace ROADMAP 11): with a :class:`~loom.Motion`, a frame's moving transforms
and spheres are written as ``motion { key { ... } ... }`` blocks, sampled at times around the
frame's instant, and everything that holds still is written exactly as without one.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, Const, TimeFn, vec, Scene, Material, Sphere, Beads, Light, Camera, Group,
    PointPath, LoopCurve, Motion,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.scene import emit_element  # noqa: E402


def _emit(el, clock, motion=None) -> str:
    return emit_element(el, EmitCtx(clock=clock, cache=Cache(), motion=motion))


def _keys(text: str):
    """The bodies of a motion block's keys."""
    return re.findall(r"key \{ ([^}]*) \}", text)


def test_motion_needs_the_instant_on_a_key():
    Motion((-0.5, 0.5), 5)
    Motion((0.0, 1.0), 2)
    with pytest.raises(ValueError):
        Motion((-0.5, 0.5), 4)        # keys at -0.5, -1/6, 1/6, 0.5: none at 0
    with pytest.raises(ValueError):
        Motion((0.2, 1.0), 3)         # the instant is outside
    with pytest.raises(ValueError):
        Motion((0.0, 1.0), 1)
    assert Motion((-0.5, 0.5), 5).offsets() == (-0.5, -0.25, 0.0, 0.25, 0.5)
    assert Motion((-0.5, 0.5), 5).instant_key() == 2


def test_still_transform_is_written_as_without_motion():
    g = Group(Sphere((0, 0, 0), 0.2, "m"), translate=(1, 2, 3), rotate=(0, 30, 0))
    clock = Clock.at_frame(3, 10)
    assert _emit(g, clock, Motion()) == _emit(g, clock)
    assert "motion" not in _emit(g, clock, Motion())


def test_moving_translate_gets_a_key_per_motion_key():
    # An open 10-frame timeline: one frame moves the phase by 1/9, so x = 9 t moves 1 per frame.
    g = Group(Sphere((0, 0, 0), 0.2, "m"),
              translate=vec(TimeFn(lambda t: 9.0 * t, periodic=False), 0.0, 0.0))
    clock = Clock.at_frame(4, 10, loop=False)
    txt = _emit(g, clock, Motion((-0.5, 0.5), 5))
    keys = _keys(txt)
    assert len(keys) == 5
    xs = [float(k.split()[1]) for k in keys]
    assert xs == pytest.approx([3.5, 3.75, 4.0, 4.25, 4.5])
    assert txt.count("{") == txt.count("}")
    # the group's own transform is in the keys, not beside them
    assert "\n    translate" not in txt


def test_rotation_keys_unwrap_across_the_loop_seam():
    # A periodic spin on a looping clock folds the phase into [0, 1), so at frame 0 the key
    # half a frame earlier reads 360 - 22.5 = 337.5: unwrapped it must be -22.5, or ftrace
    # would turn the long way round between the two keys.
    g = Group(Sphere((0, 0, 0), 0.2, "m"),
              rotate=vec(0.0, TimeFn(lambda t: 360.0 * t), 0.0))
    txt = _emit(g, Clock.at_frame(0, 8), Motion((-0.5, 0.5), 3))
    ys = [float(k.split()[2]) for k in _keys(txt)]
    assert ys == pytest.approx([-22.5, 0.0, 22.5])


def test_a_fast_spin_keeps_its_direction():
    # 1000 degrees a frame on an open 48-frame shot: 250 degrees between neighbouring keys, which
    # folding each key to within half a turn of the last would read as -110 -- a slow spin the
    # wrong way round.
    g = Group(Sphere((0, 0, 0), 0.2, "m"),
              rotate=vec(0.0, TimeFn(lambda t: 47000.0 * t, periodic=False), 0.0))
    clock = Clock.at_frame(10, 48, loop=False)
    ys = [float(k.split()[2]) for k in _keys(_emit(g, clock, Motion((-0.5, 0.5), 5)))]
    steps = [b - a for a, b in zip(ys, ys[1:])]
    assert steps == pytest.approx([250.0] * 4, rel=1e-4)


def test_moving_sphere_rides_a_keyed_group():
    s = Sphere(vec(TimeFn(lambda t: 8.0 * t, periodic=False), 0.5, 0.0), 0.25, "m")
    txt = _emit(s, Clock.at_frame(2, 9, loop=False), Motion((-0.5, 0.5), 3))
    assert txt.startswith("group { motion {")
    assert 'sphere { center 0 0 0  radius 0.25  material "m" }' in txt
    keys = _keys(txt)
    assert len(keys) == 3 and all("scale" not in k for k in keys)
    assert [float(k.split()[1]) for k in keys] == pytest.approx([1.5, 2.0, 2.5])


def test_growing_sphere_scales_in_its_keys():
    s = Sphere((0, 0, 0), TimeFn(lambda t: 1.0 + t, periodic=False), "m")
    clock = Clock(t=0.5, frame=1, frames=3, loop=False)    # one frame = 0.5 of the phase
    txt = _emit(s, clock, Motion((-0.5, 0.5), 3))
    scales = [float(k.split("scale")[1]) for k in _keys(txt)]
    assert scales == pytest.approx([1.25 / 1.5, 1.0, 1.75 / 1.5], rel=1e-5)   # written to 6 digits


def test_still_sphere_and_beads_are_unchanged():
    clock = Clock.at_frame(1, 4)
    s = Sphere((1, 2, 3), 0.5, "m")
    assert _emit(s, clock, Motion()) == _emit(s, clock)
    path = PointPath([(0, 0, 0), (1, 0, 0), (1, 1, 0)], closed=True)
    b = Beads(LoopCurve(path, Const(0.0)), 6, 0.05, "m")
    assert _emit(b, clock, Motion()) == _emit(b, clock)


def _scene(shutter=(-0.25, 0.25)):
    s = Scene(Camera(eye=(0, 0.5, 3), look_at=(0, 0.3, 0), res=(32, 24), shutter=shutter))
    s.add(Material("m", "diffuse", reflect=0.7),
          Sphere(vec(TimeFn(lambda t: 2.0 * t - 1.0, periodic=False), 0.3, 0.0), 0.3, "m"),
          Group(Sphere((0, 0, 0), 0.2, "m"), translate=(0.6, 0.2, -0.4),
                rotate=vec(0.0, TimeFn(lambda t: 90.0 * t, periodic=False), 0.0)),
          Light("area", origin="-0.5 2 -0.5", u="1 0 0", v="0 0 1", normal="0 -1 0",
                spd="preset:bb6500"))
    return s


def test_scene_writes_the_interval_and_the_shutter():
    s = _scene()
    clock = Clock.at_frame(5, 12, loop=False)
    txt = s.emit(clock, Cache(), motion=Motion())
    assert txt.splitlines()[0].endswith("motion_interval -0.5 0.5 }")
    assert "shutter_open -0.25  shutter_close 0.25" in txt
    plain = s.emit(clock, Cache())
    assert "motion" not in plain
    with pytest.raises(ValueError):
        Camera(eye=(0, 0, 1), look_at=(0, 0, 0), shutter=(0.5, -0.5))


def _ftrace():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    exe = os.path.join(root, "ftrace.exe")
    return exe if os.path.isfile(exe) else None


@pytest.mark.skipif(_ftrace() is None, reason="needs a built ftrace.exe")
def test_ftrace_loads_the_keyed_scene():
    txt = _scene().emit(Clock.at_frame(5, 12, loop=False), Cache(), motion=Motion())
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "keyed.ftsl")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(txt)
        r = subprocess.run([_ftrace(), "-in", p, "-parseonly"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
