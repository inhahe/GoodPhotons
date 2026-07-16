"""M3 tests: scene emission is valid, animated, and seamless.

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, Sine, LoopNoise, vec, PointPath, LoopCurve,
    Scene, Material, Sphere, Beads, Light, Camera, Raw,
    SignalCycleError, RefSignal,
)


def _mk_scene(res=(64, 64)) -> Scene:
    anchors = [
        vec(0.4 + LoopNoise(cells=4, seed=i, amp=0.1),
            LoopNoise(cells=4, seed=10 + i, amp=0.1),
            0.4 * math.sin(i))
        for i in range(5)
    ]
    curve = LoopCurve(PointPath(anchors, closed=True), Const(0.0))
    s = Scene(Camera(eye=(0, 0.5, 2), look_at=(0, 0, 0), fov_y=40, mode="R", res=res))
    s.add(
        Material("wire", "diffuse", reflect=0.8),
        Beads(curve, count=24, radius=0.05, material="wire"),
        Sphere(vec(Sine(cycles=1, amp=0.3), 0.0, 0.0), 0.1, "wire"),
        Light("area", origin="0 1.5 0", u="0.5 0 0", v="0 0 0.5",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return s


def test_emit_is_wellformed():
    s = _mk_scene()
    text = s.emit(Clock(t=0.2, frame=6, frames=48))
    assert text.startswith("scene {")
    # balanced braces
    assert text.count("{") == text.count("}"), (text.count("{"), text.count("}"))
    assert 'camera "cam"' in text
    assert text.count("sphere {") == 24 + 1   # beads + one explicit sphere
    assert 'material "wire"' in text
    assert "light area" in text


def test_emit_is_seamless():
    # The render path builds clocks with Clock.at_frame, which maps frame N of N
    # back to t=0.0 exactly -- so the loop closes with no seam.
    s = _mk_scene()
    a = s.emit(Clock.at_frame(0, 48))
    b = s.emit(Clock.at_frame(48, 48))   # wraps to frame 0's phase
    assert a == b, "scene must be identical at the loop wrap point (seamless)"


def test_emit_actually_animates():
    s = _mk_scene()
    a = s.emit(Clock(t=0.0, frame=0, frames=48))
    b = s.emit(Clock(t=0.5, frame=24, frames=48))
    assert a != b, "scene should differ across the loop"


def test_check_cycles_fires_on_bad_scene():
    ref = RefSignal("x")
    bad = ref + 1.0
    ref.bind(bad)
    s = Scene(Camera(eye=(0, 0, 2), look_at=(0, 0, 0), res=(16, 16)))
    s.add(Material("m", "diffuse", reflect=bad))
    try:
        s.check_cycles()
    except SignalCycleError:
        return
    raise AssertionError("check_cycles should have raised on a cyclic material prop")


def test_beads_from_pointpath():
    path = PointPath([vec(0, 0, 0), vec(1, 0, 0), vec(1, 1, 0), vec(0, 1, 0)],
                     closed=True)
    b = Beads(path, count=8, radius=0.1, material="m")
    from loom.ftsl_emit import EmitCtx
    txt = b.emit(EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    assert txt.count("sphere {") == 8


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
