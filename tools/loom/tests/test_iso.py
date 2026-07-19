"""M5 tests: animatable isosurface emission (gyroid + N-D slicer).

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, Sine, rotations, vec,
    Scene, Material, Camera, Isosurface, gyroid_surface, phase_drift,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, clock):
    return el.emit(EmitCtx(clock=clock, cache=Cache()))


def test_isosurface_emits_wellformed_block():
    iso = gyroid_surface(freq=1.5, threshold=0.0, material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert txt.startswith('gyroid = isosurface {')
    assert txt.count("{") == txt.count("}"), (txt.count("{"), txt.count("}"))
    assert 'function {' in txt and "expr" in txt
    assert "contained_by" in txt
    assert 'material "m"' in txt
    # gyroid uses sin/cos three times each
    assert txt.count("sin(") == 3 and txt.count("cos(") == 3


def test_no_plus_minus_adjacency():
    # a rotation makes coefficients negative; the emitter must not produce "+-".
    rot = rotations(3, [(0, 1, 0.7), (1, 2, -0.4)])
    iso = gyroid_surface(freq=1.0, rotation=rot, material="m")
    txt = _emit(iso, Clock(t=0.3))
    assert "+-" not in txt, txt
    assert "*-" not in txt, txt


def test_drift_animates_and_is_seamless():
    iso = gyroid_surface(freq=1.0, drift=vec(phase_drift(1.0), 0.0, 0.0),
                         material="m")
    a0 = _emit(iso, Clock.at_frame(0, 24))
    amid = _emit(iso, Clock.at_frame(12, 24))
    awrap = _emit(iso, Clock.at_frame(24, 24))
    assert a0 != amid, "drifting gyroid should change across the loop"
    assert a0 == awrap, "loop wrap must reproduce frame 0 exactly (seamless)"


def test_rotation_animates():
    rot = rotations(3, [(0, 2, Sine(cycles=1, amp=math.pi))])
    iso = gyroid_surface(freq=1.2, rotation=rot, material="m")
    a = _emit(iso, Clock(t=0.1, frame=2, frames=24))
    b = _emit(iso, Clock(t=0.3, frame=7, frames=24))
    assert a != b, "an animated rotation should tilt the field over time"


def test_sphere_container_and_open():
    iso = Isosurface("schwarz_p", freq=1.0, container="sphere",
                     center=(0, 0, 0), radius=3.0, open=True, material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert "contained_by { sphere {" in txt
    assert "open on" in txt


def test_scene_check_cycles_ok():
    s = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(16, 16)))
    rot = rotations(3, [(0, 1, Sine(cycles=1, amp=1.0))])
    s.add(Material("m", "diffuse", reflect=0.7),
          gyroid_surface(freq=Const(1.5), rotation=rot,
                         drift=vec(phase_drift(2.0), 0.0, 0.0), material="m"))
    s.check_cycles()  # must not raise
    txt = s.emit(Clock.at_frame(3, 24), Cache())
    assert 'gyroid = isosurface' in txt


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
