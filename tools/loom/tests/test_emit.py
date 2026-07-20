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
    Scene, Material, Sphere, Beads, Light, Camera, CameraCurve, Raw, Volume,
    SignalCycleError, RefSignal,
)
from loom.spatial import X, Y, Z, sin as ssin  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402


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
    assert 'cam = camera' in text
    assert text.count("sphere {") == 24 + 1   # beads + one explicit sphere
    assert 'wire = material' in text
    assert "kind area" in text


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


def _vol_emit(v: Volume, t: float = 0.0) -> str:
    return v.emit(EmitCtx(clock=Clock(t=t), cache=Cache()))


def test_volume_homogeneous():
    txt = _vol_emit(Volume(sigma_t=2.0, albedo=0.9, g=0.3))
    assert txt.startswith("medium {")
    assert txt.count("{") == txt.count("}")
    assert "sigma_t 2" in txt and "albedo 0.9" in txt and "g 0.3" in txt
    assert "bounds" not in txt and "density" not in txt


def test_volume_bounds_variants():
    box = _vol_emit(Volume(box=((0, 0, 0), (1, 1, 1))))
    assert "bounds { min 0 0 0  max 1 1 1 }" in box
    sph = _vol_emit(Volume(sphere=((0.5, 0.45, 0.5), 0.32)))
    assert "bounds { center 0.5 0.45 0.5  radius 0.32 }" in sph
    obj = _vol_emit(Volume(obj="orb"))
    assert 'bounds { object "orb" }' in obj


def test_volume_procedural_density_and_max():
    field = 0.6 + 0.4 * ssin(8.0 * X) * ssin(8.0 * Y) * ssin(8.0 * Z)
    txt = _vol_emit(Volume(sigma_t=8.0, density=field,
                           sphere=((0.5, 0.45, 0.5), 0.32), density_max=1.2))
    assert 'density "' in txt and "sin(" in txt
    assert "density_max 1.2" in txt


def test_volume_density_string_forms():
    assert "density pattern:noise" in _vol_emit(Volume(density="pattern:noise", obj="o"))
    assert "density vdb:clouds/c.nvdb" in _vol_emit(Volume(density="vdb:clouds/c.nvdb"))
    # a bare string is a raw ftsl expr -> quoted
    assert 'density "x*x+y*y"' in _vol_emit(Volume(density="x*x+y*y", box=((0, 0, 0), (1, 1, 1))))


def test_volume_animatable_sigma_t():
    v = Volume(sigma_t=Const(1.0) + Sine(cycles=1, amp=0.5), sphere=((0, 0, 0), 1.0))
    a = _vol_emit(v, t=0.0)
    b = _vol_emit(v, t=0.25)
    assert a != b, "animatable sigma_t should differ across the loop"


def test_volume_rayleigh_and_bound_guard():
    assert "rayleigh true" in _vol_emit(Volume(rayleigh=True))
    try:
        Volume(box=((0, 0, 0), (1, 1, 1)), sphere=((0, 0, 0), 1.0))
    except ValueError:
        return
    raise AssertionError("Volume must reject more than one bound")


def _cc_emit(cc: CameraCurve) -> str:
    return cc.emit(EmitCtx(clock=Clock(t=0.0), cache=Cache()))


def test_camera_curve_minimal_golden():
    cc = CameraCurve([(0, 0, 2), (1, 0, 2), (1, 0, 0)],
                     look_at=(0, 0, 0), fov_y=40, frames=8,
                     res=(320, 240), name="fly")
    txt = _cc_emit(cc)
    assert txt.count("{") == txt.count("}")
    assert txt.startswith('fly = camera_curve {')
    assert txt.count("point ") == 3
    assert "look_at 0 0 0" in txt
    assert "fov_y 40" in txt and "mode R" in txt
    assert "frames 8" in txt
    assert "film { res 320 240 }" in txt
    # No orientation keywords authored -> none emitted (legacy world behavior).
    for kw in ("frame ", "fwd_frame", "up_frame", "fwd_at", "up_at", "roll"):
        assert kw not in txt, kw


def test_camera_curve_orientation_axes():
    cc = CameraCurve(
        [(0.5, 0.75, 1.7), (1.3, 0.45, 0.5), (0.5, 0.75, -0.7), (-0.3, 0.45, 0.5)],
        frames=6, closed=True, mode="R",
        fwd_at=[(0.0, 0, 0, -1), (0.5, 0.707, 0, -0.707), (1.0, 1, 0, 0)],
        up_at=[(0.0, 0, 1, 0), (1.0, 0, 0, 1)],
        frame="world", fwd_frame="travel", up_frame="travel", name="orbit")
    txt = _cc_emit(cc)
    assert txt.count("{") == txt.count("}")
    assert "closed" in txt
    assert "frame world" in txt
    assert "fwd_frame travel" in txt
    assert "up_frame travel" in txt
    assert "fwd_at 0 0 0 -1" in txt
    assert "fwd_at 0.5 0.707 0 -0.707" in txt
    assert "up_at 0 0 1 0" in txt and "up_at 1 0 0 1" in txt


def test_camera_curve_scalar_tracks_and_density():
    cc = CameraCurve([(0, 0, 2), (2, 0, 0)], look_at=(0, 0, 0), density=0.5,
                     fov_at=[(0, 60), (1, 30)], roll_at=[(0, 0), (1, 45)],
                     focus_at=[(0, 2.0), (1, 1.0)], name="dolly")
    txt = _cc_emit(cc)
    assert "density 0.5" in txt
    assert "fov_at 0 60" in txt and "fov_at 1 30" in txt
    assert "roll_at 0 0" in txt and "roll_at 1 45" in txt
    assert "focus_at 0 2" in txt and "focus_at 1 1" in txt


def test_camera_curve_look_points_spline():
    cc = CameraCurve([(0, 0, 2), (2, 0, 0)],
                     look_points=[(0, 0, 0), (1, 0, 0)], frames=4, name="chase")
    txt = _cc_emit(cc)
    assert "look curve" in txt
    assert txt.count("look_point ") == 2


def test_camera_curve_validates():
    for bad in (
        lambda: CameraCurve([(0, 0, 0)], frames=4),                       # <2 points
        lambda: CameraCurve([(0, 0, 0), (1, 0, 0)]),                      # no frames/density
        lambda: CameraCurve([(0, 0, 0), (1, 0, 0)], frames=4,
                            look_at=(0, 0, 0), look_points=[(0, 0, 0)]),  # both aims
        lambda: CameraCurve([(0, 0, 0), (1, 0, 0)], frames=4, frame="bogus"),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("CameraCurve should have rejected the bad args")


def test_camera_curve_in_scene():
    cc = CameraCurve([(0, 0.5, 2), (2, 0.5, 0)], look_at=(0, 0, 0),
                     frames=8, name="cam")
    s = Scene(cc)
    s.add(Sphere(vec(0, 0, 0), 0.3, "m"),
          Material("m", "diffuse", reflect=0.6),
          Light("area", origin="0 1.5 0", u="0.5 0 0", v="0 0 0.5",
                normal="0 -1 0", spd="preset:bb6500"))
    text = s.emit(Clock(t=0.0, frame=0, frames=8))
    assert 'cam = camera_curve' in text
    assert text.count("{") == text.count("}")


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
