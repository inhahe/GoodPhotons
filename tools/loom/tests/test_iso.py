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
    Scene, Material, Camera, Isosurface, Room, gyroid_surface, phase_drift,
    affine, Affine,
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


def test_march_and_uv_controls_emit():
    # ftrace's addIsosurface reads samples/accuracy/refine/uv, but loom had no way to
    # emit any of them, so a sampled march was stuck on the 256-sample default. This
    # gap was found by the J3c emitter audit (scraps/emit_audit.py).
    iso = Isosurface("gyroid", freq=2.0, material="m", method="sample",
                     samples=300, accuracy=0.01, refine="regula_falsi",
                     uv="planar axis=y")
    txt = _emit(iso, Clock(t=0.0))
    for line in ("method sample", "samples 300", "refine regula_falsi",
                 "uv planar axis=y"):
        assert line in txt, (line, txt)
    assert "accuracy 0.01" in txt


def test_march_and_uv_controls_omitted_by_default():
    # None must mean "say nothing" so ftrace's own defaults still apply.
    txt = _emit(Isosurface("gyroid", freq=2.0, material="m"), Clock(t=0.0))
    for key in ("samples", "accuracy", "refine", "uv ", "method"):
        assert key not in txt, (key, txt)


def test_bareword_uv_axis_is_rejected():
    # FTSL.md §2's documented trap: `uv planar y` parses the axis as a stray
    # statement and is silently dropped, so the surface keeps default UVs. ftrace
    # now warns about it, but loom must not emit it in the first place.
    for bad in ("planar y", "planar axis=w", "spherical up", "cubic"):
        try:
            Isosurface("gyroid", freq=1.0, material="m", uv=bad)
        except ValueError:
            continue
        raise AssertionError(f"uv={bad!r} should have been rejected")
    for ok in ("planar", "spherical", "cylindrical axis=z"):
        Isosurface("gyroid", freq=1.0, material="m", uv=ok)


def test_bad_refine_and_method_are_rejected():
    for kw in ({"refine": "newton"}, {"method": "raymarch"}):
        try:
            Isosurface("gyroid", freq=1.0, material="m", **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} should have been rejected")


def test_placement_zero_is_byte_identical():
    # default placement (origin) must emit exactly like the un-placed form.
    a = _emit(gyroid_surface(freq=1.0, material="m"), Clock(t=0.0))
    b = _emit(gyroid_surface(freq=1.0, placement=(0.0, 0.0, 0.0), material="m"),
              Clock(t=0.0))
    assert a == b


def test_placement_offsets_frame_and_container():
    iso = gyroid_surface(freq=1.0, placement=(2.0, 0.0, 0.0), material="m",
                         container="box",
                         bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    txt = _emit(iso, Clock(t=0.0))
    # the coordinate frame shifts: x becomes (x-(2))
    assert "(x-(2))" in txt, txt
    # and the container box moves with it: min x = -1+2 = 1, max x = 1+2 = 3
    assert "min 1 -1 -1" in txt, txt
    assert "max 3 1 1" in txt, txt


def test_sphere_placement_moves_center():
    iso = Isosurface("schwarz_p", freq=1.0, container="sphere",
                     center=(0, 0, 0), radius=3.0,
                     placement=(1.0, 2.0, -1.0), material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert "center 1 2 -1" in txt, txt


def test_placement_animatable():
    # a Signal-driven placement should move the surface over time.
    iso = gyroid_surface(freq=1.0,
                         placement=vec(Sine(cycles=1, amp=3.0), 0.0, 0.0),
                         material="m")
    a = _emit(iso, Clock(t=0.1))
    b = _emit(iso, Clock(t=0.35))
    assert a != b, "an animated placement should move the surface"


def test_room_namespaces_child_names():
    room = Room("hall",
                gyroid_surface(freq=1.0, name="a", material="m"),
                Isosurface("schwarz_p", freq=1.0, name="b", material="m"))
    txt = _emit(room, Clock(t=0.0))
    assert "hall/a = isosurface" in txt, txt
    assert "hall/b = isosurface" in txt, txt
    # child names are restored after emit (no leakage)
    assert room.children[0].name == "a"
    assert room.children[1].name == "b"


def test_room_translation_moves_children():
    # a room translated by +5 in x moves an origin-placed child's box to x∈[4,6].
    room = Room("hall",
                gyroid_surface(freq=1.0, name="g", material="m",
                               bounds=((-1, -1, -1), (1, 1, 1))),
                frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(room, Clock(t=0.0))
    assert "min 4 -1 -1" in txt, txt
    assert "max 6 1 1" in txt, txt
    # the coordinate frame shifts too: x -> (x-(5))
    assert "(x-(5))" in txt, txt


def test_room_child_placement_composes_with_frame():
    # child placed at (2,0,0) inside a room translated (5,0,0) -> world x=7.
    child = gyroid_surface(freq=1.0, name="g", material="m",
                           placement=(2.0, 0.0, 0.0),
                           bounds=((-1, -1, -1), (1, 1, 1)))
    room = Room("hall", child, frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(room, Clock(t=0.0))
    assert "min 6 -1 -1" in txt, txt   # 7-1
    assert "max 8 1 1" in txt, txt     # 7+1
    assert "(x-(7))" in txt, txt


def test_room_frame_leaves_child_parent_unset():
    child = gyroid_surface(freq=1.0, name="g", material="m")
    room = Room("hall", child, frame=Affine.translation((3.0, 0.0, 0.0)))
    _emit(room, Clock(t=0.0))
    assert child._parent is None, "room must restore child _parent after emit"


def test_nested_rooms_stack_names_and_frames():
    inner = Room("inner",
                 gyroid_surface(freq=1.0, name="g", material="m",
                                bounds=((-1, -1, -1), (1, 1, 1))),
                 frame=Affine.translation((2.0, 0.0, 0.0)))
    outer = Room("outer", inner, frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(outer, Clock(t=0.0))
    assert "outer/inner/g = isosurface" in txt, txt
    # composed translation 5+2 = 7
    assert "min 6 -1 -1" in txt, txt
    assert "max 8 1 1" in txt, txt


def test_room_animatable_frame_moves_over_time():
    room = Room("hall",
                gyroid_surface(freq=1.0, name="g", material="m"),
                frame=affine(3, [("move", vec(Sine(cycles=1, amp=4.0), 0.0, 0.0))]))
    a = _emit(room, Clock(t=0.1))
    b = _emit(room, Clock(t=0.6))
    assert a != b, "an animated room frame should move its children"


def test_room_in_scene_emits_and_checks_cycles():
    s = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(16, 16)))
    room = Room("hall",
                gyroid_surface(freq=Const(1.5), name="a", material="m"),
                gyroid_surface(freq=1.0, name="b", material="m",
                               placement=(3.0, 0.0, 0.0)),
                frame=affine(3, [("rot", 0, 2, Sine(cycles=1, amp=1.0))]))
    s.add(Material("m", "diffuse", reflect=0.7), room)
    s.check_cycles()  # must not raise
    txt = s.emit(Clock.at_frame(3, 24), Cache())
    assert "hall/a = isosurface" in txt and "hall/b = isosurface" in txt


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
