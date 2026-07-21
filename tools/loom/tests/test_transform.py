"""Per-object Transform tests (Phase A): position / size / rotation / skew wrap
any element in an ftsl ``group { … }``, animate, nest, and are cycle-checked.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, Sine, vec, Scene, Material, Sphere, Beads, Light,
    Camera, Group, Transform, PointPath, LoopCurve, SignalCycleError, RefSignal,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, t: float = 0.0) -> str:
    from loom.scene import emit_element
    return emit_element(el, EmitCtx(clock=Clock(t=t), cache=Cache()))


def test_transform_wraps_in_group():
    s = Sphere((0, 0, 0), 1.0, "m").transformed(translate=(2, 0, 0), scale=1.5)
    txt = _emit(s)
    assert txt.startswith("group {")
    assert txt.count("{") == txt.count("}")
    assert "translate 2 0 0" in txt
    assert "scale 1.5 1.5 1.5" in txt          # scalar scale broadcasts
    assert 'sphere { center 0 0 0  radius 1  material "m" }' in txt


def test_transform_all_fields_emit():
    s = Sphere((0, 0, 0), 1.0, "m").transformed(
        translate=(1, 2, 3), rotate=(0, 45, 90), scale=(2, 3, 4), skew=(0.3, 0, 0.1))
    txt = _emit(s)
    assert "translate 1 2 3" in txt
    assert "rotate 0 45 90" in txt
    assert "scale 2 3 4" in txt
    assert "shear 0.3 0 0.1" in txt


def test_identity_transform_no_group():
    # A Transform with no fields set must not wrap the element at all.
    s = Sphere((0, 0, 0), 1.0, "m")
    s.xf = Transform()
    txt = _emit(s)
    assert txt.startswith("sphere {")
    assert "group" not in txt


def test_no_transform_is_plain():
    txt = _emit(Sphere((0, 0, 0), 1.0, "m"))
    assert txt.startswith("sphere {")


def test_transform_animates():
    s = Sphere((0, 0, 0), 1.0, "m").transformed(scale=Const(1.0) + Sine(cycles=1, amp=0.5))
    assert _emit(s, t=0.0) != _emit(s, t=0.25)


def test_group_wraps_children_and_nests():
    g = Group(
        Sphere((0, 0, 0), 0.5, "m").transformed(scale=2.0),
        Sphere((1, 0, 0), 0.5, "m"),
        translate=(0, 1, 0), rotate=(0, 45, 0),
    )
    txt = _emit(g)
    assert txt.count("{") == txt.count("}")
    # outer group carries the cluster transform ...
    assert "translate 0 1 0" in txt and "rotate 0 45 0" in txt
    # ... the first child keeps its own inner group (nested) ...
    assert txt.count("group {") == 2
    assert "scale 2 2 2" in txt
    assert txt.count("sphere {") == 2


def test_group_in_scene_balanced():
    cam = Camera((0, 0, 5), (0, 0, 0))
    s = Scene(cam)
    s.add(Material("m", "diffuse", reflect=0.8))
    s.add(Group(Sphere((0, 0, 0), 1.0, "m"), Beads(
        LoopCurve(PointPath([vec(0, 0, 0), vec(1, 0, 0), vec(0, 1, 0)], closed=True),
                  Const(0.0)), count=6, radius=0.05, material="m"),
        translate=(0, 0, 1), scale=2.0))
    s.check_cycles()
    text = s.emit(Clock(t=0.0, frame=0, frames=48))
    assert text.count("{") == text.count("}")
    assert "group {" in text
    assert text.count("sphere {") == 1 + 6


def test_transform_roots_are_cycle_checked():
    ref = RefSignal("x")
    bad = ref + 1.0
    ref.bind(bad)
    cam = Camera((0, 0, 5), (0, 0, 0))
    s = Scene(cam)
    s.add(Material("m", "diffuse", reflect=0.8))
    # cyclic signal buried in a transform field must be caught
    s.add(Sphere((0, 0, 0), 1.0, "m").transformed(scale=vec(bad, 1.0, 1.0)))
    try:
        s.check_cycles()
    except SignalCycleError:
        return
    raise AssertionError("check_cycles must catch a cyclic transform field")


def test_scalar_field_validation():
    for bad in (
        lambda: Transform(translate=1.0),   # scalar translate not allowed
        lambda: Transform(rotate=2.0),      # scalar rotate not allowed
        lambda: Transform(skew=0.5),        # scalar skew not allowed
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("Transform must reject a scalar for translate/rotate/skew")
    # scalar scale IS allowed (uniform)
    assert Transform(scale=2.0).scale.dim == 3


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
