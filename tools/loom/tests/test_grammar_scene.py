"""J3c: grammar-backed geometry / light readers (`parse_element`).

Extends the shared EPEG grammar (`ftsl.epeg`) + ParseNode builders from
materials/textures to the scene's `sphere` and `light` blocks.  As with
materials, there is no hand-written `.parse` oracle, so the contract is
**emit is a fixed point** (emit -> parse_element -> re-emit byte-identical)
plus rebuilt-kind / salient-field round-trip.

Runnable directly (`python tests/test_grammar_scene.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Clock, Cache, Light, Camera  # noqa: E402
from loom.scene import Sphere  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.grammar.reader import parse_element  # noqa: E402


def _ctx() -> EmitCtx:
    return EmitCtx(clock=Clock(t=0.0), cache=Cache())


_SAMPLES = [
    Sphere((0.0, 1.5, -2.0), 0.75, "gold"),
    Sphere((-1, 0, 0), 1, "glass"),
    Light("point", position="0 5 0", spd="preset:bb6500", power="40"),
    Light("area", color="0.9 0.8 0.7", size="2 2"),
    Light("sky", turbidity="3"),
    Camera((0, 2, 5), (0, 0, 0)),
    Camera((3.0, -1.5, 2.25), (0, 1, 0), up=(0, 0, 1), fov_y=55.0,
           mode="A", res=(1920, 1080), name="hero"),
]


@pytest.mark.parametrize("obj", _SAMPLES)
def test_emit_is_a_fixed_point(obj):
    ctx = _ctx()
    once = obj.emit(ctx)
    twice = parse_element(once).emit(ctx)
    assert once == twice


@pytest.mark.parametrize("obj", _SAMPLES)
def test_rebuilt_kind_matches(obj):
    assert type(parse_element(obj.emit(_ctx()))) is type(obj)


def test_sphere_fields_roundtrip():
    s = Sphere((0.0, 1.5, -2.0), 0.75, "gold")
    back = parse_element(s.emit(_ctx()))
    assert isinstance(back, Sphere)
    # center/radius are VecSignal/scalar — compare via re-baked numeric emit
    assert back.emit(_ctx()) == s.emit(_ctx())
    assert back.material == "gold"


def test_light_fields_roundtrip():
    lt = Light("point", position="0 5 0", spd="preset:bb6500", power="40")
    back = parse_element(lt.emit(_ctx()))
    assert isinstance(back, Light)
    assert back.kind == "point"
    assert back.props["position"] == "0 5 0"
    assert back.props["spd"] == "preset:bb6500"
    assert back.props["power"] == "40"


def test_camera_fields_roundtrip():
    cam = Camera((3.0, -1.5, 2.25), (0, 1, 0), up=(0, 0, 1), fov_y=55.0,
                 mode="A", res=(1920, 1080), name="hero")
    back = parse_element(cam.emit(_ctx()))
    assert isinstance(back, Camera)
    assert back.name == "hero"
    assert back.mode == "A"
    assert back.res == (1920, 1080)
    assert back.emit(_ctx()) == cam.emit(_ctx())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
