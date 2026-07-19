"""J3c: grammar-backed material / texture readers
(`loom.grammar.reader.parse_element`).

The shared EPEG grammar (`ftsl.epeg`) + ParseNode->Element builders parse the
`material "name" { … }` and `texture "name" { … }` blocks that loom emits.  There
is no hand-written `Material.parse` / `Texture.parse` oracle (unlike records), so
the contract proved here is **emit is a fixed point**: emit -> parse_element ->
emit is byte-identical, and the rebuilt object reproduces the same element kind
and salient fields.  Covers image textures, procedural (rgb-function) textures,
scalar / vector / spectrum-ref / texture-ref material props, and all material
types.

Runnable directly (`python tests/test_grammar_material.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Clock, Cache, Material  # noqa: E402
from loom.scene import Texture, ProcTexture  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.grammar.reader import parse_element  # noqa: E402


def _ctx() -> EmitCtx:
    return EmitCtx(clock=Clock(t=0.0), cache=Cache())


# A battery spanning every material/texture form loom emits.
_SAMPLES = [
    # scalar prop + spectrum-ref colour
    Material("gold", "diffuse", reflect="spectrum:gold", roughness="0.2"),
    # vector (space-separated) colour + scalar ior, glass type
    Material("m", "glass", ior="1.5", reflect="0.8 0.7 0.2"),
    # material bound to a texture by ref
    Material("t", "diffuse", reflect="texture:hide"),
    # metal type, no extra props beyond type
    Material("bare", "metal"),
    # image texture, all defaults
    Texture("hide", "wood.png"),
    # image texture, non-default encoding/filter/wrap
    Texture("map", "textures/cow.png", encoding="linear",
            filter="nearest", wrap="clamp"),
    # procedural rgb-function texture
    ProcTexture("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)",
                res=256, wrap="clamp"),
    # procedural texture, all defaults
    ProcTexture("plain", "u", "v", "0.5"),
]


@pytest.mark.parametrize("obj", _SAMPLES)
def test_emit_is_a_fixed_point(obj):
    ctx = _ctx()
    once = obj.emit(ctx)
    twice = parse_element(once).emit(ctx)
    assert once == twice


@pytest.mark.parametrize("obj", _SAMPLES)
def test_rebuilt_kind_matches(obj):
    back = parse_element(obj.emit(_ctx()))
    assert type(back) is type(obj)
    assert back.name == obj.name


def test_material_fields_roundtrip():
    m = Material("gold", "glass", ior="1.5", reflect="spectrum:gold",
                 roughness="0.2")
    back = parse_element(m.emit(_ctx()))
    assert isinstance(back, Material)
    assert back.mtype == "glass"
    assert back.props["ior"] == "1.5"
    assert back.props["reflect"] == "spectrum:gold"
    assert back.props["roughness"] == "0.2"


def test_texture_fields_roundtrip():
    t = Texture("map", "textures/cow.png", encoding="linear",
                filter="nearest", wrap="clamp")
    back = parse_element(t.emit(_ctx()))
    assert isinstance(back, Texture)
    assert back.file == "textures/cow.png"
    assert back.encoding == "linear"
    assert back.filter == "nearest"
    assert back.wrap == "clamp"


def test_proctexture_fields_roundtrip():
    t = ProcTexture("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)",
                    res=256, filter="nearest", wrap="clamp")
    back = parse_element(t.emit(_ctx()))
    assert isinstance(back, ProcTexture)
    assert (back.r, back.g, back.b) == ("u", "v", "0.5+0.5*sin(2*pi*8*u)")
    assert back.res == 256
    assert back.filter == "nearest"
    assert back.wrap == "clamp"


def test_reader_rejects_non_element():
    with pytest.raises(ValueError):
        parse_element("this is not an element at all")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
