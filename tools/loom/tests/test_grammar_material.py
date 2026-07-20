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
    # tagged (space-separated) colour + scalar ior, glass type
    Material("m", "glass", ior="1.5", reflect="rgb 0.8 0.7 0.2"),
    # keyword-headed spectrum expressions in purely-spectral fields
    Material("fl", "fluorescent", absorb="shortpass edge=490 slope=0.15 amp=1",
             emit="gaussian center=560 sigma=25 amp=1"),
    # blackbody / whitewall / wall spectra
    Material("lamp", "diffuse", emit="blackbody 3200", reflect="whitewall 0.6"),
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


# ---- keyword-headed spectrum expressions parse (grammar) + validate (shape) ---

def test_keyword_headed_spectrum_values_parse():
    m = parse_element("m = material { type diffuse  emit gaussian center=560 sigma=25 amp=1 }")
    assert isinstance(m, Material)
    assert m.props["emit"] == "gaussian center=560 sigma=25 amp=1"


def test_word_run_stops_at_next_key():
    # `emit blackbody 6500` must NOT swallow the following `ior 1.5` key.
    m = parse_element("m = material { type dielectric  emit blackbody 6500  ior 1.5 }")
    assert m.props["emit"] == "blackbody 6500"
    assert m.props["ior"] == "1.5"


def test_reader_rejects_bad_spectral_field():
    # `ior` is purely spectral, so a non-spectrum value is a shape error at parse.
    with pytest.raises(ValueError):        # ShapeError is a ValueError
        parse_element("m = material { type dielectric  ior wombat }")


def test_reader_rejects_untagged_reflect_and_ior():
    # Both `reflect` and `ior` reject a bare (untagged) colour triple — ftrace does too
    # (`unrecognized spectrum expression '0.8'`).  `reflect` accepts a `texture:` bind
    # on top of a spectrum (the binding union), but an untagged triple is neither.
    with pytest.raises(ValueError):
        parse_element("m = material { type diffuse  reflect 0.8 0.7 0.2 }")
    with pytest.raises(ValueError):
        parse_element("m = material { type dielectric  ior 0.8 0.7 0.2 }")


def test_reflect_accepts_texture_bind_and_spectrum():
    # `reflect` is colour-bindable: a `texture:<name>` bind OR a spectrum expression.
    assert parse_element("m = material { type diffuse  reflect texture:hide }") \
        .props["reflect"] == "texture:hide"
    assert parse_element("m = material { type diffuse  reflect rgb 0.8 0.7 0.2 }") \
        .props["reflect"] == "rgb 0.8 0.7 0.2"


def test_reflect_rejects_pattern_bind():
    # `reflect` binds only a UV texture (bindReflectTexture), never a `pattern:` — a
    # bare pattern ref would fall through to spectrumParam and be rejected.
    with pytest.raises(ValueError):
        parse_element("m = material { type diffuse  reflect pattern:wiggle }")


def test_roughness_accepts_scalar_and_binds():
    # `roughness` is scalar-bindable: a number, a `pattern:<name>` or a `texture:<name>`.
    assert parse_element("m = material { type glossy  roughness 0.2 }") \
        .props["roughness"] == "0.2"
    assert parse_element("m = material { type glossy  roughness pattern:wiggle }") \
        .props["roughness"] == "pattern:wiggle"
    assert parse_element("m = material { type glossy  roughness texture:rough }") \
        .props["roughness"] == "texture:rough"


def test_roughness_rejects_spectrum_and_triple():
    # A scalar field is one number, not a spectrum / colour — `roughness rgb 1 0 0`
    # and an untagged triple both reject.
    with pytest.raises(ValueError):
        parse_element("m = material { type glossy  roughness rgb 1 0 0 }")
    with pytest.raises(ValueError):
        parse_element("m = material { type glossy  roughness 0.2 0.3 0.4 }")


def test_map_field_requires_texture_or_pattern_bind():
    # A `*_map` field (film_thickness_map / weight_map) accepts only a `pattern:` or
    # `texture:` bind — no numeric fallback.
    assert parse_element(
        "m = material { type thinfilm  film_thickness_map texture:prof }") \
        .props["film_thickness_map"] == "texture:prof"
    assert parse_element(
        "m = material { type thinfilm  film_thickness_map pattern:prof }") \
        .props["film_thickness_map"] == "pattern:prof"
    with pytest.raises(ValueError):
        parse_element("m = material { type thinfilm  film_thickness_map 300 }")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
