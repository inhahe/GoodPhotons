"""``Color.parse`` / ``parse_color`` / ``parse_color_list`` — reading ``.ftsl``
colour tokens back into animatable :class:`loom.Color` objects.

These wire the locked color-vector / array grammar (``loom.grammar.values``) into a
real loom API: the inverse of :meth:`Color.token`.  A colour parsed from its own
emitted token round-trips exactly; a bare / bracketed triple reads in the default
colorspace; and the shape rules are enforced (a 6-vector or a colour *list* handed
to the single-colour reader is a :class:`ShapeError`, an unbalanced bracket is a
plain ``ValueError``).

Runnable directly (``python tests/test_color_parse.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Clock, Cache, Color, parse_color, parse_color_list, rgb, hsv, hsl  # noqa: E402
from loom.grammar.values import ShapeError  # noqa: E402


def _clk() -> Clock:
    return Clock(t=0.0, frame=0, frames=1, fps=30.0)


def _tok(c: Color) -> str:
    return c.token(_clk(), Cache())


# ---- one colour: parse reads the authored colorspace + components ----------

def test_parse_bare_triple_is_rgb_by_default():
    c = parse_color("0.8 0.7 0.2")
    assert c.mode == "rgb"
    assert _tok(c) == "rgb 0.8 0.7 0.2"


def test_parse_default_space_override():
    c = parse_color("0.5 1 0.5", default_space="hsl")
    assert c.mode == "hsl"
    assert _tok(c) == "hsl 0.5 1 0.5"


def test_parse_explicit_tag_beats_default():
    assert parse_color("hsv 0.6 0.8 0.9", default_space="rgb").mode == "hsv"
    assert parse_color("hsl 0.6 0.7 0.5").mode == "hsl"


def test_parse_bracket_identity():
    # `[X] ≡ X` — a lone bracket is transparent.
    assert _tok(parse_color("[1, 0, 0]")) == "rgb 1 0 0"
    assert _tok(parse_color("rgb [0.2 0.4 0.6]")) == "rgb 0.2 0.4 0.6"


# ---- the round-trip: token() -> parse -> token() is identity ---------------

@pytest.mark.parametrize("color", [
    rgb(0.9, 0.4, 0.1),
    hsv(0.6, 0.8, 0.9),
    hsl(0.6, 0.7, 0.5),
])
def test_token_parse_round_trip(color):
    assert _tok(Color.parse(_tok(color))) == _tok(color)


# ---- shape errors: single-colour reader rejects non-colours ----------------

def test_parse_color_rejects_six_vector():
    with pytest.raises(ShapeError):
        parse_color("2 0 0 3 0 0")


def test_parse_color_rejects_a_list():
    with pytest.raises(ShapeError):
        parse_color("2 0 0, 3 0 0")


def test_parse_color_rejects_ref_channels():
    # a colour needs numeric channels, not a spectrum reference atom
    with pytest.raises(ShapeError):
        parse_color("spectrum:gold")


def test_unbalanced_bracket_is_valueerror():
    with pytest.raises(ValueError):
        parse_color("[1 0 0")


# ---- flat palette: parse_color_list accepts every locked spelling ----------

def _tokens(cs):
    return [_tok(c) for c in cs]


@pytest.mark.parametrize("text", [
    "2 0 0, 3 0 0",
    "[2 0 0] [3 0 0]",
    "[2 0 0], [3 0 0]",
    "[2, 0, 0], [3, 0, 0]",
])
def test_color_list_equivalent_spellings(text):
    assert _tokens(parse_color_list(text)) == ["rgb 2 0 0", "rgb 3 0 0"]


def test_color_list_lone_colour_is_one_element():
    # [X] ≡ X — a single colour reads as a one-element palette.
    assert _tokens(parse_color_list("1 0 0")) == ["rgb 1 0 0"]


def test_color_list_mixes_colorspaces_via_rle_tags():
    cs = parse_color_list("rgb 1 0 0, 2 0 0, hsl 4 0 0, 5 0 0")
    assert _tokens(cs) == ["rgb 1 0 0", "rgb 2 0 0", "hsl 4 0 0", "hsl 5 0 0"]


def test_color_list_rejects_list_of_lists():
    with pytest.raises(ShapeError):
        parse_color_list("rgb [1 0 0, 2 0 0], hsl [3 0 0, 4 0 0]")


def test_parsed_colors_are_real_animatable_colors():
    # a parsed Color IS a resolved-RGB VecSignal, usable everywhere loom takes one
    c = parse_color("hsl 0 0 1")            # white
    r, g, b = c.at(_clk(), Cache())
    assert (round(r, 6), round(g, 6), round(b, 6)) == (1.0, 1.0, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
