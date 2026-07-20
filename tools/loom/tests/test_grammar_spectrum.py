"""The ``.ftsl`` spectral-value grammar (`loom.grammar.spectrum`).

Pins the accepted spectrum-expression forms against ftrace's ``evalSpectrum``
(``src/ftsl.h`` ~1106): a constant, ``blackbody``/``ior``/``whitewall`` (with
defaults), the bareword walls, the ``gaussian``/``shortpass`` bands, tagged
``rgb``/``hsv``/``hsl`` colours, the library ``prefix:name`` references, and a
record channel reference — plus the rejection of an untagged bare colour and of a
truly unrecognized head.

Runnable directly (`python tests/test_grammar_spectrum.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.grammar.spectrum import (  # noqa: E402
    Band, Blackbody, ColorSpec, Const, Ior, LibRef, LineSpec, NamedWall,
    RecordRef, WhiteWall, as_spectrum, parse_spectrum,
)
from loom.grammar.values import ShapeError  # noqa: E402


# ---- constant + defaults ---------------------------------------------------

def test_bare_number_is_constant():
    assert parse_spectrum("0.75") == Const(0.75)
    assert parse_spectrum("-1") == Const(-1.0)


def test_blackbody_default_and_explicit():
    assert parse_spectrum("blackbody") == Blackbody(6500.0)
    assert parse_spectrum("blackbody 3200") == Blackbody(3200.0)


def test_ior_default_and_explicit():
    assert parse_spectrum("ior") == Ior(1.5)
    assert parse_spectrum("ior 1.6") == Ior(1.6)


def test_whitewall_default_and_explicit():
    assert parse_spectrum("whitewall") == WhiteWall(0.75)
    assert parse_spectrum("whitewall 0.5") == WhiteWall(0.5)


def test_bareword_walls():
    assert parse_spectrum("redwall") == NamedWall("redwall")
    assert parse_spectrum("greenwall") == NamedWall("greenwall")


# ---- parametric bands (key=value words, any order, defaults) ---------------

def test_gaussian_full():
    assert parse_spectrum("gaussian center=560 sigma=25 amp=1") == \
        Band("gaussian", 560.0, 25.0, 1.0)


def test_gaussian_defaults_and_order_independent():
    assert parse_spectrum("gaussian") == Band("gaussian", 0.0, 0.0, 1.0)
    assert parse_spectrum("gaussian amp=0.5 center=490") == \
        Band("gaussian", 490.0, 0.0, 0.5)


def test_shortpass_uses_edge_slope():
    assert parse_spectrum("shortpass edge=490 slope=0.15 amp=1") == \
        Band("shortpass", 490.0, 0.15, 1.0)


# ---- colours delegate to the colour-vector validator -----------------------

def test_rgb_hsv_hsl_colours():
    assert parse_spectrum("rgb 0.8 0.7 0.2") == ColorSpec("rgb", (0.8, 0.7, 0.2))
    assert parse_spectrum("hsv 0.6 0.8 0.9") == ColorSpec("hsv", (0.6, 0.8, 0.9))
    assert parse_spectrum("hsl 0.6 0.7 0.5") == ColorSpec("hsl", (0.6, 0.7, 0.5))


def test_bracketed_colour_still_parses():
    # colour delegates to as_color, so the colour-vector bracket syntax works too
    assert parse_spectrum("rgb [0.2, 0.4, 0.6]") == ColorSpec("rgb", (0.2, 0.4, 0.6))


def test_colour_wrong_arity_is_shape_error():
    with pytest.raises(ShapeError):
        parse_spectrum("rgb 0.8 0.8")          # a colour needs 3 components


# ---- dominant-wavelength emission heads (K3) -------------------------------

def test_line_heads_parse():
    # `rgbline r g b` (and hsvline/hslline) → LineSpec, sigma optional
    assert parse_spectrum("rgbline 1 0 0") == LineSpec("rgb", (1.0, 0.0, 0.0), None)
    assert parse_spectrum("hsvline 0.6 0.8 0.9") == LineSpec("hsv", (0.6, 0.8, 0.9), None)
    assert parse_spectrum("hslline 0.6 0.7 0.5") == LineSpec("hsl", (0.6, 0.7, 0.5), None)


def test_line_head_explicit_sigma():
    assert parse_spectrum("rgbline 0 0 1 6") == LineSpec("rgb", (0.0, 0.0, 1.0), 6.0)


def test_line_head_wrong_arity_is_shape_error():
    with pytest.raises(ShapeError):
        parse_spectrum("rgbline 0 1")          # needs 3 components


# ---- untagged bare colour is NOT a spectrum (matches ftrace) ---------------

def test_untagged_triple_is_rejected():
    # ftrace: `unrecognized spectrum expression '0.8'` for a bare `0.8 0.8 0.8`
    with pytest.raises(ShapeError):
        parse_spectrum("0.8 0.8 0.8")


# ---- library references ----------------------------------------------------

@pytest.mark.parametrize("text,kind,name", [
    ("metal:gold", "metal", "gold"),
    ("glass:BK7", "glass", "BK7"),
    ("reflectance:skin", "reflectance", "skin"),
    ("filter:cyan", "filter", "cyan"),
    ("preset:d65", "preset", "d65"),
    ("spectrum:steel", "spectrum", "steel"),
    ("file:curves/led.csv", "file", "curves/led.csv"),
])
def test_library_references(text, kind, name):
    assert parse_spectrum(text) == LibRef(kind, name)


# ---- record channel reference (constant value site) ------------------------

def test_record_channel_reference_forms():
    assert parse_spectrum("grad.reflect[0]") == RecordRef("grad.reflect[0]")
    assert parse_spectrum("grad.reflect(0.5)") == RecordRef("grad.reflect(0.5)")
    assert parse_spectrum("grad.reflect") == RecordRef("grad.reflect")


# ---- unrecognized head fails like ftrace -----------------------------------

def test_unknown_head_is_shape_error():
    with pytest.raises(ShapeError):
        parse_spectrum("wombat 1 2 3")
    with pytest.raises(ShapeError):
        parse_spectrum("nonsense")


def test_as_spectrum_is_parse_spectrum():
    assert as_spectrum("metal:copper") == parse_spectrum("metal:copper")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
