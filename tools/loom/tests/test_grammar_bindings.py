"""J3c: per-field *binding-union* validators (`loom.grammar.bindings`).

`reflect` / `roughness` / `*_map` don't take a bare spectrum — they take a union of
binding forms (a `texture:` / `pattern:` ref, a scalar number, a spectrum).  These
validators mirror ftrace's `bindReflectTexture` / `bindScalarTexture` /
`bindScalarPattern` + `spectrumParam` / `dblParam` (`src/ftsl.h`), confirming the
token *shape* (a bound name's scene membership is a later, scene-aware check).

Runnable directly (`python tests/test_grammar_bindings.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.grammar.bindings import (  # noqa: E402
    as_color_binding, as_scalar_binding, as_map_binding,
)
from loom.grammar.spectrum import ColorSpec, Const, LibRef, WhiteWall  # noqa: E402


# ---- colour bind (reflect): texture: | spectrum ---------------------------

def test_color_binding_texture():
    assert as_color_binding("texture:hide") == ("texture", "hide")


def test_color_binding_spectrum_forms():
    assert as_color_binding("rgb 0.8 0.7 0.2") == ColorSpec("rgb", (0.8, 0.7, 0.2))
    assert as_color_binding("whitewall 0.6") == WhiteWall(0.6)
    assert as_color_binding("spectrum:gold") == LibRef("spectrum", "gold")
    assert as_color_binding("0.75") == Const(0.75)


def test_color_binding_rejects_pattern():
    # reflect binds only a UV texture, not a pattern (pattern: is not a spectrum head).
    with pytest.raises(ValueError):
        as_color_binding("pattern:wiggle")


def test_color_binding_rejects_untagged_triple():
    with pytest.raises(ValueError):
        as_color_binding("0.8 0.7 0.2")


# ---- scalar bind (roughness): pattern: | texture: | number ----------------

def test_scalar_binding_number():
    assert as_scalar_binding("0.2") == ("scalar", 0.2)


def test_scalar_binding_pattern_and_texture():
    assert as_scalar_binding("pattern:wiggle") == ("pattern", "wiggle")
    assert as_scalar_binding("texture:rough") == ("texture", "rough")


def test_scalar_binding_rejects_spectrum_and_multi():
    with pytest.raises(ValueError):
        as_scalar_binding("rgb 1 0 0")
    with pytest.raises(ValueError):
        as_scalar_binding("0.2 0.3")
    with pytest.raises(ValueError):
        as_scalar_binding("blackbody 6500")


# ---- scalar map (*_map): pattern: | texture: only -------------------------

def test_map_binding_forms():
    assert as_map_binding("pattern:prof") == ("pattern", "prof")
    assert as_map_binding("texture:prof") == ("texture", "prof")


def test_map_binding_rejects_number():
    with pytest.raises(ValueError):
        as_map_binding("300")


# ---- malformed ref names --------------------------------------------------

def test_ref_name_must_be_single_identifier():
    with pytest.raises(ValueError):
        as_color_binding("texture:")            # empty name
    with pytest.raises(ValueError):
        as_scalar_binding("pattern:a b")        # multi-word
    with pytest.raises(ValueError):
        as_map_binding("texture:1bad")          # not an identifier


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
