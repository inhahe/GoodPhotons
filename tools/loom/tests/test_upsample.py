"""User-supplied RGB→spectral upsamplers (K1) — ``NamedSpectrum`` / ``Upsample``.

ftrace has five *built-in* upsamplers reached by glued colour heads (``rgb`` =
Jakob-Hanika, ``rgbillum``, ``rgbsmits``, ``rgbbox``, ``rgbmeng``).  The sixth is
open-ended: a scene declares ``upsample "name" { expr "f(r,g,b,w)" }`` and a material
slot names it with the colon-separated head ``rgb:<name> r g b``.

These tests pin the **emitted text**, because that text is the contract — ftrace's
loader is the consumer and the two halves are checked against each other by
``ftrace -checkupsample`` (section h), which loads the very same spellings.  What is
worth pinning here is what loom could plausibly get wrong on its own: the block
syntax, the emission ORDER (a `spectrum` an upsample body samples, and the upsample
itself, both ahead of the materials that name them), expression validation on
``NamedSpectrum``, and that a `rgb:<name>` head survives every place a colour head is
accepted — including a record channel's inline-colour tag, whose accepted-head set is
the one most likely to drift out of sync.

Runnable directly (`python tests/test_upsample.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Camera, Material, NamedSpectrum, Scene, Sphere, Upsample  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.grammar.values import ShapeError  # noqa: E402
from loom.record import Record, RecordChannel, RecordStop  # noqa: E402
from loom.signals.core import Clock  # noqa: E402


def _ctx():
    return EmitCtx(clock=Clock(0.0), cache=None)


def _scene():
    return Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))


# ---- block emission --------------------------------------------------------

def test_upsample_block_syntax():
    assert Upsample("basis", "r*spec:sr(w)").emit(_ctx()) == \
        'upsample "basis" { expr "r*spec:sr(w)" }'


def test_named_spectrum_block_syntax():
    assert NamedSpectrum("sr", "gaussian center=620 sigma=40").emit(_ctx()) == \
        'spectrum "sr" = gaussian center=620 sigma=40'


def test_named_spectrum_validates_its_expression():
    # A bad spectrum expression should fail HERE, in Python, not at render time —
    # that is the whole reason NamedSpectrum parses on construction.
    with pytest.raises(ShapeError):
        NamedSpectrum("bad", "wombat 1 2 3")
    # ...and an untagged bare colour is not a spectrum in ftrace either.
    with pytest.raises(ShapeError):
        NamedSpectrum("bad", "0.8 0.8 0.8")


def test_upsample_rejects_an_empty_body():
    with pytest.raises(ValueError):
        Upsample("empty", "   ")


# ---- emission order --------------------------------------------------------

def test_spectra_and_upsamplers_precede_materials():
    s = _scene()
    s.add(NamedSpectrum("sr", "gaussian center=620 sigma=40"),
          Upsample("basis", "r*spec:sr(w)"),
          Material("m", "diffuse", reflect="rgb:basis 0.3 0.6 0.1"),
          Sphere((0, 0, 0), 1, "m"))
    text = s.emit(Clock(0.0))
    i_spec = text.index('spectrum "sr"')
    i_ups = text.index('upsample "basis"')
    i_mat = text.index("rgb:basis")
    assert i_spec < i_ups < i_mat


def test_the_three_lists_are_separate():
    s = _scene()
    s.add(NamedSpectrum("sr", "0.5"), Upsample("u", "r"))
    assert [e.name for e in s.spectra] == ["sr"]
    assert [e.name for e in s.upsamplers] == ["u"]
    assert s.patterns == [] and s.textures == [] and s.materials == []


def test_a_measured_basis_scene_round_trips_to_text():
    # The motivating example from the Upsample docstring, end to end.
    s = _scene()
    s.add(NamedSpectrum("sr", "gaussian center=620 sigma=40"),
          NamedSpectrum("sg", "gaussian center=540 sigma=40"),
          NamedSpectrum("sb", "gaussian center=460 sigma=40"),
          Upsample("basis", "r*spec:sr(w) + g*spec:sg(w) + b*spec:sb(w)"),
          Material("m", "diffuse", reflect="rgb:basis 0.3 0.6 0.1"),
          Sphere((0, 0, 0), 1, "m"))
    text = s.emit(Clock(0.0))
    for want in ('spectrum "sr" = gaussian center=620 sigma=40',
                 'spectrum "sg" = gaussian center=540 sigma=40',
                 'spectrum "sb" = gaussian center=460 sigma=40',
                 'upsample "basis" { expr "r*spec:sr(w) + g*spec:sg(w) + b*spec:sb(w)" }'):
        assert want in text
    assert "rgb:basis 0.3 0.6 0.1" in text


# ---- the colour-head set -----------------------------------------------------
# ftrace keeps ONE `isColourHead` shared by `evalSpectrum` and the record-channel
# inline-colour tag, precisely so the two can't drift; loom mirrors that with one
# `is_colour_space`.  These pin the mirror.

@pytest.mark.parametrize("space", ["rgb:mine", "hsv:mine", "hsl:mine"])
def test_user_head_is_a_legal_record_channel_tag(space):
    ch = RecordChannel("reflect", [RecordStop([0.0, 0.0, 0.0], 0.0),
                                   RecordStop([1.0, 1.0, 1.0], 1.0)], space)
    assert ch.is_inline_colour and ch.space == space


def test_user_head_survives_a_record_channel_line_round_trip():
    rec = Record.parse(
        "grad = range 0 1 [\n"
        "  reflect  rgb:mine 0 0 0, 1 1 1\n"
        "]\n")
    ch = rec.channels[0]
    assert ch.space == "rgb:mine"
    assert [tuple(s.as_vector()) for s in ch.stops] == [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def test_a_bogus_channel_tag_is_still_rejected():
    with pytest.raises(ValueError):
        RecordChannel("reflect", [RecordStop([0.0, 0.0, 0.0], 0.0)], "wombat")
    # a colon head in a space that isn't one of the three is not a colour head either
    with pytest.raises(ValueError):
        RecordChannel("reflect", [RecordStop([0.0, 0.0, 0.0], 0.0)], "xyz:mine")


def test_a_user_head_channel_lowers_verbatim():
    # Lowering an inline-colour channel to `spectrum:` refs must pass a NON-plain head
    # through untouched: loom cannot re-derive a user upsampler's output, so converting
    # or re-deriving would silently change the colour.  Same rule as `rgbmeng`.
    rec = Record.parse(
        "grad = range 0 1 [\n"
        "  reflect  rgb:mine 0.2 0.5 0.9, 0.2 0.5 0.9\n"
        "]\n")
    decls, lowered = rec.lower_colours()
    assert len(decls) == 1                       # the two identical stops dedupe
    assert decls[0].startswith('spectrum "grad_c0" = rgb:mine ')
    assert "0.2 0.5 0.9" in decls[0]
    assert all(s.components[0].startswith("spectrum:")
               for s in lowered.channels[0].stops)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
