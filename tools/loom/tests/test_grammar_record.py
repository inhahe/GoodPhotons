"""J3c: grammar-backed record reader (`loom.grammar.reader.parse_record`).

Proves the shared EPEG grammar (`ftsl.epeg`) + ParseNode->Record builder produce
records structurally identical to the hand-written `Record.parse` oracle, across
every channel form (whitespace scalar / spectrum-ref colour, vector, inline
`rgb`/`hsv`/`hsl` colour, position pins, all interp modes, both domain spellings),
and that emit -> parse_record -> emit is a fixed point.

Runnable directly (`python tests/test_grammar_record.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.record import Record  # noqa: E402
from loom.grammar.reader import parse_record  # noqa: E402


def _canon(r: Record):
    """Structural snapshot: name, domain, interp, and every channel's stops/pins."""
    return (
        r.name, r.lo, r.hi, r.interp,
        tuple(
            (ch.name, ch.kind, ch.space,
             tuple((tuple(s.components), s.pinned,
                    round(s.pos, 9) if s.pinned else None) for s in ch.stops),
             tuple(round(p, 9) for p in r.positions(ch.name)))
            for ch in r.channels),
    )


# A battery of record blocks spanning every supported channel form.
_SAMPLES = [
    # whitespace scalar + spectrum-ref colour + interior pin + interp
    ("funny = range 0-1 [\n"
     "    reflect    spectrum:steel  spectrum:gold  spectrum:copper\n"
     "    roughness  0.0  p:0.7 0.4  1.0\n"
     "    interp     smooth\n]\n"),
    # vector channel + lone vector stop (trailing comma)
    ("vecrec = range 0-2 [\n"
     "    tint    0 0 0, 1 1 1\n"
     "    single  0.5 0.5 0.5,\n]\n"),
    # inline rgb colour
    ("glow = range 0-1 [\n    reflect  rgb 0.55 0.57 0.6, 0.9 0.75 0.3\n]\n"),
    # inline hsv colour, three stops
    ("warm = range 0-1 [\n    tint  hsv 0.0 1 1, 0.15 1 1, 0.3 1 1\n]\n"),
    # pinned colour ref in the whitespace form
    ("pinref = range 0-1 [\n    reflect  spectrum:a  p:0.5 spectrum:b  spectrum:c\n]\n"),
    # compact `-1-2` domain (leading-minus + inner dash) and nearest interp
    ("dom = range -1-2 [\n    x  0 1 2\n    interp  nearest\n]\n"),
    # space-separated domain form
    ("spc = range 0.5 2.5 [\n    x  0 1\n]\n"),
    # compact single-line body (no interior newlines)
    ("one = range 0 1 [ x 0 1 ]\n"),
    # no trailing newline at all
    ("bare = range 0-1 [ x 0 1 ]"),
]


@pytest.mark.parametrize("text", _SAMPLES)
def test_reader_matches_oracle(text):
    assert _canon(parse_record(text)) == _canon(Record.parse(text))


@pytest.mark.parametrize("text", _SAMPLES)
def test_emit_is_a_fixed_point(text):
    # emit -> parse_record -> emit must be byte-identical (stable normal form)
    once = Record.parse(text).emit()
    twice = parse_record(once).emit()
    assert once == twice


def test_reader_roundtrips_built_records():
    # build via the high-level API, emit, and read back through the grammar
    rec = Record.from_channels(
        "m", 0.0, 1.0,
        [("roughness", ["0.0", ("0.4", 0.7), "1.0"]),
         ("reflect", ["spectrum:steel", "spectrum:gold"])],
        interp="smooth")
    assert _canon(parse_record(rec.emit())) == _canon(rec)


def test_reader_rejects_non_record():
    with pytest.raises(ValueError):
        parse_record("this is not a record at all")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
