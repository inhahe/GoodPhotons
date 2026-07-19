"""J3a tests: loom twin of the FTSL parametric record — model, emit, parse,
round-trip against the real ``scenes/_record_*.ftsl`` fixtures, and the numeric
scalar sampler (mirrors ftrace ``recSampleScalar``).

Runnable directly (``python tests/test_record.py``) or under pytest.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Record, RecordChannel, RecordStop  # noqa: E402


# repo root = .../forward raytracer  (tests/ -> loom/ -> tools/ -> repo root)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_SCENES = os.path.join(_REPO, "scenes")


def _fixture_scenes():
    return sorted(glob.glob(os.path.join(_SCENES, "_record_*.ftsl")))


def _canon(rec: Record):
    """Comparable structural snapshot: (name, lo, hi, interp, channels)."""
    return (rec.name, rec.lo, rec.hi, rec.interp,
            tuple((ch.name, tuple(s.token for s in ch.stops),
                   tuple(round(p, 9) for p in rec.positions(ch.name)))
                  for ch in rec.channels))


# ---------------------------------------------------------------------------
# construction + emit

def test_from_channels_and_emit():
    rec = Record.from_channels("frost", 0, 1, [("rough", ["0.0", "0.6"])])
    text = rec.emit()
    assert text.startswith("frost = range 0-1 [")
    assert "rough" in text
    assert text.rstrip().endswith("]")


def test_emit_pinned_stop_and_interp():
    rec = Record.from_channels(
        "m", 0, 1, [("roughness", ["0.0", ("0.4", 0.7), "1.0"])], interp="smooth")
    text = rec.emit()
    assert "p:0.7  0.4" in text
    assert "interp" in text and "smooth" in text


def test_emit_omits_default_linear_interp():
    rec = Record.from_channels("m", 0, 1, [("rough", ["0.0", "1.0"])])
    assert "interp" not in rec.emit()


# ---------------------------------------------------------------------------
# parse

def test_parse_basic():
    rec = Record.parse("frost = range 0-1 [\n    rough  0.0  0.6\n]")
    assert rec.name == "frost"
    assert (rec.lo, rec.hi) == (0.0, 1.0)
    assert rec.interp == "linear"
    assert [c.name for c in rec.channels] == ["rough"]
    assert [s.token for s in rec.channel("rough").stops] == ["0.0", "0.6"]


def test_parse_pins_and_interp_and_comments():
    txt = (
        "funny = range 0-1 [\n"
        "    reflect    spectrum:steel  spectrum:gold  spectrum:copper\n"
        "    roughness  0.0  p:0.7 0.4  1.0   # a pinned interior stop\n"
        "    interp     smooth\n"
        "]\n")
    rec = Record.parse(txt)
    assert rec.interp == "smooth"
    rough = rec.channel("roughness")
    assert [s.token for s in rough.stops] == ["0.0", "0.4", "1.0"]
    assert rough.stops[1].pinned and abs(rough.stops[1].pos - 0.7) < 1e-12
    assert rec.channel("reflect").kind == "colour"


def test_parse_domain_forms():
    assert Record.parse("a = range 0 1 [ x 0 1 ]").hi == 1.0
    assert Record.parse("a = range -1-2 [ x 0 1 ]").lo == -1.0
    r = Record.parse("a = range 0.5 2.5 [ x 0 1 ]")
    assert (r.lo, r.hi) == (0.5, 2.5)


def test_parse_rejects_bad_range():
    with pytest.raises(ValueError):
        Record.parse("a = range 1-0 [ x 0 1 ]")   # HI must exceed LO
    with pytest.raises(ValueError):
        Record.parse("not a record at all")


# ---------------------------------------------------------------------------
# redistribution

def test_redistribution_even_spread():
    rec = Record.from_channels("a", 0, 1, [("v", ["0", "1", "2", "3", "4"])])
    assert rec.positions("v") == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_redistribution_pinned_interior():
    rec = Record.from_channels("a", 0, 1, [("v", ["0", ("9", 0.7), "1"])])
    assert rec.positions("v") == pytest.approx([0.0, 0.7, 1.0])


def test_redistribution_run_between_pins():
    # two pins at 0.2 and 0.8 with two unpinned stops between -> spread evenly
    rec = Record.from_channels(
        "a", 0, 1, [("v", [("0", 0.2), "1", "2", ("3", 0.8)])])
    assert rec.positions("v") == pytest.approx([0.2, 0.4, 0.6, 0.8])


# ---------------------------------------------------------------------------
# numeric sampler (mirrors ftrace recSampleScalar)

def test_sample_linear():
    rec = Record.from_channels("frost", 0, 1, [("rough", ["0.0", "0.6"])])
    assert rec.sample("rough", 0.0) == pytest.approx(0.0)
    assert rec.sample("rough", 0.5) == pytest.approx(0.3)
    assert rec.sample("rough", 1.0) == pytest.approx(0.6)
    # clamps outside the domain
    assert rec.sample("rough", -5.0) == pytest.approx(0.0)
    assert rec.sample("rough", 9.0) == pytest.approx(0.6)


def test_sample_exact_at_nodes_all_interps():
    for mode in ("nearest", "linear", "smooth"):
        rec = Record.from_channels(
            "a", 0, 1, [("v", ["0.0", ("0.4", 0.7), "1.0"])], interp=mode)
        assert rec.sample("v", 0.0) == pytest.approx(0.0)
        assert rec.sample("v", 0.7) == pytest.approx(0.4)
        assert rec.sample("v", 1.0) == pytest.approx(1.0)


def test_sample_nearest_picks_neighbour():
    rec = Record.from_channels("a", 0, 1, [("v", ["10", "20"])], interp="nearest")
    assert rec.sample("v", 0.4) == pytest.approx(10.0)
    assert rec.sample("v", 0.6) == pytest.approx(20.0)


def test_sample_smooth_monotone_no_overshoot():
    # a monotone ramp must stay within neighbour bounds under Fritsch-Carlson
    rec = Record.from_channels("a", 0, 1, [("v", ["0", "1", "2", "3", "4"])],
                               interp="smooth")
    for i in range(0, 101):
        d = i / 100.0
        s = rec.sample("v", d)
        assert -1e-9 <= s <= 4.0 + 1e-9


def test_sample_rejects_colour_and_expression():
    rec = Record.from_channels(
        "a", 0, 1, [("reflect", ["spectrum:steel", "spectrum:gold"]),
                    ("rough", ["sin(v)", "0.5"])])
    with pytest.raises(TypeError):
        rec.sample("reflect", 0.5)     # colour channel
    with pytest.raises(TypeError):
        rec.sample("rough", 0.5)       # expression stop


# ---------------------------------------------------------------------------
# channel kind / homogeneity

def test_channel_kind_and_mixed_rejected():
    assert RecordChannel("c", [RecordStop("spectrum:steel")]).kind == "colour"
    assert RecordChannel("s", [RecordStop("0.5")]).kind == "scalar"
    with pytest.raises(ValueError):
        RecordChannel("bad", [RecordStop("spectrum:steel"),
                              RecordStop("0.5")]).kind


# ---------------------------------------------------------------------------
# round-trip: emit -> parse -> emit is stable, and structure survives

def test_roundtrip_emit_parse_stable():
    rec = Record.from_channels(
        "funny_mirror", 0, 1,
        [("reflect", ["spectrum:steel", "spectrum:gold", "spectrum:copper"]),
         ("roughness", ["0.0", ("0.4", 0.7), "1.0"])],
        interp="smooth")
    once = rec.emit()
    twice = Record.parse(once).emit()
    assert once == twice
    assert _canon(rec) == _canon(Record.parse(once))


@pytest.mark.parametrize("path", _fixture_scenes())
def test_roundtrip_fixture_scenes(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    recs = Record.parse_all(text)
    assert recs, f"no record found in {os.path.basename(path)}"
    for rec in recs:
        # emit -> parse must reproduce the structure exactly
        reparsed = Record.parse(rec.emit())
        assert _canon(rec) == _canon(reparsed), os.path.basename(path)


# ---------------------------------------------------------------------------
# J3b — arbitrary-arity (vector) channels

def _canon_gen(rec: Record):
    """Structural snapshot including per-stop components (for vector channels)."""
    return (rec.name, rec.lo, rec.hi, rec.interp,
            tuple((ch.name, ch.arity,
                   tuple((tuple(s.components), s.pos) for s in ch.stops))
                  for ch in rec.channels))


def test_stop_arity_and_accessors():
    s = RecordStop(["0", "1", "2"])
    assert s.arity == 3
    assert s.components == ["0", "1", "2"]
    assert s.as_vector() == [0.0, 1.0, 2.0]
    with pytest.raises(TypeError):
        s.token          # a vector stop has no single token
    scalar = RecordStop("0.5")
    assert scalar.arity == 1 and scalar.token == "0.5"


def test_vector_channel_kind_and_ragged_rejected():
    ch = RecordChannel("tint", [RecordStop(["0", "0", "0"]), RecordStop(["1", "1", "1"])])
    assert ch.kind == "vector" and ch.arity == 3
    with pytest.raises(ValueError):
        RecordChannel("bad", [RecordStop(["0", "0"]), RecordStop(["1", "1", "1"])]).kind


def test_from_channels_vector_and_pins():
    rec = Record.from_channels(
        "grad", 0, 1,
        [("tint", [["0", "0", "0"], (["1", "0", "0"], 0.7), ["1", "1", "1"]])])
    ch = rec.channel("tint")
    assert ch.arity == 3
    assert [s.components for s in ch.stops] == \
        [["0", "0", "0"], ["1", "0", "0"], ["1", "1", "1"]]
    assert rec.positions("tint") == pytest.approx([0.0, 0.7, 1.0])


def test_sample_vec_per_component():
    rec = Record.from_channels("grad", 0, 1, [("tint", [["0", "0", "0"], ["1", "2", "4"]])])
    assert rec.sample_vec("tint", 0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert rec.sample_vec("tint", 0.5) == pytest.approx([0.5, 1.0, 2.0])
    assert rec.sample_vec("tint", 1.0) == pytest.approx([1.0, 2.0, 4.0])
    # scalar channel still works through sample() and sample_vec()
    r2 = Record.from_channels("a", 0, 1, [("v", ["0.0", "0.6"])])
    assert r2.sample("v", 0.5) == pytest.approx(0.3)
    assert r2.sample_vec("v", 0.5) == pytest.approx([0.3])


def test_sample_rejects_vector_scalar_api():
    rec = Record.from_channels("grad", 0, 1, [("tint", [["0", "0"], ["1", "1"]])])
    with pytest.raises(TypeError):
        rec.sample("tint", 0.5)          # scalar API on a vector channel


def test_emit_rejects_vector_but_generalized_roundtrips():
    rec = Record.from_channels(
        "grad", 0, 1,
        [("tint", [["0", "0", "0"], (["1", "0", "0"], 0.7), ["1", "1", "1"]]),
         ("rough", ["0.0", "0.5", "1.0"])],
        interp="smooth")
    with pytest.raises(TypeError):
        rec.emit()                       # vector channel: not current-FTSL representable
    text = rec.emit_generalized()
    assert "0 0 0, p:0.7 1 0 0, 1 1 1" in text
    reparsed = Record.parse_generalized(text)
    assert _canon_gen(rec) == _canon_gen(reparsed)


def test_generalized_roundtrip_scalar_and_colour():
    # generalized emit/parse must also round-trip plain scalar + colour-ref channels
    rec = Record.from_channels(
        "m", 0, 1,
        [("reflect", ["spectrum:steel", "spectrum:gold"]),
         ("rough", ["0.0", ("0.4", 0.7), "1.0"])],
        interp="linear")
    reparsed = Record.parse_generalized(rec.emit_generalized())
    assert _canon_gen(rec) == _canon_gen(reparsed)
    assert reparsed.channel("reflect").kind == "colour"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
