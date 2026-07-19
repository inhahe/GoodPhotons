"""Tests for the delimiter-precedence-ladder parser (loom J3b item 2).

Covers the ladder semantics locked in ROADMAP_records.md §3.1: whitespace binds
like ``×``, comma like ``+``, brackets are parens; structure is recoverable from
delimiters alone; bracketing one level is idempotent; parens are opaque atoms.

Runnable directly (``python tests/test_ladder.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.ladder import parse_ladder, emit_ladder, shape  # noqa: E402


# ---------------------------------------------------------------------------
# leaves and flat vectors

def test_scalar_leaf():
    assert parse_ladder(".5") == ".5"
    assert parse_ladder("spectrum:steel") == "spectrum:steel"


def test_flat_vector_is_space_product():
    assert parse_ladder("1 1 1") == ["1", "1", "1"]
    assert parse_ladder("0.55 0.57 0.6") == ["0.55", "0.57", "0.6"]


# ---------------------------------------------------------------------------
# the locked equivalences (§3.1)

def test_bracketing_one_level_is_idempotent():
    assert parse_ladder("[1 1 1]") == parse_ladder("1 1 1") == ["1", "1", "1"]


def test_sum_of_products_equals_product_of_groups():
    a = parse_ladder("1 1 1, 2 2 2, 3 3 3")
    b = parse_ladder("[1 1 1] [2 2 2] [3 3 3]")
    assert a == b == [["1", "1", "1"], ["2", "2", "2"], ["3", "3", "3"]]


def test_comma_binds_looser_than_space():
    # 1, 2 3  ==  (1) + (2 3)  ->  a scalar sibling next to a 2-vector
    assert parse_ladder("1, 2 3") == ["1", ["2", "3"]]


def test_precedence_matches_arithmetic_analogy():
    # 1 1, 2 2  ==  (1·1) + (2·2)
    assert parse_ladder("1 1, 2 2") == [["1", "1"], ["2", "2"]]


# ---------------------------------------------------------------------------
# nesting depth

def test_explicit_deep_nesting():
    v = parse_ladder("[1 1, 2 2], 3 3")
    assert v == [[["1", "1"], ["2", "2"]], ["3", "3"]]


def test_empty_brackets():
    assert parse_ladder("[]") == []


def test_nested_brackets_group():
    assert parse_ladder("[[1 2] [3 4]] [5 6]") == \
        [[["1", "2"], ["3", "4"]], ["5", "6"]]


# ---------------------------------------------------------------------------
# parens are opaque (not a ladder delimiter)

def test_parens_are_opaque_atoms():
    assert parse_ladder("sin(v)") == "sin(v)"
    # comma inside parens does NOT split
    assert parse_ladder("clamp(x,0,1)") == "clamp(x,0,1)"
    # a vector of expression atoms
    assert parse_ladder("sin(u) cos(v)") == ["sin(u)", "cos(v)"]


def test_parens_with_internal_space_and_brackets():
    assert parse_ladder("mix(a + b, [1,2])") == "mix(a + b, [1,2])"


# ---------------------------------------------------------------------------
# errors

def test_rejects_empty_and_stray_delims():
    for bad in ("", "   ", ",", "]", "1 1 ]", "[1 1"):
        with pytest.raises(ValueError):
            parse_ladder(bad)


# ---------------------------------------------------------------------------
# shape

def test_shape():
    assert shape(parse_ladder(".5")) == ()
    assert shape(parse_ladder("1 1 1")) == (3,)
    assert shape(parse_ladder("1 1, 2 2, 3 3")) == (3, 2)
    with pytest.raises(ValueError):
        shape(parse_ladder("1 1, 2"))          # ragged: a 2-vec beside a scalar


# ---------------------------------------------------------------------------
# emit round-trips

@pytest.mark.parametrize("text", [
    ".5",
    "1 1 1",
    "1 1 1, 2 2 2, 3 3 3",
    "1, 2 3",
    "1 1, 2 2",
    "[1 1, 2 2], 3 3",
    "sin(v)",
    "clamp(x,0,1) 0.5",
])
def test_emit_reparses_to_same_tree(text):
    v = parse_ladder(text)
    assert parse_ladder(emit_ladder(v)) == v


def test_emit_canonical_forms():
    assert emit_ladder(parse_ladder("[1 1 1]")) == "1 1 1"
    assert emit_ladder(parse_ladder("[1 1] [2 2]")) == "1 1, 2 2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
