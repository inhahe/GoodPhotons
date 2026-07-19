"""The locked color-vector / array syntax (`loom.grammar.values`).

Pins the syntax decision recorded in TODO.md ("DECISION — color-vector / array
syntax", locked 2026-07-19): how whitespace, commas, brackets and the `rgb` /
`hsl` / `hsv` colorspace tags group scalars into vectors and vectors into
(possibly nested) lists.  Every example the decision spells out is asserted
here, plus the per-field shape validators.

Runnable directly (`python tests/test_grammar_values.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.grammar.values import (  # noqa: E402
    Arr, Ref, ShapeError, Vec,
    as_color, as_color_list, as_scalar, as_vector, parse_value,
)


# ---- the value tree: whitespace only joins scalars inside one vector -------

def test_single_vector():
    assert parse_value("2 0 0") == Vec([2.0, 0.0, 0.0])


def test_whitespace_never_crosses_a_boundary():
    # `2 0 0 3 0 0` is ONE 6-number vector, not two colors.
    assert parse_value("2 0 0 3 0 0") == Vec([2.0, 0.0, 0.0, 3.0, 0.0, 0.0])


def test_lone_scalar_commas_are_component_separators():
    # `1, 0, 0` == the single vector `1 0 0` (all operands lone scalars).
    assert parse_value("1, 0, 0") == Vec([1.0, 0.0, 0.0])
    assert parse_value("1, 0, 0") == parse_value("1 0 0")


def test_grouped_commas_are_array_separators():
    # `2 0 0, 3 0 0` == two vectors (operands are space-grouped groups).
    assert parse_value("2 0 0, 3 0 0") == Arr([Vec([2.0, 0.0, 0.0]),
                                               Vec([3.0, 0.0, 0.0])])


def test_all_lone_scalars_make_one_long_vector():
    # The documented corner: `1, 0, 0, 2, 0, 0` is a single 6-vector (invalid as
    # a color), NOT two colors.
    assert parse_value("1, 0, 0, 2, 0, 0") == Vec([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])


# ---- the four equivalent spellings of the two colors 2 0 0 and 3 0 0 -------

@pytest.mark.parametrize("text", [
    "2 0 0, 3 0 0",
    "[2 0 0] [3 0 0]",
    "[2 0 0], [3 0 0]",
    "[2, 0, 0], [3, 0, 0]",
])
def test_two_color_spellings_all_equivalent(text):
    assert parse_value(text) == Arr([Vec([2.0, 0.0, 0.0]), Vec([3.0, 0.0, 0.0])])


def test_bracketed_pair_differs_from_bare_six_vector():
    # `[2 0 0] [3 0 0]` != `2 0 0 3 0 0` (brackets are load-bearing with a sibling)
    assert parse_value("[2 0 0] [3 0 0]") != parse_value("2 0 0 3 0 0")


# ---- rgb / hsl / hsv are inline RLE-style tags, not array-openers ----------

def test_flat_palette_mixes_colorspaces_depth1():
    tree = parse_value("rgb 1 0 0, 2 0 0, 3 0 0, hsl 4 0 0, 5 0 0, 6 0 0")
    assert tree == Arr([
        Vec([1.0, 0.0, 0.0], "rgb"), Vec([2.0, 0.0, 0.0], "rgb"),
        Vec([3.0, 0.0, 0.0], "rgb"), Vec([4.0, 0.0, 0.0], "hsl"),
        Vec([5.0, 0.0, 0.0], "hsl"), Vec([6.0, 0.0, 0.0], "hsl"),
    ])
    # depth-1: no item is itself a list
    assert all(isinstance(it, Vec) for it in tree.items)


# ---- [X] == X is a whole-value identity; brackets only nest ----------------

def test_lone_bracket_is_transparent():
    assert parse_value("[1 0 0]") == parse_value("1 0 0")
    assert parse_value("[1, 0, 0]") == parse_value("1 0 0")


def test_lone_bracket_transparent_under_a_tag():
    # `rgb [1 0 0]` == `rgb 1 0 0`
    assert parse_value("rgb [1 0 0]") == Vec([1.0, 0.0, 0.0], "rgb")
    assert parse_value("rgb [1 0 0]") == parse_value("rgb 1 0 0")


def test_sibling_brackets_are_load_bearing():
    # `[2 0 0] [3 0 0]` != `2 0 0 3 0 0` (already covered) — and the brackets
    # cannot be dropped once there is a sibling.
    with_sibling = parse_value("[2 0 0] [3 0 0]")
    assert isinstance(with_sibling, Arr) and len(with_sibling.items) == 2


def test_nested_brackets_are_depth2():
    tree = parse_value("rgb [1 0 0, 2 0 0, 3 0 0], hsl [1 0 0, 2 0 0, 3 0 0]")
    assert isinstance(tree, Arr) and len(tree.items) == 2
    assert all(isinstance(it, Arr) for it in tree.items)   # depth-2
    assert tree.items[0].items[0] == Vec([1.0, 0.0, 0.0], "rgb")
    assert tree.items[1].items[0] == Vec([1.0, 0.0, 0.0], "hsl")


def test_flat_form_and_nested_form_differ():
    flat = parse_value("rgb 1 0 0, 2 0 0, 3 0 0, hsl 1 0 0, 2 0 0, 3 0 0")
    nested = parse_value("rgb [1 0 0, 2 0 0, 3 0 0], hsl [1 0 0, 2 0 0, 3 0 0]")
    assert flat != nested                     # depth-1 vs depth-2


# ---- scalar atoms may be references ----------------------------------------

def test_ref_scalar():
    assert parse_value("spectrum:gold") == Vec([Ref("spectrum:gold")])


# ---- syntax vs shape: the grammar accepts, the field validates -------------

def test_as_color_accepts_a_three_vector():
    assert as_color("0.8 0.7 0.2") == ("rgb", (0.8, 0.7, 0.2))
    assert as_color("hsl 0.5 1 0.5") == ("hsl", (0.5, 1.0, 0.5))
    assert as_color("[1, 0, 0]") == ("rgb", (1.0, 0.0, 0.0))


def test_as_color_rejects_wrong_arity():
    with pytest.raises(ShapeError):
        as_color("2 0 0 3 0 0")               # a 6-vector, not a color
    with pytest.raises(ShapeError):
        as_color("2 0 0, 3 0 0")              # a list of two, not one color


def test_as_color_list_accepts_flat_list_and_lone_color():
    assert as_color_list("2 0 0, 3 0 0") == [("rgb", (2.0, 0.0, 0.0)),
                                             ("rgb", (3.0, 0.0, 0.0))]
    # a lone color is a one-element list ([X] == X)
    assert as_color_list("1 0 0") == [("rgb", (1.0, 0.0, 0.0))]


def test_as_color_list_rejects_list_of_lists():
    with pytest.raises(ShapeError) as ei:
        as_color_list("rgb [1 0 0, 2 0 0], hsl [3 0 0, 4 0 0]")
    assert "list-of-lists" in str(ei.value)


def test_as_scalar_and_as_vector():
    assert as_scalar("5") == 5.0
    assert as_scalar("[5]") == 5.0
    assert as_vector("1 2 3 4", 4) == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ShapeError):
        as_scalar("1 2")
    with pytest.raises(ShapeError):
        as_vector("1 2 3", 4)


def test_syntax_error_is_valueerror_not_shapeerror():
    with pytest.raises(ValueError):
        parse_value("[1 0 0")                 # unbalanced bracket


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
