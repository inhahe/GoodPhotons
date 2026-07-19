"""Generic ``.ftsl`` value tree — the locked color-vector / array syntax.

Parses a bare value (a color, a vector, a list of colors, a palette-of-palettes,
…) with the shared EPEG grammar's ``value`` rule and normalizes it to a small
canonical tree, then lets each *field* shape-check that tree.  This is the
reference implementation of the syntax decision recorded in ``TODO.md``
("DECISION — color-vector / array syntax", locked 2026-07-19):

* **Whitespace only joins scalars inside one vector.**  ``vnums`` greedily eats a
  whitespace run, so ``2 0 0 3 0 0`` is a *single* 6-number vector, never two
  colors.
* **Array boundaries are marked** by a comma, by brackets, or by a colorspace
  keyword (``rgb`` / ``hsl`` / ``hsv``).
* **The comma's role is decided by its operands.**  Between lone scalars it is a
  component separator (``1, 0, 0`` == the one vector ``1 0 0``); between
  space-grouped groups it is an array separator (``2 0 0, 3 0 0`` == two vectors).
  Falls out: ``1, 0, 0, 2, 0, 0`` (all lone) is one 6-vector, *not* two colors.
* **``rgb`` / ``hsl`` / ``hsv`` are inline modal RLE-style tags**, not
  array-openers: they set the colorspace for the run of vectors that follows
  until the next tag, so a flat palette can mix source colorspaces with no
  brackets.
* **``[X] ≡ X`` is a whole-value identity** — a lone bracket is transparent; the
  instant it has a sibling the brackets are load-bearing (they *are* the separate
  arrays).  Only brackets nest.

Syntax stays split from shape: the grammar is context-free and this module builds
*any* nested value-tree; :func:`as_color`, :func:`as_color_list`, … then validate
the tree against what a particular field wants, with shape (not parse) errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Union

from . import load_grammar
from .reader import _GRAMMAR_PATH, _kid, _kids

# Colorspace tags a vector may carry (None = untagged; a field supplies a default).
COLOR_SPACES = ("rgb", "hsl", "hsv")


# ---- canonical value-tree nodes -------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A scalar reference atom, e.g. ``spectrum:gold`` (kept verbatim)."""
    name: str


Scalar = Union[float, Ref]


@dataclass
class Vec:
    """A whitespace vector — an ordered run of scalar components (a color
    candidate).  ``space`` is its colorspace tag (``rgb`` / ``hsl`` / ``hsv``) or
    ``None`` when untagged (the field then supplies its default)."""
    comps: List[Scalar]
    space: Optional[str] = None


@dataclass
class Arr:
    """An explicit array — a list of child nodes (:class:`Vec` or nested
    :class:`Arr`).  Depth grows only through brackets (and comma/keyword lists at
    the top level)."""
    items: List["Node"]


Node = Union[Vec, Arr]


class ShapeError(ValueError):
    """A value is well-formed syntax but the wrong *shape* for the field.

    Deliberately distinct from a parse error: ``[[c,c,c],[c,c,c]]`` handed to a
    field that wants a flat color list is a shape error, not a syntax error.
    """


# ---- grammar parse --------------------------------------------------------

@lru_cache(maxsize=1)
def _value_parser():
    """The shared grammar parsed with ``value`` as the start rule.

    A separate parser instance from :func:`loom.grammar.reader._parser` so the
    start-rule override never leaks into element parsing.
    """
    with open(_GRAMMAR_PATH, "r", encoding="utf-8") as fh:
        sc = load_grammar(fh.read())
    sc.parser.start_rule = "value"
    return sc


def _scalar(node) -> Scalar:
    """A NUMBER / REF leaf -> ``float`` or :class:`Ref`."""
    return Ref(node.value) if node.name == "REF" else float(node.value)


def _vnums(vnums_node) -> List[Scalar]:
    """A ``vnums`` node -> its ordered scalar components."""
    return [_scalar(c) for c in vnums_node.children if c.name in ("NUMBER", "REF")]


def _piece(p):
    """One ``vpiece`` -> ``('tag', space)`` / ``('nums', [scalar…])`` /
    ``('bracket', inner_value_node)``."""
    ct = _kid(p, "colour_tag")
    if ct is not None:
        space = ct.value if ct.value is not None else ct.children[0].value
        return ("tag", space)
    vb = _kid(p, "vbracket")
    if vb is not None:
        return ("bracket", _kid(vb, "value"))
    return ("nums", _vnums(_kid(p, "vnums")))


def _normalize(node, inherited_space: Optional[str] = None) -> Node:
    """A ``value`` ParseNode -> canonical :class:`Vec` / :class:`Arr` tree.

    ``inherited_space`` is the colorspace in force from an enclosing tag (a tag
    before a bracket colors the bracket's contents until an inner tag overrides).
    """
    runs = [[_piece(p) for p in _kids(vr, "vpiece")] for vr in _kids(node, "vrun")]
    has_struct = any(k in ("tag", "bracket") for run in runs for (k, _) in run)

    if not has_struct:
        # Every run is exactly one bare `nums` piece.  The comma-role rule:
        # a single run is one vector; several runs that are ALL lone scalars
        # merge into one vector; otherwise the runs are a list of vectors.
        run_scalars = [run[0][1] for run in runs]
        if len(run_scalars) == 1:
            return Vec(run_scalars[0], inherited_space)
        if all(len(s) == 1 for s in run_scalars):
            return Vec([s[0] for s in run_scalars], inherited_space)
        return Arr([Vec(s, inherited_space) for s in run_scalars])

    # Structure present (tags and/or brackets): walk pieces left-to-right,
    # applying RLE colorspace tags and recursing into brackets.  A lone item
    # (single bracket, or `tag vec`) collapses — that is the `[X] ≡ X` identity.
    items: List[Node] = []
    cur_space = inherited_space
    for run in runs:
        for kind, payload in run:
            if kind == "tag":
                cur_space = payload
            elif kind == "nums":
                items.append(Vec(payload, cur_space))
            else:  # bracket
                items.append(_normalize(payload, cur_space))
    return items[0] if len(items) == 1 else Arr(items)


def parse_value(text: str) -> Node:
    """Parse a bare ``.ftsl`` value into its canonical :class:`Vec` / :class:`Arr`
    tree.  Raises :class:`ValueError` on a syntax error (a *shape* mismatch is only
    raised later, by the ``as_*`` validators)."""
    try:
        tree = _value_parser().parse(text)
    except SyntaxError as exc:
        raise ValueError(f"not a valid .ftsl value: {exc}") from exc
    if tree is None:
        raise ValueError("not a valid .ftsl value")
    return _normalize(tree)


# ---- per-field shape validators -------------------------------------------
# The grammar accepts any nested value-tree; these give each field a schema and
# a good "expected X, got Y" message.  They take either raw ``.ftsl`` text or an
# already-parsed :class:`Node`, so a caller can validate a value it just parsed.

def _describe(node: Node) -> str:
    if isinstance(node, Vec):
        if len(node.comps) == 1:
            return "a scalar"
        return f"a {len(node.comps)}-vector"
    if isinstance(node, Arr):
        if any(isinstance(it, Arr) for it in node.items):
            return "a list-of-lists"
        return f"a flat list of {len(node.items)} vectors"
    return "an unknown value"


def _as_node(value: Union[str, Node]) -> Node:
    return parse_value(value) if isinstance(value, str) else value


def _numeric(comps: List[Scalar]) -> bool:
    return all(isinstance(c, float) for c in comps)


def as_scalar(value: Union[str, Node]) -> float:
    """A single number (``5`` or ``[5]``)."""
    node = _as_node(value)
    if isinstance(node, Vec) and len(node.comps) == 1 and _numeric(node.comps):
        return node.comps[0]
    raise ShapeError(f"expected a scalar, got {_describe(node)}")


def as_vector(value: Union[str, Node], n: int) -> List[float]:
    """A plain numeric ``n``-vector (no colorspace nesting)."""
    node = _as_node(value)
    if isinstance(node, Vec) and len(node.comps) == n and _numeric(node.comps):
        return list(node.comps)
    raise ShapeError(f"expected a {n}-vector, got {_describe(node)}")


def as_color(value: Union[str, Node], default_space: str = "rgb"):
    """One color -> ``(space, (a, b, c))``.  A 3-component vector; its colorspace
    is its tag or ``default_space``."""
    node = _as_node(value)
    if isinstance(node, Vec) and len(node.comps) == 3 and _numeric(node.comps):
        return (node.space or default_space, tuple(node.comps))
    raise ShapeError(f"expected a color (3 components), got {_describe(node)}")


def as_color_list(value: Union[str, Node], default_space: str = "rgb"):
    """A flat list of colors -> ``[(space, (a, b, c)), …]``.  A lone color counts
    as a one-element list (the ``[X] ≡ X`` identity); a list-of-lists is the wrong
    shape here (it is valid for a field that wants a list of palettes)."""
    node = _as_node(value)
    if isinstance(node, Vec):
        return [as_color(node, default_space)]
    if isinstance(node, Arr) and all(isinstance(it, Vec) for it in node.items):
        return [as_color(it, default_space) for it in node.items]
    raise ShapeError(f"expected a flat list of colors, got {_describe(node)}")
