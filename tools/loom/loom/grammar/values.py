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
* **A trailing tuple on an array is the sample call, not a label** (ADDENDUM —
  "call = sample", 2026-07-20): ``[0 1](u)`` reads the array at the driver ``u``,
  exactly as loom's ``grid(x, y)`` does; ``(a=u)`` rebinds a *named* axis.  The
  tuple hangs off a piece, so calls compose (``[[0 1](u), [2 3](v)]``,
  ``n(m(u), v)``).  An array written *without* one is **unsaturated** — legal to
  declare, an error only when a field tries to sample it (:func:`as_sampled`).

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


@dataclass
class Arg:
    """One argument of a sample call.

    ``formal`` is ``None`` for a positional argument (it binds the next axis, left
    to right) or the array-side axis name for a keyword rebind — ``(a=u)`` reads
    *formal* ``a`` = *driver* ``u``.  ``driver`` is what supplies the coordinate:
    an axis/driver name (``"u"``), a constant, or a nested :class:`Call`
    (composition, ``n(m(u), v)``).
    """
    formal: Optional[str]
    driver: "Coord"


@dataclass
class Call:
    """A *sampled* value — an array (or a named array parameter) together with the
    coordinates it is sampled at.  ``[0 1](u)`` is ``Call(Vec([0, 1]), [Arg(None,
    "u")])``.

    The trailing tuple is not a label on the literal; it *is* the sample call — the
    same operation as loom's ``grid(x, y)``.  An array written without one is not a
    ``Call`` at all, it is a bare :class:`Vec` / :class:`Arr`: an **unsaturated**
    value, legal to declare and an error only when something tries to render it.
    :func:`as_sampled` is where that error lives.
    """
    target: Union["Node", str]      # an array literal, or a NAME being called
    args: List[Arg]


Node = Union[Vec, Arr, Call]
Coord = Union[str, float, Call]


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
    ``('sampled', vsampled_node)``."""
    ct = _kid(p, "colour_tag")
    if ct is not None:
        space = ct.value if ct.value is not None else ct.children[0].value
        return ("tag", space)
    vs = _kid(p, "vsampled")
    if vs is not None:
        return ("sampled", vs)
    return ("nums", _vnums(_kid(p, "vnums")))


# ---- the sample call ------------------------------------------------------
# `vsampled = vbracket axistuple? | NAME axistuple` — a trailing tuple is not a
# label on the array, it is the *sample call*.  Without one, a bracket is just a
# bracket and the `[X] ≡ X` identity still applies, so `_sampled` returns the
# inner node unchanged in that case and a :class:`Call` only when a tuple is
# actually present.

def _check_args(args: List["Arg"]) -> List["Arg"]:
    """Positionals-before-keywords and no-duplicate-formals.

    The grammar deliberately does not enforce either (``arg = NAME '=' coord |
    coord`` in any order), so that the error can be phrased in terms of axes
    instead of arriving as an opaque parse failure.
    """
    seen_keyword = False
    formals: set = set()
    for a in args:
        if a.formal is None:
            if seen_keyword:
                raise ShapeError(
                    "positional axis arguments must come before keyword ones: "
                    "write `(u, a=v)`, not `(a=v, u)`")
        else:
            seen_keyword = True
            if a.formal in formals:
                raise ShapeError(
                    f"axis '{a.formal}' is bound twice in one sample call")
            formals.add(a.formal)
    return args


def _sampled(vs, space: Optional[str]) -> Node:
    """A ``vsampled`` node -> a :class:`Call`, or — when there is no trailing
    tuple — the bracket's own normalized node."""
    vb = _kid(vs, "vbracket")
    target: Union[Node, str]
    if vb is not None:
        target = _normalize(_kid(vb, "value"), space)
    else:
        target = _kid(vs, "NAME").value
    tup = _kid(vs, "axistuple")
    if tup is None:
        return target
    return Call(target, _check_args([_arg(a, space) for a in _kids(tup, "arg")]))


def _arg(a, space: Optional[str]) -> "Arg":
    """One ``arg`` -> :class:`Arg`.  A direct ``NAME`` child (as opposed to one
    nested under ``coord``) is the keyword form's *formal*."""
    formal = _kid(a, "NAME")
    return Arg(formal.value if formal is not None else None,
               _coord(_kid(a, "coord"), space))


def _coord(c, space: Optional[str]) -> "Coord":
    """One ``coord`` -> a driver name, a constant, or a nested call."""
    vs = _kid(c, "vsampled")
    if vs is not None:
        return _sampled(vs, space)
    nm = _kid(c, "NAME")
    if nm is not None:
        return nm.value
    return float(_kid(c, "NUMBER").value)


def _normalize(node, inherited_space: Optional[str] = None) -> Node:
    """A ``value`` ParseNode -> canonical :class:`Vec` / :class:`Arr` tree.

    ``inherited_space`` is the colorspace in force from an enclosing tag (a tag
    before a bracket colors the bracket's contents until an inner tag overrides).
    """
    runs = [[_piece(p) for p in _kids(vr, "vpiece")] for vr in _kids(node, "vrun")]
    has_struct = any(k in ("tag", "sampled") for run in runs for (k, _) in run)

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
            else:  # bracket, with or without a trailing sample call
                items.append(_sampled(payload, cur_space))
    return items[0] if len(items) == 1 else Arr(items)


def parse_value(text: str) -> Node:
    """Parse a bare ``.ftsl`` value into its canonical :class:`Vec` / :class:`Arr` /
    :class:`Call` tree.  Raises :class:`ValueError` on a syntax error; a *field*
    shape mismatch is only raised later, by the ``as_*`` validators.  (The one
    thing normalization itself rejects is an ill-formed argument list — argument
    order is deliberately not a grammar rule so the error can name the axis.)"""
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
    if isinstance(node, Call):
        n = len(node.args)
        return f"a sample call with {n} coordinate{'' if n == 1 else 's'}"
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


def as_sampled(value: Union[str, Node]) -> Call:
    """A *sampled* array — an array together with the coordinates it is read at,
    ``[0 1](u)`` or ``ramp(u, v)``.

    This is where the **unsaturated** error lives: an array on its own is a legal
    thing to write down, but nothing can be rendered from it until a site says
    *where* to sample it, so a bare array reaching a field that samples is an
    error with a fix in the message rather than a silent default.
    """
    node = _as_node(value)
    if isinstance(node, Call):
        return node
    raise ShapeError(
        f"expected a sampled array like `[0 1](u)`, got {_describe(node)} — an "
        "array is unsaturated until it is called at a coordinate; add `(u)` "
        "(or `(u, v)` for a 2-D array)")
