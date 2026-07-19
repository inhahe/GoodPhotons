"""
Delimiter **precedence ladder** parser for the generalized record stop grammar
(``ROADMAP_records.md`` §3.1; TODO §J3b item 2).

The generalized grammar authors an arbitrary-arity nested value with three
delimiters that form a *precedence ladder* — not an interchangeable free-order
set:

* **whitespace binds tightest — like ``×``** (juxtaposition → a vector),
* **comma binds looser — like ``+``** (opens a new outer level),
* **brackets ``[ ]`` are parentheses** — an explicit level anywhere.

so ``1 1 1, 2 2 2`` parses exactly like ``(1·1·1) + (2·2·2)`` → two groups of
three.  Because the ladder has fixed precedence, **structure is recoverable from
the delimiters alone** (the channel's declared arity is only used to *validate*).
All of these therefore denote the same tree::

    1 1 1                 ==  [1 1 1]                       # bracketing one level is idempotent
    1 1 1, 2 2 2, 3 3 3   ==  [1 1 1] [2 2 2] [3 3 3]       # sum-of-products == product-of-groups

Parens ``( )`` are **not** a ladder delimiter — they are reserved for expression
grouping / the §3.2 application surface (``sin(v)``, ``prop(2)``, ``gold.color(u=x)``).
The tokenizer treats a parenthesised run as an *opaque expression atom*, so
``clamp(x,0,1)`` stays a single leaf token (its internal comma is not a delimiter).

The parse result is a plain nested Python structure: a **leaf** is the raw token
string (a number, a colour ref, or an expression), a **group** is a ``list``.
Single-element groups unwrap (idempotent bracketing).  :func:`emit_ladder` renders
such a structure back to canonical ladder text, and :func:`shape` reports the
(rectangular) nested dimensions.

This is loom-only authoring (the J3b superset); current ftrace cannot parse it
(its tokenizer is not comma-aware — see ``ROADMAP_records.md`` §5).
"""

from __future__ import annotations

from typing import List, Tuple, Union

# A parsed ladder value: a leaf token (str) or a nested list of values.
Value = Union[str, List["Value"]]


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

_DELIMS = frozenset("[],")


def _tokenize(s: str) -> List[str]:
    """Split ``s`` into ``[``, ``]``, ``,`` and bareword tokens.

    Whitespace separates tokens; ``[`` ``]`` ``,`` are their own tokens.  A
    parenthesised run ``(…)`` is opaque — everything inside it (including commas,
    brackets and spaces) accretes into the current bareword, so an expression
    like ``clamp(x,0,1)`` is one leaf.
    """
    toks: List[str] = []
    buf: List[str] = []
    depth = 0

    def flush() -> None:
        if buf:
            toks.append("".join(buf))
            buf.clear()

    for c in s:
        if depth > 0:
            buf.append(c)
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            continue
        if c == "(":
            buf.append(c)
            depth += 1
        elif c.isspace():
            flush()
        elif c in _DELIMS:
            flush()
            toks.append(c)
        else:
            buf.append(c)
    flush()
    return toks


# ---------------------------------------------------------------------------
# recursive-descent parser (sum -> product -> factor)
# ---------------------------------------------------------------------------

class _Cursor:
    __slots__ = ("toks", "i")

    def __init__(self, toks: List[str]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Union[str, None]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> str:
        t = self.toks[self.i]
        self.i += 1
        return t


def _parse_sum(c: _Cursor) -> Value:
    """comma level (``+``): products separated by ``,`` → a list (unwrapped if 1)."""
    items = [_parse_product(c)]
    while c.peek() == ",":
        c.next()
        items.append(_parse_product(c))
    return items[0] if len(items) == 1 else items


def _parse_product(c: _Cursor) -> Value:
    """whitespace level (``×``): juxtaposed factors → a list (unwrapped if 1)."""
    items = [_parse_factor(c)]
    while c.peek() not in (None, ",", "]"):
        items.append(_parse_factor(c))
    return items[0] if len(items) == 1 else items


def _parse_factor(c: _Cursor) -> Value:
    """atom leaf or an explicit ``[ … ]`` group (parens of the ladder)."""
    t = c.peek()
    if t == "[":
        c.next()
        if c.peek() == "]":
            c.next()
            return []
        v = _parse_sum(c)
        if c.peek() != "]":
            raise ValueError("ladder: unclosed '['")
        c.next()
        return v
    if t in ("]", ",", None):
        raise ValueError(f"ladder: unexpected {t!r}")
    return c.next()


def parse_ladder(s: str) -> Value:
    """Parse a flexible-delimiter value string into a nested structure.

    Raises :class:`ValueError` on an empty string or leftover/mismatched tokens.
    """
    toks = _tokenize(s)
    if not toks:
        raise ValueError("ladder: empty value")
    c = _Cursor(toks)
    v = _parse_sum(c)
    if c.peek() is not None:
        raise ValueError(f"ladder: trailing tokens {toks[c.i:]!r}")
    return v


# ---------------------------------------------------------------------------
# emit + shape
# ---------------------------------------------------------------------------

def _is_multilevel(v: Value) -> bool:
    """True when ``v`` is a list that itself contains a list (depth ≥ 2)."""
    return isinstance(v, list) and any(isinstance(g, list) for g in v)


def emit_ladder(v: Value) -> str:
    """Render a nested value back to canonical ladder text.

    Canonical style: a list of all-leaves is space-joined (``1 1 1``); a list with
    any list child is comma-joined, and any multi-level child is bracket-wrapped so
    the lower comma level can't swallow it.  Round-trips through :func:`parse_ladder`.
    """
    if isinstance(v, str):
        return v
    if any(isinstance(c, list) for c in v):
        parts: List[str] = []
        for c in v:
            s = emit_ladder(c)
            if _is_multilevel(c):
                s = "[" + s + "]"
            parts.append(s)
        return ", ".join(parts)
    return " ".join(v)  # all leaves → a flat vector


def shape(v: Value) -> Tuple[int, ...]:
    """Nested dimension tuple of a *rectangular* value (``()`` for a leaf).

    Raises :class:`ValueError` on a ragged (non-rectangular) structure.
    """
    if isinstance(v, str):
        return ()
    subs = [shape(c) for c in v]
    if subs and any(s != subs[0] for s in subs):
        raise ValueError(f"ladder: ragged shape (children differ): {v!r}")
    inner = subs[0] if subs else ()
    return (len(v),) + inner
