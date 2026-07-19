"""Grammar-driven ``.ftsl`` reader (J3c, option-a).

Parses ``.ftsl`` text with the single shared EPEG grammar (``ftsl.epeg``, via the
vendored GPDA parser) and builds loom ``Element`` objects from the resulting
``ParseNode`` tree.  This is the grammar-backed replacement for the hand-written
readers (starting with :class:`loom.record.Record`), so one grammar is the single
source of truth for both loom's Python side and (later) ftrace's C++ front-end.

Scope today: the parametric ``record`` block — :func:`parse_record` produces a
``Record`` structurally identical to :meth:`loom.record.Record.parse`.  The
grammar (and this reader) grow toward the full scene.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from . import load_grammar
from ..record import Record
from ..scene import Material, Texture

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "ftsl.epeg")


@lru_cache(maxsize=1)
def _parser():
    """Load + cache the shared grammar parser (built once)."""
    with open(_GRAMMAR_PATH, "r", encoding="utf-8") as fh:
        return load_grammar(fh.read())


# ---- ParseNode helpers ----------------------------------------------------

def _kids(node, name):
    return [c for c in node.children if c.name == name]


def _kid(node, name):
    for c in node.children:
        if c.name == name:
            return c
    return None


def _terminals(node):
    """Flatten every leaf value under ``node`` in order (name, value) pairs.

    GPDA collapses a single-terminal rule's value onto the rule node *and* keeps
    the child terminal leaf, so a ``node.value`` on an interior (has-children)
    node is a duplicate of a leaf below it — only true leaves contribute here.
    """
    if node.children:
        out = []
        for c in node.children:
            out.extend(_terminals(c))
        return out
    return [(node.name, node.value)] if node.value is not None else []


# ---- ParseNode -> Record --------------------------------------------------

def _pin_pos(pin_value: str) -> float:
    # PIN token is `p:<pos>`
    return float(pin_value[2:])


def _ws_stop_spec(node):
    """One whitespace ``ws_stop`` -> a from_channels stop spec (token or (token,pos))."""
    if node.value is not None:            # single terminal collapsed onto the node
        return node.value
    pin = _kid(node, "PIN")
    tok = None
    for c in node.children:
        if c.name in ("NUMBER", "REF"):
            tok = c.value
    if pin is not None:
        return (tok, _pin_pos(pin.value))
    return tok


def _vstop_spec(node):
    """One ``vstop`` (space-separated components) -> [comp,…] or ([comp,…], pos)."""
    comps = [c.value for c in node.children if c.name == "NUMBER"]
    pin = _kid(node, "PIN")
    if pin is not None:
        return (comps, _pin_pos(pin.value))
    return comps


def _channel_spec(node):
    """A ``channel`` ParseNode -> a from_channels channel spec tuple."""
    name = node.children[0].value          # leading NAME
    comma = _kid(node, "comma_body")
    if comma is not None:
        tag_node = _kid(comma, "colour_tag")
        space = tag_node.value if tag_node is not None else None
        stops = [_vstop_spec(v) for v in _kids(comma, "vstop")]
        if space is not None:
            return (name, stops, space)
        return (name, stops)
    ws = _kid(node, "ws_stops")
    stops = [_ws_stop_spec(s) for s in _kids(ws, "ws_stop")]
    return (name, stops)


def _domain(node):
    """A ``domain`` ParseNode -> (lo, hi) via the existing loom domain parser."""
    if node.value is not None:             # compact DOMAIN token, e.g. '0-1' / '-1-2'
        words = [node.value]
    else:                                  # NUMBER NUMBER form
        words = [c.value for c in node.children if c.name == "NUMBER"]
    return Record._parse_domain(words)


def _parse_tree(text: str):
    """Parse ``text`` and return the single element node (unwrap the ``element`` start)."""
    try:
        tree = _parser().parse(text)
    except SyntaxError as exc:              # normalise parser errors to ValueError
        raise ValueError(f"not a valid .ftsl element: {exc}") from exc
    if tree is None:
        raise ValueError("not a valid .ftsl element")
    # start rule `element` wraps the one alternative; unwrap to the concrete node
    return tree.children[0] if tree.name == "element" else tree


def _build_record(node) -> Record:
    name = node.children[0].value          # first NAME child is the record name
    lo, hi = _domain(_kid(node, "domain"))

    interp = "linear"
    channels = []
    lines = _kid(node, "lines")
    for line in _kids(lines, "line"):
        inner = line.children[0]
        if inner.name == "interp_line":
            mode = _kid(inner, "interp_mode")
            interp = mode.value if mode.value is not None else mode.children[0].value
        else:                               # channel
            channels.append(_channel_spec(inner))
    return Record.from_channels(name, lo, hi, channels, interp=interp)


def _unquote(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


def _props(node):
    """A material/texture body -> ordered list of (key, [raw token, …])."""
    out = []
    for prop in _kids(node, "prop"):
        key = prop.children[0].value        # leading NAME key
        pval = _kid(prop, "pvalue")
        toks = [t[1] for t in _terminals(pval)]
        out.append((key, toks))
    return out


def _build_material(node) -> Material:
    name = _unquote(_kid(node, "STRING").value)
    mtype = "diffuse"
    props = {}
    for key, toks in _props(_kid(node, "mbody")):
        if key == "type":
            mtype = toks[0]
        else:
            # store the raw emitted token(s) (verbatim through value_token) so
            # emit -> parse -> emit is stable; strings keep their quotes stripped
            props[key] = " ".join(_unquote(t) for t in toks)
    return Material(name, mtype, **props)


def _build_texture(node):
    name = _unquote(_kid(node, "STRING").value)
    fields = {k: [_unquote(t) for t in toks] for k, toks in _props(_kid(node, "mbody"))}
    if "rgb" in fields:                     # procedural (function) skin -> ProcTexture
        from ..scene import ProcTexture
        r, g, b = fields["rgb"][:3]
        kw = {}
        if "res" in fields:
            kw["res"] = int(float(fields["res"][0]))
        if "filter" in fields:
            kw["filter"] = fields["filter"][0]
        if "wrap" in fields:
            kw["wrap"] = fields["wrap"][0]
        return ProcTexture(name, r, g, b, **kw)
    kw = {}
    if "encoding" in fields:
        kw["encoding"] = fields["encoding"][0]
    if "filter" in fields:
        kw["filter"] = fields["filter"][0]
    if "wrap" in fields:
        kw["wrap"] = fields["wrap"][0]
    return Texture(name, fields["file"][0], **kw)


_BUILDERS = {"record": _build_record, "material": _build_material, "texture": _build_texture}


def parse_element(text: str):
    """Parse a single top-level ``.ftsl`` element (record / material / texture)."""
    node = _parse_tree(text)
    build = _BUILDERS.get(node.name)
    if build is None:
        raise ValueError(f"unsupported .ftsl element: {node.name!r}")
    return build(node)


def parse_record(text: str) -> Record:
    """Parse a single ``NAME = range LO-HI [ … ]`` block into a :class:`Record`.

    Grammar-backed twin of :meth:`loom.record.Record.parse` — same result, but the
    structure comes from the shared ``ftsl.epeg`` grammar rather than hand-written
    string splitting.
    """
    node = _parse_tree(text)
    if node.name != "record":
        raise ValueError("not a record declaration (expected `NAME = range LO-HI [`)")
    return _build_record(node)
