"""Grammar-driven ``.ftsl`` reader (J3c, option-a).

Parses ``.ftsl`` text with the single shared EPEG grammar (``ftsl.epeg``, via the
vendored GPDA parser) and builds loom ``Element`` objects from the resulting
``ParseNode`` tree.  This is the grammar-backed replacement for the hand-written
readers (starting with :class:`loom.record.Record`), so one grammar is the single
source of truth for both loom's Python side and (later) ftrace's C++ front-end.

Three entry points, narrowest first: :func:`parse_record` (one ``range`` block, structurally
identical to :meth:`loom.record.Record.parse`), :func:`parse_element` (one element of any
kind), and :func:`parse_document` (a whole file, keeping the text *between* elements so it
re-emits byte for byte).  A kind that maps onto an authoring class without loss becomes that
class; the *baked* kinds become a layout-preserving :class:`loom.block.Block` (see
:mod:`loom.block` for why the read direction cannot be the inverse of ``emit``).

**Scope.** ``ftsl.epeg`` is loom's *typed* grammar of the element kinds loom emits — it keeps
real ``NUMBER``/``REF``/``PIN``/``STRING`` terminals so the record / material / spectrum
validators can shape-check a value.  It is deliberately **not** the whole ftrace language, and
adopting ftrace's catch-all ``WORD`` tokenizer to make it so would erase exactly those
terminals.  Hand-authored full-language constructs (``/``-containing names, ``§3.2``
per-property access, expression arguments, ``[…]`` array literals at block value sites) are
therefore parse errors here; their home is ``ftsl_scene.epeg``, the grammar compiled into
ftrace's own front-end.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from . import load_grammar
from ..block import Document
from ..record import Record
from ..scene import Material, Texture

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "ftsl.epeg")


@lru_cache(maxsize=1)
def _grammar_text() -> str:
    with open(_GRAMMAR_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


@lru_cache(maxsize=1)
def _parser():
    """Load + cache the shared grammar parser (built once)."""
    return load_grammar(_grammar_text())


@lru_cache(maxsize=1)
def _file_parser():
    """The same grammar with the start rule moved to ``elements`` (a whole file).

    A second parser rather than a second grammar: the rules are identical, only the
    entry point differs, so a file and a lone element can never disagree about what
    a block means.
    """
    sc = load_grammar(_grammar_text())
    sc.parser.start_rule = "elements"
    return sc


# ---- ParseNode helpers ----------------------------------------------------

def _kids(node, name):
    return [c for c in node.children if c.name == name]


def _kid(node, name):
    for c in node.children:
        if c.name == name:
            return c
    return None


class _Src:
    """The source text plus a line-offset table, so a (line, col) span can be sliced
    back out verbatim.  Whitespace is a skip token, so this is the only way to recover
    the spacing *inside* a statement (a hand-aligned ``units    meters``)."""

    __slots__ = ("text", "starts")

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.starts.append(i + 1)

    def slice(self, start, end) -> Optional[str]:
        """The text between two (line, col) positions, or None if either is unknown or
        they straddle a line (a statement never legally does)."""
        if not start or not end or start[0] != end[0] or not start[0]:
            return None
        if start[0] > len(self.starts):
            return None
        base = self.starts[start[0] - 1]
        return self.text[base + start[1] - 1: base + end[1] - 1]

    def offset(self, pos) -> Optional[int]:
        """A (line, col) position as an index into :attr:`text`."""
        if not pos or not pos[0] or pos[0] > len(self.starts):
            return None
        return self.starts[pos[0] - 1] + pos[1] - 1

    def between(self, start, end) -> str:
        """The verbatim text from ``start`` to ``end``, **spanning lines** — the blank
        lines and top-level comments that sit between two elements, which no element can
        hold because they belong to neither."""
        a = 0 if start is None else self.offset(start)
        b = len(self.text) if end is None else self.offset(end)
        return self.text[a:b] if a is not None and b is not None else ""

    def line(self, n: int) -> str:
        """Line ``n`` (1-based), without its newline."""
        base = self.starts[n - 1]
        end = self.starts[n] - 1 if n < len(self.starts) else len(self.text)
        return self.text[base:end].rstrip("\r")

    def lines(self, first: int, last: int) -> List[str]:
        """Lines ``first`` … ``last`` inclusive (empty when the range is empty)."""
        return [self.line(n) for n in range(max(1, first), min(last, len(self.starts)) + 1)]

    def tail(self, pos) -> str:
        """The rest of ``pos``'s line, from its column on — a trailing ``# comment``
        and the whitespace before it, which the lexer skipped."""
        return self.line(pos[0])[pos[1] - 1:] if pos and pos[0] else ""


def _span_end(node):
    """(line, col-just-past) of a subtree's **last** terminal, or None if it has none.

    The mirror of :meth:`ParseNode.first_pos`, and the second half of what it takes to
    recover a shared line's inter-statement spacing: whitespace is a skip token, so the
    only record of a gap is the distance between one entry's end column and the next
    entry's start column.
    """
    if not node.children:
        return (node.line, node.col + len(node.value)) if node.line else None
    for c in reversed(node.children):
        end = _span_end(c)
        if end is not None:
            return end
    return None


def _terminal_nodes(node):
    """Every positioned leaf under ``node``, in source order."""
    if node.children:
        out = []
        for c in node.children:
            out.extend(_terminal_nodes(c))
        return out
    return [node] if node.line else []


def _core_span(node):
    """``(start, end)`` of an element's own text, with the ``nl?`` padding every element
    rule carries on both sides trimmed off.

    Without the trim an element would claim the blank line that separates it from its
    neighbour, and whichever of the two got it first would swallow it.
    """
    toks = _terminal_nodes(node)
    lo, hi = 0, len(toks)
    while lo < hi and toks[lo].name == "NEWLINE":
        lo += 1
    while hi > lo and toks[hi - 1].name == "NEWLINE":
        hi -= 1
    if lo >= hi:
        return None
    first, last = toks[lo], toks[hi - 1]
    return ((first.line, first.col), (last.line, last.col + len(last.value or "")))


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


def _binder_name(node) -> Optional[str]:
    """The bound NAME of a `NAME = KIND { … }` header, or None if anonymous.

    The unified element header carries its name in an optional `binder` node
    (`binder = NAME '='`); an anonymous `KIND { … }` has no binder.
    """
    b = _kid(node, "binder")
    if b is None:
        return None
    nm = _kid(b, "NAME")
    return nm.value if nm is not None else None


def _props(node):
    """A material/texture body -> ordered list of (key, [raw token, …])."""
    out = []
    for prop in _kids(node, "prop"):
        key = prop.children[0].value        # leading NAME key
        pval = _kid(prop, "pvalue")
        toks = [t[1] for t in _terminals(pval)]
        out.append((key, toks))
    return out


# Material/light fields that accept ONLY a spectrum expression, so the shared
# spectrum grammar (loom.grammar.spectrum) is their complete, correct validator.
# `transmit` is deliberately NOT here: like `reflect` it goes through ftrace's
# pattern-aware `patternedSpectrumParam`, so `transmit pattern:p` is legal there.
_SPECTRAL_ONLY_FIELDS = ("ior", "absorb", "substrate_k", "emit")

# Binding-union fields: they accept more than a bare spectrum (a `texture:` /
# `pattern:` bind, a scalar number, …) — validated by loom.grammar.bindings, a mirror
# of ftrace's bindReflectTexture / bindScalarTexture / bindScalarPattern.  `reflect`
# is colour-bindable (`texture:` | spectrum); `roughness` is scalar-bindable
# (`pattern:` | `texture:` | number); any `*_map` key is a scalar map (`pattern:` |
# `texture:`).  A record-driven override (`reflect = REC.chan`) is a *whole-block*
# form (`isRecordOverrideBlock`) loom does not emit, so it is out of scope here.
_COLOR_BIND_FIELDS = ("reflect",)
# Same union minus the image albedo — ftrace binds a `texture:` only on `reflect`.
_PATTERNED_SPECTRAL_FIELDS = ("transmit",)
_SCALAR_BIND_FIELDS = ("roughness",)


def _validate_spectral(props, fields) -> None:
    """Shape-check each present *purely-spectral* field's value against the shared
    ``.ftsl`` spectrum grammar (raising :class:`ShapeError` on a bad expression).

    Non-destructive — the verbatim value is left untouched so emit round-trips; this
    only rejects a value that ftrace's ``evalSpectrum`` would also reject."""
    from .spectrum import as_spectrum   # lazy: avoids the reader<->values import cycle
    for key in fields:
        if key in props:
            as_spectrum(props[key])


def _validate_bindings(props) -> None:
    """Shape-check the material binding-union fields (`reflect` / `roughness` /
    `*_map`) against ftrace's per-field binding grammar (loom.grammar.bindings).

    Like :func:`_validate_spectral`, this is non-destructive shape-only validation:
    a bound name's scene membership is left to the renderer, but a value ftrace's
    ``bind*`` / ``spectrumParam`` / ``dblParam`` would reject is rejected here."""
    from .bindings import as_color_binding, as_scalar_binding, as_map_binding
    for key in _COLOR_BIND_FIELDS:
        if key in props:
            as_color_binding(props[key])
    for key in _PATTERNED_SPECTRAL_FIELDS:
        if key in props:
            as_color_binding(props[key], texture=False)
    for key in _SCALAR_BIND_FIELDS:
        if key in props:
            as_scalar_binding(props[key])
    for key, val in props.items():
        if key.endswith("_map"):
            as_map_binding(val)


def _build_material(node) -> Material:
    name = _binder_name(node)
    ordered = _props(_kid(node, "mbody"))
    for key, toks in ordered:
        if key == "type" and toks[:1] == ["mix"]:
            return _build_mix(name, ordered)
    mtype = "diffuse"
    props = {}
    for key, toks in ordered:
        if key == "type":
            mtype = toks[0]
        else:
            # store the raw emitted token(s) (verbatim through value_token) so
            # emit -> parse -> emit is stable; strings keep their quotes stripped
            props[key] = " ".join(_unquote(t) for t in toks)
    _validate_spectral(props, _SPECTRAL_ONLY_FIELDS)
    _validate_bindings(props)
    return Material(name, mtype, **props)


def _build_spectrum_decl(node):
    from ..scene import NamedSpectrum
    name = _unquote(_kid(node, "STRING").value)
    return NamedSpectrum(name, " ".join(t[1] for t in _terminals(_kid(node, "pvalue"))))


def _build_mix(name, ordered):
    """A `type mix` material body -> :class:`~loom.material.MixMaterial`.

    A mix is the one material whose body has a *repeated* key: each ``layer "n" w``
    line is one layer, so it has to be read from the ordered property list — folding
    the body into a dict (as every other material can be) would keep only the last
    layer and silently produce a one-layer mix.
    """
    from ..material import MixMaterial
    layers, weight_map = [], None
    for key, toks in ordered:
        if key == "layer":
            layers.append((_unquote(toks[0]), float(toks[1])))
        elif key == "weight_map":
            val = _unquote(toks[0])
            # emitted as `pattern:<name>`; MixMaterial stores the bare pattern name
            weight_map = val.split(":", 1)[1] if val.startswith("pattern:") else val
    return MixMaterial(name, layers, weight_map=weight_map)


def _build_texture(node):
    name = _binder_name(node)
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


def _build_sphere(node):
    from ..scene import Sphere
    nums = [c.value for c in node.children if c.name == "NUMBER"]
    cx, cy, cz, r = (float(v) for v in nums[:4])
    mat = _unquote(_kid(node, "STRING").value)
    return Sphere((cx, cy, cz), r, mat)


def _build_light(node):
    from ..scene import Light
    # Unified header `[NAME =] light { kind <subtype>  … }`: the subtype rides a
    # `kind` property in the body rather than a bareword after the KIND.
    kind = None
    props = {}
    for key, toks in _props(_kid(node, "mbody")):
        val = " ".join(_unquote(t) for t in toks)
        if key == "kind":
            kind = val
        else:
            props[key] = val
    _validate_spectral(props, ("spd",))     # a light's `spd` is purely spectral
    return Light(kind, **props)


def _vec3n(node):
    """A ``vec3n`` (three space-separated NUMBERs) -> (x, y, z) floats."""
    return tuple(float(c.value) for c in node.children if c.name == "NUMBER")


def _build_camera(node):
    from ..scene import Camera
    name = _binder_name(node)
    view = _kid(node, "cam_view")
    eye, look_at, up = (_vec3n(v) for v in _kids(view, "vec3n"))
    fov_y = float(_kid(view, "NUMBER").value)
    mode = _kid(node, "cam_mode").value
    film = _kid(node, "cam_film")
    w, h = (int(float(c.value)) for c in film.children if c.name == "NUMBER")
    return Camera(eye, look_at, up=up, fov_y=fov_y, mode=mode, res=(w, h), name=name)


def _attach_trivia(entries, spans, brace, close, src: "_Src") -> List[str]:
    """Hang each entry's surrounding **trivia** — blank lines and ``# comments`` — off
    the entries, and return the trivia sitting between the last one and the ``}``.

    Comments are a lexer skip, so they are absent from the parse tree entirely; the
    only place they still exist is the source, which is why this works off spans
    rather than nodes.  Dropping them would be the single most destructive thing a
    round-trip could do to a hand-written scene — ``scenes/flythrough.ftsl`` explains
    each ``point`` of its camera path in a trailing comment — so an editor that loaded
    and saved a file would silently strip its documentation.

    A comment/blank *line* attaches to the entry **below** it (``before``), and the
    tail of a line attaches to the entry that ends that line (``trail``).
    """
    for i, (e, (start, end)) in enumerate(zip(entries, spans)):
        prev_line = spans[i - 1][1][0] if i else brace[0]
        if start[0] > prev_line:                    # this entry opens a new line
            e.before = src.lines(prev_line + 1, start[0] - 1)
        next_line = spans[i + 1][0][0] if i + 1 < len(spans) else close[0]
        if end[0] < next_line:                      # nothing else shares its line
            e.trail = src.tail(end)
    last_line = spans[-1][1][0] if spans else brace[0]
    return src.lines(last_line + 1, close[0] - 1)


def _build_block(node, src: Optional["_Src"] = None):
    """A generic ``block`` node -> a :class:`loom.block.Block` (J3c read direction).

    Layout is recovered from the ``nl`` separators the grammar keeps inside ``bbody``
    (``bbody = (nl? bentry)* nl?``): an entry preceded by a newline owns its line, one
    that isn't shares the previous entry's line.  That is what lets a round-trip
    reproduce loom's single-line ``sphere { … }`` and indented multi-line ``medium { … }``
    instead of reflowing every block into one house style.

    ``src`` (when given) additionally recovers the spacing the token stream throws
    away: the gap between two statements sharing a line, the indent of the body, and
    each statement's verbatim slice.  Without it a block still round-trips
    *structurally*, just in loom's own house formatting.
    """
    from ..block import Block, Stmt, GAP, INDENT

    # A top-level `block` is `nl? bcore nl?`; a nested one *is* the bare `bcore` (see
    # the grammar's note on why the padding is factored out).  Accept either.
    if node.name == "block":
        node = _kid(node, "bcore")
    head = _kid(node, "bhead")
    quoted = _kid(head, "STRING")
    # Spaces between the header and its `{` (the literal right after `bhead`).
    head_end, brace = _span_end(head), node.children[1].first_pos()
    brace_gap = max(1, brace[1] - head_end[1]) if head_end and head_end[0] == brace[0] else 1
    # `bhead = binder NAME subtype? | NAME STRING? subtype?`: the kind is the LAST direct
    # NAME child, so neither the binder's own NAME (`iso` in `iso = isosurface {`) nor a
    # trailing bareword subtype is mistaken for it.
    names = [c for c in head.children if c.name == "NAME"]
    kind = names[-1].value if names else None
    if kind is None:
        raise ValueError("block has no kind keyword")

    entries = []
    spans = []                                      # per entry: (start, end) positions
    indent, first_start, last_end = None, None, None
    body = _kid(node, "bbody")
    if body is not None:
        pending_nl = False
        prev_end = None                             # (line, col) just past the last entry
        for child in body.children:
            if child.name == "nl":
                pending_nl = True
                continue
            if child.name != "bentry":
                continue
            inner = child.children[0]
            # Spaces between this entry and the previous one on the same line: the
            # whitespace itself is skipped by the lexer, so it is recovered from the
            # columns.  Only meaningful within a line, hence the line-number check.
            gap = GAP
            start = inner.first_pos()
            if not pending_nl and prev_end is not None and prev_end[0] == start[0]:
                gap = max(1, start[1] - prev_end[1])
            if pending_nl and indent is None:       # first entry to claim a line
                indent = max(1, start[1] - head.first_pos()[1])
            if inner.name == "bcore":
                sub = _build_block(inner, src)
                sub.own_line, sub.gap = pending_nl, gap
                entries.append(sub)
            else:                                   # bprop
                key = inner.children[0].value
                pval = _kid(inner, "pvalue")
                words = [t[1] for t in _terminals(pval)] if pval is not None else []
                raw = src.slice(start, _span_end(inner)) if src is not None else None
                entries.append(Stmt(key, words, own_line=pending_nl, gap=gap, raw=raw))
            prev_end = last_end = _span_end(inner)
            spans.append((start, prev_end))
            if first_start is None:
                first_start = start
            pending_nl = False

    # Padding just inside the braces (only observable on a one-line body).
    close = node.children[-1].first_pos()
    lpad = (first_start[1] - brace[1] - 1) if first_start and first_start[0] == brace[0] else 1
    rpad = (close[1] - last_end[1]) if last_end and last_end[0] == close[0] else 1

    tail_before = _attach_trivia(entries, spans, brace, close, src) if src else ()

    # The binder lives inside `bhead`, not alongside it as in the fixed-shape element
    # rules, so it has to be read from there (`_kid` only looks at direct children).
    sub = _kid(head, "subtype")
    return Block(kind, entries, name=_binder_name(head),
                 quoted_name=_unquote(quoted.value) if quoted is not None else None,
                 subtype=_kid(sub, "NAME").value if sub is not None else None,
                 indent=INDENT if indent is None else indent,
                 brace_gap=brace_gap, pad=(lpad, rpad),
                 tail_before=tail_before)


_BUILDERS = {
    "record": _build_record, "spectrum_decl": _build_spectrum_decl,
    "material": _build_material,
    "texture": _build_texture, "sphere": _build_sphere, "light": _build_light,
    "camera": _build_camera, "block": _build_block,
}


def parse_element(text: str):
    """Parse a single top-level ``.ftsl`` element.

    Blocks that map onto an authoring class without loss become that class
    (:class:`~loom.record.Record`, :class:`~loom.scene.Material`,
    :class:`~loom.scene.Texture`, :class:`~loom.scene.Sphere`,
    :class:`~loom.scene.Light`, :class:`~loom.scene.Camera`); every other block —
    ``isosurface``, ``medium``, ``pattern``, ``camera_curve``, ``mesh``, ``group``,
    ``quad``, ``box``, the CSG combinators, … — becomes a structure-preserving
    :class:`loom.block.Block`, because their emitted form is a baked snapshot that no
    reader can turn back into the animated element that produced it (see
    :mod:`loom.block`).
    """
    return _build_element(_parse_tree(text), _Src(text))


def _build_element(node, src: "_Src"):
    build = _BUILDERS.get(node.name)
    if build is None:
        raise ValueError(f"unsupported .ftsl element: {node.name!r}")
    # Only the generic block preserves layout; the typed classes have their own
    # emitters and a fixed shape, so they take no source.
    return build(node, src) if build is _build_block else build(node)


def parse_document(text: str) -> Document:
    """Parse a whole ``.ftsl`` file into a :class:`~loom.block.Document` — its elements
    **plus the verbatim text between them**.

    This is the read side of a *file*, not of a scene: it does not resolve the
    ``scene { … }`` header against the rest, look names up, or rebuild a live
    :class:`~loom.scene.Scene`.  What it does give is the whole file back byte for byte
    (``parse_document(t).emit(ctx) == t``), because the blank lines that group elements
    and the top-level comments that head a file's sections belong to *neither*
    neighbouring element and would otherwise be dropped on the way through.
    """
    tree, src, spans = _parse_file(text)
    els = [_build_element(_unwrap(el), src) for el in _kids(tree, "element")]
    gaps = [src.between(None, spans[0][0] if spans else None)]
    for i, (_, end) in enumerate(spans):
        gaps.append(src.between(end, spans[i + 1][0] if i + 1 < len(spans) else None))
    return Document(els, gaps)


def parse_elements(text: str) -> List:
    """Parse a whole ``.ftsl`` file into a list of elements (same rules as
    :func:`parse_element`, applied to each top-level block in source order).

    The elements only; use :func:`parse_document` when the text *between* them (blank
    lines, top-level comments) has to survive the round-trip too.
    """
    tree, src, _ = _parse_file(text)
    return [_build_element(_unwrap(el), src) for el in _kids(tree, "element")]


def _unwrap(el):
    return el.children[0] if el.name == "element" else el


def _parse_file(text: str):
    """``(tree, src, spans)`` for a whole file — one parse, one line table, and each
    element's own span with its ``nl?`` padding trimmed."""
    try:
        tree = _file_parser().parse(text)
    except SyntaxError as exc:
        raise ValueError(f"not a valid .ftsl file: {exc}") from exc
    if tree is None:
        raise ValueError("not a valid .ftsl file")
    src = _Src(text)
    spans = [s for s in (_core_span(el) for el in _kids(tree, "element")) if s]
    return tree, src, spans


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
