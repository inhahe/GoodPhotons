"""
Loom twin of the FTSL **parametric record** (see ``ROADMAP_records.md`` / TODO §J3).

A record is a *bank of named per-channel curves over a shared scalar driver domain*
``[lo, hi]``.  Each channel is named after a real destination slot (``reflect``,
``roughness``, …) and holds an ordered list of **stops**; a driver scalar samples
every channel at once (``nearest`` / ``linear`` / ``smooth`` interpolation).  This
mirrors ftrace's ``Record`` (``src/record.h`` + the ``NAME = range LO-HI [ … ]``
grammar in ``src/ftsl.h``) closely enough to **read, represent and re-emit** the
records in a ``.ftsl`` scene (J3a — the round-trip goal).

Model (matching the generalized spec, ``ROADMAP_records.md`` §3):

* a **channel** outputs a value of some arity ``D``.  These kinds are supported:
  a *scalar* channel (``D==1``; every stop is a numeric literal or a pattern
  expression), a *colour* channel — either the ftrace-native ``spectrum:<name>``-ref
  form (every stop a ``:``-ref, interpolated in linear-RGB → Jakob–Hanika) **or** the
  J3b *inline-colour* form (a channel-level ``rgb``/``hsv``/``hsl`` tag over arity-3
  numeric triples; see :meth:`Record.lower_colours`) — and a *vector* channel
  (``D`` ≥ 2; every stop is an arity-``D`` numeric/expression tuple — the **J3b**
  generalized channel).  A channel must be homogeneous.  ftrace materializes only
  scalar + ``spectrum:``-ref colour today; inline-colour and vector channels are
  loom-only (inline colour *lowers* to ``spectrum:``-refs ftrace can parse).
* a **stop** carries its raw component tokens (``.components``; ``.token`` is the single
  component of an arity-1 stop) preserved verbatim for faithful re-emit, and an optional
  pinned domain position (author ``p:<pos>`` prefix).  Unpinned stops are spread evenly
  between their pinned/anchor neighbours exactly as ftrace does.

One backward-compatible ladder grammar (:meth:`Record.emit` / :meth:`Record.parse`
/ :meth:`parse_all`) that is a strict **additive superset** of current FTSL — not a
breaking change.  Each channel line is dispatched on the presence of a top-level
comma (:meth:`_split_top_commas`):

* **no top-level comma → whitespace stops** (the current-FTSL path).  Every
  *whitespace*-word is its own stop and a stop is colour only when its token
  contains ``':'`` — so ``reflect  spectrum:steel spectrum:gold`` is two colour
  stops and ``rough  0 0 0`` is three scalar stops, exactly as ftrace reads them.
  Every real ``scenes/_record_*.ftsl`` round-trips through this path unchanged.
* **top-level comma present → comma stops** (the J3b generalized superset,
  ``ROADMAP_records.md`` §3.1).  Stops are *comma*-separated and vector components
  *space*-separated (``tint  0 0 0, 1 1 1`` is a 2-stop arity-3 channel), parsed via
  the delimiter precedence ladder (:mod:`loom.ladder`).  A *lone* vector stop is
  written with a trailing comma (``tint  0 0 0,``) so it can't be misread as N
  scalar stops.  ftrace's own tokenizer is not comma-aware, so a record that
  actually uses comma lines is loom-only until ftrace's parser is upgraded (J3c).

An **inline-colour** channel opts in with a leading ``rgb``/``hsv``/``hsl`` tag word
(``reflect  rgb 0 0 0, 1 1 1``): the tag fixes arity 3, so each comma-group is one
colour stop (a lone tagged stop needs no trailing comma).  :meth:`Record.lower_colours`
rewrites such channels to synthesized ``spectrum "<name>" = rgb …`` decls +
``spectrum:<name>`` refs so the result is parseable by current ftrace.

:meth:`emit` picks the right form per channel automatically: scalar/``spectrum:``-ref
colour channels emit as whitespace lines, vector channels as comma lines, inline-colour
channels as a tagged comma line.

What this module also does **not** do (deferred to J3c's full pattern VM): evaluate
*expression* stops.  The numeric :meth:`Record.sample` sampler works on all-numeric
scalar channels; colour and expression channels are represented and re-emitted
faithfully but not evaluated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Number
from .scene import Element
from .ftsl_emit import EmitCtx, fmt
from .ladder import parse_ladder, Value


_INTERP = ("nearest", "linear", "smooth")


def _is_number(tok: str) -> bool:
    """True if ``tok`` is a plain numeric literal (mirrors ftrace's ``isNumber``)."""
    try:
        float(tok)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class RecordStop:
    """One stop in a channel LUT — an arity-``D`` tuple of component tokens.

    ``value`` is either a single token ``str`` (arity 1: a numeric literal, a scalar
    pattern expression, or a colour ref containing ``':'``) or a ``list`` of component
    tokens (arity ``D`` ≥ 2 — the J3b generalized vector stop, e.g. ``["0","0","0"]``
    for an inline ``rgb`` triple).  Each component is kept verbatim so a parsed record
    re-emits equivalently.  ``pos`` is the pinned domain position when the author wrote a
    ``p:<pos>`` prefix, else ``None`` (redistributed on demand).
    """

    value: Union[str, List[str]]
    pos: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.value, str):
            self._components: List[str] = [self.value]
        else:
            self._components = [str(c) for c in self.value]
        if not self._components:
            raise ValueError("record stop has no components")

    @property
    def components(self) -> List[str]:
        """The ``D`` component tokens (a 1-list for a scalar/colour-ref stop)."""
        return self._components

    @property
    def arity(self) -> int:
        return len(self._components)

    @property
    def token(self) -> str:
        """The single token of an arity-1 stop (raises for a vector stop)."""
        if len(self._components) != 1:
            raise TypeError(
                f"vector stop (arity {len(self._components)}) has no single token; "
                "use .components / .as_vector()")
        return self._components[0]

    @property
    def pinned(self) -> bool:
        return self.pos is not None

    @property
    def is_colour(self) -> bool:
        # ftrace: a stop is a colour ref iff a component contains ':'
        return any(":" in c for c in self._components)

    def as_number(self) -> float:
        if len(self._components) != 1 or not _is_number(self._components[0]):
            raise TypeError(f"stop {self._components!r} is not a plain numeric literal")
        return float(self._components[0])

    def as_vector(self) -> List[float]:
        """The stop as a list of ``D`` floats (raises on any non-numeric component)."""
        if not all(_is_number(c) for c in self._components):
            raise TypeError(f"stop {self._components!r} has non-numeric components")
        return [float(c) for c in self._components]


_COLOUR_SPACES = ("rgb", "hsv", "hsl")


@dataclass
class RecordChannel:
    """A named LUT: ``channelname stop stop …`` (auto-bound to a like-named slot).

    ``space`` is an optional **colour-space tag** (``rgb`` / ``hsv`` / ``hsl``): when
    set, the channel is an *inline colour* channel whose stops are arity-3 numeric
    triples interpreted in that colour space (the J3b superset — authored as
    ``reflect  rgb 0 0 0, 1 1 1`` rather than a chain of ``spectrum:<name>`` refs).
    ``space is None`` is every other channel: scalar, vector, or a ``spectrum:``-ref
    colour channel (the ftrace-native colour form).
    """

    name: str
    stops: List[RecordStop] = field(default_factory=list)
    space: Optional[str] = None

    def __post_init__(self) -> None:
        if self.space is not None and self.space not in _COLOUR_SPACES:
            raise ValueError(
                f"record channel {self.name!r}: colour space must be one of "
                f"{_COLOUR_SPACES} (got {self.space!r})")

    @property
    def is_inline_colour(self) -> bool:
        """True for an ``rgb``/``hsv``/``hsl``-tagged inline-colour channel (J3b)."""
        return self.space is not None

    @property
    def kind(self) -> str:
        """Channel kind, enforcing homogeneity:

        * ``"colour"`` — either a ``spectrum:``-ref channel (every stop contains
          ``':'``; the ftrace-native form) **or** an inline-colour channel tagged
          ``rgb``/``hsv``/``hsl`` (every stop an arity-3 numeric/expression triple).
        * ``"scalar"`` — every stop is a single non-colour token (arity 1).
        * ``"vector"`` — every stop is an arity-``D`` (``D`` ≥ 2) numeric/expression
          tuple, all the same ``D`` (the J3b generalized channel).

        Raises on a mix of colour and non-colour stops, or ragged vector arity.
        """
        if self.space is not None:
            # inline colour channel: arity-3 numeric/expression triples, no :-refs
            if any(s.is_colour for s in self.stops):
                raise ValueError(
                    f"record channel {self.name!r}: an {self.space} colour channel "
                    "can't hold spectrum:<name> refs (drop the tag or the refs)")
            arities = {s.arity for s in self.stops}
            if arities != {3}:
                raise ValueError(
                    f"record channel {self.name!r}: an {self.space} colour channel "
                    f"needs arity-3 stops, got arities {sorted(arities)}")
            return "colour"
        colour = any(s.is_colour for s in self.stops)
        noncolour = any(not s.is_colour for s in self.stops)
        if colour and noncolour:
            raise ValueError(
                f"record channel {self.name!r} mixes colour (spectrum:…) and scalar stops"
            )
        if colour:
            return "colour"
        arities = {s.arity for s in self.stops}
        if len(arities) != 1:
            raise ValueError(
                f"record channel {self.name!r} mixes stop arities {sorted(arities)}")
        return "scalar" if arities == {1} else "vector"

    @property
    def arity(self) -> int:
        """Component count ``D`` of every stop (1 for scalar/``spectrum:``-ref
        channels, 3 for an inline-colour channel)."""
        arities = {s.arity for s in self.stops}
        if len(arities) != 1:
            raise ValueError(
                f"record channel {self.name!r} mixes stop arities {sorted(arities)}")
        return next(iter(arities))

    @property
    def is_numeric(self) -> bool:
        """True when every stop is all-numeric (scalar or vector; sampler-eligible)."""
        return bool(self.stops) and all(
            all(_is_number(c) for c in s.components) for s in self.stops)


# ---------------------------------------------------------------------------
# position redistribution (port of ftrace redistributeStops)
# ---------------------------------------------------------------------------

def _redistribute(stops: Sequence[RecordStop], lo: float, hi: float) -> List[float]:
    """Effective domain positions: pinned stops keep ``pos``; the first/last
    unpinned stops anchor to ``lo``/``hi``; interior unpinned runs spread evenly
    between their fixed neighbours.  Mirrors ftrace's ``redistributeStops``."""
    n = len(stops)
    pos: List[float] = [s.pos if s.pinned else 0.0 for s in stops]
    fixed: List[bool] = [s.pinned for s in stops]
    if n == 1:
        if not fixed[0]:
            pos[0] = lo
        return pos
    if not fixed[0]:
        pos[0] = lo
        fixed[0] = True
    if not fixed[n - 1]:
        pos[n - 1] = hi
        fixed[n - 1] = True
    a = 0
    while a < n:
        if not fixed[a]:
            a += 1
            continue
        b = a + 1
        while b < n and not fixed[b]:
            b += 1
        if b < n and b > a + 1:
            pa, pb = pos[a], pos[b]
            gaps = b - a
            for j in range(a + 1, b):
                pos[j] = pa + (pb - pa) * (j - a) / gaps
        a = b
    return pos


# ---------------------------------------------------------------------------
# numeric sampler (port of recSampleScalar for all-numeric scalar channels)
# ---------------------------------------------------------------------------

def _fc_tangent(p: Sequence[float], v: Sequence[float], sec: Sequence[float],
                n: int, k: int) -> float:
    """Fritsch–Carlson monotone-cubic tangent at node ``k`` (matches ftrace)."""
    if k == 0:
        return sec[0]
    if k == n - 1:
        return sec[n - 2]
    s0, s1 = sec[k - 1], sec[k]
    if s0 * s1 <= 0.0:
        return 0.0
    h0 = p[k] - p[k - 1]
    h1 = p[k + 1] - p[k]
    w0 = 2.0 * h1 + h0
    w1 = h1 + 2.0 * h0
    return (w0 + w1) / (w0 / s0 + w1 / s1)


def _sample_numeric(pos: Sequence[float], vals: Sequence[float], d: float,
                    interp: str) -> float:
    """Sample a numeric LUT at driver ``d`` per ``interp`` (bit-mirrors ftrace)."""
    n = len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return vals[0]
    lo, hi = pos[0], pos[n - 1]
    if d < lo:
        d = lo
    elif d > hi:
        d = hi
    # locate interval [i, i+1]
    i = 0
    while i < n - 2 and d > pos[i + 1]:
        i += 1
    span = pos[i + 1] - pos[i]
    t = (d - pos[i]) / span if span > 1e-12 else 0.0
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

    if interp == "nearest":
        return vals[i if t < 0.5 else i + 1]
    v0, v1 = vals[i], vals[i + 1]
    if interp == "linear":
        return v0 + (v1 - v0) * t
    # smooth: monotone cubic Hermite
    sec = [((vals[k + 1] - vals[k]) / (pos[k + 1] - pos[k]))
           if (pos[k + 1] - pos[k]) > 1e-12 else 0.0 for k in range(n - 1)]
    mk = _fc_tangent(pos, vals, sec, n, i)
    mk1 = _fc_tangent(pos, vals, sec, n, i + 1)
    h = pos[i + 1] - pos[i]
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * v0 + h10 * h * mk + h01 * v1 + h11 * h * mk1


# ---------------------------------------------------------------------------
# Record element
# ---------------------------------------------------------------------------

# One channel line as authored: (name, [stop, …]) or (name, [stop, …], space) where a
# stop is a scalar token ``str``, a vector ``List[str]`` of component tokens, or a
# ``(value, pos)`` pin tuple, and ``space`` is an optional ``rgb``/``hsv``/``hsl`` tag.
_StopSpec = Union[str, List[str], Tuple[Union[str, List[str]], Optional[float]]]
ChannelSpec = Union[Tuple[str, Sequence[_StopSpec]],
                    Tuple[str, Sequence[_StopSpec], Optional[str]]]


class Record(Element):
    """A parametric record — emits ``NAME = range LO-HI [ … ]`` and parses one back.

    Construct directly with :class:`RecordChannel` objects, from lightweight tuples
    via :meth:`from_channels`, or parse an existing block with :meth:`parse`.
    """

    def __init__(self, name: str, lo: float, hi: float,
                 channels: Sequence[RecordChannel],
                 *, interp: str = "linear") -> None:
        if hi <= lo:
            raise ValueError(f"record {name!r}: range needs HI > LO (got {lo}-{hi})")
        if interp not in _INTERP:
            raise ValueError(f"record {name!r}: interp must be one of {_INTERP}")
        if not channels:
            raise ValueError(f"record {name!r}: has no channels")
        self.name = name
        self.lo = float(lo)
        self.hi = float(hi)
        self.interp = interp
        self.channels: List[RecordChannel] = list(channels)
        for ch in self.channels:
            if not ch.stops:
                raise ValueError(f"record {name!r} channel {ch.name!r}: has no stops")
            ch.kind  # noqa: B018 — trigger the homogeneity check
        self._validate_positions()

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_channels(cls, name: str, lo: float, hi: float,
                      channels: Sequence[ChannelSpec],
                      *, interp: str = "linear") -> "Record":
        """Build from ``(chan_name, [token | (token, pos), …])`` tuples.

        A channel spec may be a 3-tuple ``(chan_name, stops, space)`` to tag an
        inline-colour channel (``space`` ∈ ``rgb``/``hsv``/``hsl``; stops are arity-3
        component lists)."""
        def _as_value(v) -> Union[str, List[str]]:
            # a list is a vector stop (components); a scalar str is a single token
            return [str(c) for c in v] if isinstance(v, list) else str(v)

        chans: List[RecordChannel] = []
        for spec in channels:
            if len(spec) == 3:
                cname, raw, space = spec
            else:
                cname, raw = spec
                space = None
            stops: List[RecordStop] = []
            for item in raw:
                if isinstance(item, tuple):        # (value, pos) pin — value str or list
                    val, pos = item
                    stops.append(RecordStop(_as_value(val),
                                            None if pos is None else float(pos)))
                else:                              # bare str (scalar) or list (vector)
                    stops.append(RecordStop(_as_value(item)))
            chans.append(RecordChannel(str(cname), stops,
                                       None if space is None else str(space)))
        return cls(name, lo, hi, chans, interp=interp)

    # -- validation ----------------------------------------------------------

    def _validate_positions(self) -> None:
        for ch in self.channels:
            pos = _redistribute(ch.stops, self.lo, self.hi)
            for p in pos:
                if p < self.lo - 1e-9 or p > self.hi + 1e-9:
                    raise ValueError(
                        f"record {self.name!r} channel {ch.name!r}: "
                        f"stop position {p} is outside the domain [{self.lo}, {self.hi}]")
            for i in range(1, len(pos)):
                if pos[i] < pos[i - 1] - 1e-12:
                    raise ValueError(
                        f"record {self.name!r} channel {ch.name!r}: "
                        "stop positions must be non-decreasing")

    def channel(self, name: str) -> RecordChannel:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"record {self.name!r} has no channel {name!r}")

    def positions(self, name: str) -> List[float]:
        """Effective (redistributed) domain positions of a channel's stops."""
        return _redistribute(self.channel(name).stops, self.lo, self.hi)

    # -- numeric sampling (all-numeric scalar channels only) -----------------

    def sample(self, name: str, d: float) -> float:
        """Sample an all-numeric **scalar** (arity-1) channel at driver ``d``.

        Raises :class:`TypeError` for colour channels, expression stops (need the
        pattern VM; J3c), or vector channels (use :meth:`sample_vec`)."""
        ch = self.channel(name)
        if ch.arity != 1:
            raise TypeError(
                f"channel {name!r} has arity {ch.arity} (not a scalar) — use sample_vec()")
        return self.sample_vec(name, d)[0]

    def sample_vec(self, name: str, d: float) -> List[float]:
        """Sample an all-numeric channel at driver ``d``, per-component.

        Works for scalar (arity 1 → 1-list), vector (arity ``D`` → ``D``-list) and
        **inline `rgb` colour** channels; each component is interpolated independently
        exactly as ftrace samples a scalar LUT (per-component linear interpolation of
        rgb == ftrace's linear-RGB colour interpolation).  Raises for `hsv`/`hsl`
        colour channels (lower to `rgb` first), `spectrum:`-ref colour channels (need
        the spectrum table), and expression stops."""
        ch = self.channel(name)
        if ch.space in ("hsv", "hsl"):
            raise TypeError(
                f"channel {name!r} is an {ch.space} colour channel — lower to rgb "
                "(lower_colours) before numeric sampling")
        if ch.kind == "colour" and ch.space is None:
            raise TypeError(
                f"channel {name!r} is a spectrum:-ref colour channel — "
                "not numerically sampleable")
        if not ch.is_numeric:
            raise TypeError(
                f"channel {name!r} has expression stops — sampling needs the pattern VM (J3c)")
        pos = _redistribute(ch.stops, self.lo, self.hi)
        D = ch.arity
        out: List[float] = []
        for c in range(D):
            vals = [s.as_vector()[c] for s in ch.stops]
            out.append(_sample_numeric(pos, vals, d, self.interp))
        return out

    # -- emit ----------------------------------------------------------------

    def roots(self) -> List:
        return []

    @property
    def has_vector_channel(self) -> bool:
        """True if any channel is an arity-``D`` (``D`` ≥ 2) vector channel (J3b)."""
        return any(ch.kind == "vector" for ch in self.channels)

    def _domain_str(self) -> str:
        # range: prefer the compact `LO-HI` when lo >= 0 (unambiguous), else `LO HI`.
        if self.lo >= 0:
            return f"{fmt(self.lo)}-{fmt(self.hi)}"
        return f"{fmt(self.lo)} {fmt(self.hi)}"

    def emit(self, ctx: Optional[EmitCtx] = None) -> str:
        """Emit the ``NAME = range LO-HI [ … ]`` block in the one backward-compatible
        ladder grammar.  Each channel picks its form automatically: a scalar or
        ``spectrum:``-ref colour channel emits whitespace-separated stops (identical to
        current FTSL); a vector channel emits comma-separated stops with space-separated
        components (a lone vector stop gets a trailing comma so it can't be misread as N
        scalar stops); an inline-colour channel emits its ``rgb``/``hsv``/``hsl`` tag
        then comma-separated arity-3 triples.  Round-trips through :meth:`parse`."""
        dom = self._domain_str()
        # pad channel names (+ the interp keyword) to a common width for tidy columns
        names = [ch.name for ch in self.channels]
        width = max([len(n) for n in names] + [len("interp")])
        lines = [f"{self.name} = range {dom} ["]
        for ch in self.channels:
            lines.append(f"    {ch.name.ljust(width)}  " + self._emit_channel_body(ch))
        if self.interp != "linear":
            lines.append(f"    {'interp'.ljust(width)}  {self.interp}")
        lines.append("]")
        return "\n".join(lines)

    @staticmethod
    def _emit_channel_body(ch: RecordChannel) -> str:
        """The stop list of one channel line (whitespace form for scalar / ``spectrum:``
        colour, tagged comma form for inline colour, comma form for a vector channel)."""
        if ch.space is not None:
            # inline colour channel: `<tag> r g b, r g b, …` (the tag fixes arity 3, so
            # a lone stop needs no trailing comma to disambiguate)
            stop_strs: List[str] = []
            for s in ch.stops:
                comp = " ".join(s.components)
                stop_strs.append(f"p:{fmt(s.pos)} {comp}" if s.pinned else comp)
            return f"{ch.space} " + ", ".join(stop_strs)
        if ch.kind == "vector":
            stop_strs = []
            for s in ch.stops:
                comp = " ".join(s.components)          # a flat vector (space-joined)
                stop_strs.append(f"p:{fmt(s.pos)} {comp}" if s.pinned else comp)
            body = ", ".join(stop_strs)
            if len(ch.stops) == 1:
                body += ","          # disambiguate a lone vector stop from N scalars
            return body
        # scalar / spectrum:-ref colour channel: whitespace-separated tokens (current-FTSL)
        toks: List[str] = []
        for s in ch.stops:
            if s.pinned:
                toks.append(f"p:{fmt(s.pos)}")
            toks.append(s.token)
        return "  ".join(toks)

    # -- lowering (inline colour -> spectrum:-ref, for ftrace) ----------------

    @property
    def has_inline_colour(self) -> bool:
        """True if any channel is an inline ``rgb``/``hsv``/``hsl`` colour channel."""
        return any(ch.is_inline_colour for ch in self.channels)

    @staticmethod
    def _to_rgb(space: str, comps: Sequence[float]) -> Tuple[float, float, float]:
        """Convert one static arity-3 colour stop to linear RGB (via loom's own
        conversion so there is a single source of truth for the hue maths)."""
        r, g, b = float(comps[0]), float(comps[1]), float(comps[2])
        if space == "rgb":
            return (r, g, b)
        from .color import hsv_to_rgb, hsl_to_rgb
        from .signals.core import Clock
        vec = (hsv_to_rgb if space == "hsv" else hsl_to_rgb)(r, g, b)
        out = vec.at(Clock(t=0.0))
        return (float(out[0]), float(out[1]), float(out[2]))

    def lower_colours(self, *, prefix: Optional[str] = None
                      ) -> Tuple[List[str], "Record"]:
        """Lower every inline ``rgb``/``hsv``/``hsl`` colour channel to a
        ``spectrum:``-ref channel ftrace can parse.

        Returns ``(decls, lowered_record)``: ``decls`` is a list of top-level
        ``spectrum "<name>" = rgb r g b`` declaration strings (one per **unique**
        colour, deduped across the whole record; ``hsv``/``hsl`` components are
        converted to rgb), and ``lowered_record`` is a copy whose inline-colour
        channels now reference those spectra by name.  Scalar / vector /
        ``spectrum:``-ref channels pass through unchanged.  Raises
        :class:`TypeError` on an inline-colour channel with expression stops (not
        constant-lowerable — needs the pattern VM, J3c)."""
        base = prefix if prefix is not None else self.name
        decls: List[str] = []
        name_of: dict = {}                          # rgb key -> spectrum name
        new_channels: List[RecordChannel] = []
        for ch in self.channels:
            if not ch.is_inline_colour:
                new_channels.append(RecordChannel(
                    ch.name, [RecordStop(list(s.components) if s.arity > 1
                                         else s.components[0], s.pos)
                              for s in ch.stops], ch.space))
                continue
            if not ch.is_numeric:
                raise TypeError(
                    f"record {self.name!r} channel {ch.name!r}: inline-colour channel "
                    "has expression stops — can't lower to constant spectra (needs the "
                    "pattern VM, J3c)")
            new_stops: List[RecordStop] = []
            for s in ch.stops:
                rgb = self._to_rgb(ch.space, s.as_vector())
                key = tuple(round(c, 6) for c in rgb)
                spec_name = name_of.get(key)
                if spec_name is None:
                    spec_name = f"{base}_c{len(name_of)}"
                    name_of[key] = spec_name
                    decls.append(
                        f'spectrum "{spec_name}" = rgb {fmt(rgb[0])} {fmt(rgb[1])} '
                        f'{fmt(rgb[2])}')
                new_stops.append(RecordStop(f"spectrum:{spec_name}", s.pos))
            new_channels.append(RecordChannel(ch.name, new_stops))
        lowered = Record(self.name, self.lo, self.hi, new_channels, interp=self.interp)
        return decls, lowered

    def lower_ftsl(self, *, prefix: Optional[str] = None) -> str:
        """Convenience: the lowered record's ``.ftsl`` text with its synthesized
        ``spectrum`` decls prepended (a self-contained block ftrace can parse, as long
        as the record has no loom-only *vector* channel)."""
        decls, lowered = self.lower_colours(prefix=prefix)
        body = lowered.emit()
        return ("\n".join(decls) + "\n\n" + body) if decls else body

    # -- parse ---------------------------------------------------------------

    _HEADER = re.compile(
        r"(?P<name>\w+)\s*=\s*range\s+(?P<dom>[^\[]+?)\s*\[", re.DOTALL)

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Blank out ``#…`` comments (keeping newlines so offsets/lines are stable)."""
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    @staticmethod
    def _parse_domain(words: Sequence[str]) -> Tuple[float, float]:
        """Mirror ftrace's ``parseRecordDomain``: ``LO-HI`` or ``LO HI``."""
        if len(words) == 2 and _is_number(words[0]) and _is_number(words[1]):
            lo, hi = float(words[0]), float(words[1])
            if hi > lo:
                return lo, hi
        if len(words) == 1:
            s = words[0]
            for k in range(1, len(s)):
                if s[k] != "-":
                    continue
                p = s[k - 1]
                if p in ("e", "E", "+", "-"):
                    continue
                a, b = s[:k], s[k + 1:]
                if _is_number(a) and _is_number(b):
                    lo, hi = float(a), float(b)
                    if hi > lo:
                        return lo, hi
        raise ValueError(f"bad record range {' '.join(words)!r} (need LO-HI or LO HI, HI>LO)")

    @classmethod
    def parse(cls, text: str) -> "Record":
        """Parse a single ``NAME = range LO-HI [ … ]`` block back into a Record.

        One backward-compatible ladder grammar: each channel line is dispatched on
        whether it contains a top-level comma.  A comma-free line uses the current-FTSL
        whitespace-stop path (``rough  0 0 0`` = three scalar stops); a comma line uses
        the generalized ladder path (``tint  0 0 0, 1 1 1`` = two arity-3 vector stops),
        with a trailing comma marking a lone vector stop (``tint  0 0 0,``).  A leading
        ``rgb``/``hsv``/``hsl`` word is an **inline-colour tag**: it fixes arity 3, so
        each comma-group is one colour stop (``reflect  rgb 0 0 0, 1 1 1`` = two rgb
        stops; a lone ``reflect  rgb .5 .5 .5`` needs no trailing comma)."""
        text = cls._strip_comments(text)
        m = cls._HEADER.search(text)
        if not m:
            raise ValueError("not a record declaration (expected `NAME = range LO-HI [`)")
        name = m.group("name")
        dom_words = m.group("dom").split()
        lo, hi = cls._parse_domain(dom_words)
        # body = between the header '[' and the first ']'
        body_start = m.end()
        close = text.find("]", body_start)
        if close < 0:
            raise ValueError(f"record {name!r}: missing closing ']'")
        body = text[body_start:close]

        interp = "linear"
        chans: List[RecordChannel] = []
        for raw_line in body.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            key = parts[0]
            value_text = parts[1].strip() if len(parts) > 1 else ""
            if key == "interp":
                if value_text not in _INTERP:
                    raise ValueError(
                        f"record {name!r}: interp must be one of {_INTERP}")
                interp = value_text
                continue
            if not value_text:
                raise ValueError(f"record {name!r} channel {key!r}: has no stops")
            # inline-colour tag: a leading rgb/hsv/hsl word fixes arity 3, so every
            # comma-group (or the lone group) is one colour stop via the comma path.
            space: Optional[str] = None
            tag_split = value_text.split(None, 1)
            if tag_split[0] in _COLOUR_SPACES:
                space = tag_split[0]
                value_text = tag_split[1].strip() if len(tag_split) > 1 else ""
                if not value_text:
                    raise ValueError(
                        f"record {name!r} channel {key!r}: {space} tag with no stops")
                comma_parts = cls._split_top_commas(value_text)
                if comma_parts and comma_parts[-1] == "" and len(comma_parts) > 1:
                    comma_parts = comma_parts[:-1]     # tolerate a trailing comma
                stops = cls._parse_comma_stops(name, key, comma_parts)
                chans.append(RecordChannel(key, stops, space))
                continue
            comma_parts = cls._split_top_commas(value_text)
            # a trailing top-level comma forces the comma path (its empty tail is the
            # disambiguator for a lone vector stop) — drop that empty tail.
            trailing = len(comma_parts) > 1 and comma_parts[-1] == ""
            if trailing:
                comma_parts = comma_parts[:-1]
            if len(comma_parts) > 1 or trailing:
                stops = cls._parse_comma_stops(name, key, comma_parts)
            else:
                stops = cls._parse_ws_stops(name, key, value_text)
            chans.append(RecordChannel(key, stops))
        return cls(name, lo, hi, chans, interp=interp)

    @staticmethod
    def _parse_ws_stops(name: str, key: str, value_text: str) -> List[RecordStop]:
        """Current-FTSL whitespace path: every word is a scalar stop; a ``p:<pos>``
        word pins the stop that follows it."""
        stops: List[RecordStop] = []
        pin: Optional[float] = None
        for w in value_text.split():
            if w.startswith("p:"):
                pv = w[2:]
                if not _is_number(pv):
                    raise ValueError(
                        f"record {name!r} channel {key!r}: bad p:<pos> {w!r}")
                pin = float(pv)
                continue
            stops.append(RecordStop(w, pin))
            pin = None
        if pin is not None:
            raise ValueError(
                f"record {name!r} channel {key!r}: trailing p:<pos> with no value")
        if not stops:
            raise ValueError(f"record {name!r} channel {key!r}: has no stops")
        return stops

    @classmethod
    def _parse_comma_stops(cls, name: str, key: str,
                           chunks: Sequence[str]) -> List[RecordStop]:
        """Generalized comma path: each chunk is one stop (a leading ``p:<pos>`` pins
        it), ladder-parsed into a scalar or a flat vector (:mod:`loom.ladder`)."""
        stops: List[RecordStop] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                raise ValueError(
                    f"record {name!r} channel {key!r}: empty stop (stray comma?)")
            pin: Optional[float] = None
            words = chunk.split(maxsplit=1)
            if words and words[0].startswith("p:"):
                pv = words[0][2:]
                if not _is_number(pv):
                    raise ValueError(
                        f"record {name!r} channel {key!r}: bad p:<pos> {words[0]!r}")
                pin = float(pv)
                chunk = words[1] if len(words) > 1 else ""
                if not chunk.strip():
                    raise ValueError(
                        f"record {name!r} channel {key!r}: p:<pos> with no value")
            stops.append(cls._stop_from_ladder(parse_ladder(chunk), pin))
        if not stops:
            raise ValueError(f"record {name!r} channel {key!r}: has no stops")
        return stops

    @staticmethod
    def _split_top_commas(s: str) -> List[str]:
        """Split ``s`` on top-level commas, ignoring commas inside ``[…]``/``(…)``."""
        parts: List[str] = []
        buf: List[str] = []
        depth = 0
        for c in s:
            if c in "[(":
                depth += 1
                buf.append(c)
            elif c in "])":
                depth = max(0, depth - 1)
                buf.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(buf))
                buf.clear()
            else:
                buf.append(c)
        parts.append("".join(buf))
        return [p.strip() for p in parts]

    @staticmethod
    def _stop_from_ladder(v: Value, pin: Optional[float]) -> RecordStop:
        """Build a stop from a parsed ladder value (a leaf or a flat vector)."""
        if isinstance(v, str):
            return RecordStop(v, pin)
        if all(isinstance(c, str) for c in v):
            return RecordStop(list(v), pin)
        raise ValueError(
            "record stop: a single stop must be a scalar or a flat vector "
            f"(nested stop value {v!r} not supported)")

    @classmethod
    def parse_all(cls, text: str) -> List["Record"]:
        """Parse every ``NAME = range … [ … ]`` block found in a larger ``.ftsl`` text."""
        text = cls._strip_comments(text)
        out: List["Record"] = []
        for m in cls._HEADER.finditer(text):
            close = text.find("]", m.end())
            if close < 0:
                raise ValueError(
                    f"record {m.group('name')!r}: missing closing ']'")
            out.append(cls.parse(text[m.start():close + 1]))
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"Record(name={self.name!r}, range={self.lo}-{self.hi}, "
                f"interp={self.interp!r}, channels={[c.name for c in self.channels]})")
