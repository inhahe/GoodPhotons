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

* a **channel** outputs a value of some arity ``D``.  ftrace materializes exactly
  two: a *scalar* channel (``D==1``; every stop is a numeric literal or a pattern
  expression) and a *colour* channel (``D==3``; every stop is a ``spectrum:<name>``
  / ``metal:<name>`` / ``rgb:<…>`` ref, interpolated in linear-RGB → Jakob–Hanika).
  A channel must be homogeneous (all-scalar or all-colour).  Higher arities are the
  loom-only superset (J3b) and are not emitted here.
* a **stop** carries its raw ``token`` (preserved verbatim for faithful re-emit) and
  an optional pinned domain position (author ``p:<pos>`` prefix).  Unpinned stops are
  spread evenly between their pinned/anchor neighbours exactly as ftrace does.

What this module does **not** do (deferred to J3c's full pattern VM): evaluate
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
    """One stop in a channel LUT.

    ``token`` is the raw authored value word (a numeric literal, a scalar pattern
    expression, or a colour ref containing ``':'``) — kept verbatim so a parsed
    record re-emits equivalently.  ``pos`` is the pinned domain position when the
    author wrote a ``p:<pos>`` prefix, else ``None`` (redistributed on demand).
    """

    token: str
    pos: Optional[float] = None

    @property
    def pinned(self) -> bool:
        return self.pos is not None

    @property
    def is_colour(self) -> bool:
        # ftrace: a stop is a colour ref iff its token contains ':'
        return ":" in self.token

    def as_number(self) -> float:
        if not _is_number(self.token):
            raise TypeError(f"stop {self.token!r} is not a plain numeric literal")
        return float(self.token)


@dataclass
class RecordChannel:
    """A named LUT: ``channelname stop stop …`` (auto-bound to a like-named slot)."""

    name: str
    stops: List[RecordStop] = field(default_factory=list)

    @property
    def kind(self) -> str:
        """``"colour"`` if any stop is a colour ref, else ``"scalar"``."""
        colour = any(s.is_colour for s in self.stops)
        scalar = any(not s.is_colour for s in self.stops)
        if colour and scalar:
            raise ValueError(
                f"record channel {self.name!r} mixes colour (spectrum:…) and scalar stops"
            )
        return "colour" if colour else "scalar"

    @property
    def is_numeric(self) -> bool:
        """True when every stop is a plain numeric literal (sampler-eligible)."""
        return bool(self.stops) and all(_is_number(s.token) for s in self.stops)


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

# One channel line as authored: (name, [(token, pinned_pos_or_None), …]).
ChannelSpec = Tuple[str, Sequence[Union[str, Tuple[str, Optional[float]]]]]


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
        """Build from ``(chan_name, [token | (token, pos), …])`` tuples."""
        chans: List[RecordChannel] = []
        for cname, raw in channels:
            stops: List[RecordStop] = []
            for item in raw:
                if isinstance(item, tuple):
                    tok, pos = item
                    stops.append(RecordStop(str(tok), None if pos is None else float(pos)))
                else:
                    stops.append(RecordStop(str(item)))
            chans.append(RecordChannel(str(cname), stops))
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
        """Sample an all-numeric scalar channel at driver ``d``.

        Raises :class:`TypeError` for colour channels or channels with expression
        stops (those need the pattern VM; deferred to J3c)."""
        ch = self.channel(name)
        if ch.kind == "colour":
            raise TypeError(
                f"channel {name!r} is a colour channel — not numerically sampleable")
        if not ch.is_numeric:
            raise TypeError(
                f"channel {name!r} has expression stops — sampling needs the pattern VM (J3c)")
        pos = _redistribute(ch.stops, self.lo, self.hi)
        vals = [s.as_number() for s in ch.stops]
        return _sample_numeric(pos, vals, d, self.interp)

    # -- emit ----------------------------------------------------------------

    def roots(self) -> List:
        return []

    def emit(self, ctx: Optional[EmitCtx] = None) -> str:
        # range: prefer the compact `LO-HI` when lo >= 0 (unambiguous), else `LO HI`.
        if self.lo >= 0:
            dom = f"{fmt(self.lo)}-{fmt(self.hi)}"
        else:
            dom = f"{fmt(self.lo)} {fmt(self.hi)}"
        # pad channel names (+ the interp keyword) to a common width for tidy columns
        names = [ch.name for ch in self.channels]
        width = max([len(n) for n in names] + [len("interp")])
        lines = [f"{self.name} = range {dom} ["]
        for ch in self.channels:
            toks: List[str] = []
            for s in ch.stops:
                if s.pinned:
                    toks.append(f"p:{fmt(s.pos)}")
                toks.append(s.token)
            lines.append(f"    {ch.name.ljust(width)}  " + "  ".join(toks))
        if self.interp != "linear":
            lines.append(f"    {'interp'.ljust(width)}  {self.interp}")
        lines.append("]")
        return "\n".join(lines)

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
        """Parse a single ``NAME = range LO-HI [ … ]`` block back into a Record."""
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
            words = line.split()
            key, rest = words[0], words[1:]
            if key == "interp":
                if not rest or rest[0] not in _INTERP:
                    raise ValueError(
                        f"record {name!r}: interp must be one of {_INTERP}")
                interp = rest[0]
                continue
            stops: List[RecordStop] = []
            pin: Optional[float] = None
            for w in rest:
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
            chans.append(RecordChannel(key, stops))
        return cls(name, lo, hi, chans, interp=interp)

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
