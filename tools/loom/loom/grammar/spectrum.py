"""The ``.ftsl`` spectral-value grammar — a faithful reference of ftrace's
``evalSpectrum`` (``src/ftsl.h``).

Material and light *spectral* fields (``reflect`` / ``emit`` / ``transmit`` /
``absorb`` / ``ior`` / ``substrate_k`` / ``spd``) do not take a bare colour — they
take a **spectrum expression**, a richer grammar of which the colour-vector syntax
(:mod:`loom.grammar.values`) is only the ``rgb`` / ``hsv`` / ``hsl`` subset.  This
module parses that grammar to a small canonical tree and is the single source of
truth mirrored by ftrace's hand-written ``evalSpectrum`` today (and, later, by the
shared C++ front-end at the J3c port).

The accepted forms — exactly ftrace's ``evalSpectrum`` (``src/ftsl.h`` ~1106) — are:

* a bare **number** → a constant grey spectrum (``0.75``);
* **`blackbody [K]`** (default 6500), **`ior [n]`** (default 1.5),
  **`whitewall [r]`** (default 0.75);
* the bareword walls **`redwall`** / **`greenwall`**;
* the parametric bands **`gaussian center=.. sigma=.. amp=..`** and
  **`shortpass edge=.. slope=.. amp=..`** (``key=value`` words, any order, all
  optional);
* a **colour** — **`rgb r g b`** / **`hsv h s v`** / **`hsl h s l`** (delegated to
  :func:`loom.grammar.values.as_color`, so a bracketed / comma colour also parses);
* the dominant-wavelength emission heads **`rgbline r g b [sigma]`** /
  **`hsvline …`** / **`hslline …`** → a narrow spectral line at the colour's dominant
  wavelength (:class:`LineSpec`);
* a library **reference** — ``glass:`` / ``metal:`` / ``reflectance:`` / ``filter:``
  / ``preset:`` / ``file:`` / ``spectrum:`` followed by a name / path;
* a **record channel reference** used as a constant — ``RECORD.channel[i]`` or
  ``RECORD.channel(driver)`` (the syntactic form is accepted here; whether it is a
  compile-time constant is a later, record-aware check).

A bare colour with **no** ``rgb`` / ``hsv`` / ``hsl`` tag (e.g. ``0.8 0.8 0.8``) is
**not** a spectrum — ftrace rejects it (``unrecognized spectrum expression '0.8'``),
so we raise :class:`~loom.grammar.values.ShapeError` to match.  ``table { λ:v … }``
is the one *block* form and is handled where blocks are parsed, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .values import ShapeError, as_color

# The colorspace-tag heads and library-reference prefixes ftrace recognizes.
_COLOR_HEADS = ("rgb", "hsv", "hsl")
# The dominant-wavelength emission heads (K3): `rgbline r g b [sigma]`, etc.
_LINE_HEADS = {"rgbline": "rgb", "hsvline": "hsv", "hslline": "hsl"}
_LIB_PREFIXES = ("glass:", "metal:", "reflectance:", "filter:", "preset:",
                 "file:", "spectrum:")

# A record channel reference used as a constant value: `RECORD.channel`, optionally
# indexed `[i]` or driven `(expr)`.  Matched as a whole single word.
_RECORD_RE = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*(\[.*\]|\(.*\))?$")


# ---- canonical spectrum-expression nodes ----------------------------------

@dataclass(frozen=True)
class Const:
    """A bare number → a constant grey spectrum."""
    value: float


@dataclass(frozen=True)
class Blackbody:
    """``blackbody [K]`` — a Planckian emitter at ``kelvin`` (default 6500)."""
    kelvin: float = 6500.0


@dataclass(frozen=True)
class Ior:
    """``ior [n]`` — a constant index of refraction (default 1.5)."""
    n: float = 1.5


@dataclass(frozen=True)
class WhiteWall:
    """``whitewall [r]`` — a flat grey reflectance (default 0.75)."""
    r: float = 0.75


@dataclass(frozen=True)
class NamedWall:
    """The bareword Cornell walls ``redwall`` / ``greenwall``."""
    name: str


@dataclass(frozen=True)
class Band:
    """A parametric band: ``gaussian`` (center/sigma/amp) or ``shortpass``
    (edge/slope/amp).  Missing keys default (a=0, b=0, amp=1) exactly as ftrace."""
    kind: str                      # "gaussian" | "shortpass"
    a: float = 0.0                 # center / edge
    b: float = 0.0                 # sigma / slope
    amp: float = 1.0


@dataclass(frozen=True)
class ColorSpec:
    """A tagged colour ``rgb`` / ``hsv`` / ``hsl`` upsampled to a reflectance."""
    space: str
    comps: Tuple[float, float, float]


@dataclass(frozen=True)
class LineSpec:
    """The dominant-wavelength *emission* form ``rgbline r g b [sigma]`` (and
    ``hsvline``/``hslline``): a narrow Gaussian at the colour's dominant wavelength
    (ftrace's ``rgbToLineEmission`` / K3).  ``sigma`` (nm) is the optional forced line
    width; ``None`` = derive it from the colour's saturation.  A *head keyword* (not a
    trailing ``line`` modifier) because ftrace's parser ends a value at the next
    bareword, so a trailing word would be dropped."""
    space: str
    comps: Tuple[float, float, float]
    sigma: float | None = None


@dataclass(frozen=True)
class LibRef:
    """A library reference ``<prefix>:<name>`` (``metal:gold``, ``preset:d65``,
    ``file:curve.csv``, ``spectrum:steel``, …).  ``kind`` is the prefix without
    its colon."""
    kind: str
    name: str


@dataclass(frozen=True)
class RecordRef:
    """A record channel reference used as a value (``R.chan[i]`` / ``R.chan(t)``),
    kept verbatim; a later record-aware pass resolves / constant-checks it."""
    text: str


# ---- parse ----------------------------------------------------------------

def _num(tok: str) -> float:
    return float(tok)


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _band(kind: str, words) -> Band:
    """``gaussian`` / ``shortpass`` key=value words → a :class:`Band` (ftrace's
    ``splitEq`` loop: unknown / malformed words are skipped, keys default)."""
    a, b, amp = 0.0, 0.0, 1.0
    for w in words:
        if "=" not in w:
            continue
        key, _, val = w.partition("=")
        if not _is_number(val):
            continue
        x = float(val)
        if key in ("center", "edge"):
            a = x
        elif key in ("sigma", "slope"):
            b = x
        elif key == "amp":
            amp = x
    return Band(kind, a, b, amp)


def parse_spectrum(text: str):
    """Parse a bare ``.ftsl`` spectrum expression to its canonical node.

    Raises :class:`~loom.grammar.values.ShapeError` for a value that is well-formed
    text but not a spectrum expression (mirrors ftrace's ``fail`` on an unrecognized
    head), matching the ``as_*`` shape-validators' error type.
    """
    words = text.split()
    if not words:
        raise ShapeError("empty spectrum expression")
    head = words[0]

    # ---- single-word forms (checked before keyword heads) ----
    if len(words) == 1:
        if _is_number(head):
            return Const(_num(head))
        for pre in _LIB_PREFIXES:
            if head.startswith(pre):
                return LibRef(pre[:-1], head[len(pre):])
        if _RECORD_RE.match(head):
            return RecordRef(head)

    # ---- keyword-headed forms ----
    if head == "blackbody":
        return Blackbody(_num(words[1]) if len(words) > 1 else 6500.0)
    if head == "ior":
        return Ior(_num(words[1]) if len(words) > 1 else 1.5)
    if head == "whitewall":
        return WhiteWall(_num(words[1]) if len(words) > 1 else 0.75)
    if head == "redwall":
        return NamedWall("redwall")
    if head == "greenwall":
        return NamedWall("greenwall")
    if head in ("gaussian", "shortpass"):
        return _band(head, words[1:])
    if head in _LINE_HEADS:
        # `rgbline r g b [sigma]` (hsvline/hslline) → dominant-wavelength emission.
        # The 3 colour components (+ optional sigma) follow as plain numbers.
        space = _LINE_HEADS[head]
        nums = [w for w in words[1:] if _is_number(w)]
        if len(nums) < 3:
            raise ShapeError(f"{head} needs 3 components")
        _sp, comps = as_color(space + " " + " ".join(nums[:3]), default_space=space)
        sigma = _num(nums[3]) if len(nums) > 3 else None
        return LineSpec(space, comps, sigma)
    if head in _COLOR_HEADS:
        space, comps = as_color(text, default_space=head)
        return ColorSpec(space, comps)

    # A multi-word library prefix (e.g. a stray `metal: gold`) or anything else is
    # not a recognized spectrum expression — same failure ftrace raises.
    raise ShapeError(
        f"unrecognized spectrum expression '{head}'"
        + ("" if len(words) == 1 else f" (in '{text.strip()}')"))


def as_spectrum(value: str):
    """Validate ``value`` as a spectrum expression, returning its canonical node.

    Thin wrapper over :func:`parse_spectrum` giving spectral fields the same
    ``as_*``-validator surface the colour-vector fields have."""
    return parse_spectrum(value)
