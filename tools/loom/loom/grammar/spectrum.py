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
* the Jakob-Hanika *illuminant* heads **`rgbillum r g b`** / **`hsvillum …`** /
  **`hslillum …`** → a smooth full-spectrum *emission* SPD reproducing the colour under
  the bare observer (:class:`IllumSpec`), the emitter analogue of ``rgb``;
* the Smits 1999 reflectance heads **`rgbsmits r g b`** / **`hsvsmits …`** /
  **`hslsmits …`** → the colour upsampled via the classic tabulated Smits basis
  (:class:`SmitsSpec`), a selectable lower-fidelity alternative to ``rgb``;
* the plain 3-box reflectance heads **`rgbbox r g b`** / **`hsvbox …`** /
  **`hslbox …`** → the colour upsampled to three calibrated rectangular bands
  (:class:`BoxSpec`), the cheapest selectable alternative to ``rgb``;
* the Meng 2015 smoothest-spectrum heads **`rgbmeng r g b`** / **`hsvmeng …`** /
  **`hslmeng …`** → the colour upsampled to the *smoothest* reflectance realising
  it (:class:`MengSpec`), the highest-fidelity selectable alternative to ``rgb``;
* the **user-declared** upsampler heads **`rgb:<name> r g b`** / **`hsv:<name> …`** /
  **`hsl:<name> …`** → the colour upsampled by a scene-declared
  ``upsample "<name>" { expr "f(r,g,b,w)" }`` block (:class:`UserSpec`), the open-ended
  counterpart to the five built-in upsamplers above;
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
# The Jakob-Hanika illuminant (full-spectrum emission) heads (K1): `rgbillum r g b`, etc.
_ILLUM_HEADS = {"rgbillum": "rgb", "hsvillum": "hsv", "hslillum": "hsl"}
# The Smits 1999 reflectance upsampler heads (K1): `rgbsmits r g b`, etc.
_SMITS_HEADS = {"rgbsmits": "rgb", "hsvsmits": "hsv", "hslsmits": "hsl"}
# The plain calibrated 3-box reflectance upsampler heads (K1): `rgbbox r g b`, etc.
_BOX_HEADS = {"rgbbox": "rgb", "hsvbox": "hsv", "hslbox": "hsl"}
# The Meng 2015 smoothest-spectrum upsampler heads (K1): `rgbmeng r g b`, etc.
_MENG_HEADS = {"rgbmeng": "rgb", "hsvmeng": "hsv", "hslmeng": "hsl"}
_LIB_PREFIXES = ("glass:", "metal:", "reflectance:", "filter:", "preset:",
                 "file:", "spectrum:")
# A USER-declared upsampler head (K1): `<space>:<name> r g b`, naming an
# `upsample "<name>"` block.  A colon rather than yet another glued suffix because the
# built-in suffixes are a closed set a reader can memorise while a user name is
# open-ended — mirrors ftrace's `isCustomColourHead` (src/ftsl.h), which is likewise
# SHAPE-only: whether the name resolves is a scene-level question, not a grammar one.
_USER_HEAD_RE = re.compile(r"^(rgb|hsv|hsl):(\w+)$")

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
class IllumSpec:
    """The Jakob-Hanika *illuminant* emission form ``rgbillum r g b`` (and
    ``hsvillum``/``hslillum``): a smooth, full-spectrum emission SPD (``A·sigmoid``)
    whose integral under the bare CIE observer reproduces the colour — ftrace's
    ``rgbToIlluminantJH`` / K1, the emitter analogue of :class:`ColorSpec`.  A *head
    keyword* (not a trailing modifier) for the same parser reason as :class:`LineSpec`."""
    space: str
    comps: Tuple[float, float, float]


@dataclass(frozen=True)
class SmitsSpec:
    """The Smits 1999 reflectance form ``rgbsmits r g b`` (and ``hsvsmits``/
    ``hslsmits``): the colour upsampled to a reflectance via ftrace's classic
    tabulated Smits basis (``rgbToReflectanceSmits`` / K1) — a selectable,
    lower-fidelity alternative to the default Jakob-Hanika :class:`ColorSpec`.  A
    *head keyword* (not a trailing modifier) for the same parser reason as
    :class:`LineSpec`."""
    space: str
    comps: Tuple[float, float, float]


@dataclass(frozen=True)
class BoxSpec:
    """The plain 3-box reflectance form ``rgbbox r g b`` (and ``hsvbox``/
    ``hslbox``): the colour upsampled to three calibrated rectangular reflectance
    bands (ftrace's ``rgbToReflectanceBox`` / K1) — the cheapest selectable
    alternative to the default Jakob-Hanika :class:`ColorSpec`.  A *head keyword*
    (not a trailing modifier) for the same parser reason as :class:`LineSpec`."""
    space: str
    comps: Tuple[float, float, float]


@dataclass(frozen=True)
class MengSpec:
    """The Meng 2015 form ``rgbmeng r g b`` (and ``hsvmeng``/``hslmeng``): the
    colour upsampled to the *smoothest* reflectance realising it — the minimum
    of sum (s[i+1]-s[i])^2 over all physical reflectances of that colour, read
    from ftrace's baked table (``rgbToReflectanceMeng`` / K1).  The
    highest-fidelity selectable alternative to the default Jakob-Hanika
    :class:`ColorSpec`, and the one to prefer when a reflectance will be
    re-illuminated by a strongly non-D65 light or dispersed.  A *head keyword*
    (not a trailing modifier) for the same parser reason as :class:`LineSpec`."""
    space: str
    comps: Tuple[float, float, float]


@dataclass(frozen=True)
class UserSpec:
    """A USER-declared upsampler: ``rgb:<name> r g b`` (and ``hsv:``/``hsl:``), naming
    an ``upsample "<name>" { expr "f(r,g,b,w)" }`` block in the same scene — ftrace's
    ``applyUpsample`` / K1.  The last of the K1 upsamplers and the open-ended one: the
    other five name a *built-in* fit, this one names a scene-supplied function of the
    colour and the wavelength, so a scene can plug in e.g. a measured three-spectrum
    basis (``r*spec:red(w) + g*spec:green(w) + b*spec:blue(w)``).

    ``space`` is the colour space of the *written* triple; ftrace converts to linear
    sRGB **before** calling the body, so all three heads feed the same ``r, g, b`` and
    an upsampler never has to know which was used.  ``name`` is kept verbatim and is
    NOT resolved here — mirroring ftrace, which likewise checks only the shape at the
    grammar level and reports an unknown upsampler from the loader."""
    space: str
    name: str
    comps: Tuple[float, float, float]


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
    if head in _ILLUM_HEADS:
        # `rgbillum r g b` (hsvillum/hslillum) → full-spectrum illuminant emission.
        space = _ILLUM_HEADS[head]
        _sp, comps = as_color(space + " " + " ".join(words[1:]), default_space=space)
        return IllumSpec(space, comps)
    if head in _SMITS_HEADS:
        # `rgbsmits r g b` (hsvsmits/hslsmits) → Smits 1999 reflectance upsample.
        space = _SMITS_HEADS[head]
        _sp, comps = as_color(space + " " + " ".join(words[1:]), default_space=space)
        return SmitsSpec(space, comps)
    if head in _BOX_HEADS:
        # `rgbbox r g b` (hsvbox/hslbox) → plain calibrated 3-box reflectance.
        space = _BOX_HEADS[head]
        _sp, comps = as_color(space + " " + " ".join(words[1:]), default_space=space)
        return BoxSpec(space, comps)
    if head in _MENG_HEADS:
        # `rgbmeng r g b` (hsvmeng/hslmeng) → Meng 2015 smoothest-spectrum upsample.
        space = _MENG_HEADS[head]
        _sp, comps = as_color(space + " " + " ".join(words[1:]), default_space=space)
        return MengSpec(space, comps)
    m = _USER_HEAD_RE.match(head)
    if m:
        # `rgb:<name> r g b` (hsv:/hsl:) → a scene-declared `upsample` block.  Checked
        # after every built-in head so a future built-in can never be shadowed by a
        # user name — the built-ins are glued suffixes and these are colon-separated,
        # so the two spellings cannot collide, but the ordering makes that explicit.
        space, name = m.group(1), m.group(2)
        _sp, comps = as_color(space + " " + " ".join(words[1:]), default_space=space)
        return UserSpec(space, name, comps)
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
