"""
Loom colour model — RGB, HSV *and* HSL, as first-class **animatable DAG nodes**.

A :class:`Color` is a 3-component :class:`~loom.signals.vector.VecSignal`, so it
drops into every place Loom already takes a colour with **no special casing**:

- **2-D graphics** (:mod:`loom.canvas`) — a marker / stroke / background colour;
- **3-D graphics** (:mod:`loom.scene`) — a material colour / spectral param;
- and, because it remembers *how it was authored*, it emits the matching **``.ftsl``
  colour token** (``rgb r g b`` / ``hsv h s v`` / ``hsl h s l``) so a surface's
  colour round-trips to the renderer in whichever model you wrote it in.

Both cylindrical models are kept because they answer different questions and both
are standard: **HSV** ("value") matches painterly colour pickers (V=1 is the most
vivid), while **HSL** ("lightness") is symmetric about a neutral grey (L=0.5 is the
pure hue, L→1 white, L→0 black) and matches CSS.  They share the same hue wheel.

Authoring is any model, interchangeably::

    Color.rgb(0.9, 0.4, 0.1)          # or  rgb(0.9, 0.4, 0.1)
    Color.hsv(Sine(cycles=1)*0.5+0.5, 0.8, 1.0)   # or  hsv(...)
    Color.hsl(0.6, 0.7, 0.5)          # or  hsl(...)

Every channel may be a :class:`~loom.signals.core.Signal` or number, so a colour
animates like anything else — and because **hue is in ``[0, 1]`` and wraps**, a hue
driven by a 1-periodic leaf cycles the whole wheel and returns bit-for-bit at the
loop seam (seamless colour cycling).  ``s``/``v``/``l`` are clamped to ``[0, 1]``.

The conversion is done in the graph (:func:`hsv_to_rgb` / :func:`hsl_to_rgb` and
their inverses), so an HSV/HSL colour is a real RGB node downstream — a
:class:`Color` *is* its resolved RGB, while separately remembering its authored
channels for token emission.
"""

from __future__ import annotations

import math
from typing import Optional, Union

from .signals.core import Signal, Clock, Cache, as_signal, Number
from .signals.vector import VecSignal, Vecish


# ---------------------------------------------------------------------------
# per-channel conversion nodes (pure functions of a clock, cycle-checked)
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class _HsvToRgbChannel(Signal):
    """One RGB channel (``ch`` 0/1/2) of an HSV triple.  ``h`` wraps in ``[0,1)``."""

    def __init__(self, h: Signal, s: Signal, v: Signal, ch: int) -> None:
        super().__init__()
        self.h = as_signal(h)
        self.s = as_signal(s)
        self.v = as_signal(v)
        self.ch = int(ch)

    def children(self):
        return (self.h, self.s, self.v)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        h = self.h.at(clock, cache)
        s = _clamp01(self.s.at(clock, cache))
        v = _clamp01(self.v.at(clock, cache))
        h -= math.floor(h)                      # wrap hue into [0, 1)
        x = h * 6.0
        i = int(math.floor(x)) % 6
        f = x - math.floor(x)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        rgb = ((v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q))[i]
        return rgb[self.ch]


class _RgbToHsvChannel(Signal):
    """One HSV channel (``ch`` 0=h/1=s/2=v) of an RGB triple.  ``h`` in ``[0,1)``."""

    def __init__(self, r: Signal, g: Signal, b: Signal, ch: int) -> None:
        super().__init__()
        self.r = as_signal(r)
        self.g = as_signal(g)
        self.b = as_signal(b)
        self.ch = int(ch)

    def children(self):
        return (self.r, self.g, self.b)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        r = _clamp01(self.r.at(clock, cache))
        g = _clamp01(self.g.at(clock, cache))
        b = _clamp01(self.b.at(clock, cache))
        mx = max(r, g, b)
        mn = min(r, g, b)
        d = mx - mn
        if self.ch == 2:
            return mx
        if self.ch == 1:
            return 0.0 if mx <= 0.0 else d / mx
        # hue
        if d <= 0.0:
            return 0.0
        if mx == r:
            h = ((g - b) / d) % 6.0
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        return (h / 6.0) % 1.0


def hsv_to_rgb(h: Union[Signal, Number], s: Union[Signal, Number],
               v: Union[Signal, Number]) -> VecSignal:
    """HSV channels → an animatable RGB :class:`VecSignal` (hue in ``[0,1]``, wraps)."""
    hs, ss, vs = as_signal(h), as_signal(s), as_signal(v)
    return VecSignal([_HsvToRgbChannel(hs, ss, vs, ch) for ch in range(3)])


def rgb_to_hsv(r: Union[Signal, Number], g: Union[Signal, Number],
               b: Union[Signal, Number]) -> VecSignal:
    """RGB channels → an animatable HSV :class:`VecSignal` (hue in ``[0,1)``)."""
    rs, gs, bs = as_signal(r), as_signal(g), as_signal(b)
    return VecSignal([_RgbToHsvChannel(rs, gs, bs, ch) for ch in range(3)])


class _HslToRgbChannel(Signal):
    """One RGB channel (``ch`` 0/1/2) of an HSL triple.  ``h`` wraps in ``[0,1)``."""

    def __init__(self, h: Signal, s: Signal, l: Signal, ch: int) -> None:
        super().__init__()
        self.h = as_signal(h)
        self.s = as_signal(s)
        self.l = as_signal(l)
        self.ch = int(ch)

    def children(self):
        return (self.h, self.s, self.l)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        h = self.h.at(clock, cache)
        s = _clamp01(self.s.at(clock, cache))
        ll = _clamp01(self.l.at(clock, cache))
        h -= math.floor(h)                      # wrap hue into [0, 1)
        c = (1.0 - abs(2.0 * ll - 1.0)) * s     # chroma
        x = h * 6.0
        i = int(math.floor(x)) % 6
        second = c * (1.0 - abs((x % 2.0) - 1.0))
        m = ll - 0.5 * c
        rgb = ((c, second, 0.0), (second, c, 0.0), (0.0, c, second),
               (0.0, second, c), (second, 0.0, c), (c, 0.0, second))[i]
        return rgb[self.ch] + m


class _RgbToHslChannel(Signal):
    """One HSL channel (``ch`` 0=h/1=s/2=l) of an RGB triple.  ``h`` in ``[0,1)``."""

    def __init__(self, r: Signal, g: Signal, b: Signal, ch: int) -> None:
        super().__init__()
        self.r = as_signal(r)
        self.g = as_signal(g)
        self.b = as_signal(b)
        self.ch = int(ch)

    def children(self):
        return (self.r, self.g, self.b)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        r = _clamp01(self.r.at(clock, cache))
        g = _clamp01(self.g.at(clock, cache))
        b = _clamp01(self.b.at(clock, cache))
        mx = max(r, g, b)
        mn = min(r, g, b)
        d = mx - mn
        ll = 0.5 * (mx + mn)
        if self.ch == 2:
            return ll
        if self.ch == 1:
            denom = 1.0 - abs(2.0 * ll - 1.0)
            return 0.0 if denom <= 0.0 else d / denom
        # hue (same wheel as HSV)
        if d <= 0.0:
            return 0.0
        if mx == r:
            h = ((g - b) / d) % 6.0
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        return (h / 6.0) % 1.0


def hsl_to_rgb(h: Union[Signal, Number], s: Union[Signal, Number],
               l: Union[Signal, Number]) -> VecSignal:
    """HSL channels → an animatable RGB :class:`VecSignal` (hue in ``[0,1]``, wraps)."""
    hs, ss, ls = as_signal(h), as_signal(s), as_signal(l)
    return VecSignal([_HslToRgbChannel(hs, ss, ls, ch) for ch in range(3)])


def rgb_to_hsl(r: Union[Signal, Number], g: Union[Signal, Number],
               b: Union[Signal, Number]) -> VecSignal:
    """RGB channels → an animatable HSL :class:`VecSignal` (hue in ``[0,1)``)."""
    rs, gs, bs = as_signal(r), as_signal(g), as_signal(b)
    return VecSignal([_RgbToHslChannel(rs, gs, bs, ch) for ch in range(3)])


# ---------------------------------------------------------------------------
# Color — a VecSignal that *is* its resolved RGB, remembering its authored model
# ---------------------------------------------------------------------------

class Color(VecSignal):
    """A 3-component colour authored as RGB, HSV or HSL.

    A ``Color`` **is** a :class:`VecSignal` whose components are the resolved
    **RGB** (an HSV/HSL colour is converted in the graph), so it works everywhere
    Loom takes a colour.  It also remembers its authored ``mode`` and channels
    (:attr:`src`), so :func:`loom.ftsl_emit.value_token` can emit the matching
    ``.ftsl`` token — ``rgb r g b`` / ``hsv h s v`` / ``hsl h s l`` — for a
    surface's colour.
    """

    def __init__(self, components: Vecish, mode: str) -> None:
        if mode not in ("rgb", "hsv", "hsl"):
            raise ValueError('color mode must be "rgb", "hsv" or "hsl"')
        src = VecSignal.of(components)
        if src.dim != 3:
            raise ValueError(f"a colour needs 3 components, got {src.dim}")
        self.mode = mode
        self.src = src                          # authored channels (for token emit)
        c = src.components
        if mode == "rgb":
            rgb = list(c)
        elif mode == "hsv":
            rgb = [_HsvToRgbChannel(c[0], c[1], c[2], ch) for ch in range(3)]
        else:
            rgb = [_HslToRgbChannel(c[0], c[1], c[2], ch) for ch in range(3)]
        super().__init__(rgb)                   # this node's value == resolved RGB

    @classmethod
    def rgb(cls, r: Union[Signal, Number], g: Union[Signal, Number],
            b: Union[Signal, Number]) -> "Color":
        return cls((r, g, b), "rgb")

    @classmethod
    def hsv(cls, h: Union[Signal, Number], s: Union[Signal, Number],
            v: Union[Signal, Number]) -> "Color":
        return cls((h, s, v), "hsv")

    @classmethod
    def hsl(cls, h: Union[Signal, Number], s: Union[Signal, Number],
            l: Union[Signal, Number]) -> "Color":
        return cls((h, s, l), "hsl")

    def token(self, clock: Clock, cache: Optional[Cache] = None) -> str:
        """The ``.ftsl`` colour token in the authored model, e.g. ``hsl 0.6 0.7 0.5``."""
        vals = self.src.at(clock, cache)
        return f"{self.mode} " + " ".join(f"{c:.6g}" for c in vals)

    @classmethod
    def parse(cls, text: str, default_space: str = "rgb") -> "Color":
        """Read one ``.ftsl`` colour token back into a :class:`Color`.

        The inverse of :meth:`token` — ``Color.parse("hsl 0.6 0.7 0.5")`` round-trips
        an emitted token, and a bare ``"0.8 0.7 0.2"`` (or bracketed ``"[1, 0, 0]"``)
        is read in ``default_space``.  Backed by the shared ``.ftsl`` grammar's
        :func:`loom.grammar.values.as_color`, so it enforces the locked color-vector
        shape rules: a syntax error raises :class:`ValueError`, and a well-formed but
        wrong-shaped value (a 6-vector, a list of colours) raises
        :class:`~loom.grammar.values.ShapeError`.
        """
        from .grammar.values import as_color   # lazy: avoids a module import cycle
        space, comps = as_color(text, default_space)
        return cls(comps, space)


def rgb(r: Union[Signal, Number], g: Union[Signal, Number],
        b: Union[Signal, Number]) -> Color:
    """Author an RGB :class:`Color` (usable in 2-D, 3-D, and ``.ftsl`` surfaces)."""
    return Color.rgb(r, g, b)


def hsv(h: Union[Signal, Number], s: Union[Signal, Number],
        v: Union[Signal, Number]) -> Color:
    """Author an HSV :class:`Color` — hue in ``[0, 1]`` wraps (seamless colour loops)."""
    return Color.hsv(h, s, v)


def hsl(h: Union[Signal, Number], s: Union[Signal, Number],
        l: Union[Signal, Number]) -> Color:
    """Author an HSL :class:`Color` — hue in ``[0, 1]`` wraps; ``l`` is lightness
    (0.5 = the pure hue, 1 = white, 0 = black)."""
    return Color.hsl(h, s, l)


def parse_color(text: str, default_space: str = "rgb") -> Color:
    """Read one ``.ftsl`` colour token (``rgb r g b`` / ``hsl h s l`` / a bare or
    bracketed triple) into a :class:`Color`.  Alias for :meth:`Color.parse`."""
    return Color.parse(text, default_space)


def parse_color_list(text: str, default_space: str = "rgb") -> "list[Color]":
    """Read a flat ``.ftsl`` colour list into a list of :class:`Color`.

    Accepts every spelling the locked color-vector syntax allows for a flat palette
    — comma-separated groups (``"2 0 0, 3 0 0"``), bracketed siblings
    (``"[2 0 0] [3 0 0]"``) and inline RLE colorspace tags
    (``"rgb 1 0 0, 2 0 0, hsl 4 0 0"``) — and a lone colour reads as a one-element
    list (the ``[X] ≡ X`` identity).  A list-of-lists (depth-2 palette-of-palettes)
    is the wrong shape here and raises :class:`~loom.grammar.values.ShapeError`.
    """
    from .grammar.values import as_color_list   # lazy: avoids a module import cycle
    return [Color(comps, space) for space, comps in as_color_list(text, default_space)]
