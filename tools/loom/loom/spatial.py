"""
Loom spatial-expression tree (M10.5) — one pattern definition, evaluated **two
ways**: numerically over pixels (the 2-D backend) *and* emitted as an ftsl string
(the 3-D isosurface / material backend).

DESIGN.md §11.10: loom's temporal DAG is a function of *time* (the clock), cached
per frame.  A *field* is a function of *space* (x, y, z) — a different axis — so it
does **not** belong in the time-DAG (its per-frame cache would be wrong: one value
per pixel, not per frame).  Coordinates live here, in a small **spatial** algebra
whose leaves are the coordinate variables :data:`X`, :data:`Y`, :data:`Z` and the
loop phase :data:`T`, and whose *coefficients* may be temporal :class:`Signal`\\s
(baked per frame — exactly how the 3-D side already animates a static formula).

A :class:`SpatialExpr` evaluates two ways:

- :meth:`SpatialExpr.eval_np` — numerically over numpy coordinate arrays (2-D
  raster fields), and
- :meth:`SpatialExpr.emit` — as an ftsl expression string in ``x``/``y``/``z``
  with the temporal coefficients baked to numbers (3-D isosurfaces / patterns).

Because it also exposes ``build(cx, cy, cz, ctx)`` and ``param_signals()`` it drops
straight into :class:`loom.Isosurface` and :class:`loom.FuncPattern` through their
existing duck-typed template protocol — no changes there.  Every function name
emitted (:func:`sin`, :func:`sign`, :func:`clamp`, …) is a real ftsl pattern
builtin (``src/pattern.h``), so the emitted string always parses; the numpy path
computes the *same* mathematics (``noise`` is intentionally absent — ftrace's value
noise has no bit-identical numpy twin, so it would break the "one definition, two
backends" honesty).
"""

from __future__ import annotations

import operator
from typing import Callable, List, Sequence, Tuple, Union

from .signals.core import Signal, Number
from .ftsl_emit import fmt

try:  # numpy is only needed for the 2-D numeric path
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------

class SpatialExpr:
    """A scalar function of space (and optionally the loop phase ``T``).

    Build with the leaves :data:`X`/:data:`Y`/:data:`Z`/:data:`T`, Python numbers,
    temporal :class:`~loom.signals.core.Signal`\\s (used as animated coefficients),
    the arithmetic operators, and the module math functions.
    """

    # ---- operators (coerce the other operand into the spatial algebra) ----
    def __add__(self, o): return _Bin("+", self, _coerce(o))
    def __radd__(self, o): return _Bin("+", _coerce(o), self)
    def __sub__(self, o): return _Bin("-", self, _coerce(o))
    def __rsub__(self, o): return _Bin("-", _coerce(o), self)
    def __mul__(self, o): return _Bin("*", self, _coerce(o))
    def __rmul__(self, o): return _Bin("*", _coerce(o), self)
    def __truediv__(self, o): return _Bin("/", self, _coerce(o))
    def __rtruediv__(self, o): return _Bin("/", _coerce(o), self)
    def __neg__(self): return _Neg(self)
    def __pow__(self, o): return spow(self, o)
    def __abs__(self): return sabs(self)

    # ---- tree walk (leaves override the hooks) ----------------------------
    def children(self) -> Tuple["SpatialExpr", ...]:
        return ()

    def _time_signal(self):
        return None  # a _Sig leaf returns its wrapped Signal

    def _is_time(self) -> bool:
        return False  # T and _Sig leaves are time-dependent

    def _walk(self):
        stack: List[SpatialExpr] = [self]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children())

    def time_signals(self) -> List[Signal]:
        """The temporal Signals embedded as coefficients (deduped) — the DAG roots
        an :class:`Isosurface`/:class:`FuncPattern` must expose for cycle/cache."""
        out: List[Signal] = []
        seen = set()
        for n in self._walk():
            s = n._time_signal()
            if s is not None and id(s) not in seen:
                seen.add(id(s))
                out.append(s)
        return out

    def uses_time(self) -> bool:
        """True if any ``T`` leaf or temporal-Signal coefficient is present — i.e.
        the field varies over the loop (else a 2-D raster can be baked once)."""
        return any(n._is_time() for n in self._walk())

    # ---- evaluation (subclasses implement) --------------------------------
    def emit(self, coords: Tuple[str, str, str], ctx) -> str:
        raise NotImplementedError

    def eval_np(self, coords, clock, cache):
        raise NotImplementedError

    # ---- duck-typed template protocol (Isosurface / FuncPattern) ----------
    def build(self, cx: str, cy: str, cz: str, ctx) -> str:
        return self.emit((cx, cy, cz), ctx)

    def param_signals(self) -> List[Signal]:
        return self.time_signals()


def _coerce(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    if isinstance(v, SpatialExpr):
        return v
    if isinstance(v, Signal):
        return _Sig(v)
    if isinstance(v, (int, float)):
        return _Const(float(v))
    raise TypeError(f"cannot use {type(v).__name__} in a spatial expression")


def sexpr(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    """Coerce a number / temporal Signal / SpatialExpr into the spatial algebra."""
    return _coerce(v)


# ---------------------------------------------------------------------------
# leaves
# ---------------------------------------------------------------------------

class _Const(SpatialExpr):
    def __init__(self, v: float) -> None:
        self.v = float(v)

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.v)})"

    def eval_np(self, coords, clock, cache):
        return self.v


class _Coord(SpatialExpr):
    """A coordinate variable: axis 0/1/2 -> x/y/z."""

    def __init__(self, axis: int, label: str) -> None:
        self.axis = axis
        self.label = label

    def emit(self, coords, ctx) -> str:
        return f"({coords[self.axis]})"

    def eval_np(self, coords, clock, cache):
        return coords[self.axis]


class _Time(SpatialExpr):
    """The loop phase ``t`` in [0, 1) at the current frame."""

    def emit(self, coords, ctx) -> str:
        return f"({fmt(ctx.clock.t)})"

    def eval_np(self, coords, clock, cache):
        return clock.t

    def _is_time(self) -> bool:
        return True


class _Sig(SpatialExpr):
    """A temporal Signal used as an animated coefficient (baked per frame)."""

    def __init__(self, sig: Signal) -> None:
        self.sig = sig

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.sig.at(ctx.clock, ctx.cache))})"

    def eval_np(self, coords, clock, cache):
        return self.sig.at(clock, cache)

    def _time_signal(self):
        return self.sig

    def _is_time(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

_BINOPS = {"+": operator.add, "-": operator.sub,
           "*": operator.mul, "/": operator.truediv}


class _Bin(SpatialExpr):
    def __init__(self, op: str, a: SpatialExpr, b: SpatialExpr) -> None:
        self.op = op
        self.a = a
        self.b = b

    def children(self):
        return (self.a, self.b)

    def emit(self, coords, ctx) -> str:
        return f"({self.a.emit(coords, ctx)}{self.op}{self.b.emit(coords, ctx)})"

    def eval_np(self, coords, clock, cache):
        return _BINOPS[self.op](self.a.eval_np(coords, clock, cache),
                                self.b.eval_np(coords, clock, cache))


class _Neg(SpatialExpr):
    def __init__(self, a: SpatialExpr) -> None:
        self.a = a

    def children(self):
        return (self.a,)

    def emit(self, coords, ctx) -> str:
        return f"(-({self.a.emit(coords, ctx)}))"

    def eval_np(self, coords, clock, cache):
        return -self.a.eval_np(coords, clock, cache)


class _Fn(SpatialExpr):
    """An ftsl builtin call — ``name`` must exist in ``src/pattern.h``; ``npfn`` is
    its numpy twin (same argument order/semantics)."""

    def __init__(self, name: str, args: Sequence[SpatialExpr], npfn: Callable) -> None:
        self.name = name
        self.args = list(args)
        self.npfn = npfn

    def children(self):
        return tuple(self.args)

    def emit(self, coords, ctx) -> str:
        inner = ",".join(a.emit(coords, ctx) for a in self.args)
        return f"{self.name}({inner})"

    def eval_np(self, coords, clock, cache):
        return self.npfn(*[a.eval_np(coords, clock, cache) for a in self.args])


# ---------------------------------------------------------------------------
# leaf singletons + math functions (each emits a real ftsl pattern builtin)
# ---------------------------------------------------------------------------

X = _Coord(0, "x")
Y = _Coord(1, "y")
Z = _Coord(2, "z")
T = _Time()


def _mk(name: str, npfn: Callable) -> Callable[..., SpatialExpr]:
    def f(*args):
        return _Fn(name, [_coerce(a) for a in args], npfn)
    f.__name__ = name
    return f


def _np_step(edge, x):
    return _np.where(_np.asarray(x) >= edge, 1.0, 0.0)


def _np_clamp(x, lo, hi):
    return _np.minimum(hi, _np.maximum(lo, x))


def _np_mix(a, b, t):
    return a + (b - a) * t


def _np_smoothstep(e0, e1, x):
    t = _np.clip((_np.asarray(x) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _np_fract(x):
    return x - _np.floor(x)


# unary
sin = _mk("sin", lambda x: _np.sin(x))
cos = _mk("cos", lambda x: _np.cos(x))
tan = _mk("tan", lambda x: _np.tan(x))
sqrt = _mk("sqrt", lambda x: _np.sqrt(x))
exp = _mk("exp", lambda x: _np.exp(x))
log = _mk("log", lambda x: _np.log(x))
floor = _mk("floor", lambda x: _np.floor(x))
fract = _mk("fract", _np_fract)
sign = _mk("sign", lambda x: _np.sign(x))
saturate = _mk("saturate", lambda x: _np.clip(x, 0.0, 1.0))
sabs = _mk("abs", lambda x: _np.abs(x))       # ``abs`` shadows the builtin -> sabs
# binary
smin = _mk("min", lambda a, b: _np.minimum(a, b))
smax = _mk("max", lambda a, b: _np.maximum(a, b))
spow = _mk("pow", lambda a, b: _np.power(a, b))
atan2 = _mk("atan2", lambda a, b: _np.arctan2(a, b))
step = _mk("step", _np_step)
# ternary
clamp = _mk("clamp", _np_clamp)
mix = _mk("mix", _np_mix)
smoothstep = _mk("smoothstep", _np_smoothstep)


# ---------------------------------------------------------------------------
# preset patterns (shared 2-D-numeric / 3-D-emitted) — compose your own too
# ---------------------------------------------------------------------------

def waves(freq: Union[SpatialExpr, Signal, Number] = 1.0, axis: int = 0) -> SpatialExpr:
    """1-D sinusoid along one axis, remapped to [0, 1]."""
    c = (X, Y, Z)[axis]
    return 0.5 + 0.5 * sin(freq * c)


def rings(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """Concentric shells ``0.5 + 0.5 sin(freq |p|)`` in [0, 1]."""
    return 0.5 + 0.5 * sin(freq * sqrt(X * X + Y * Y + Z * Z))


def checker(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """3-D checkerboard in [0, 1] (sign of the product of three sines)."""
    return 0.5 + 0.5 * sign(sin(freq * X) * sin(freq * Y) * sin(freq * Z))


def gyroid(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """Schoen gyroid field ``sin x cos y + sin y cos z + sin z cos x`` (an
    isosurface field at ``=0``, or a signed pattern)."""
    fx, fy, fz = freq * X, freq * Y, freq * Z
    return sin(fx) * cos(fy) + sin(fy) * cos(fz) + sin(fz) * cos(fx)


SPATIAL_PATTERNS = {
    "waves": waves, "rings": rings, "checker": checker, "gyroid": gyroid,
}
