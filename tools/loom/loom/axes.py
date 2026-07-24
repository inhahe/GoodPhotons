"""
Loom axis-typed signals (roadmap §E5) — one influence model.

This is the E5 foundation: every value-producing node declares the **set of
axes it depends on** (its free variables), and *composition alone* decides how
two nodes combine.  It resolves E5's deferred "open q" — the concrete node
taxonomy and how axis-set inference is represented — with the small, additive
node set below.  The legacy scalar :class:`~loom.signals.core.Signal` DAG is a
function of the clock's single ``t`` axis; :class:`Lift` bridges it in, so this
layer sits *on top of* the existing graph without touching it (all 891 existing
loom tests stay green).

The model (DESIGN.md §E5)
-------------------------
An :class:`AxSignal` is a pure function of a **point** — a mapping from axis
names (``'t'``, ``'s'``, ``'u'``, ``'v'``, grid axes ``'a'``/``'b'``/…) to
scalar coordinates.  Each node carries ``.axes`` (a ``frozenset[str]``) telling
which axes it actually reads.  "A influences B" means *evaluate A at the point
where B is being evaluated*, and the axis sets decide the shape:

- **Broadcast** on axes A lacks — a ``{t}`` node evaluated at a ``{s,t}`` point
  ignores ``s`` and returns the same value for every ``s`` (a time-curve shifts
  the whole elevation of a spatial curve over time).  Free, pure, implicit.
- **Pointwise** on shared axes — two ``{t}`` nodes combine at the *same* ``t``.
  The illegal "run over the whole of B across time" op is **inexpressible** (a
  function of ``t`` simply cannot see any ``t`` but the current one), not
  caught after the fact — so there is no ``t``-influences-``t`` detector and no
  split into separate spatial/temporal signal types.

The **only** node that crosses an axis (reads inputs at points other than the
current one) is the explicit :class:`Reduce` — never smuggled in implicitly.
The space-time "video node" (cross-``t``) already lives in :mod:`loom.xvideo`.

Edges carry two orthogonal attributes: broadcast (implied by the axis sets,
above) and a **combine-mode** ``pin | mod`` with a **gain**, where the *target*
declares its quantity kind (additive / gain / bipolar) and hence its neutral
element and accumulate operator (:class:`Target`, :func:`combine`).

One sample/select grammar (records, curves, grids, scatters): sample with
``(...)`` (continuous, interpolated → :func:`sample` / :class:`Sample`), index
with ``[...]`` (discrete constant selector → :func:`select`), pick a component
with ``.name`` (→ :meth:`AxSignal.comp`).  ``curve(t)`` yields a value *at the
current t*, so it broadcasts across the target's other axes and lands on the
free side by construction.  :func:`sample` folds loom's own clock-parameterized
producers into that grammar — a :class:`~loom.interp.LoopCurve` / other curve
(:class:`CurveSample`, which threads the clock axis so an animated spatial curve
types as ``{s, t}``) and a :class:`~loom.record.Record` (:class:`RecordSample`,
a static ``{driver}`` LUT) — so the caller binds a real curve's parameter axis
directly instead of pre-baking a bare callable that could not see the clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, alloc_id, Number

Point = Mapping[str, float]
Numeric = Union["AxSignal", Number]

# Standard axis names.  Not an enum — axes are just strings, so grid axes
# ('a', 'b', 'c', …) and any user axis compose without a registration step.
AXIS_T = "t"   # time / normalized loop phase
AXIS_S = "s"   # arclength / curve parameter
AXIS_U = "u"   # surface u
AXIS_V = "v"   # surface v


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class AxSignal:
    """A scalar (or fixed-vector) pure function of a :class:`Point`.

    Subclasses set ``self.axes`` (a ``frozenset[str]``) and implement
    :meth:`_eval`.  The public :meth:`eval` ignores any extra axes a point
    supplies (**broadcast**) and raises on a missing required axis.
    """

    axes: FrozenSet[str] = frozenset()

    def __init__(self) -> None:
        self._id = alloc_id()

    # ---- node protocol (matches Signal for cycle detection / walk) ----------
    @property
    def id(self) -> int:
        return self._id

    def children(self) -> Tuple["AxSignal", ...]:
        return ()

    def _eval(self, point: Point):
        raise NotImplementedError

    def eval(self, point: Optional[Point] = None, **coords: float):
        """Evaluate at ``point`` (a dict of axis→coord); ``**coords`` merges in.

        Extra axes are ignored (broadcast).  A required axis missing from the
        point raises :class:`ValueError` naming it.
        """
        pt: Dict[str, float] = {}
        if point:
            pt.update(point)
        if coords:
            pt.update(coords)
        missing = self.axes - pt.keys()
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs axes {sorted(self.axes)}; "
                f"point is missing {sorted(missing)}"
            )
        v = self._eval(pt)
        return v

    # ---- operator overloads -------------------------------------------------
    def __add__(self, other: Numeric) -> "AxSignal":
        return _binop(self, other, "+", lambda a, b: a + b)

    def __radd__(self, other: Numeric) -> "AxSignal":
        return _binop(other, self, "+", lambda a, b: a + b)

    def __sub__(self, other: Numeric) -> "AxSignal":
        return _binop(self, other, "-", lambda a, b: a - b)

    def __rsub__(self, other: Numeric) -> "AxSignal":
        return _binop(other, self, "-", lambda a, b: a - b)

    def __mul__(self, other: Numeric) -> "AxSignal":
        return _binop(self, other, "*", lambda a, b: a * b)

    def __rmul__(self, other: Numeric) -> "AxSignal":
        return _binop(other, self, "*", lambda a, b: a * b)

    def __truediv__(self, other: Numeric) -> "AxSignal":
        return _binop(self, other, "/", _safe_div)

    def __rtruediv__(self, other: Numeric) -> "AxSignal":
        return _binop(other, self, "/", _safe_div)

    def __neg__(self) -> "AxSignal":
        return AFn("neg", lambda v: -v, self)

    # ---- sample/select grammar ---------------------------------------------
    def comp(self, i: int) -> "AxSignal":
        """Pick component ``i`` of a vector-valued node (``curve(t).y``)."""
        return _Comp(self, int(i))


def as_ax(x: Numeric) -> "AxSignal":
    """Coerce a number to :class:`AConst`; pass an :class:`AxSignal` through."""
    return x if isinstance(x, AxSignal) else AConst(float(x))


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        raise ValueError("Div by zero in AxSignal graph")
    return a / b


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------

class Ax(AxSignal):
    """The coordinate identity for one axis: ``eval`` returns ``point[name]``."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = str(name)
        self.axes = frozenset({self.name})

    def _eval(self, point: Point) -> float:
        return float(point[self.name])


class AConst(AxSignal):
    """A constant (axis set ∅ → broadcasts everywhere)."""

    def __init__(self, value: Number) -> None:
        super().__init__()
        self.value = float(value)
        self.axes = frozenset()

    def _eval(self, point: Point) -> float:
        return self.value


class Lift(AxSignal):
    """Bridge a legacy scalar :class:`Signal` (a function of ``t``) into the
    axis-typed layer as a ``{t}``-typed node.

    The wrapped signal is evaluated with a fresh :class:`Clock` at
    ``t = point['t']``; ``loop`` is carried so periodic leaves wrap correctly.
    """

    def __init__(self, signal: Signal, *, loop: bool = True) -> None:
        super().__init__()
        self.signal = signal
        self.loop = bool(loop)
        self.axes = frozenset({AXIS_T})

    def children(self):
        return (self.signal,)

    def _eval(self, point: Point) -> float:
        return float(self.signal.at(Clock(t=float(point[AXIS_T]), loop=self.loop)))


# ---------------------------------------------------------------------------
# Composition — axes inferred by union (broadcast implicit)
# ---------------------------------------------------------------------------

class AFn(AxSignal):
    """Apply ``fn`` elementwise over one or more child nodes.

    ``axes`` is the union of the children's axes: composition broadcasts on
    unshared axes and combines pointwise on shared ones, automatically.
    """

    def __init__(self, name: str, fn: Callable[..., float], *args: Numeric) -> None:
        super().__init__()
        self.name = str(name)
        self.fn = fn
        self.args = tuple(as_ax(a) for a in args)
        self.axes = frozenset().union(*(a.axes for a in self.args)) \
            if self.args else frozenset()

    def children(self):
        return self.args

    def _eval(self, point: Point) -> float:
        return float(self.fn(*(a._eval(point) for a in self.args)))


def _binop(a: Numeric, b: Numeric, sym: str, fn) -> AxSignal:
    aa, bb = as_ax(a), as_ax(b)
    return AFn(sym, fn, aa, bb)


class _Comp(AxSignal):
    """Pick a component of a vector-valued node (see :meth:`AxSignal.comp`)."""

    def __init__(self, node: AxSignal, i: int) -> None:
        super().__init__()
        self.node = node
        self.i = int(i)
        self.axes = node.axes

    def children(self):
        return (self.node,)

    def _eval(self, point: Point) -> float:
        v = self.node._eval(point)
        try:
            return float(v[self.i])
        except (TypeError, IndexError) as e:
            raise ValueError(
                f"component [{self.i}] of a non-indexable/short value {v!r}"
            ) from e


# ---------------------------------------------------------------------------
# Sample / select grammar
# ---------------------------------------------------------------------------

class Sample(AxSignal):
    """Continuous sample ``fn(arg)`` — the ``curve(t)`` form.

    ``fn`` is any callable of one scalar (a baked loom curve's sampling
    function, an interpolant, …) returning a scalar or a fixed sequence.  The
    curve's own parameter axis is *bound* by ``arg``, so the result's axes are
    exactly ``arg``'s axes: a ``curve(t)`` sampled at the current ``t`` yields a
    value at that ``t`` and broadcasts across every other axis.  Pick a
    component of a vector result with :meth:`AxSignal.comp` (``curve(t).y``).
    """

    def __init__(self, fn: Callable[[float], Union[float, Sequence[float]]],
                 arg: Numeric) -> None:
        super().__init__()
        self.fn = fn
        self.arg = as_ax(arg)
        self.axes = self.arg.axes

    def children(self):
        return (self.arg,)

    def _eval(self, point: Point):
        p = self.arg._eval(point)
        return self.fn(p)


class CurveSample(AxSignal):
    """Sample a **clock-parameterized loom curve** at a bound parameter axis.

    ``curve`` is any loom curve exposing ``.sample(u, clock, cache) -> value``
    (a :class:`~loom.interp.LoopCurve`, a :class:`~loom.interp.TrackedCurve`
    track, a :class:`~loom.interp.FieldCurve` position, …).  ``arg`` binds the
    curve's own parameter axis — write ``CurveSample(loop, Ax('s'))`` for the
    ``curve(s)`` form.  Unlike a plain :class:`Sample` over a bare callable, this
    threads the **clock axis** (default ``'t'``) into the curve's control-point
    Signals, so an *animated* spatial curve is correctly typed ``{s, t}`` (its
    shape moves over time) while a static one just broadcasts trivially over
    ``t``.  The result is whatever ``.sample`` returns (a vector for a position
    curve); pick a component with :meth:`AxSignal.comp` (``curve(s).y``).
    """

    def __init__(self, curve, arg: Numeric, *, clock_axis: str = AXIS_T,
                 loop: bool = True) -> None:
        super().__init__()
        if not callable(getattr(curve, "sample", None)):
            raise TypeError(
                "CurveSample needs a loom curve with .sample(u, clock, cache)")
        self.curve = curve
        self.arg = as_ax(arg)
        self.clock_axis = str(clock_axis)
        self.loop = bool(loop)
        self.axes = self.arg.axes | {self.clock_axis}

    def children(self):
        # thread the loom curve node in too (its control points are part of the
        # DAG); the axis-layer walk duck-types over loom Signals, as Lift does.
        return (self.arg, self.curve)

    def _eval(self, point: Point):
        u = self.arg._eval(point)
        clk = Clock(t=float(point[self.clock_axis]), loop=self.loop)
        v = self.curve.sample(u, clk)
        return tuple(v) if isinstance(v, list) else v


class RecordSample(AxSignal):
    """Sample a loom :class:`~loom.record.Record` channel at a bound driver axis.

    A Record is a **static** LUT keyed by a driver in ``[lo, hi]`` (no clock), so
    ``arg`` binds that driver axis and the result's axes are exactly ``arg``'s —
    this is the record leaf of the shared sample grammar (``R(driver).chan``).
    Scalar channels return a float; vector channels a tuple (pick a component
    with :meth:`AxSignal.comp`).  Colour / expression channels are rejected by
    the underlying :meth:`Record.sample_vec`.
    """

    def __init__(self, record, channel: str, arg: Numeric) -> None:
        super().__init__()
        if not callable(getattr(record, "sample_vec", None)):
            raise TypeError("RecordSample needs a loom Record (with .sample_vec)")
        self.record = record
        self.channel = str(channel)
        self.arg = as_ax(arg)
        self.axes = self.arg.axes

    def children(self):
        return (self.arg,)

    def _eval(self, point: Point):
        d = self.arg._eval(point)
        vec = self.record.sample_vec(self.channel, d)
        return vec[0] if len(vec) == 1 else tuple(vec)


def sample(obj, arg: Numeric, *, channel: Optional[str] = None,
           clock_axis: str = AXIS_T, loop: bool = True) -> AxSignal:
    """The unified continuous ``obj(arg)`` sample — one grammar over every loom
    value producer, binding ``obj``'s own parameter axis to ``arg``:

    - a **loom Record** (has ``.sample_vec``) → :class:`RecordSample` (needs
      ``channel=``; static LUT, axes = ``arg``'s);
    - a **clock-parameterized loom curve** (has ``.sample(u, clock, cache)``) →
      :class:`CurveSample` (threads ``clock_axis``, so animated ⇒ ``{s, t}``);
    - a **plain callable** of one scalar → :class:`Sample` (the low-level form).

    This is the fold that lets the sample grammar bind a real loom curve's param
    axis directly rather than forcing the caller to pre-bake a bare callable
    (which could not thread the clock the curve's control points depend on).
    """
    if callable(getattr(obj, "sample_vec", None)):          # a Record
        if channel is None:
            raise ValueError("sampling a Record needs channel=<name>")
        return RecordSample(obj, channel, arg)
    if callable(getattr(obj, "sample", None)):              # a clock-param curve
        return CurveSample(obj, arg, clock_axis=clock_axis, loop=loop)
    if callable(obj):                                        # a bare callable
        return Sample(obj, arg)
    raise TypeError(
        "sample(obj, …): obj must be a loom Record, a loom curve with "
        ".sample(u, clock, cache), or a plain callable")


def select(items: Sequence[Numeric], i: int) -> AxSignal:
    """Discrete constant selector ``items[i]`` — the ``R.chan[i]`` / ``[...]``
    form.  ``i`` is a fixed Python int (last-write-wins constant), not an axis;
    this mirrors records' discrete stop-select.  Returns the chosen node."""
    n = len(items)
    if not (-n <= i < n):
        raise IndexError(f"select index {i} out of range for {n} items")
    return as_ax(items[i])


# ---------------------------------------------------------------------------
# The one cross-axis node: explicit reduction over an axis
# ---------------------------------------------------------------------------

_REDUCERS: Dict[str, Callable[[Sequence[float]], float]] = {
    "sum": lambda xs: float(sum(xs)),
    "mean": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "min": lambda xs: float(min(xs)) if xs else 0.0,
    "max": lambda xs: float(max(xs)) if xs else 0.0,
}


class Reduce(AxSignal):
    """The **only** node that crosses an axis: materialise ``body`` over a grid
    of ``samples`` values of ``axis`` in ``[lo, hi]`` and reduce them.

    ``axes = body.axes - {axis}`` — the reduced axis is consumed, so a reduce
    over ``s`` of an ``{s,t}`` body yields a ``{t}`` node.  This is the explicit
    reduction the design requires (arc length, centroid, integral, "all points
    at once as a set"); a plain :class:`AFn` can never read another point.

    ``op`` is ``'sum' | 'mean' | 'min' | 'max' | 'integral'`` or a callable
    ``Sequence[float] -> float``.  ``'integral'`` is a trapezoidal rule over
    ``[lo, hi]`` (so it approximates ∫ body d(axis)).
    """

    def __init__(self, body: AxSignal, axis: str, samples: int,
                 op: Union[str, Callable[[Sequence[float]], float]] = "sum",
                 *, lo: float = 0.0, hi: float = 1.0) -> None:
        super().__init__()
        if axis not in body.axes:
            raise ValueError(
                f"Reduce over '{axis}' but body depends only on "
                f"{sorted(body.axes)} (nothing to reduce)"
            )
        if samples < 1:
            raise ValueError("Reduce needs samples >= 1")
        self.body = body
        self.axis = str(axis)
        self.samples = int(samples)
        self.lo = float(lo)
        self.hi = float(hi)
        self.op = op
        self.axes = body.axes - {self.axis}

    def children(self):
        return (self.body,)

    def _grid(self):
        n = self.samples
        if n == 1:
            return [0.5 * (self.lo + self.hi)]
        step = (self.hi - self.lo) / (n - 1)
        return [self.lo + i * step for i in range(n)]

    def _eval(self, point: Point) -> float:
        xs = self._grid()
        base = dict(point)
        vals = []
        for x in xs:
            base[self.axis] = x
            vals.append(float(self.body._eval(base)))
        op = self.op
        if callable(op):
            return float(op(vals))
        if op == "integral":
            if len(vals) == 1:
                return vals[0] * (self.hi - self.lo)
            step = (self.hi - self.lo) / (len(vals) - 1)
            s = 0.5 * (vals[0] + vals[-1]) + sum(vals[1:-1])
            return float(s * step)
        try:
            return _REDUCERS[op](vals)
        except KeyError:
            raise ValueError(f"unknown reduce op {op!r}") from None


# ---------------------------------------------------------------------------
# Combine-mode: pin | mod, with target-declared neutral / accumulate operator
# ---------------------------------------------------------------------------

# Quantity kinds → (neutral element, accumulate(y, x, gain)).
ADDITIVE = "additive"   # position / elevation: neutral 0, y += gain*x
GAIN = "gain"           # gains / scales:       neutral 1, y *= x**gain
BIPOLAR = "bipolar"     # bipolar-[0,1]:        neutral ½, ½-centred, clamped


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


_NEUTRAL: Dict[str, float] = {ADDITIVE: 0.0, GAIN: 1.0, BIPOLAR: 0.5}


def _accumulate(kind: str, y: float, x: float, gain: float) -> float:
    if kind == ADDITIVE:
        return y + gain * x
    if kind == GAIN:
        # push in log space: gain=1 → y*=x, gain=0 → no effect
        return y * (x ** gain)
    if kind == BIPOLAR:
        return _clamp01((y - 0.5) + gain * (x - 0.5) + 0.5)
    raise ValueError(f"unknown target kind {kind!r}")


@dataclass
class Binding:
    """One DAG edge into a target: ``source`` combined via ``mode`` with ``gain``.

    - ``mode='pin'`` — replace (last-write-wins); ``gain`` blends
      ``y*(1-gain) + x*gain`` (``gain=1`` → full replace).
    - ``mode='mod'`` — accumulate toward the target's neutral element using the
      target-kind operator (additive/gain/bipolar), scaled by ``gain``.
    """

    source: AxSignal
    mode: str = "mod"
    gain: float = 1.0


class Target(AxSignal):
    """A modulated value site with a declared quantity ``kind`` and a base.

    ``combine`` folds each :class:`Binding` in order starting from ``base``
    (defaulting to the kind's neutral).  ``axes`` is the union of the base's
    and all sources' axes, so a target driven by a ``{t}`` source over an
    otherwise-``{s}`` base is ``{s,t}`` and broadcasts correctly.
    """

    def __init__(self, kind: str, bindings: Sequence[Binding],
                 base: Optional[Numeric] = None) -> None:
        super().__init__()
        if kind not in _NEUTRAL:
            raise ValueError(f"unknown target kind {kind!r}")
        self.kind = kind
        self.bindings = list(bindings)
        self.base = AConst(_NEUTRAL[kind]) if base is None else as_ax(base)
        srcs = [b.source.axes for b in self.bindings]
        self.axes = frozenset().union(self.base.axes, *srcs) if srcs else self.base.axes

    def children(self):
        return (self.base, *(b.source for b in self.bindings))

    def _eval(self, point: Point) -> float:
        y = float(self.base._eval(point))
        for b in self.bindings:
            x = float(b.source._eval(point))
            if b.mode == "pin":
                y = y * (1.0 - b.gain) + x * b.gain
            elif b.mode == "mod":
                y = _accumulate(self.kind, y, x, b.gain)
            else:
                raise ValueError(f"unknown binding mode {b.mode!r}")
        return y


def combine(kind: str, bindings: Sequence[Binding],
            base: Optional[Numeric] = None) -> Target:
    """Build a :class:`Target` — sugar for ``Target(kind, bindings, base)``."""
    return Target(kind, bindings, base)


__all__ = [
    "AxSignal", "Ax", "AConst", "Lift", "AFn", "Sample", "select", "Reduce",
    "CurveSample", "RecordSample", "sample",
    "Binding", "Target", "combine", "as_ax",
    "ADDITIVE", "GAIN", "BIPOLAR",
    "AXIS_T", "AXIS_S", "AXIS_U", "AXIS_V", "Point",
]
