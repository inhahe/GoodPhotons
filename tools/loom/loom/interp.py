"""
Loom interpolators — turn a dataset into a *field*, and every field is a Signal.

Each interpolator is itself a node in the modulation DAG, so its output can feed
another modulator ("it's just another function") and the whole thing stays
cycle-checked and cached.

1. :class:`LoopCurve`   — scribbles3's seamless closed curve, generalized to
   N-D.  A ``VecSignal`` parameterized by a scalar curve param ``u``.
2. :class:`GridField`   — N-linear interpolation of a :class:`~loom.data.Grid`
   at an animatable query point.  A scalar ``Signal``.
3. :class:`ScatterField`— smooth (inverse-distance) interpolation of a
   :class:`~loom.data.Scatter` at a query point.  A scalar ``Signal``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

from .signals.core import Signal, Clock, Cache, Number, as_signal, alloc_id
from .signals.vector import VecSignal, Vecish
from .data import PointPath, Grid, Scatter


# ---------------------------------------------------------------------------
# 1. LoopCurve — seamless closed curve (scribbles3), N-D
# ---------------------------------------------------------------------------

def _quad_bezier(p1: Tuple[float, ...], p2: Tuple[float, ...],
                 p3: Tuple[float, ...], f: float) -> Tuple[float, ...]:
    a = (1.0 - f) * (1.0 - f)
    b = 2.0 * (1.0 - f) * f
    c = f * f
    return tuple(a * x1 + b * x2 + c * x3 for x1, x2, x3 in zip(p1, p2, p3))


def _mid(p: Tuple[float, ...], q: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(0.5 * (a + b) for a, b in zip(p, q))


def eval_curve(pts: List[Tuple[float, ...]], u: float, closed: bool) -> Tuple[float, ...]:
    """Point on the midpoint-quadratic-Bezier curve through control points ``pts``
    at parameter ``u`` (wrapped to [0,1)).  ``pts`` are already-evaluated tuples."""
    n = len(pts)
    u -= math.floor(u)
    if closed:
        x = u * n
        i = int(math.floor(x)) % n
        f = x - math.floor(x)
        a0, a1, a2 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
    else:
        segs = n - 2
        if segs < 1:
            x = u * (n - 1)
            i = min(int(math.floor(x)), n - 2)
            f = x - i
            return tuple(p * (1 - f) + q * f for p, q in zip(pts[i], pts[i + 1]))
        x = u * segs
        i = min(int(math.floor(x)), segs - 1)
        f = x - i
        a0, a1, a2 = pts[i], pts[i + 1], pts[i + 2]
    return _quad_bezier(_mid(a0, a1), a1, _mid(a1, a2), f)


class _CurveComponent(Signal):
    """Scalar view of one axis of a :class:`LoopCurve` (for vector math)."""

    def __init__(self, curve: "LoopCurve", axis: int) -> None:
        super().__init__()
        self.curve = curve
        self.axis = axis

    def children(self):
        return (self.curve,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.curve.at(clock, cache)[self.axis]


class LoopCurve(VecSignal):
    """A point on the scribbles3 midpoint-quadratic-Bezier curve at param ``u``.

    For each control point ``B`` with neighbours ``A, C`` the curve draws an arc
    from ``mid(A,B)`` through ``B`` to ``mid(B,C)``; consecutive arcs share their
    join point *and* tangent, so a **closed** path is seamless with no seam angle
    to choose.  The construction is per-component, so it works in any dimension.

    ``u`` is a scalar curve parameter (a Signal or number).  Wrapped to
    ``[0, 1)``; ``u`` and ``t`` are independent (drive ``u`` from ``t`` for a
    point travelling around the loop, or hold it to pin a location).
    """

    def __init__(self, path: PointPath, u: Union[Signal, Number],
                 *, closed: Optional[bool] = None) -> None:
        # NOTE: intentionally do NOT call VecSignal.__init__ — we synthesize our
        # own component accessors and override at()/children().
        self.path = path
        self._u = as_signal(u)
        self.closed = path.closed if closed is None else bool(closed)
        self._id = alloc_id()
        self.components: List[Signal] = [_CurveComponent(self, a) for a in range(path.dim)]

    def children(self):
        kids: List = [self._u]
        for p in self.path.points:
            kids.extend(p.components)
        return tuple(kids)

    def _control_points(self, clock: Clock, cache: Optional[Cache]) -> List[Tuple[float, ...]]:
        return [p.at(clock, cache) for p in self.path.points]

    def sample(self, u_value: float, clock: Clock,
               cache: Optional[Cache] = None) -> Tuple[float, ...]:
        """Point on the curve at an explicit parameter (independent of ``self._u``)."""
        return eval_curve(self._control_points(clock, cache), u_value, self.closed)

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        pts = self._control_points(clock, cache)
        out = eval_curve(pts, self._u.at(clock, cache), self.closed)
        if cache is not None:
            cache.set(self._id, clock.frame, out)
        return out


# ---------------------------------------------------------------------------
# 2. GridField — N-linear interpolation
# ---------------------------------------------------------------------------

class GridField(Signal):
    """Scalar N-linear interpolation of a :class:`Grid` at ``query`` (a VecSignal).

    Query rank must equal the grid's ndim.  Out-of-domain queries are clamped to
    the boundary cell (edge-extend).
    """

    def __init__(self, grid: Grid, query: Vecish) -> None:
        super().__init__()
        self.grid = grid
        self.q = VecSignal.of(query)
        if self.q.dim != grid.ndim:
            raise ValueError(f"query dim {self.q.dim} != grid ndim {grid.ndim}")
        for v in grid.values:
            if not isinstance(v, Signal):
                raise TypeError("GridField requires scalar (Signal) grid values")

    def children(self):
        return tuple(self.q.components) + tuple(self.grid.values)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        g = self.grid
        coords = self.q.at(clock, cache)
        base: List[int] = []
        fracs: List[float] = []
        for axis in range(g.ndim):
            n = g.shape[axis]
            lo, hi = g.lo[axis], g.hi[axis]
            # normalized position in [0, n-1]
            p = (coords[axis] - lo) / (hi - lo) * (n - 1) if hi != lo else 0.0
            if p <= 0.0:
                base.append(0)
                fracs.append(0.0)
            elif p >= n - 1:
                base.append(n - 2)
                fracs.append(1.0)
            else:
                i = int(math.floor(p))
                base.append(i)
                fracs.append(p - i)

        total = 0.0
        for corner in range(1 << g.ndim):
            w = 1.0
            idx: List[int] = []
            for axis in range(g.ndim):
                bit = (corner >> axis) & 1
                idx.append(base[axis] + bit)
                w *= fracs[axis] if bit else (1.0 - fracs[axis])
            if w == 0.0:
                continue
            total += w * g.value_at_index(idx).at(clock, cache)  # type: ignore[union-attr]
        return total


# ---------------------------------------------------------------------------
# 3. ScatterField — inverse-distance (Shepard) interpolation
# ---------------------------------------------------------------------------

class ScatterField(Signal):
    """Scalar smooth interpolation of a :class:`Scatter` set at ``query``.

    Shepard inverse-distance weighting with exponent ``power``.  Simple and
    robust; quality/speed tradeoffs (RBF, natural neighbour) are a documented
    open item (see DESIGN.md §11).  Exactly reproduces a sample's value when the
    query coincides with that sample.
    """

    def __init__(self, scatter: Scatter, query: Vecish, *,
                 power: float = 2.0, eps: float = 1e-9) -> None:
        super().__init__()
        self.scatter = scatter
        self.q = VecSignal.of(query)
        if self.q.dim != scatter.dim:
            raise ValueError(f"query dim {self.q.dim} != scatter dim {scatter.dim}")
        for v in scatter.values:
            if not isinstance(v, Signal):
                raise TypeError("ScatterField requires scalar (Signal) values")
        self.power = float(power)
        self.eps = float(eps)

    def children(self):
        return (tuple(self.q.components)
                + tuple(c for p in self.scatter.positions for c in p.components)
                + tuple(self.scatter.values))

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        q = self.q.at(clock, cache)
        half = self.power * 0.5
        num = 0.0
        den = 0.0
        for pos, val in zip(self.scatter.positions, self.scatter.values):
            p = pos.at(clock, cache)
            d2 = sum((a - b) ** 2 for a, b in zip(q, p))
            if d2 <= self.eps:
                return val.at(clock, cache)  # type: ignore[union-attr]
            w = 1.0 / (d2 ** half)
            num += w * val.at(clock, cache)  # type: ignore[union-attr]
            den += w
        return num / den if den else 0.0
