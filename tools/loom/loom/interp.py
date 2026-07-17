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
from .data import PointPath, TrackedPath, Grid, Scatter


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
        # thread through the PointPath *node* so the dataset is part of the DAG
        # (cycle detection walks it → a control point that loops back is caught).
        return (self._u, self.path)

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
# 1b. Reparam — retime a curve by a per-waypoint speed / density track
# ---------------------------------------------------------------------------

class Reparam(Signal):
    """Map a uniform **travel** parameter ``s`` to a **curve** parameter ``u`` so a
    point dwells longer where a per-waypoint ``weights`` (speed / density) track is
    large — the toolkit analog of a `camera_curve`'s *density* track retiming a
    flyby (spend more frames where density is high).

    The curve parameter ``u ∈ [0, 1)`` is split into ``len(weights)`` equal bins.
    Bin ``i`` is given a dwell proportional to ``weights[i]``: as ``s`` sweeps
    ``[0, 1)`` uniformly it spends fraction ``weights[i]/Σweights`` of its travel in
    bin ``i`` (an inverse-CDF), so large weight ⇒ slow ``u`` ⇒ the point lingers.
    For a **closed** curve the bins line up with the control points (bin ``i`` is the
    arc around waypoint ``i+1``, matching :func:`eval_curve`'s closed mapping); an
    **open** curve reuses the same normalized bins.

    ``weights`` are animatable Signals, so the speed profile can itself modulate over
    the loop.  Non-positive weights are floored to ``eps`` to keep the map strictly
    monotonic (no zero-width dwell / division by zero).  Feed the resulting ``u`` to a
    :class:`LoopCurve` / :class:`TrackedCurve` (see :meth:`TrackedCurve.traveling`).
    """

    def __init__(self, weights: List[Union[Signal, Number]], s: Union[Signal, Number],
                 *, closed: bool = True, eps: float = 1e-9) -> None:
        super().__init__()
        self.weights: List[Signal] = [as_signal(w) for w in weights]
        if not self.weights:
            raise ValueError("Reparam needs at least one weight")
        self.s = as_signal(s)
        self.closed = bool(closed)
        self.eps = float(eps)

    def children(self):
        return tuple(self.weights) + (self.s,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        w = [max(self.eps, wi.at(clock, cache)) for wi in self.weights]
        n = len(w)
        total = math.fsum(w)
        s = self.s.at(clock, cache)
        if self.closed:
            s -= math.floor(s)                     # wrap into [0, 1)
        else:
            s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        target = s * total
        acc = 0.0
        for i in range(n):
            if i == n - 1 or target < acc + w[i]:
                frac = (target - acc) / w[i]
                frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
                return (i + frac) / n
            acc += w[i]
        return 0.0  # unreachable (loop always returns)


# ---------------------------------------------------------------------------
# 1c. TrackedCurve — sample a sequence + all its side-tracks at one parameter
# ---------------------------------------------------------------------------

class TrackedCurve:
    """Sample a :class:`~loom.data.TrackedPath` at a shared curve parameter ``u``.

    The main **point** and **every** named track ride the *same* seamless
    midpoint-quadratic-Bézier over the *same* waypoints and the *same* ``u`` — this
    is the "Y curves onto one sequence" model of a `camera_curve` carrying a position
    curve **plus** a speed curve **plus** an orientation curve.  Internally each
    track is just another :class:`LoopCurve` sharing ``u``, so it composes with the
    rest of the DAG (cycle-checked, cached) exactly like any field.

    Attributes / methods:

    - :attr:`position` — the main point as a :class:`LoopCurve` (a ``VecSignal``);
    - :meth:`track` (or ``curve[name]``) — a named track, returned as a scalar
      :class:`Signal` if it was authored scalar, else a :class:`VecSignal`;
    - :meth:`traveling` — build one whose ``u`` is **retimed** by a scalar track via
      :class:`Reparam` (the camera-curve *speed* semantics, where the track doesn't
      just get sampled but changes how fast the sequence is traversed).
    """

    def __init__(self, tracked: TrackedPath, u: Union[Signal, Number],
                 *, closed: Optional[bool] = None) -> None:
        self.tracked = tracked
        self._u = as_signal(u)
        self.closed = tracked.closed if closed is None else bool(closed)
        self.position = LoopCurve(tracked.path, self._u, closed=self.closed)
        self._curves: dict = {}
        for name, pts in tracked.tracks.items():
            pp = PointPath(pts, closed=self.closed)
            self._curves[name] = LoopCurve(pp, self._u, closed=self.closed)

    @property
    def u(self) -> Signal:
        return self._u

    def track(self, name: str) -> Union[Signal, VecSignal]:
        if name not in self._curves:
            raise KeyError(f"no track named {name!r}")
        curve = self._curves[name]
        if self.tracked.is_scalar(name):
            return curve.components[0]          # scalar view of the 1-D track curve
        return curve

    def __getitem__(self, name: str) -> Union[Signal, VecSignal]:
        return self.track(name)

    def names(self) -> Tuple[str, ...]:
        return tuple(self._curves.keys())

    @classmethod
    def traveling(cls, tracked: TrackedPath, s: Union[Signal, Number],
                  density: str, *, closed: Optional[bool] = None) -> "TrackedCurve":
        """Build a :class:`TrackedCurve` whose parameter is **retimed** by a scalar
        ``density`` track: as the travel parameter ``s`` advances uniformly, the point
        (and every track sampled off the same ``u``) dwells where ``density`` is high —
        a camera-flyby speed curve.  ``density`` names a scalar track on ``tracked``.
        """
        closed_ = tracked.closed if closed is None else bool(closed)
        u = Reparam(tracked.weights_of(density), s, closed=closed_)
        return cls(tracked, u, closed=closed_)


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
        return tuple(self.q.components) + (self.grid,)

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
        return tuple(self.q.components) + (self.scatter,)

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
