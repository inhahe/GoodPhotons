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

Grids and scatters may store **vector** values (a :class:`~loom.signals.vector.VecSignal`
per sample, an optional named-channel model).  :class:`VecGridField` /
:class:`VecScatterField` interpolate those as a ``VecSignal``, computing the shared
domain weights once and blending every channel with them.
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
# 2. Grid / Scatter fields — shared domain weights, scalar OR vector valued
# ---------------------------------------------------------------------------
#
# Both interpolators split into a purely *geometric* weight computation (which
# grid corners / scatter samples contribute, and how much) and a *value* blend
# (apply those weights to each stored channel).  The geometry depends only on the
# query point (and, for scatter, the sample positions) — never on the number of
# value channels — so scalar and vector fields share one weight kernel and the
# vector field pays for the domain math exactly once per frame, then reuses the
# weights across every channel.

def _cell_base_frac(grid: Grid, axis: int, coord: float) -> Tuple[int, float]:
    """Lower cell index ``i`` and in-cell fraction ``f in [0,1]`` for ``coord`` on
    ``axis``.  Out-of-domain coords clamp to the boundary cell (edge-extend)."""
    n = grid.shape[axis]
    lo, hi = grid.lo[axis], grid.hi[axis]
    p = (coord - lo) / (hi - lo) * (n - 1) if hi != lo else 0.0
    if p <= 0.0:
        return 0, 0.0
    if p >= n - 1:
        return n - 2, 1.0
    i = int(math.floor(p))
    return i, p - i


def _catmull_rom_axis(grid: Grid, axis: int, coord: float) -> List[Tuple[int, float]]:
    """1-D Catmull-Rom contributions ``(sample_index, weight)`` on one axis.

    Four samples at offsets ``-1,0,+1,+2`` around the cell.  Weights sum to 1 but may
    be negative (the overshoot that gives cubic its snap).  A phantom point off the
    end of the axis is **linearly extrapolated** (``p[-1] = 2·p0 − p1``), folding its
    weight back onto the two edge samples — this keeps the boundary reproducing linear
    ramps exactly, unlike a plain edge-clamp.  Axes with < 3 samples fall back to
    linear (can't form the 4-point stencil)."""
    n = grid.shape[axis]
    i, f = _cell_base_frac(grid, axis, coord)
    if n < 3:
        return [(i, 1.0 - f), (i + 1, f)]
    f2 = f * f
    f3 = f2 * f
    w = (0.5 * (-f3 + 2.0 * f2 - f),
         0.5 * (3.0 * f3 - 5.0 * f2 + 2.0),
         0.5 * (-3.0 * f3 + 4.0 * f2 + f),
         0.5 * (f3 - f2))
    acc: dict = {}
    for off, wj in zip((-1, 0, 1, 2), w):
        if wj == 0.0:
            continue
        idx = i + off
        if idx < 0:                     # phantom below 0: 2·p0 − p1
            acc[0] = acc.get(0, 0.0) + 2.0 * wj
            acc[1] = acc.get(1, 0.0) - wj
        elif idx > n - 1:               # phantom above n-1: 2·p_{n-1} − p_{n-2}
            acc[n - 1] = acc.get(n - 1, 0.0) + 2.0 * wj
            acc[n - 2] = acc.get(n - 2, 0.0) - wj
        else:
            acc[idx] = acc.get(idx, 0.0) + wj
    return [(k, v) for k, v in acc.items() if v != 0.0]


def _grid_weights(grid: Grid, coords: Tuple[float, ...],
                  cubic: bool = False) -> List[Tuple[int, float]]:
    """Separable interpolation weights for ``coords``: ``(flat_index, weight)`` list.

    ``cubic=False`` (default) is N-linear (2^ndim corners); ``cubic=True`` is
    separable **Catmull-Rom** (up to 4^ndim taps, weights may be negative).  Both
    edge-extend outside the domain and drop zero-weight taps.
    """
    if not cubic:
        # fast N-linear path (kept dedicated for the common default).
        base: List[int] = []
        fracs: List[float] = []
        for axis in range(grid.ndim):
            i, f = _cell_base_frac(grid, axis, coords[axis])
            base.append(i)
            fracs.append(f)
        out: List[Tuple[int, float]] = []
        for corner in range(1 << grid.ndim):
            w = 1.0
            idx: List[int] = []
            for axis in range(grid.ndim):
                bit = (corner >> axis) & 1
                idx.append(base[axis] + bit)
                w *= fracs[axis] if bit else (1.0 - fracs[axis])
            if w == 0.0:
                continue
            out.append((grid.flat_index(idx), w))
        return out
    # cubic: tensor product of per-axis Catmull-Rom contributions.
    combos: List[Tuple[List[int], float]] = [([], 1.0)]
    for axis in range(grid.ndim):
        contrib = _catmull_rom_axis(grid, axis, coords[axis])
        combos = [(idxs + [ci], w * cw)
                  for idxs, w in combos for ci, cw in contrib]
    return [(grid.flat_index(idxs), w) for idxs, w in combos if w != 0.0]


def _shepard_weights(scatter: Scatter, q: Tuple[float, ...], half: float, eps: float,
                     clock: Clock, cache: Optional[Cache]
                     ) -> Tuple[Optional[int], List[Tuple[int, float]]]:
    """Shepard inverse-distance weights of ``scatter`` at ``q``.

    Returns ``(coincident_index, weights)``.  If the query coincides with a
    sample (within ``eps`` squared distance) ``coincident_index`` is that sample's
    index and ``weights`` is empty — the caller returns that sample exactly.
    Otherwise ``coincident_index`` is ``None`` and ``weights`` is the (unnormalized)
    ``(index, weight)`` list.  ``half = power / 2`` (applied to squared distance).
    """
    out: List[Tuple[int, float]] = []
    for i, pos in enumerate(scatter.positions):
        p = pos.at(clock, cache)
        d2 = sum((a - b) ** 2 for a, b in zip(q, p))
        if d2 <= eps:
            return i, []
        out.append((i, 1.0 / (d2 ** half)))
    return None, out


def _parse_grid_interp(interp: str) -> bool:
    """Map a grid ``interp`` name to the ``cubic`` flag ``_grid_weights`` takes."""
    key = str(interp).lower()
    if key in ("linear", "multilinear", "nlinear"):
        return False
    if key in ("cubic", "tricubic", "catmull", "catmull-rom", "catmull_rom"):
        return True
    raise ValueError(f"unknown grid interp {interp!r} (use 'linear' or 'cubic')")


class _VecFieldComponent(Signal):
    """Scalar view of one channel of a vector field (mirrors ``_CurveComponent``)."""

    def __init__(self, field: VecSignal, axis: int) -> None:
        super().__init__()
        self.field = field
        self.axis = axis

    def children(self):
        return (self.field,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.field.at(clock, cache)[self.axis]


class GridField(Signal):
    """Scalar grid interpolation of a :class:`Grid` at ``query`` (a VecSignal).

    Query rank must equal the grid's ndim.  ``interp`` selects the kernel:
    ``"linear"`` (default, separable N-linear) or ``"cubic"`` (separable
    Catmull-Rom / tricubic — smoother, C1, may overshoot).  Out-of-domain queries
    are clamped to the boundary cell (edge-extend).  For a vector-valued grid use
    :class:`VecGridField`.
    """

    def __init__(self, grid: Grid, query: Vecish, *, interp: str = "linear") -> None:
        super().__init__()
        self.grid = grid
        self.q = VecSignal.of(query)
        if self.q.dim != grid.ndim:
            raise ValueError(f"query dim {self.q.dim} != grid ndim {grid.ndim}")
        if grid.is_vector:
            raise TypeError("GridField requires scalar grid values; "
                            "use VecGridField for a vector-valued Grid")
        self._cubic = _parse_grid_interp(interp)

    def children(self):
        return tuple(self.q.components) + (self.grid,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        g = self.grid
        coords = self.q.at(clock, cache)
        return math.fsum(w * g.values[fi].at(clock, cache)  # type: ignore[union-attr]
                         for fi, w in _grid_weights(g, coords, self._cubic))


class VecGridField(VecSignal):
    """Vector grid interpolation of a vector-valued :class:`Grid`.

    Every channel is blended with the *same* interpolation weights (computed once
    per frame), so this is a true vector field — not N independent scalar fields
    recomputing the domain math.  ``interp`` is ``"linear"`` (default) or ``"cubic"``
    (Catmull-Rom / tricubic).  ``.channel(name_or_index)`` returns a scalar view of
    one channel (by name if the grid was built with ``channels=``, else by index).
    """

    def __init__(self, grid: Grid, query: Vecish, *, interp: str = "linear") -> None:
        # like LoopCurve: synthesize component views, override at()/children().
        self.grid = grid
        self.q = VecSignal.of(query)
        if self.q.dim != grid.ndim:
            raise ValueError(f"query dim {self.q.dim} != grid ndim {grid.ndim}")
        if not grid.is_vector:
            raise TypeError("VecGridField requires a vector-valued Grid; "
                            "use GridField for a scalar Grid")
        self._cubic = _parse_grid_interp(interp)
        self._vdim = grid.value_dim
        self._id = alloc_id()
        self.components: List[Signal] = [
            _VecFieldComponent(self, a) for a in range(self._vdim)]

    def children(self):
        return tuple(self.q.components) + (self.grid,)

    def channel(self, channel) -> Signal:
        return self.components[self.grid.channel_index(channel)]

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        g = self.grid
        coords = self.q.at(clock, cache)
        acc = [0.0] * self._vdim
        for fi, w in _grid_weights(g, coords, self._cubic):
            vv = g.values[fi].at(clock, cache)  # tuple (VecSignal value)
            for a in range(self._vdim):
                acc[a] += w * vv[a]
        out = tuple(acc)
        if cache is not None:
            cache.set(self._id, clock.frame, out)
        return out


# ---------------------------------------------------------------------------
# 3. ScatterField — inverse-distance (Shepard) interpolation
# ---------------------------------------------------------------------------

class ScatterField(Signal):
    """Scalar smooth interpolation of a :class:`Scatter` set at ``query``.

    Shepard inverse-distance weighting with exponent ``power``.  Simple and
    robust; quality/speed tradeoffs (RBF, natural neighbour) are a documented
    open item (see DESIGN.md §11).  Exactly reproduces a sample's value when the
    query coincides with that sample.  For vector-valued samples use
    :class:`VecScatterField`.
    """

    def __init__(self, scatter: Scatter, query: Vecish, *,
                 power: float = 2.0, eps: float = 1e-9) -> None:
        super().__init__()
        self.scatter = scatter
        self.q = VecSignal.of(query)
        if self.q.dim != scatter.dim:
            raise ValueError(f"query dim {self.q.dim} != scatter dim {scatter.dim}")
        if scatter.is_vector:
            raise TypeError("ScatterField requires scalar values; "
                            "use VecScatterField for a vector-valued Scatter")
        self.power = float(power)
        self.eps = float(eps)

    def children(self):
        return tuple(self.q.components) + (self.scatter,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        sc = self.scatter
        q = self.q.at(clock, cache)
        coincident, weights = _shepard_weights(
            sc, q, self.power * 0.5, self.eps, clock, cache)
        if coincident is not None:
            return sc.values[coincident].at(clock, cache)  # type: ignore[union-attr]
        num = math.fsum(w * sc.values[i].at(clock, cache)  # type: ignore[union-attr]
                        for i, w in weights)
        den = math.fsum(w for _, w in weights)
        return num / den if den else 0.0


class VecScatterField(VecSignal):
    """Vector Shepard interpolation of a vector-valued :class:`Scatter`.

    All channels share the *same* inverse-distance weights (computed once per
    frame).  Reproduces a sample exactly when the query coincides with it.
    ``.channel(name_or_index)`` returns a scalar view of one channel.
    """

    def __init__(self, scatter: Scatter, query: Vecish, *,
                 power: float = 2.0, eps: float = 1e-9) -> None:
        self.scatter = scatter
        self.q = VecSignal.of(query)
        if self.q.dim != scatter.dim:
            raise ValueError(f"query dim {self.q.dim} != scatter dim {scatter.dim}")
        if not scatter.is_vector:
            raise TypeError("VecScatterField requires a vector-valued Scatter; "
                            "use ScatterField for scalar values")
        self.power = float(power)
        self.eps = float(eps)
        self._vdim = scatter.value_dim
        self._id = alloc_id()
        self.components: List[Signal] = [
            _VecFieldComponent(self, a) for a in range(self._vdim)]

    def children(self):
        return tuple(self.q.components) + (self.scatter,)

    def channel(self, channel) -> Signal:
        return self.components[self.scatter.channel_index(channel)]

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        sc = self.scatter
        q = self.q.at(clock, cache)
        coincident, weights = _shepard_weights(
            sc, q, self.power * 0.5, self.eps, clock, cache)
        if coincident is not None:
            out = sc.values[coincident].at(clock, cache)  # type: ignore[union-attr]
        else:
            acc = [0.0] * self._vdim
            den = 0.0
            for i, w in weights:
                vv = sc.values[i].at(clock, cache)
                for a in range(self._vdim):
                    acc[a] += w * vv[a]
                den += w
            out = tuple(a / den for a in acc) if den else tuple(acc)
        if cache is not None:
            cache.set(self._id, clock.frame, out)
        return out
