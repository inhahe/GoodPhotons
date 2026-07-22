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


def _local_query(dataset, q: VecSignal) -> VecSignal:
    """If ``dataset`` carries a placement Transform, inverse-map the world-space
    query ``q`` into the dataset's local frame; otherwise return ``q`` unchanged.

    This is what decouples a sampling curve from a data object: the curve stays in
    world space while the dataset can be moved / resized / skewed, and the field
    reads whatever local coordinate the world point maps to (see
    :meth:`loom.transform.Transform.inverse_apply`)."""
    xf = getattr(dataset, "xf", None)
    if xf is not None and not xf.is_identity():
        return xf.inverse_apply(q)
    return q


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

_GRID_OUTSIDE = frozenset(("clamp", "raise", "wrap"))
_GRID_OOB_TOL = 1e-9


def _parse_on_outside(on_outside: str) -> str:
    """Validate a Grid out-of-domain policy name (``clamp``/``raise``/``wrap``)."""
    key = str(on_outside).lower()
    if key not in _GRID_OUTSIDE:
        raise ValueError(f"unknown on_outside {on_outside!r}; "
                         f"choose from {sorted(_GRID_OUTSIDE)}")
    return key


def _cell_base_frac(grid: Grid, axis: int, coord: float,
                    on_outside: str = "clamp") -> Tuple[int, float]:
    """Lower cell index ``i`` and in-cell fraction ``f in [0,1]`` for ``coord`` on
    ``axis``.

    ``on_outside`` picks the out-of-domain behaviour: ``"clamp"`` (default) pins to
    the boundary cell (edge-extend); ``"raise"`` errors past the domain; ``"wrap"``
    folds the coordinate periodically (period ``hi-lo``, so sample ``n-1`` aliases
    sample ``0`` — the stencil index wrap is applied by the callers)."""
    n = grid.shape[axis]
    lo, hi = grid.lo[axis], grid.hi[axis]
    if hi == lo:
        return 0, 0.0
    p = (coord - lo) / (hi - lo) * (n - 1)
    if on_outside == "wrap":
        span = n - 1
        p -= math.floor(p / span) * span          # fold into [0, n-1)
        i = int(math.floor(p))
        if i >= span:                              # numerical guard at the seam
            return 0, 0.0
        return i, p - i
    if p <= 0.0:
        if on_outside == "raise" and p < -_GRID_OOB_TOL:
            raise ValueError(f"grid query {coord} is below axis {axis} domain "
                             f"[{lo}, {hi}] (on_outside='raise')")
        return 0, 0.0
    if p >= n - 1:
        if on_outside == "raise" and p > (n - 1) + _GRID_OOB_TOL:
            raise ValueError(f"grid query {coord} is above axis {axis} domain "
                             f"[{lo}, {hi}] (on_outside='raise')")
        return n - 2, 1.0
    i = int(math.floor(p))
    return i, p - i


def _catmull_rom_axis(grid: Grid, axis: int, coord: float,
                      on_outside: str = "clamp") -> List[Tuple[int, float]]:
    """1-D Catmull-Rom contributions ``(sample_index, weight)`` on one axis.

    Four samples at offsets ``-1,0,+1,+2`` around the cell.  Weights sum to 1 but may
    be negative (the overshoot that gives cubic its snap).  Off the end of the axis the
    ``on_outside`` policy applies: ``"clamp"``/``"raise"`` **linearly extrapolate** a
    phantom point (``p[-1] = 2·p0 − p1``), folding its weight back onto the two edge
    samples — this keeps the boundary reproducing linear ramps exactly, unlike a plain
    edge-clamp; ``"wrap"`` folds the stencil index periodically (index ``n-1`` aliases
    ``0``).  Axes with < 3 samples fall back to linear (can't form the 4-point
    stencil)."""
    n = grid.shape[axis]
    i, f = _cell_base_frac(grid, axis, coord, on_outside)
    if n < 3:
        i1 = (i + 1) % (n - 1) if on_outside == "wrap" else i + 1
        return [(i, 1.0 - f), (i1, f)]
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
        if on_outside == "wrap":        # periodic: n-1 aliases 0
            k = idx % (n - 1)
            acc[k] = acc.get(k, 0.0) + wj
        elif idx < 0:                   # phantom below 0: 2·p0 − p1
            acc[0] = acc.get(0, 0.0) + 2.0 * wj
            acc[1] = acc.get(1, 0.0) - wj
        elif idx > n - 1:               # phantom above n-1: 2·p_{n-1} − p_{n-2}
            acc[n - 1] = acc.get(n - 1, 0.0) + 2.0 * wj
            acc[n - 2] = acc.get(n - 2, 0.0) - wj
        else:
            acc[idx] = acc.get(idx, 0.0) + wj
    return [(k, v) for k, v in acc.items() if v != 0.0]


def _grid_weights(grid: Grid, coords: Tuple[float, ...],
                  cubic: bool = False, on_outside: str = "clamp"
                  ) -> List[Tuple[int, float]]:
    """Separable interpolation weights for ``coords``: ``(flat_index, weight)`` list.

    ``cubic=False`` (default) is N-linear (2^ndim corners); ``cubic=True`` is
    separable **Catmull-Rom** (up to 4^ndim taps, weights may be negative).
    ``on_outside`` selects the out-of-domain behaviour (``clamp``/``raise``/``wrap``,
    see :func:`_cell_base_frac`).  Zero-weight taps are dropped.
    """
    wrap = (on_outside == "wrap")
    if not cubic:
        # fast N-linear path (kept dedicated for the common default).
        base: List[int] = []
        fracs: List[float] = []
        for axis in range(grid.ndim):
            i, f = _cell_base_frac(grid, axis, coords[axis], on_outside)
            base.append(i)
            fracs.append(f)
        out: List[Tuple[int, float]] = []
        for corner in range(1 << grid.ndim):
            w = 1.0
            idx: List[int] = []
            for axis in range(grid.ndim):
                bit = (corner >> axis) & 1
                k = base[axis] + bit
                if wrap:                       # periodic: n-1 aliases 0
                    k %= (grid.shape[axis] - 1)
                idx.append(k)
                w *= fracs[axis] if bit else (1.0 - fracs[axis])
            if w == 0.0:
                continue
            out.append((grid.flat_index(idx), w))
        return out
    # cubic: tensor product of per-axis Catmull-Rom contributions.
    combos: List[Tuple[List[int], float]] = [([], 1.0)]
    for axis in range(grid.ndim):
        contrib = _catmull_rom_axis(grid, axis, coords[axis], on_outside)
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
    Catmull-Rom / tricubic — smoother, C1, may overshoot).  ``on_outside`` picks the
    out-of-domain policy: ``"clamp"`` (default, edge-extend), ``"raise"`` (error past
    the domain — the guard you want when a curve must stay inside the box), or
    ``"wrap"`` (periodic fold, apt for a triply-periodic field like a gyroid).  For a
    vector-valued grid use :class:`VecGridField`.
    """

    def __init__(self, grid: Grid, query: Vecish, *, interp: str = "linear",
                 on_outside: str = "clamp") -> None:
        super().__init__()
        self.grid = grid
        self.q = VecSignal.of(query)
        if self.q.dim != grid.ndim:
            raise ValueError(f"query dim {self.q.dim} != grid ndim {grid.ndim}")
        if grid.is_vector:
            raise TypeError("GridField requires scalar grid values; "
                            "use VecGridField for a vector-valued Grid")
        self.q = _local_query(grid, self.q)
        self._cubic = _parse_grid_interp(interp)
        self._outside = _parse_on_outside(on_outside)

    def children(self):
        return tuple(self.q.components) + (self.grid,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        g = self.grid
        coords = self.q.at(clock, cache)
        return math.fsum(w * g.values[fi].at(clock, cache)  # type: ignore[union-attr]
                         for fi, w in _grid_weights(g, coords, self._cubic, self._outside))


class VecGridField(VecSignal):
    """Vector grid interpolation of a vector-valued :class:`Grid`.

    Every channel is blended with the *same* interpolation weights (computed once
    per frame), so this is a true vector field — not N independent scalar fields
    recomputing the domain math.  ``interp`` is ``"linear"`` (default) or ``"cubic"``
    (Catmull-Rom / tricubic).  ``on_outside`` picks the out-of-domain policy
    (``"clamp"`` default / ``"raise"`` / ``"wrap"``, see :class:`GridField`).
    ``.channel(name_or_index)`` returns a scalar view of one channel (by name if the
    grid was built with ``channels=``, else by index).
    """

    def __init__(self, grid: Grid, query: Vecish, *, interp: str = "linear",
                 on_outside: str = "clamp") -> None:
        # like LoopCurve: synthesize component views, override at()/children().
        self.grid = grid
        self.q = VecSignal.of(query)
        if self.q.dim != grid.ndim:
            raise ValueError(f"query dim {self.q.dim} != grid ndim {grid.ndim}")
        if not grid.is_vector:
            raise TypeError("VecGridField requires a vector-valued Grid; "
                            "use GridField for a scalar Grid")
        self.q = _local_query(grid, self.q)
        self._cubic = _parse_grid_interp(interp)
        self._outside = _parse_on_outside(on_outside)
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
        for fi, w in _grid_weights(g, coords, self._cubic, self._outside):
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
        self.q = _local_query(scatter, self.q)
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
        self.q = _local_query(scatter, self.q)
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


# ---------------------------------------------------------------------------
# 3b. RBF scatter interpolation — radial basis functions (scipy-backed)
# ---------------------------------------------------------------------------
#
# Shepard is robust but low-order (only C0, and it flattens toward the mean far
# from samples).  Radial-basis interpolation is smooth, **exact at the data points**
# (smoothing=0), works in any N-D with no meshing, and — via a single kernel
# factorization with a **multi-RHS** solve — interpolates all channels of a vector
# scatter for the price of one.  We defer to ``scipy.interpolate.RBFInterpolator``.
#
# Caveat baked into the API: an RBF **extrapolates (and overshoots) outside the
# convex hull** of the samples.  ``on_outside`` controls that — ``"clamp"`` (default)
# clips an out-of-hull result to the per-channel data range, ``"raise"`` errors,
# ``"extrapolate"`` returns the raw RBF value.  (No NaN-flag mode: loom's Signal
# contract forbids non-finite values, so "flag" is a hard ``raise``.)

_RBF_KERNELS = frozenset((
    "linear", "thin_plate_spline", "cubic", "quintic",
    "multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian",
))
_RBF_OUTSIDE = frozenset(("clamp", "raise", "extrapolate"))


def _require_scipy():
    try:
        import numpy as np  # noqa: F401
        from scipy.interpolate import RBFInterpolator  # noqa: F401
    except Exception as exc:  # pragma: no cover - only when scipy missing
        raise ImportError(
            "RBF scatter fields need numpy + scipy "
            "(pip install scipy) — not available: " + str(exc)) from exc


def _make_hull(P):
    """A cheap point-in-hull tester for the sample positions ``P`` (M x dim)."""
    import numpy as np
    dim = P.shape[1]
    if dim == 1:
        return ("1d", float(P[:, 0].min()), float(P[:, 0].max()))
    try:
        from scipy.spatial import Delaunay
        return ("delaunay", Delaunay(P))
    except Exception:
        # degenerate (collinear / coplanar / too few points) — fall back to bbox.
        return ("bbox", P.min(axis=0), P.max(axis=0))


def _inside_hull(hull, q, tol: float = 1e-9) -> bool:
    kind = hull[0]
    if kind == "1d":
        return hull[1] - tol <= float(q[0]) <= hull[2] + tol
    if kind == "delaunay":
        return bool(hull[1].find_simplex(q) >= 0)
    lo, hi = hull[1], hull[2]
    return bool((q >= lo - tol).all() and (q <= hi + tol).all())


class _RbfEngine:
    """Per-field RBF builder + evaluator that rebuilds **only when the sampled
    positions or values actually change**.

    The kernel factorization depends on the (animatable) sample positions and the
    values are its right-hand side.  Both are Signals, so in principle they can
    move every frame — but in practice a scatter field is usually authored with
    *static* positions (fixed sample sites), and very often static values too, and
    is then queried at many frames (e.g. sampled along a moving path).  Rebuilding
    the ``O(M^3)`` interpolator once per frame in that case is pure waste.

    So instead of a bare per-frame gate we cache the last built interpolator
    together with the exact position/value arrays it was built from.  When the
    frame advances we re-evaluate the sample arrays and, if they are bit-for-bit
    identical to the cached ones, reuse the interpolator untouched (bit-identical
    output).  Only a genuine change triggers a rebuild.  A whole vector scatter is
    one ``RBFInterpolator`` with a multi-column ``d`` (multi-RHS), which is why
    scalar and vector fields share this engine.
    """

    def __init__(self, scatter: Scatter, kernel: str, epsilon, smoothing,
                 degree, neighbors, on_outside: str) -> None:
        if kernel not in _RBF_KERNELS:
            raise ValueError(f"unknown RBF kernel {kernel!r}; "
                             f"choose from {sorted(_RBF_KERNELS)}")
        if on_outside not in _RBF_OUTSIDE:
            raise ValueError(f"unknown on_outside {on_outside!r}; "
                             f"choose from {sorted(_RBF_OUTSIDE)}")
        _require_scipy()
        self.scatter = scatter
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = float(smoothing)
        self.degree = degree
        self.neighbors = neighbors
        self.on_outside = on_outside
        self._frame: Optional[int] = None
        self._state = None
        self._P = None      # position array the cached interpolator was built from
        self._D = None      # value array the cached interpolator was built from
        self._builds = 0    # rebuild counter (tests assert the cache is reused)

    def _refresh(self, clock: Clock, cache: Optional[Cache]) -> None:
        """Ensure ``self._state`` reflects the samples at ``clock``, rebuilding the
        interpolator only if the position/value arrays changed since the last build."""
        import numpy as np
        sc = self.scatter
        P = np.asarray([pos.at(clock, cache) for pos in sc.positions], dtype=float)
        D = np.asarray([val.at(clock, cache) for val in sc.values], dtype=float)
        if (self._state is not None and self._P is not None
                and P.shape == self._P.shape and D.shape == self._D.shape
                and np.array_equal(P, self._P) and np.array_equal(D, self._D)):
            return  # samples unchanged — reuse the cached factorization verbatim
        self._build(P, D)
        self._P = P
        self._D = D

    def _build(self, P, D) -> None:
        import numpy as np
        from scipy.interpolate import RBFInterpolator
        rbf = RBFInterpolator(P, D, kernel=self.kernel, epsilon=self.epsilon,
                              smoothing=self.smoothing, degree=self.degree,
                              neighbors=self.neighbors)
        dmin = np.atleast_1d(D.min(axis=0))
        dmax = np.atleast_1d(D.max(axis=0))
        hull = None if self.on_outside == "extrapolate" else _make_hull(P)
        self._state = (rbf, dmin, dmax, hull)
        self._builds += 1

    def evaluate(self, q: Tuple[float, ...], clock: Clock, cache: Optional[Cache]):
        import numpy as np
        if self._frame != clock.frame or self._state is None:
            self._refresh(clock, cache)
            self._frame = clock.frame
        rbf, dmin, dmax, hull = self._state
        qa = np.asarray(q, dtype=float).reshape(1, -1)
        out = np.atleast_1d(np.asarray(rbf(qa)[0], dtype=float))
        if hull is not None and not _inside_hull(hull, qa[0]):
            if self.on_outside == "clamp":
                out = np.minimum(np.maximum(out, dmin), dmax)
            elif self.on_outside == "raise":
                raise ValueError(f"RBF query {q} is outside the sample convex hull")
        return out


class RbfScatterField(Signal):
    """Scalar RBF interpolation of a :class:`Scatter` at ``query`` (scipy-backed).

    Smooth, exact at the samples (``smoothing=0``), meshless, works in any N-D.
    ``kernel`` defaults to the parameter-free ``"thin_plate_spline"``; other scipy
    kernels (``multiquadric``/``gaussian``/… need ``epsilon``).  ``on_outside`` guards
    the convex-hull extrapolation (``"clamp"`` default; ``"raise"`` / ``"extrapolate"``).
    For vector samples use
    :class:`VecRbfScatterField`.
    """

    def __init__(self, scatter: Scatter, query: Vecish, *,
                 kernel: str = "thin_plate_spline", epsilon=None,
                 smoothing: float = 0.0, degree=None, neighbors=None,
                 on_outside: str = "clamp") -> None:
        super().__init__()
        self.scatter = scatter
        self.q = VecSignal.of(query)
        if self.q.dim != scatter.dim:
            raise ValueError(f"query dim {self.q.dim} != scatter dim {scatter.dim}")
        if scatter.is_vector:
            raise TypeError("RbfScatterField requires scalar values; "
                            "use VecRbfScatterField for a vector-valued Scatter")
        self.q = _local_query(scatter, self.q)
        self._eng = _RbfEngine(scatter, kernel, epsilon, smoothing,
                               degree, neighbors, on_outside)

    def children(self):
        return tuple(self.q.components) + (self.scatter,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        q = self.q.at(clock, cache)
        return float(self._eng.evaluate(q, clock, cache)[0])


class VecRbfScatterField(VecSignal):
    """Vector RBF interpolation of a vector-valued :class:`Scatter`.

    One ``RBFInterpolator`` with a multi-column right-hand side interpolates every
    channel from a single kernel factorization (the multi-RHS win).
    ``.channel(name_or_index)`` returns a scalar view of one channel.  See
    :class:`RbfScatterField` for the kernel / ``on_outside`` options.
    """

    def __init__(self, scatter: Scatter, query: Vecish, *,
                 kernel: str = "thin_plate_spline", epsilon=None,
                 smoothing: float = 0.0, degree=None, neighbors=None,
                 on_outside: str = "clamp") -> None:
        self.scatter = scatter
        self.q = VecSignal.of(query)
        if self.q.dim != scatter.dim:
            raise ValueError(f"query dim {self.q.dim} != scatter dim {scatter.dim}")
        if not scatter.is_vector:
            raise TypeError("VecRbfScatterField requires a vector-valued Scatter; "
                            "use RbfScatterField for scalar values")
        self.q = _local_query(scatter, self.q)
        self._eng = _RbfEngine(scatter, kernel, epsilon, smoothing,
                               degree, neighbors, on_outside)
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
        q = self.q.at(clock, cache)
        out = tuple(float(x) for x in self._eng.evaluate(q, clock, cache))
        if cache is not None:
            cache.set(self._id, clock.frame, out)
        return out


# ---------------------------------------------------------------------------
# 4. FieldCurve — a curve routed through a field (poll = coords + {channel: value})
# ---------------------------------------------------------------------------

class _MutableComp(Signal):
    """Scalar view of one axis of a :class:`_MutableVec` (never cached: it reads a
    value the owner mutates between evaluations, so caching would freeze it)."""

    def __init__(self, owner: "_MutableVec", axis: int) -> None:
        super().__init__()
        self.owner = owner
        self.axis = axis

    def children(self):
        return ()

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> float:
        return self.owner._val[self.axis]


class _MutableVec(VecSignal):
    """A settable query point used only for :meth:`FieldCurve.sample` probing."""

    def __init__(self, dim: int) -> None:
        self._val: Tuple[float, ...] = tuple(0.0 for _ in range(dim))
        self._id = alloc_id()
        self.components: List[Signal] = [_MutableComp(self, a) for a in range(dim)]

    def set(self, v) -> None:
        self._val = tuple(float(x) for x in v)

    def children(self):
        return tuple(self.components)

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        return self._val


class FieldCurve:
    """A loom curve **routed through a field**: at a progression index it yields the
    curve's N spatial coordinates *and* the field's ``{channel: value}`` there.

    Give it a ``PointPath`` (or a ready position ``VecSignal``) and a ``field``
    *builder* — a callable mapping a query ``VecSignal`` to a field node, e.g.
    ``lambda q: VecGridField(grid, q, interp="cubic")`` or ``lambda q: ScatterField(sc, q)``.
    Both the position and the sampled value are real DAG nodes, so either can drive
    scene variables (this is the object §E2's "curve variables drive scene variables"
    and §F6's inspection build on):

    - :attr:`position` — the spatial coordinate ``VecSignal`` (a ``LoopCurve`` when built
      from a ``PointPath``);
    - :attr:`value` — the field output (a scalar ``Signal`` or a ``VecSignal``);
    - :meth:`channel` — a scalar view of one value channel (by name if the dataset was
      built with ``channels=``, else by index);
    - :meth:`sample` — poll at an explicit progression ``u`` → ``(coords, {channel: value})``.
    """

    def __init__(self, curve, field, u=None, *, closed: Optional[bool] = None) -> None:
        if isinstance(curve, PointPath):
            if u is None:
                raise ValueError("FieldCurve over a PointPath needs a progression u")
            self.position = LoopCurve(curve, u, closed=closed)
        elif isinstance(curve, VecSignal):
            self.position = curve                       # a ready position signal
        else:
            raise TypeError("curve must be a PointPath or a VecSignal position")
        if not callable(field):
            raise TypeError(
                "field must be a builder callable(query_vecsignal) -> field node")
        self._build = field
        try:
            self.value = field(self.position)           # DAG-facing sampled value
        except ValueError as exc:
            # sharpen the common "curve dim != field domain dim" mismatch
            raise ValueError(
                f"FieldCurve: curve position (dim {self.position.dim}) is "
                f"incompatible with the field — {exc}") from exc
        self.is_vector = isinstance(self.value, VecSignal)
        # discover channel names from the underlying dataset, if any.
        ds = getattr(self.value, "grid", None)
        if ds is None:
            ds = getattr(self.value, "scatter", None)
        self.channel_names: Optional[Tuple[str, ...]] = (
            tuple(ds.channels) if (ds is not None and getattr(ds, "channels", None)) else None)
        # a separate probe field over a mutable query, for explicit-u polling.
        self._poke = _MutableVec(self.position.dim)
        self._probe = field(self._poke)

    # ---- DAG accessors ------------------------------------------------------
    @property
    def coords(self) -> VecSignal:
        return self.position

    def channel(self, channel) -> Signal:
        if isinstance(self.value, VecSignal):
            if hasattr(self.value, "channel"):
                return self.value.channel(channel)      # named/indexed view
            return self.value.components[int(channel)]
        if channel in (0, None):
            return self.value                           # scalar field
        raise KeyError(f"scalar field has no channel {channel!r}")

    def channels(self) -> Optional[Tuple[str, ...]]:
        return self.channel_names

    # ---- polling ------------------------------------------------------------
    def _value_map(self, v) -> dict:
        if isinstance(v, tuple):
            if self.channel_names:
                return dict(zip(self.channel_names, v))
            return {i: x for i, x in enumerate(v)}
        return {self.channel_names[0] if self.channel_names else 0: v}

    def sample(self, u_value: float, clock: Clock, cache: Optional[Cache] = None):
        """Poll at progression ``u_value``: returns ``(coords_tuple, {channel: value})``.

        ``coords`` are the curve's spatial coordinates at ``u_value``; the value map is
        the field sampled at those coordinates.  Requires a curve that supports explicit
        parameter sampling (a ``LoopCurve``, i.e. built from a ``PointPath``)."""
        if not hasattr(self.position, "sample"):
            raise TypeError("FieldCurve.sample needs a LoopCurve position "
                            "(build the FieldCurve from a PointPath)")
        coords = self.position.sample(u_value, clock, cache)
        self._poke.set(coords)
        v = self._probe.at(clock, None)     # uncached: reflect the just-set poke
        return coords, self._value_map(v)
