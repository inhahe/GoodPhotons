"""
Loom datasets — three ways to store N-D values, every one *animatable*.

Each stored value may itself be a :class:`~loom.signals.core.Signal` or
:class:`~loom.signals.vector.VecSignal`, so control points / grid samples /
scatter values can be driven by modulators.  The datasets are plain containers;
the *interpolators* that turn them into fields live in :mod:`loom.interp`.

1. :class:`PointPath` — ordered N-D points (a curve's control points).
2. :class:`Grid`      — N-D values on a regular lattice of arbitrary rarity.
3. :class:`Scatter`   — N-D values at arbitrary positions (no lattice).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Number, as_signal, alloc_id
from .signals.vector import VecSignal, Vecish


def _infer_value_dim(values: Sequence[Union[Signal, VecSignal]]) -> Tuple[int, bool]:
    """Inspect a dataset's stored values and return ``(value_dim, is_vector)``.

    A dataset is either **all scalar** (every value a :class:`Signal`, giving
    ``value_dim == 1`` and ``is_vector == False``) or **all vector** (every value a
    :class:`VecSignal` sharing one dimension, giving that dimension and
    ``is_vector == True``).  Mixing the two, or mixing vector dimensions, is a
    construction-time error — a field can only interpolate a uniform channel model.
    """
    vec: Optional[bool] = None
    dim: Optional[int] = None
    for v in values:
        this_vec = isinstance(v, VecSignal)
        this_dim = v.dim if this_vec else 1
        if vec is None:
            vec, dim = this_vec, this_dim
        elif this_vec != vec:
            raise ValueError("dataset values must be all scalar or all vector")
        elif this_dim != dim:
            raise ValueError(
                f"all vector values must share a dimension ({dim} vs {this_dim})")
    return int(dim or 1), bool(vec)


def _check_channels(channels: Optional[Sequence[str]], value_dim: int) -> Optional[Tuple[str, ...]]:
    if channels is None:
        return None
    names = tuple(str(c) for c in channels)
    if len(names) != value_dim:
        raise ValueError(
            f"channels has {len(names)} names but values have dimension {value_dim}")
    if len(set(names)) != len(names):
        raise ValueError("channel names must be unique")
    return names


class _Transformable:
    """Mixin giving a dataset an optional local→world :class:`~loom.transform.Transform`.

    The transform does **not** move the stored samples; it defines how the dataset's
    fixed *local* frame is placed in world space.  A field sampled by a world-space
    curve inverse-maps the query through this transform (see
    :meth:`loom.transform.Transform.inverse_apply`), so transforming the data object
    changes what values the (independent) curve reads back — while the curve itself
    stays put.  Every transform parameter is signal-modulatable.
    """

    xf = None  # type: ignore[assignment]

    def transformed(self, transform=None, *, translate=None, rotate=None,
                    scale=None, skew=None):
        """Attach a :class:`~loom.transform.Transform` (position / size / rotation /
        skew, all signal-modulatable) placing this dataset's local frame in world
        space, and return ``self`` for chaining.  2-D datasets use only the in-plane
        parameters (translate/scale XY, rotate about Z, skew X-along-Y)."""
        from .transform import Transform
        self.xf = transform if transform is not None else Transform(
            translate=translate, rotate=rotate, scale=scale, skew=skew)
        return self


def _resolve_channel(channels: Optional[Tuple[str, ...]], value_dim: int,
                     channel: Union[int, str]) -> int:
    """Map a channel selector (index or name) to a component index in range."""
    if isinstance(channel, str):
        if channels is None:
            raise KeyError(f"dataset has no named channels (asked for {channel!r})")
        try:
            return channels.index(channel)
        except ValueError:
            raise KeyError(f"no channel named {channel!r} (have {list(channels)})")
    i = int(channel)
    if i < 0:
        i += value_dim
    if not (0 <= i < value_dim):
        raise IndexError(f"channel index {channel} out of range for dim {value_dim}")
    return i


class PointPath:
    """An ordered sequence of N-D points (each an animatable ``VecSignal``).

    A dataset is a **node in the modulation DAG** (it carries an ``id`` and
    ``children()``), so it can be both *modulable* (its stored control points are
    Signals/VecSignals driven by modulators) **and** a *modulator* (an interpolator
    over it is a Signal that can feed other nodes).  Because it is a real node,
    :func:`~loom.signals.core.detect_signal_cycle` walks through the dataset and
    catches any loop that passes through a control point.
    """

    def __init__(self, points: Iterable[Vecish], *, closed: bool = True) -> None:
        self.points: List[VecSignal] = [VecSignal.of(p) for p in points]
        if len(self.points) < 2:
            raise ValueError("PointPath needs at least 2 points")
        self.dim = self.points[0].dim
        for p in self.points:
            if p.dim != self.dim:
                raise ValueError("all PointPath points must share a dimension")
        self.closed = bool(closed)
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, i: int) -> VecSignal:
        return self.points[i]

    def children(self) -> Tuple[VecSignal, ...]:
        return tuple(self.points)


class TrackedPath:
    """A :class:`PointPath` that carries **extra per-waypoint tracks**.

    This is the toolkit analog of a `camera_curve`: one ordered set of control
    points (the *sequence*) where each waypoint bundles not just a main N-D point
    but any number of side values — a scalar **speed / density** track, a vector
    **orientation** track, a **scale** or **colour** track, whatever you key.  A
    track is one value *per control point* (so it has the same length and the same
    ``closed``-ness as the main path), and every track is sampled on the **same**
    seamless curve parameter as the main point by :class:`~loom.interp.TrackedCurve`
    — exactly the way a camera flyby's speed and look-direction curves ride along
    its position curve.

    Each track value may be:

    - **scalar** — a :class:`~loom.signals.core.Signal` or plain number (stored as a
      1-D vector internally, read back out as a scalar ``Signal``); or
    - **vector** — a :class:`~loom.signals.vector.VecSignal` or a sequence
      (numbers / Signals), e.g. an N-D orientation.

    All values within one track must share a dimension.  Track values are
    animatable like everything else in Loom (a per-point speed can itself be
    driven by a modulator).
    """

    def __init__(self, points: Iterable[Vecish], *,
                 tracks: Optional[dict] = None, closed: bool = True) -> None:
        self.path = PointPath(points, closed=closed)
        self.closed = self.path.closed
        self.dim = self.path.dim
        self.tracks: dict = {}          # name -> List[VecSignal] (per control point)
        self._scalar: dict = {}         # name -> bool (was authored as a scalar track)
        for name, values in (tracks or {}).items():
            self.add_track(name, values)
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def __len__(self) -> int:
        return len(self.path)

    @property
    def npoints(self) -> int:
        return len(self.path)

    def add_track(self, name: str, values: Iterable) -> "TrackedPath":
        """Attach one value per control point under ``name`` (scalar or vector)."""
        vals = list(values)
        if len(vals) != self.npoints:
            raise ValueError(
                f"track {name!r} has {len(vals)} values but the path has "
                f"{self.npoints} control points")
        scalar = all(isinstance(v, (Signal, int, float)) for v in vals)
        pts: List[VecSignal] = []
        for v in vals:
            if scalar:
                pts.append(VecSignal([as_signal(v)]))       # 1-D
            else:
                pts.append(VecSignal.of(v))
        d = pts[0].dim
        if any(p.dim != d for p in pts):
            raise ValueError(f"all values of track {name!r} must share a dimension")
        self.tracks[name] = pts
        self._scalar[name] = scalar
        return self

    def track_points(self, name: str) -> List[VecSignal]:
        if name not in self.tracks:
            raise KeyError(f"no track named {name!r}")
        return self.tracks[name]

    def is_scalar(self, name: str) -> bool:
        return bool(self._scalar[name])

    def weights_of(self, name: str) -> List[Signal]:
        """Per-control-point scalar values of a scalar track (e.g. a speed/density
        track), as a list of :class:`Signal` — the input to a reparameterization."""
        if not self._scalar.get(name, False):
            raise ValueError(f"track {name!r} is not a scalar track")
        return [p.components[0] for p in self.tracks[name]]

    def children(self):
        kids: List = [self.path]
        for pts in self.tracks.values():
            kids.extend(pts)
        return tuple(kids)


def _query_of(query: tuple):
    """Interpret a dataset-call argument list into a query the field builders accept.

    ``ds(x, y)`` → components ``(x, y)``; ``ds(vec(...))`` / ``ds([x, y])`` → the given
    vector/sequence verbatim; ``ds(x)`` (a lone scalar, a 1-D field) → the 1-tuple
    ``(x,)`` (so it is NOT mistaken for an already-assembled vector)."""
    if len(query) == 1:
        a = query[0]
        return a if isinstance(a, (VecSignal, list, tuple)) else query
    return query


def _resolve_lo(lo, ndim: int) -> Tuple[float, ...]:
    """Normalize a ``lo`` corner: ``None`` → all-zeros; a scalar → broadcast to every
    axis; a sequence → per-axis (length must match ``ndim``)."""
    if lo is None:
        return tuple(0.0 for _ in range(ndim))
    if isinstance(lo, (int, float)):
        return tuple(float(lo) for _ in range(ndim))
    seq = tuple(float(x) for x in lo)
    if len(seq) != ndim:
        raise ValueError(f"lo has {len(seq)} values but grid is {ndim}-D")
    return seq


def _resolve_hi(hi, lo: Tuple[float, ...], shape: Tuple[int, ...]) -> Tuple[float, ...]:
    """Normalize a ``hi`` corner.

    - ``None`` → a **unit-spacing index lattice** (``hi[a] = lo[a] + shape[a]-1``), so a
      query coordinate equals a sample index (spacing 1 on every axis).
    - a scalar (or a length-1 sequence) → the **axis-0 upper corner**; every other axis
      is derived to make a **uniform lattice** (one spacing ``h`` on all axes, so the
      interpolated field is geometrically isotropic):
      ``h = (hi - lo[0]) / (shape[0]-1)``, ``hi[a] = lo[a] + h·(shape[a]-1)``.
    - a full sequence → per-axis corners verbatim (the exact box; allows deliberately
      anisotropic cells).
    """
    ndim = len(shape)
    if hi is None:
        return tuple(lo[a] + (shape[a] - 1) for a in range(ndim))
    if isinstance(hi, (int, float)):
        scalar = float(hi)
    else:
        seq = tuple(float(x) for x in hi)
        if len(seq) == 1:
            scalar = seq[0]
        elif len(seq) != ndim:
            raise ValueError(f"hi has {len(seq)} values but grid is {ndim}-D")
        else:
            return seq
    h = (scalar - lo[0]) / (shape[0] - 1)          # uniform lattice spacing
    return tuple(lo[a] + h * (shape[a] - 1) for a in range(ndim))


def _is_grid_leaf(v) -> bool:
    """True if ``v`` is a grid *value* (a leaf), not a structural axis container.

    Vectors are :class:`VecSignal` (built with ``vec(...)``), and scalars are
    numbers / :class:`Signal`\\ s — none of which are bare ``list``/``tuple``\\ s.  So a
    bare ``list`` or ``tuple`` in a values tree always denotes a **grid axis**, never a
    stored value.  (To store a vector value, wrap the components in ``vec(...)``.)
    """
    return not isinstance(v, (list, tuple))


def _flatten_nested(values) -> Tuple[Tuple[int, ...], list]:
    """Flatten a possibly-nested values container to ``(shape, flat_C_order_list)``.

    The nesting itself carries the lattice ``shape`` — a flat list is 1-D, a
    list-of-rows is 2-D (``[[a b c][d e f]]`` → shape ``(2, 3)``), and so on — so an
    explicit ``shape=`` is redundant when the data is written out structurally.  The
    nesting must be **rectangular**: every sibling subtree must share a shape, else the
    grid is ragged and a :class:`Scatter` is the right container instead.
    """
    def rec(node) -> Tuple[Tuple[int, ...], list]:
        if _is_grid_leaf(node):
            return (), [node]
        children = list(node)
        if not children:
            raise ValueError("grid axis has zero length")
        flat: list = []
        first: Optional[Tuple[int, ...]] = None
        for c in children:
            s, f = rec(c)
            if first is None:
                first = s
            elif s != first:
                raise ValueError(
                    f"ragged grid values: rows have differing shapes ({first} vs {s}) "
                    "— a Grid lattice must be rectangular (use Scatter for ragged data)")
            flat.extend(f)
        return (len(children),) + (first or ()), flat

    return rec(list(values))


class Grid(_Transformable):
    """N-D scalar-or-vector values on a **regular, fixed** lattice.

    ``values`` holds the samples, as Signals (scalar field) or VecSignals (vector
    field).  It may be written **nested** — ``[[0 1 2][3 4 5]]`` — in which case the
    nesting *is* the shape (here ``(2, 3)``); no ``shape=`` is needed because the data
    already makes the lattice obvious.  A flat list is read as 1-D unless you pass an
    explicit ``shape=`` to fold it into N-D (``Grid([0,1,2,3,4,5], shape=(2, 3))``).
    Bare lists always mean *axes*; to store a vector value wrap it in ``vec(...)``.

    ``shape`` (keyword, optional) — samples per axis; inferred from the nesting when
    omitted.  Give it only to reshape a flat list, or to assert an expected shape.

    ``lo``/``hi`` place the lattice in space and are both optional/broadcastable:

    - ``lo`` — ``None`` (default) → all-zeros; a scalar → broadcast to every axis; a
      sequence → per-axis corners.
    - ``hi`` — ``None`` (default) → a unit-spacing index lattice (``lo[a]+shape[a]-1``);
      a scalar (or length-1) → the axis-0 upper corner with the other axes derived as a
      **uniform lattice** (single isotropic spacing); a full sequence → the exact box.

    The lattice **positions are deliberately fixed** — that regular structure is the
    whole point of a Grid (it buys the fast separable N-linear interpolation).  Only
    the *values* at those positions are modulable.  If you want moving sample
    *positions*, that is exactly what :class:`Scatter` is for.

    Call the grid like a function of position — ``grid(x, y)`` (or ``grid(vec(x, y))``)
    — to build the interpolating field :class:`~loom.interp.GridField` /
    :class:`~loom.interp.VecGridField` (a Signal, matching ftsl's ``n(x,y)``), or
    :meth:`sample` for an eager numeric read at an explicit point.
    """

    def __init__(self, values: Iterable[Union[Signal, VecSignal, Number]],
                 *, shape: Optional[Sequence[int]] = None,
                 lo: Optional[Union[float, Sequence[float]]] = None,
                 hi: Optional[Union[float, Sequence[float]]] = None,
                 channels: Optional[Sequence[str]] = None):
        inferred_shape, flat = _flatten_nested(values)
        self.shape: Tuple[int, ...] = (
            inferred_shape if shape is None else tuple(int(s) for s in shape))
        if any(s < 2 for s in self.shape):
            raise ValueError("each grid axis needs >= 2 samples")
        self.ndim = len(self.shape)
        self.lo = _resolve_lo(lo, self.ndim)
        self.hi = _resolve_hi(hi, self.lo, self.shape)
        n = 1
        for s in self.shape:
            n *= s
        if len(flat) != n:
            raise ValueError(f"expected {n} values, got {len(flat)}")
        self.values: List[Union[Signal, VecSignal]] = [
            v if isinstance(v, (Signal, VecSignal)) else as_signal(v) for v in flat
        ]
        self.value_dim, self.is_vector = _infer_value_dim(self.values)
        self.channels = _check_channels(channels, self.value_dim)
        # C-order strides
        self._strides: List[int] = [1] * self.ndim
        for a in range(self.ndim - 2, -1, -1):
            self._strides[a] = self._strides[a + 1] * self.shape[a + 1]
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def __call__(self, *query, interp: str = "linear", on_outside: str = "clamp"):
        """Sample this grid as a field of position — in **either tier**, chosen by what
        the query is made of.

        * A *temporal* query (numbers, :class:`~loom.signals.core.Signal`\\s,
          ``vec(...)``) → a :class:`~loom.interp.GridField` /
          :class:`~loom.interp.VecGridField`: a Signal node in the modulation DAG.
          Call ``.at(clock)`` to evaluate it, or :meth:`sample` for an eager number.
        * A *spatial* query — any :class:`~loom.spatial.SpatialExpr` argument, e.g.
          ``grid(X, Y)`` → a :class:`~loom.spatial.GridSample`: a term of the field
          algebra that **renders**, emitting ftsl's ``grid:<name>(c0, …)`` table call
          with its ``grid { … }`` block collected automatically by
          :meth:`loom.Scene.add`.

        Both tiers interpolate identically (loom's ``_grid_weights`` and ftrace's
        ``patGridSample`` are documented twins), so the choice is only about *where*
        the answer is consumed, not what it is."""
        from .interp import GridField, VecGridField   # lazy: avoid data<->interp cycle
        from .spatial import SpatialExpr, GridSample
        q = _query_of(query)
        if isinstance(q, (list, tuple)) and any(isinstance(c, SpatialExpr) for c in q):
            return GridSample(self, q, interp=interp, on_outside=on_outside)
        Field = VecGridField if self.is_vector else GridField
        return Field(self, q, interp=interp, on_outside=on_outside)

    def sample(self, *query, clock=None, interp: str = "linear",
               on_outside: str = "clamp"):
        """Eagerly read the grid at an explicit query point: returns a ``float`` (scalar
        grid) or a component ``tuple`` (vector grid).  ``clock`` defaults to a static
        frame-0 clock (fine for non-animated grids); pass one to evaluate animated
        values at a specific frame."""
        from .signals.core import Clock
        field = self(*query, interp=interp, on_outside=on_outside)
        clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
        return field.at(clk)

    def flat_index(self, idx: Sequence[int]) -> int:
        if len(idx) != self.ndim:
            raise ValueError("index rank mismatch")
        return sum(i * s for i, s in zip(idx, self._strides))

    def value_at_index(self, idx: Sequence[int]) -> Union[Signal, VecSignal]:
        return self.values[self.flat_index(idx)]

    def axis_coords(self, axis: int) -> List[float]:
        n = self.shape[axis]
        lo, hi = self.lo[axis], self.hi[axis]
        return [lo + (hi - lo) * (k / (n - 1)) for k in range(n)]

    def channel_index(self, channel: Union[int, str]) -> int:
        return _resolve_channel(self.channels, self.value_dim, channel)

    def children(self):
        return tuple(self.values)


class Scatter(_Transformable):
    """N-D values at arbitrary positions (positions animatable too).

    Like :class:`Grid`, a Scatter is callable as a field of position —
    ``scatter(x, y)`` (or ``scatter(vec(x, y))``) builds the interpolating
    :class:`~loom.interp.ScatterField` / :class:`~loom.interp.VecScatterField` Signal
    (ftsl-style ``n(x,y)``); :meth:`sample` reads it eagerly."""

    def __init__(self, samples: Iterable[Tuple[Vecish, Union[Signal, VecSignal, Number]]],
                 *, channels: Optional[Sequence[str]] = None):
        self.positions: List[VecSignal] = []
        self.values: List[Union[Signal, VecSignal]] = []
        for pos, val in samples:
            self.positions.append(VecSignal.of(pos))
            self.values.append(val if isinstance(val, (Signal, VecSignal)) else as_signal(val))
        if not self.positions:
            raise ValueError("Scatter needs at least one sample")
        self.dim = self.positions[0].dim
        for p in self.positions:
            if p.dim != self.dim:
                raise ValueError("all Scatter positions must share a dimension")
        self.value_dim, self.is_vector = _infer_value_dim(self.values)
        self.channels = _check_channels(channels, self.value_dim)
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def channel_index(self, channel: Union[int, str]) -> int:
        return _resolve_channel(self.channels, self.value_dim, channel)

    def __call__(self, *query, power: float = 2.0, eps: float = 1e-9):
        """Sample this scatter set as a field of position — in **either tier**, exactly
        like :meth:`Grid.__call__`.

        * A *temporal* query (numbers, Signals, ``vec(...)``) → a
          :class:`~loom.interp.ScatterField` / :class:`~loom.interp.VecScatterField`:
          a Signal node (Shepard inverse-distance, exponent ``power``).
        * A *spatial* query (any :class:`~loom.spatial.SpatialExpr`, e.g.
          ``scatter(X, Y)``) → a :class:`~loom.spatial.ScatterSample`, a renderable
          term emitting ftsl's ``scatter:<name>(c0, …)`` with its ``scatter { … }``
          block collected by :meth:`loom.Scene.add`."""
        from .interp import ScatterField, VecScatterField  # lazy: avoid data<->interp cycle
        from .spatial import SpatialExpr, ScatterSample
        q = _query_of(query)
        if isinstance(q, (list, tuple)) and any(isinstance(c, SpatialExpr) for c in q):
            return ScatterSample(self, q, power=power, eps=eps)
        Field = VecScatterField if self.is_vector else ScatterField
        return Field(self, q, power=power, eps=eps)

    def sample(self, *query, clock=None, power: float = 2.0, eps: float = 1e-9):
        """Eagerly read the scatter field at an explicit query point: returns a ``float``
        (scalar) or a component ``tuple`` (vector).  ``clock`` defaults to a static
        frame-0 clock; pass one to evaluate animated samples at a specific frame."""
        from .signals.core import Clock
        field = self(*query, power=power, eps=eps)
        clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
        return field.at(clk)

    def __len__(self) -> int:
        return len(self.positions)

    def children(self):
        return tuple(self.positions) + tuple(self.values)
