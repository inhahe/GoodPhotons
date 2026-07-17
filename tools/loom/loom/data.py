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


class Grid:
    """N-D scalar-or-vector values on a regular lattice.

    ``shape`` is the number of samples per axis (arbitrary rarity).  ``lo``/``hi``
    are the domain corners.  ``values`` is a flat, C-order list of length
    ``prod(shape)`` of Signals (scalar field) or VecSignals (vector field).
    """

    def __init__(self, shape: Sequence[int], lo: Sequence[float],
                 hi: Sequence[float], values: Iterable[Union[Signal, VecSignal, Number]]):
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        if any(s < 2 for s in self.shape):
            raise ValueError("each grid axis needs >= 2 samples")
        self.ndim = len(self.shape)
        if len(lo) != self.ndim or len(hi) != self.ndim:
            raise ValueError("lo/hi must match grid ndim")
        self.lo = tuple(float(x) for x in lo)
        self.hi = tuple(float(x) for x in hi)
        n = 1
        for s in self.shape:
            n *= s
        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"expected {n} values, got {len(vals)}")
        self.values: List[Union[Signal, VecSignal]] = [
            v if isinstance(v, (Signal, VecSignal)) else as_signal(v) for v in vals
        ]
        # C-order strides
        self._strides: List[int] = [1] * self.ndim
        for a in range(self.ndim - 2, -1, -1):
            self._strides[a] = self._strides[a + 1] * self.shape[a + 1]
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

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

    def children(self):
        return tuple(self.values)


class Scatter:
    """N-D values at arbitrary positions (positions animatable too)."""

    def __init__(self, samples: Iterable[Tuple[Vecish, Union[Signal, VecSignal, Number]]]):
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
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def __len__(self) -> int:
        return len(self.positions)

    def children(self):
        return tuple(self.positions) + tuple(self.values)
