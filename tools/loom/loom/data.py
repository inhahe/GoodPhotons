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

from .signals.core import Signal, Number, as_signal
from .signals.vector import VecSignal, Vecish


class PointPath:
    """An ordered sequence of N-D points (each an animatable ``VecSignal``)."""

    def __init__(self, points: Iterable[Vecish], *, closed: bool = True) -> None:
        self.points: List[VecSignal] = [VecSignal.of(p) for p in points]
        if len(self.points) < 2:
            raise ValueError("PointPath needs at least 2 points")
        self.dim = self.points[0].dim
        for p in self.points:
            if p.dim != self.dim:
                raise ValueError("all PointPath points must share a dimension")
        self.closed = bool(closed)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, i: int) -> VecSignal:
        return self.points[i]

    def children(self) -> Tuple[VecSignal, ...]:
        return tuple(self.points)


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

    def __len__(self) -> int:
        return len(self.positions)

    def children(self):
        return tuple(self.positions) + tuple(self.values)
