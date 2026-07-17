"""
Loom N-D vector signals.

A :class:`VecSignal` is an N-D vector whose components are each a scalar
:class:`~loom.signals.core.Signal` (or a broadcast scalar).  It shares the same
DAG, the same node-id space, and the same :func:`detect_signal_cycle` /
:func:`walk` machinery as scalar signals — a ``VecSignal`` is just a node whose
value is a tuple.  All the vector math you need for animation lives here as
graph-building operations (elementwise ``+ - *``, scalar broadcast, dot,
length, normalize, lerp), so an entire animated point/curve/field pipeline is
one cycle-checked graph.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .core import Signal, Clock, Cache, Const, as_signal, alloc_id, Number

Vecish = Union["VecSignal", Sequence[Union[Signal, Number]]]


class VecSignal:
    """An ordered tuple of scalar Signals, evaluated together per frame."""

    def __init__(self, components: Iterable[Union[Signal, Number]]) -> None:
        self.components: List[Signal] = [as_signal(c) for c in components]
        if not self.components:
            raise ValueError("VecSignal needs at least one component")
        self._id = alloc_id()

    # ---- node protocol (matches Signal for cycle detection / walk) ----------
    @property
    def id(self) -> int:
        return self._id

    @property
    def dim(self) -> int:
        return len(self.components)

    def children(self) -> Tuple[Signal, ...]:
        return tuple(self.components)

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        v = tuple(c.at(clock, cache) for c in self.components)
        if cache is not None:
            cache.set(self._id, clock.frame, v)
        return v

    # ---- construction helpers ----------------------------------------------
    @classmethod
    def of(cls, v: Vecish) -> "VecSignal":
        return v if isinstance(v, VecSignal) else cls(list(v))

    def __len__(self) -> int:
        return self.dim

    def __getitem__(self, i: int) -> Signal:
        return self.components[i]

    def __iter__(self):
        return iter(self.components)

    # ---- vector math (graph-building) --------------------------------------
    def _binary(self, other: Union[Vecish, Signal, Number], op) -> "VecSignal":
        if isinstance(other, (Signal, int, float)):  # broadcast scalar
            s = as_signal(other)
            return VecSignal([op(c, s) for c in self.components])
        o = VecSignal.of(other)
        if o.dim != self.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {o.dim}")
        return VecSignal([op(a, b) for a, b in zip(self.components, o.components)])

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return VecSignal.of(other).__sub__(self) if not isinstance(
            other, (Signal, int, float)) else VecSignal(
            [as_signal(other) - c for c in self.components])

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __neg__(self) -> "VecSignal":
        return VecSignal([-c for c in self.components])

    def dot(self, other: Vecish) -> Signal:
        o = VecSignal.of(other)
        if o.dim != self.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {o.dim}")
        terms = [a * b for a, b in zip(self.components, o.components)]
        acc = terms[0]
        for t in terms[1:]:
            acc = acc + t
        return acc

    def length_sq(self) -> Signal:
        return self.dot(self)

    def length(self) -> Signal:
        return _Sqrt(self.length_sq())

    def normalized(self, eps: float = 1e-12) -> "VecSignal":
        inv = _InvLength(self, eps)
        return VecSignal([c * inv for c in self.components])

    def with_dim(self, n: int, fill: Number = 0.0) -> "VecSignal":
        """Pad or truncate to exactly ``n`` components (for N-D slicing)."""
        if n == self.dim:
            return self
        if n < self.dim:
            return VecSignal(self.components[:n])
        return VecSignal(self.components + [Const(fill)] * (n - self.dim))


def vec(*components: Union[Signal, Number]) -> VecSignal:
    return VecSignal(list(components))


def lerp(a: Vecish, b: Vecish, t: Union[Signal, Number]) -> VecSignal:
    """Elementwise ``a*(1-t) + b*t`` between two vectors."""
    a = VecSignal.of(a)
    b = VecSignal.of(b)
    ts = as_signal(t)
    return VecSignal([ac * (1.0 - ts) + bc * ts
                      for ac, bc in zip(a.components, b.components)])


# ---------------------------------------------------------------------------
# Scalar helpers used by vector ops (kept here to avoid bloating core)
# ---------------------------------------------------------------------------

class _Sqrt(Signal):
    def __init__(self, x: Signal) -> None:
        super().__init__()
        self.x = x

    def children(self):
        return (self.x,)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return math.sqrt(max(0.0, self.x.at(clock, cache)))


class _InvLength(Signal):
    """1 / |v| of a VecSignal, guarded by eps.  Depends on all components."""

    def __init__(self, v: VecSignal, eps: float) -> None:
        super().__init__()
        self.v = v
        self.eps = float(eps)

    def children(self):
        return tuple(self.v.components)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        s = sum(c.at(clock, cache) ** 2 for c in self.v.components)
        return 1.0 / max(self.eps, math.sqrt(s))
