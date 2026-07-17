"""
Loom modulation DAG — scalar Signal graph.

Vendored + trimmed + generalized from
``soundshop/juce_client/signals/core.py``.  The audio-specific machinery
(MIDI/param/plugin dataclasses, beat/tempo, wavetable oscillators, the CSE
canonicalizer) is dropped; the DAG core, operator overloading, per-frame
caching, and the cycle detector are kept and re-based on a **normalized loop
clock** instead of an audio sample counter.

A :class:`Signal` is a *pure function of a clock* and a node in a directed
acyclic graph.  "Modulators modulating modulators" is just more edges.  Before
any render the whole graph is checked with :func:`detect_signal_cycle`, so a bad
graph fails loudly instead of hanging or overflowing the stack.

The clock is a normalized loop phase ``t in [0, 1)`` (one loop).  An optional
``frame``/``frames``/``fps`` mapping is carried for real-seconds export and is
the natural cache key (one value per node per frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List, Union

Number = Union[int, float]

# ---------------------------------------------------------------------------
# Shared node-id allocator
# ---------------------------------------------------------------------------
# Scalar Signals *and* VecSignals (see vector.py) draw ids from this single
# counter so that ids are globally unique and the cycle detector can walk a
# mixed scalar/vector graph by duck-typing ``.id`` / ``.children()``.

_next_id = 1


def alloc_id() -> int:
    global _next_id
    nid = _next_id
    _next_id += 1
    return nid


# ---------------------------------------------------------------------------
# Clock + cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Clock:
    """A moment in a (possibly looping) animation.

    ``t`` is the normalized phase.  Looping is a **choice**, not a baked-in
    invariant (DESIGN.md §11.6): the frame→``t`` mapping depends on ``loop``.

    - **closed** (``loop=True``, the default): ``t = (frame % frames) / frames``
      wraps into ``[0, 1)``, so frame ``N`` maps to ``t=0`` — the wrap point is
      byte-identical to frame 0.  Compose periodic leaves (:class:`Sine`,
      :class:`LoopNoise`) and a closed :class:`~loom.LoopCurve` to loop
      seamlessly.
    - **open** (``loop=False``): ``t = frame / (frames - 1)`` spans ``[0, 1]``
      *inclusive* with no modulo, so the endpoints are distinct and there is no
      phantom duplicate frame ``N``.  A one-shot timeline: use non-periodic
      leaves (:class:`Ramp`, :class:`Ease`) and an open path.

    ``frame`` is the integer frame index and the cache key; ``loop`` is carried
    so drivers/tests know whether the seam is meant to close.
    """

    t: float
    frame: int = 0
    frames: int = 1
    fps: float = 30.0
    loop: bool = True

    @property
    def seconds(self) -> float:
        return self.frame / self.fps if self.fps else 0.0

    @classmethod
    def at_frame(cls, frame: int, frames: int, fps: float = 30.0,
                 loop: bool = True) -> "Clock":
        frames = max(1, int(frames))
        if loop:
            t = (frame % frames) / frames
        else:
            # span [0, 1] inclusive; endpoints distinct, no phantom frame N
            t = frame / (frames - 1) if frames > 1 else 0.0
        return cls(t=t, frame=frame, frames=frames, fps=fps, loop=loop)


class Cache:
    """Memoize one value per (node id, frame).

    Shared sub-graphs (a modulator feeding several parameters) are then
    evaluated once per frame.  Pass a fresh :class:`Cache` per frame, or reuse
    one keyed by frame index.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[int, int], object] = {}

    def get(self, node_id: int, frame: int):
        return self._store.get((node_id, frame))

    def set(self, node_id: int, frame: int, value) -> None:
        self._store[(node_id, frame)] = value


# ---------------------------------------------------------------------------
# Signal base
# ---------------------------------------------------------------------------

class Signal:
    """A scalar pure function of a :class:`Clock` and a node in the DAG."""

    def __init__(self) -> None:
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    def children(self) -> Tuple["Signal", ...]:
        return ()

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        raise NotImplementedError

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> float:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        v = float(self._eval(clock, cache))
        if not math.isfinite(v):
            raise ValueError(f"{type(self).__name__} produced a non-finite value")
        if cache is not None:
            cache.set(self._id, clock.frame, v)
        return v

    # ---- operator overloads -------------------------------------------------
    # Binary operators coerce numbers/Signals; for any *other* operand type they
    # return NotImplemented so Python defers to that operand's reflected operator
    # (e.g. a SpatialExpr on the right of ``Signal * expr``), instead of raising.
    def __add__(self, other: Union["Signal", Number]) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Add(self, as_signal(other))

    def __radd__(self, other: Number) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Add(as_signal(other), self)

    def __sub__(self, other: Union["Signal", Number]) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Sub(self, as_signal(other))

    def __rsub__(self, other: Number) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Sub(as_signal(other), self)

    def __mul__(self, other: Union["Signal", Number]) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Mul(self, as_signal(other))

    def __rmul__(self, other: Number) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Mul(as_signal(other), self)

    def __truediv__(self, other: Union["Signal", Number]) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Div(self, as_signal(other))

    def __rtruediv__(self, other: Number) -> "Signal":
        if not isinstance(other, (Signal, int, float)):
            return NotImplemented
        return Div(as_signal(other), self)

    def __neg__(self) -> "Signal":
        return Neg(self)


def as_signal(x: Union[Signal, Number]) -> Signal:
    return x if isinstance(x, Signal) else Const(float(x))


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------

class Const(Signal):
    def __init__(self, value: Number) -> None:
        super().__init__()
        self.value = float(value)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.value


class TimeFn(Signal):
    """Wrap an arbitrary ``f(t) -> float`` of the normalized loop phase.

    If ``periodic`` (default) ``t`` is wrapped into ``[0, 1)`` first, which
    keeps user shapes loop-safe.
    """

    def __init__(self, fn: Callable[[float], float], *, periodic: bool = True) -> None:
        super().__init__()
        self.fn = fn
        self.periodic = bool(periodic)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        t = clock.t - math.floor(clock.t) if self.periodic else clock.t
        return float(self.fn(t))


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

class _Binary(Signal):
    def __init__(self, a: Signal, b: Signal) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def children(self) -> Tuple[Signal, ...]:
        return (self.a, self.b)


class Add(_Binary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.a.at(clock, cache) + self.b.at(clock, cache)


class Sub(_Binary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.a.at(clock, cache) - self.b.at(clock, cache)


class Mul(_Binary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.a.at(clock, cache) * self.b.at(clock, cache)


class Div(_Binary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        d = self.b.at(clock, cache)
        if d == 0.0:
            raise ValueError("Div by zero in Signal graph")
        return self.a.at(clock, cache) / d


class _Unary(Signal):
    def __init__(self, x: Signal) -> None:
        super().__init__()
        self.x = x

    def children(self) -> Tuple[Signal, ...]:
        return (self.x,)


class Neg(_Unary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return -self.x.at(clock, cache)


class Clamp(_Unary):
    def __init__(self, x: Signal, lo: Number, hi: Number) -> None:
        super().__init__(x)
        self.lo = float(lo)
        self.hi = float(hi)
        if self.hi < self.lo:
            raise ValueError("Clamp requires hi >= lo")

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        v = self.x.at(clock, cache)
        return self.lo if v < self.lo else (self.hi if v > self.hi else v)


class Rectify(_Unary):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return abs(self.x.at(clock, cache))


class Power(_Unary):
    """``x ** gamma``.  Unipolar clamps to [0,1]; bipolar keeps sign."""

    def __init__(self, x: Signal, gamma: Number, *, unipolar: bool = True) -> None:
        super().__init__(x)
        if gamma <= 0 or not math.isfinite(gamma):
            raise ValueError("gamma must be finite and > 0")
        self.gamma = float(gamma)
        self.unipolar = bool(unipolar)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        v = self.x.at(clock, cache)
        if self.unipolar:
            if v <= 0.0:
                return 0.0
            if v >= 1.0:
                return 1.0
            return v ** self.gamma
        if v == 0.0:
            return 0.0
        return math.copysign(abs(v) ** self.gamma, v)


class MapRange(_Unary):
    def __init__(self, x: Signal, in_min: Number, in_max: Number,
                 out_min: Number, out_max: Number, *, clamp: bool = False) -> None:
        super().__init__(x)
        if in_max == in_min:
            raise ValueError("in_max must differ from in_min")
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.clamp = bool(clamp)
        self.scale = (self.out_max - self.out_min) / (self.in_max - self.in_min)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        y = self.out_min + (self.x.at(clock, cache) - self.in_min) * self.scale
        if self.clamp:
            lo, hi = min(self.out_min, self.out_max), max(self.out_min, self.out_max)
            if y < lo:
                return lo
            if y > hi:
                return hi
        return y


class Sin(_Unary):
    """``sin(x)`` with the argument in radians (generic, unlike periodic.Sine)."""

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return math.sin(self.x.at(clock, cache))


class Cos(_Unary):
    """``cos(x)`` with the argument in radians."""

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return math.cos(self.x.at(clock, cache))


class Mix(Signal):
    """Linear blend ``a*(1-amount) + b*amount``."""

    def __init__(self, a: Signal, b: Signal, amount: Union[Signal, Number]) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.amount = as_signal(amount)

    def children(self) -> Tuple[Signal, ...]:
        return (self.a, self.b, self.amount)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        m = self.amount.at(clock, cache)
        return self.a.at(clock, cache) * (1.0 - m) + self.b.at(clock, cache) * m


class RefSignal(Signal):
    """A named handle to a shared sub-graph, bound later via :meth:`bind`.

    Useful for reusing one modulator by name across a scene.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self._target: Optional[Signal] = None

    def bind(self, target: Signal) -> None:
        self._target = target

    @property
    def target(self) -> Signal:
        if self._target is None:
            raise ValueError(f"Unresolved RefSignal('{self.key}')")
        return self._target

    def children(self) -> Tuple[Signal, ...]:
        return (self._target,) if self._target is not None else ()

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.target.at(clock, cache)


# ---------------------------------------------------------------------------
# Cycle detection (the loop detector) — works on scalar *and* vector nodes
# ---------------------------------------------------------------------------

class SignalCycleError(ValueError):
    pass


def detect_signal_cycle(root) -> None:
    """Raise :class:`SignalCycleError` if a cycle is reachable from ``root``.

    3-color DFS (0=unvisited, 1=on-stack, 2=done).  Duck-typed on ``.id`` and
    ``.children()`` so it walks a mixed scalar-``Signal`` / ``VecSignal`` graph.
    Run before every render so a bad graph fails loudly.
    """
    color: Dict[int, int] = {}
    stack: List[int] = []

    def dfs(node) -> None:
        nid = node.id
        c = color.get(nid, 0)
        if c == 1:
            start = stack.index(nid) if nid in stack else 0
            raise SignalCycleError(
                f"Cycle detected in Signal graph (node ids): {stack[start:] + [nid]}"
            )
        if c == 2:
            return
        color[nid] = 1
        stack.append(nid)
        for ch in node.children():
            dfs(ch)
        stack.pop()
        color[nid] = 2

    dfs(root)


def walk(root):
    """Yield every unique node reachable from ``root`` (scalar or vector)."""
    seen = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n.id in seen:
            continue
        seen.add(n.id)
        yield n
        stack.extend(n.children())
