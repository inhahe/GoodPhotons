"""
Loom periodic leaves — the building blocks of *seamless* loops.

Everything here is periodic in the normalized loop phase ``t in [0, 1)``, so a
graph built out of these leaves loops with no seam:

- :class:`Sine`     — ``amp*sin(2*pi*(cycles*t + phase)) + bias``.
- :class:`Cosine`   — same, cosine.
- :class:`LoopNoise`— seeded smooth value-noise on a ring (wraps seamlessly).

For the seamless *closed curve* (scribbles3), see
:class:`loom.interp.LoopCurve`, which is periodic by construction too.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Union

from .core import Signal, Clock, Cache, Number, as_signal


class Sine(Signal):
    def __init__(self, cycles: Union[Signal, Number] = 1.0,
                 phase: Union[Signal, Number] = 0.0,
                 amp: Union[Signal, Number] = 1.0,
                 bias: Union[Signal, Number] = 0.0) -> None:
        super().__init__()
        self.cycles = as_signal(cycles)
        self.phase = as_signal(phase)
        self.amp = as_signal(amp)
        self.bias = as_signal(bias)

    def children(self):
        return (self.cycles, self.phase, self.amp, self.bias)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        c = self.cycles.at(clock, cache)
        p = self.phase.at(clock, cache)
        a = self.amp.at(clock, cache)
        b = self.bias.at(clock, cache)
        return a * math.sin(2.0 * math.pi * (c * clock.t + p)) + b


class Cosine(Sine):
    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        c = self.cycles.at(clock, cache)
        p = self.phase.at(clock, cache)
        a = self.amp.at(clock, cache)
        b = self.bias.at(clock, cache)
        return a * math.cos(2.0 * math.pi * (c * clock.t + p)) + b


class LoopNoise(Signal):
    """Seamless, deterministic smooth noise over one loop.

    ``cells`` random values are placed evenly around a ring and interpolated
    with a Catmull-Rom spline using modulo-wraparound, so the first and last
    values join smoothly — the noise is periodic in ``t`` and repeats exactly
    for a given ``seed``.  ``freq`` is how many times the ring is traversed per
    loop (keep it an integer to stay seamless).
    """

    def __init__(self, cells: int = 8, seed: int = 0, *,
                 freq: int = 1, amp: Union[Signal, Number] = 1.0,
                 bias: Union[Signal, Number] = 0.0) -> None:
        super().__init__()
        if cells < 2:
            raise ValueError("LoopNoise needs cells >= 2")
        self.cells = int(cells)
        self.freq = int(freq)
        self.amp = as_signal(amp)
        self.bias = as_signal(bias)
        rng = random.Random(f"loomnoise:{int(seed)}:{self.cells}:{self.freq}")
        self._vals: List[float] = [rng.uniform(-1.0, 1.0) for _ in range(self.cells)]

    def children(self):
        return (self.amp, self.bias)

    def _sample_ring(self, u: float) -> float:
        # u in [0,1) around the ring; Catmull-Rom over 4 neighbouring cells.
        n = self.cells
        x = (u - math.floor(u)) * n
        i = int(math.floor(x))
        f = x - i
        p0 = self._vals[(i - 1) % n]
        p1 = self._vals[i % n]
        p2 = self._vals[(i + 1) % n]
        p3 = self._vals[(i + 2) % n]
        f2 = f * f
        f3 = f2 * f
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * f
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3
        )

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        a = self.amp.at(clock, cache)
        b = self.bias.at(clock, cache)
        return a * self._sample_ring(clock.t * self.freq) + b
