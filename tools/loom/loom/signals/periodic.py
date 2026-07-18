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

    ``dist`` shapes the *distribution* of the cell values (the texture of the
    noise), independent of ``amp``/``bias``:

    - ``"uniform"`` (default) — cells drawn flat over ``[-1, 1]``; the noise
      roams its whole range evenly.
    - ``"gauss"`` — cells drawn from a **bell curve** ``N(0, width)``: the noise
      dwells near the centre and only occasionally wanders out, with ``width``
      (the standard deviation, default ``0.4``) setting how far it strays.
      Because a Gaussian is unbounded, cells are clamped to ``±clip`` standard
      deviations (default 3) so ``amp`` stays a meaningful bound and the spline
      can't overshoot to infinity.  Pass ``clip=None`` to keep the raw unclamped
      tails (an occasional big spike) — the drawn sequence is identical, only the
      clamp is dropped, so a value is no longer bounded by ``clip*width``.

    In every mode the output is ``amp * <ring sample> + bias`` — so ``bias`` is
    the shift (mean offset) and ``amp`` the overall scale, on top of the shape
    ``dist``/``width`` choose.
    """

    def __init__(self, cells: int = 8, seed: int = 0, *,
                 freq: int = 1, amp: Union[Signal, Number] = 1.0,
                 bias: Union[Signal, Number] = 0.0,
                 dist: str = "uniform", width: float = 0.4,
                 clip: Optional[float] = 3.0) -> None:
        super().__init__()
        if cells < 2:
            raise ValueError("LoopNoise needs cells >= 2")
        if dist not in ("uniform", "gauss"):
            raise ValueError("LoopNoise dist must be 'uniform' or 'gauss'")
        if dist == "gauss" and not (width > 0.0):
            raise ValueError("LoopNoise gauss width must be > 0")
        if clip is not None and not (clip > 0.0):
            raise ValueError("LoopNoise clip must be > 0 or None (unclamped)")
        self.cells = int(cells)
        self.freq = int(freq)
        self.amp = as_signal(amp)
        self.bias = as_signal(bias)
        self.dist = dist
        self.width = float(width)
        self.clip = None if clip is None else float(clip)
        # The uniform key is byte-identical to the original (pre-`dist`) LoopNoise so
        # existing seeds reproduce exactly; gauss uses its own extended key.  clip is
        # applied *after* the draw, so it never perturbs the drawn sequence (and stays
        # out of the key): clip=None just skips the clamp on the same values.
        key = f"loomnoise:{int(seed)}:{self.cells}:{self.freq}"
        if dist == "gauss":
            rng = random.Random(f"{key}:gauss:{self.width}")
            lim = None if self.clip is None else self.clip * self.width
            self._vals: List[float] = [
                rng.gauss(0.0, self.width) if lim is None
                else max(-lim, min(lim, rng.gauss(0.0, self.width)))
                for _ in range(self.cells)]
        else:
            rng = random.Random(key)
            self._vals = [rng.uniform(-1.0, 1.0) for _ in range(self.cells)]

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
