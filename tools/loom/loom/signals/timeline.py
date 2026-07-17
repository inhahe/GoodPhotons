"""
Loom open-timeline leaves — the building blocks of *one-shot* (non-looping)
animation.

Where :mod:`loom.signals.periodic` is periodic in ``t`` (so it loops
seamlessly), everything here is **monotone / non-periodic** in ``t in [0, 1]``:
it deliberately differs at ``t=0`` and ``t=1``, so it makes sense only on an
**open** clock (:meth:`loom.Clock.at_frame` with ``loop=False``), where the
endpoints are distinct frames.  Feeding these to a closed loop would produce a
visible seam — that's the point: looping is a choice (DESIGN.md §11.6).

- :class:`Ramp` — linear interpolation ``start + (end-start)*t``.
- :class:`Ease` — smoothstep-eased interpolation (``in`` / ``out`` / ``in-out``).
"""

from __future__ import annotations

from typing import Optional, Union

from .core import Signal, Clock, Cache, Number, as_signal


class Ramp(Signal):
    """A linear ramp ``start + (end - start) * t`` over the open timeline.

    ``start``/``end`` may themselves be Signals.  Non-periodic by construction:
    ``Ramp(0, 1)`` is ``0`` at ``t=0`` and ``1`` at ``t=1``.
    """

    def __init__(self, start: Union[Signal, Number] = 0.0,
                 end: Union[Signal, Number] = 1.0) -> None:
        super().__init__()
        self.start = as_signal(start)
        self.end = as_signal(end)

    def children(self):
        return (self.start, self.end)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        s = self.start.at(clock, cache)
        e = self.end.at(clock, cache)
        return s + (e - s) * clock.t


class Ease(Signal):
    """A smoothstep-eased interpolation from ``start`` to ``end`` over ``t``.

    ``mode`` selects the shape:
    - ``"in-out"`` (default): classic smoothstep ``3t^2 - 2t^3`` (slow ends).
    - ``"in"``: accelerate from rest (ease-in only).
    - ``"out"``: decelerate to rest (ease-out only).

    Non-periodic: like :class:`Ramp` the endpoints differ, so it belongs on an
    open clock.
    """

    _MODES = ("in-out", "in", "out")

    def __init__(self, start: Union[Signal, Number] = 0.0,
                 end: Union[Signal, Number] = 1.0, *,
                 mode: str = "in-out") -> None:
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"Ease mode must be one of {self._MODES}, got {mode!r}")
        self.start = as_signal(start)
        self.end = as_signal(end)
        self.mode = mode

    def children(self):
        return (self.start, self.end)

    def _shape(self, t: float) -> float:
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        if self.mode == "in-out":
            return t * t * (3.0 - 2.0 * t)          # smoothstep
        if self.mode == "in":
            return t * t                            # quadratic ease-in
        # "out": mirror of ease-in
        return t * (2.0 - t)                        # quadratic ease-out

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        s = self.start.at(clock, cache)
        e = self.end.at(clock, cache)
        return s + (e - s) * self._shape(clock.t)
