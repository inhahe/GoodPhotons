"""
Loom retime nodes — sample a sub-graph at a **different time**.

A :class:`~loom.signals.core.Signal` is a *pure, stateless function of a
:class:`~loom.signals.core.Clock`*, so evaluating one at some other phase is
well-defined and cheap: build a clock at that phase and evaluate.  Nothing in
the graph ever needed to be restricted to "the current frame" — the only thing
that assumed one-value-per-node-per-frame was the memo, and
:meth:`Cache.scope` fixes that (see below).  What was missing was a *node* that
says which phase to sample at, and a way to write the phase down as a value
(:class:`~loom.signals.core.Phase`).

With those two, the whole family falls out of one mechanism:

- **freeze** — :func:`freeze`, ``x`` pinned at one phase (``sig(0)``).
- **echo / delay** — :func:`delay`, ``sig(t - dt)`` (wraps on a closed clock, so
  a seamless loop stays seamless).
- **time-warp** — :func:`warp`, ``sig(g(t))`` for any shaping function ``g``.
- **4-D time-shear** — a *spatially varying* sample phase.  That one lives on
  the spatial tier, where a coordinate is in scope: see
  :class:`loom.spatial.SigAt` (``sig.at(t = T - X/c)``).

Two things this had to get right, both called out in the roadmap:

1. **The cache.**  :class:`~loom.signals.core.Cache` keys on
   ``(node id, frame)``.  A retimed subtree is evaluated at a *different* phase
   inside the same frame, so its values must not land in that store.  Every
   retime node evaluates its child through ``cache.scope(...)`` — a nested cache
   keyed by ``(node id, frame, sample phase)``.  Sharing still works *within* one
   retimed evaluation; nothing leaks between sample points or out to the frame.
2. **Cycles.**  Both edges — the retimed subtree *and* the driver that computes
   the phase — are ordinary structural edges reported by :meth:`children`, so
   :func:`~loom.signals.core.detect_signal_cycle` keeps owning them and a knot
   still fails loudly.  A retime is **not** a recurrence: it reads a pure
   function at another point, it does not read its own past.  Recurrent
   (delayed-feedback) nodes would need a separate causality guard and are
   deliberately not built here.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

from .core import (Signal, Clock, Cache, Number, TimeFn, Phase, as_signal,
                   alloc_id)
from .vector import VecSignal, Vecish


def retimed_clock(clock: Clock, t: float, wrap: Optional[bool] = None) -> Clock:
    """The clock ``clock`` moved to phase ``t``.

    ``wrap`` folds ``t`` back into ``[0, 1)``; the default (``None``) wraps iff
    the clock is a **closed** loop — which is what keeps ``sig(t - dt)``
    seamless on a loop and honest (off the end, un-folded) on an open timeline.

    ``frame`` is carried along as the nearest integer frame index for the new
    phase, so ``clock.seconds`` and any frame-curious consumer stay sane; it is
    *not* the cache key for the retimed subtree (:meth:`Cache.scope` is).
    """
    if wrap is None:
        wrap = clock.loop
    if wrap:
        t = t - math.floor(t)
    frames = max(1, int(clock.frames))
    if clock.loop:
        frame = int(round(t * frames)) % frames
    else:
        span = frames - 1
        frame = int(round(t * span)) if span > 0 else 0
        frame = 0 if frame < 0 else (span if frame > span else frame)
    return Clock(t=t, frame=frame, frames=clock.frames, fps=clock.fps,
                 loop=clock.loop)


class _RetimeBase:
    """Shared phase resolution + cache scoping for the scalar/vector nodes."""

    when: Signal
    wrap: Optional[bool]

    def _sample_clock(self, clock: Clock,
                      cache: Optional[Cache]) -> Tuple[Clock, Optional[Cache]]:
        # The driver is evaluated at the CURRENT clock (an instantaneous edge),
        # in the CURRENT cache — it is an ordinary modulator, not a retimed one.
        # It is finite by construction: Signal.at already rejects a non-finite
        # result, so a bad phase fails at its own node, where it can be named.
        t = self.when.at(clock, cache)
        rc = retimed_clock(clock, t, self.wrap)
        sub = None if cache is None else cache.scope((self.id, clock.frame, rc.t))
        return rc, sub


class Retime(_RetimeBase, Signal):
    """Scalar ``x`` sampled at the phase ``when`` instead of the current one.

    ``when`` is any Signal (or number) — usually built out of
    :class:`~loom.signals.core.Phase`.  ``wrap`` is passed to
    :func:`retimed_clock`.
    """

    def __init__(self, x: Signal, when: Union[Signal, Number], *,
                 wrap: Optional[bool] = None) -> None:
        Signal.__init__(self)
        if not isinstance(x, Signal):
            raise TypeError(
                "Retime expects a scalar Signal; use VecRetime (or retime()) "
                f"for a VecSignal, got {type(x).__name__}")
        self.x = x
        self.when = as_signal(when)
        self.wrap = None if wrap is None else bool(wrap)

    def children(self) -> Tuple[Signal, ...]:
        return (self.x, self.when)

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        rc, sub = self._sample_clock(clock, cache)
        return self.x.at(rc, sub)


class VecRetime(_RetimeBase):
    """The :class:`Retime` of a :class:`~loom.signals.vector.VecSignal`.

    Retiming a vector as a *whole* (rather than each component separately) is
    the right shape: every component then reads the same sample phase, and a
    sub-graph shared between components is evaluated once per sample point.
    """

    def __init__(self, v: Vecish, when: Union[Signal, Number], *,
                 wrap: Optional[bool] = None) -> None:
        self.v = VecSignal.of(v)
        self.when = as_signal(when)
        self.wrap = None if wrap is None else bool(wrap)
        self._id = alloc_id()

    @property
    def id(self) -> int:
        return self._id

    @property
    def dim(self) -> int:
        return self.v.dim

    def children(self) -> Tuple[object, ...]:
        return tuple(self.v.components) + (self.when,)

    def at(self, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
        if cache is not None:
            hit = cache.get(self._id, clock.frame)
            if hit is not None:
                return hit  # type: ignore[return-value]
        rc, sub = self._sample_clock(clock, cache)
        v = self.v.at(rc, sub)
        if cache is not None:
            cache.set(self._id, clock.frame, v)
        return v

    def as_vec(self) -> VecSignal:
        """This retimed vector re-exposed as a plain :class:`VecSignal`.

        Each component is a scalar :class:`Retime` of the matching component at
        the *same* driver, so the result composes with all the vector math.  Use
        it when a consumer insists on a real ``VecSignal``; prefer the node
        itself when you want one shared sub-evaluation.
        """
        return VecSignal([Retime(c, self.when, wrap=self.wrap)
                          for c in self.v.components])


def retime(x, when: Union[Signal, Number], *,
           wrap: Optional[bool] = None):
    """:class:`Retime` for a scalar, :class:`VecRetime` for a vector."""
    if isinstance(x, Signal):
        return Retime(x, when, wrap=wrap)
    return VecRetime(x, when, wrap=wrap)


def freeze(x, at: Union[Signal, Number] = 0.0, *, wrap: Optional[bool] = None):
    """``x`` held at a fixed phase — the animation, stopped.

    ``at`` may itself be animated, which is the useful case: a *scrubbable*
    freeze (hold, then release) is ``freeze(x, at=held_phase)``.
    """
    return retime(x, at, wrap=wrap)


def delay(x, dt: Union[Signal, Number], *, wrap: Optional[bool] = None):
    """``x`` as it was ``dt`` of a loop ago — ``x(t - dt)``.

    On a closed clock the phase wraps, so a delayed copy of a seamless loop is
    still seamless (it is the same loop, rotated).  A negative ``dt`` looks
    *ahead*, which is equally well-defined here: the graph is a pure function of
    the clock, not a stream.
    """
    return retime(x, Phase() - as_signal(dt), wrap=wrap)


def warp(x, g: Union[Signal, Callable[[float], float]], *,
         wrap: Optional[bool] = None):
    """``x(g(t))`` — an arbitrary reparameterization of time.

    ``g`` is either a Signal already written in terms of
    :class:`~loom.signals.core.Phase`, or a plain ``f(t) -> t'`` callable (wrapped
    in a non-periodic :class:`~loom.signals.core.TimeFn`, so it sees the raw
    phase and may legitimately return one outside ``[0, 1)``).

    A ``g`` that is *monotone with ``g(1) - g(0)`` an integer* keeps a closed
    loop closed (ease into and out of a loop); anything else re-times freely and
    is a one-shot.
    """
    if isinstance(g, Signal):
        drive: Signal = g
    elif callable(g):
        drive = TimeFn(g, periodic=False)
    else:
        drive = as_signal(g)
    return retime(x, drive, wrap=wrap)
