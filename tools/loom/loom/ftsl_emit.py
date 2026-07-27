"""
Loom -> .ftsl text emission helpers.

Small, dependency-free utilities to evaluate animatable fields at a clock and
format them as ftrace scene-language tokens.  The scene model (:mod:`loom.scene`)
uses these to turn a snapshot into a ``.ftsl`` string per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, Cache, Number, lower_axsignal
from .signals.vector import VecSignal
from .color import Color

Animatable = Union[Signal, VecSignal, Number, Sequence[Number], str]


def site_node(x):
    """The :class:`Signal` / :class:`VecSignal` a value-site sees for ``x``.

    Passes a Signal/VecSignal straight through and **lowers** an axis-typed
    node (:mod:`loom.axes` — a ``Target``, a ``CurveSample``, …) into one, so
    every scene value-site accepts the E5 influence model uniformly.  Returns
    ``None`` for a plain number / sequence / string, which the callers format
    directly.
    """
    if isinstance(x, (Signal, VecSignal)):
        return x
    return lower_axsignal(x)


@dataclass
class EmitCtx:
    """Context for one frame's emission.

    ``assets_dir`` is where file-backed elements (e.g. a swept mesh) write their
    OBJ; ``tag`` disambiguates per-frame filenames.  When ``assets_dir`` is None
    (a stdout preview), file-backed elements fall back to a temp dir.
    """

    clock: Clock
    cache: Optional[Cache] = None
    assets_dir: Optional[Path] = None
    tag: str = ""

    def asset_path(self, name: str, ext: str) -> Path:
        import tempfile
        d = self.assets_dir if self.assets_dir is not None else Path(tempfile.gettempdir())
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}{self.tag}.{ext}"


def num(x: Union[Signal, Number], clock: Clock, cache: Optional[Cache] = None) -> float:
    n = site_node(x)
    if n is None:
        return float(x)
    if isinstance(n, VecSignal):
        raise ValueError(
            f"a scalar value-site got a {n.dim}-vector; pick a component "
            f"(e.g. .comp(1) on an axis node)")
    return n.at(clock, cache)


def vecn(x, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
    n = site_node(x)
    if isinstance(n, VecSignal):
        return n.at(clock, cache)
    if isinstance(n, Signal):
        raise ValueError("a vector value-site got a scalar")
    return tuple(num(c, clock, cache) for c in x)


def vec3(x, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, float, float]:
    t = vecn(x, clock, cache)
    if len(t) < 3:
        raise ValueError("expected a >=3-D vector")
    return (t[0], t[1], t[2])


def fmt(v: float) -> str:
    return f"{v:.6g}"


def fmt3(t: Sequence[float]) -> str:
    return " ".join(fmt(float(c)) for c in t[:3])


def value_token(v, clock: Clock, cache: Optional[Cache] = None) -> str:
    """Format an arbitrary animatable/scalar/vector/string property value."""
    if isinstance(v, str):
        return v
    if isinstance(v, Color):
        return v.token(clock, cache)            # "rgb r g b" or "hsv h s v"
    n = site_node(v)                            # lowers an axis-typed node
    if isinstance(n, VecSignal):
        return fmt3(n.at(clock, cache))
    if isinstance(n, Signal):
        return fmt(n.at(clock, cache))
    if isinstance(v, (list, tuple)):
        return " ".join(fmt(num(c, clock, cache)) for c in v)
    return fmt(float(v))
