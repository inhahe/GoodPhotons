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

from .signals.core import Signal, Clock, Cache, Number
from .signals.vector import VecSignal

Animatable = Union[Signal, VecSignal, Number, Sequence[Number], str]


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
    return x.at(clock, cache) if isinstance(x, Signal) else float(x)


def vecn(x, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
    if isinstance(x, VecSignal):
        return x.at(clock, cache)
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
    if isinstance(v, VecSignal):
        return fmt3(v.at(clock, cache))
    if isinstance(v, Signal):
        return fmt(v.at(clock, cache))
    if isinstance(v, (list, tuple)):
        return " ".join(fmt(num(c, clock, cache)) for c in v)
    return fmt(float(v))
