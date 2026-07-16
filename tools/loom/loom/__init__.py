"""Loom — programmatic-first procedural animation / geometry toolkit.

See DESIGN.md for the architecture.  M1 (Foundation) exposes the modulation DAG,
N-D vector signals, periodic leaves, the three datasets and the three
interpolators.
"""

from .signals import (
    Signal, Clock, Cache, Const, TimeFn,
    Add, Sub, Mul, Div, Neg, Clamp, Rectify, Power, MapRange, Mix, RefSignal,
    Sin, Cos,
    as_signal, Number,
    SignalCycleError, detect_signal_cycle, walk,
    VecSignal, vec, lerp,
    Sine, Cosine, LoopNoise,
)
from .data import PointPath, Grid, Scatter
from .interp import LoopCurve, GridField, ScatterField
from .mathnd import Mat, rotation, rotations, slice3

__all__ = [
    "Signal", "Clock", "Cache", "Const", "TimeFn",
    "Add", "Sub", "Mul", "Div", "Neg", "Clamp", "Rectify", "Power",
    "MapRange", "Mix", "RefSignal", "Sin", "Cos",
    "as_signal", "Number",
    "SignalCycleError", "detect_signal_cycle", "walk",
    "VecSignal", "vec", "lerp",
    "Sine", "Cosine", "LoopNoise",
    "PointPath", "Grid", "Scatter",
    "LoopCurve", "GridField", "ScatterField",
    "Mat", "rotation", "rotations", "slice3",
]
