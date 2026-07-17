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
    Ramp, Ease,
)
from .data import PointPath, Grid, Scatter
from .interp import LoopCurve, GridField, ScatterField, eval_curve
from .mathnd import Mat, rotation, rotations, slice3, Affine, affine
from .scene import (
    Scene, Material, Sphere, Beads, Raw, Light, Camera, Element, Pattern,
    SweptMesh, IsoMesh, ribbon, tube, blob, fan,
)
from .mcubes import mesh_field
from .material import (
    FuncPattern, MixMaterial, PATTERNS,
    waves, checker, rings, blobs,
)
from .sweep import (
    rmf_frames, tangents, sweep_rings, skin, circle_profile, line_profile,
    write_obj,
)
from .iso import (
    Isosurface, gyroid_surface, phase_drift, FIELDS,
    gyroid, schwarz_p, schwarz_d, neovius,
)
from .pov import pov, PovFn, POV_FUNCS, POV_ND_GENERALIZABLE
from .drive import (
    render_range, render_still, emit_frames, assemble_gif, find_ftrace,
)
from .preview import PreviewServer, preview_range
from .spatial import (
    SpatialExpr, sexpr, X, Y, Z, T, SPATIAL_PATTERNS,
    sin, cos, tan, sqrt, exp, log, floor, fract, sign, saturate, sabs,
    smin, smax, spow, atan2, step, clamp, mix, smoothstep,
)
from .canvas import (
    Canvas2D, Marker, Stroke, curve_points,
    render_canvas, render_canvas_still,
)
from .xvideo import Clip, spacetime_rotate, spacetime_shear

__all__ = [
    "Signal", "Clock", "Cache", "Const", "TimeFn",
    "Add", "Sub", "Mul", "Div", "Neg", "Clamp", "Rectify", "Power",
    "MapRange", "Mix", "RefSignal", "Sin", "Cos",
    "as_signal", "Number",
    "SignalCycleError", "detect_signal_cycle", "walk",
    "VecSignal", "vec", "lerp",
    "Sine", "Cosine", "LoopNoise",
    "Ramp", "Ease",
    "PointPath", "Grid", "Scatter",
    "LoopCurve", "GridField", "ScatterField", "eval_curve",
    "Mat", "rotation", "rotations", "slice3", "Affine", "affine",
    "Scene", "Material", "Sphere", "Beads", "Raw", "Light", "Camera", "Element",
    "Pattern", "FuncPattern", "MixMaterial", "PATTERNS",
    "waves", "checker", "rings", "blobs",
    "SweptMesh", "IsoMesh", "ribbon", "tube", "blob", "fan",
    "mesh_field",
    "rmf_frames", "tangents", "sweep_rings", "skin", "circle_profile",
    "line_profile", "write_obj",
    "Isosurface", "gyroid_surface", "phase_drift", "FIELDS",
    "gyroid", "schwarz_p", "schwarz_d", "neovius",
    "pov", "PovFn", "POV_FUNCS", "POV_ND_GENERALIZABLE",
    "render_range", "render_still", "emit_frames", "assemble_gif", "find_ftrace",
    "PreviewServer", "preview_range",
    "SpatialExpr", "sexpr", "X", "Y", "Z", "T", "SPATIAL_PATTERNS",
    "sin", "cos", "tan", "sqrt", "exp", "log", "floor", "fract", "sign",
    "saturate", "sabs", "smin", "smax", "spow", "atan2", "step", "clamp",
    "mix", "smoothstep",
    "Canvas2D", "Marker", "Stroke", "curve_points",
    "render_canvas", "render_canvas_still",
    "Clip", "spacetime_rotate", "spacetime_shear",
]
