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
from .data import PointPath, TrackedPath, Grid, Scatter
from .color import (
    Color, rgb, hsv, hsl, hsv_to_rgb, rgb_to_hsv, hsl_to_rgb, rgb_to_hsl,
    parse_color, parse_color_list,
)
from .interp import (
    LoopCurve, TrackedCurve, Reparam, GridField, ScatterField,
    VecGridField, VecScatterField, RbfScatterField, VecRbfScatterField,
    FieldCurve, eval_curve,
)
from .mathnd import Mat, rotation, rotations, slice3, Affine, affine
from .scene import (
    Scene, Material, Texture, skin, ProcTexture, func_skin, Sphere, Beads, Raw,
    Light, Camera, CameraCurve, Element, Group,
    Pattern, SweptMesh, IsoMesh, ribbon, tube, blob, fan, Volume,
)
from .transform import Transform
from .mcubes import mesh_field
from .material import (
    FuncPattern, MixMaterial, PATTERNS,
    waves, checker, rings, blobs,
)
from .record import Record, RecordChannel, RecordStop
from .ladder import parse_ladder, emit_ladder, shape as ladder_shape
from .sweep import (
    rmf_frames, tangents, sweep_rings, skin_rings, circle_profile, line_profile,
    write_obj,
)
from .iso import (
    Isosurface, gyroid_surface, phase_drift, FIELDS,
    gyroid, schwarz_p, schwarz_d, neovius,
)
from .pov import (
    pov, PovFn, POV_FUNCS, POV_ND_GENERALIZABLE, POV_PARAMS, pov_params,
)
from .pov_nd import nd_field_expr, nd_field_eval, nd_grad_bound_xi
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
from .audio import SampleBuffer

__all__ = [
    "Signal", "Clock", "Cache", "Const", "TimeFn",
    "Add", "Sub", "Mul", "Div", "Neg", "Clamp", "Rectify", "Power",
    "MapRange", "Mix", "RefSignal", "Sin", "Cos",
    "as_signal", "Number",
    "SignalCycleError", "detect_signal_cycle", "walk",
    "VecSignal", "vec", "lerp",
    "Sine", "Cosine", "LoopNoise",
    "Ramp", "Ease",
    "PointPath", "TrackedPath", "Grid", "Scatter",
    "Color", "rgb", "hsv", "hsl",
    "hsv_to_rgb", "rgb_to_hsv", "hsl_to_rgb", "rgb_to_hsl",
    "parse_color", "parse_color_list",
    "LoopCurve", "TrackedCurve", "Reparam", "GridField", "ScatterField",
    "VecGridField", "VecScatterField", "RbfScatterField", "VecRbfScatterField",
    "FieldCurve", "eval_curve",
    "Mat", "rotation", "rotations", "slice3", "Affine", "affine",
    "Scene", "Material", "Texture", "skin", "ProcTexture", "func_skin",
    "Sphere", "Beads", "Raw", "Light",
    "Camera", "CameraCurve", "Element", "Group", "Transform",
    "Pattern", "FuncPattern", "MixMaterial", "PATTERNS",
    "waves", "checker", "rings", "blobs",
    "Record", "RecordChannel", "RecordStop",
    "parse_ladder", "emit_ladder", "ladder_shape",
    "SweptMesh", "IsoMesh", "ribbon", "tube", "blob", "fan", "Volume",
    "mesh_field",
    "rmf_frames", "tangents", "sweep_rings", "skin_rings", "circle_profile",
    "line_profile", "write_obj",
    "Isosurface", "gyroid_surface", "phase_drift", "FIELDS",
    "gyroid", "schwarz_p", "schwarz_d", "neovius",
    "pov", "PovFn", "POV_FUNCS", "POV_ND_GENERALIZABLE", "POV_PARAMS", "pov_params",
    "nd_field_expr", "nd_field_eval", "nd_grad_bound_xi",
    "render_range", "render_still", "emit_frames", "assemble_gif", "find_ftrace",
    "PreviewServer", "preview_range",
    "SpatialExpr", "sexpr", "X", "Y", "Z", "T", "SPATIAL_PATTERNS",
    "sin", "cos", "tan", "sqrt", "exp", "log", "floor", "fract", "sign",
    "saturate", "sabs", "smin", "smax", "spow", "atan2", "step", "clamp",
    "mix", "smoothstep",
    "Canvas2D", "Marker", "Stroke", "curve_points",
    "render_canvas", "render_canvas_still",
    "Clip", "spacetime_rotate", "spacetime_shear",
    "SampleBuffer",
]
