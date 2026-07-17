"""
Loom scene model — animatable geometry + materials + lights + camera, emitted
to ftrace's ``.ftsl`` scene language one frame at a time.

Every element field may be a plain number, a :class:`~loom.signals.core.Signal`,
or a :class:`~loom.signals.vector.VecSignal`, so the whole scene animates.  A
:class:`Scene` knows how to (a) collect every modulator root for a pre-render
:func:`~loom.signals.core.detect_signal_cycle` check and (b) emit the ``.ftsl``
text for a given :class:`~loom.signals.core.Clock`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, Cache, Const, detect_signal_cycle
from .signals.vector import VecSignal
from .interp import LoopCurve
from .data import PointPath
from .ftsl_emit import EmitCtx, num, vec3, fmt, fmt3, value_token
from . import sweep as _sweep


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Element:
    """Base scene element.  Emits ftsl text and exposes its modulator roots."""

    def roots(self) -> List:
        """Every Signal / VecSignal stored on this element (for cycle checking)."""
        out: List = []
        for v in vars(self).values():
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        raise NotImplementedError


class Pattern(Element):
    """Marker base for procedural pattern blocks (see :mod:`loom.material`).

    A pattern is emitted before materials so a material may bind it via
    ``key pattern:<name>``.  The concrete :class:`~loom.material.FuncPattern`
    lives in ``material.py`` to avoid an import cycle; this base is what
    :class:`Scene` routes on."""


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class Texture(Element):
    """An image **skin**: ``texture "name" { file "path" … }``.

    Bind it to a surface by pointing a material's colour at it — the whole point of
    a skin is a spatially-varying diffuse albedo, so
    ``Material("hide", "diffuse", reflect="texture:<name>")`` wraps the image around
    the geometry (sampled at each hit's UV).  ``encoding`` (``srgb``/``linear``),
    ``filter`` (``bilinear``/``nearest``) and ``wrap`` (``repeat``/``clamp``/
    ``mirror``) pass straight through to ftrace.  A ``Texture`` holds no modulators
    (the file is fixed), so it is emitted once, before the materials that reference
    it.  Use :func:`skin` to make the texture *and* its material in one call.
    """

    def __init__(self, name: str, file, *, encoding: str = "srgb",
                 filter: str = "bilinear", wrap: str = "repeat") -> None:
        self.name = name
        # normalise to forward slashes so the emitted path is portable
        self.file = str(file).replace("\\", "/")
        if encoding not in ("srgb", "linear"):
            raise ValueError('texture encoding must be "srgb" or "linear"')
        if filter not in ("bilinear", "nearest"):
            raise ValueError('texture filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('texture wrap must be "repeat", "clamp" or "mirror"')
        self.encoding = encoding
        self.filter = filter
        self.wrap = wrap

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return (f'texture "{self.name}" {{ file "{self.file}"  '
                f'encoding {self.encoding}  filter {self.filter}  '
                f'wrap {self.wrap} }}')


class Material(Element):
    def __init__(self, name: str, mtype: str = "diffuse", **props) -> None:
        self.name = name
        self.mtype = mtype
        self.props = props

    def roots(self) -> List:
        return [v for v in self.props.values() if isinstance(v, (Signal, VecSignal))]

    def emit(self, ctx: EmitCtx) -> str:
        parts = [f"type {self.mtype}"]
        for k, v in self.props.items():
            parts.append(f"{k} {value_token(v, ctx.clock, ctx.cache)}")
        return f'material "{self.name}" {{ ' + "  ".join(parts) + " }"


def skin(name: str, image, *, mtype: str = "diffuse", encoding: str = "srgb",
         filter: str = "bilinear", wrap: str = "repeat",
         **props) -> Tuple["Texture", "Material"]:
    """Wrap an image over a surface: build the :class:`Texture` **and** a
    :class:`Material` bound to it, ready to drop into a scene::

        scene.add(*skin("hide", "textures/cow.png"), Sphere((0,0,0), 1, "hide"))

    The material's ``reflect`` is the image (a spatially-varying diffuse albedo,
    ftrace's ``reflect texture:<name>``); extra ``props`` (e.g. ``roughness=…``) pass
    through to the material, and the texture and material share ``name``.
    """
    tex = Texture(name, image, encoding=encoding, filter=filter, wrap=wrap)
    mat = Material(name, mtype, reflect=f"texture:{name}", **props)
    return tex, mat


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Sphere(Element):
    def __init__(self, center, radius, material: str) -> None:
        self.center = VecSignal.of(center) if not isinstance(center, VecSignal) \
            else center
        self.radius = radius
        self.material = material

    def roots(self) -> List:
        out: List = [self.center]
        if isinstance(self.radius, Signal):
            out.append(self.radius)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        c = vec3(self.center, ctx.clock, ctx.cache)
        r = num(self.radius, ctx.clock, ctx.cache)
        return f'sphere {{ center {fmt3(c)}  radius {fmt(r)}  material "{self.material}" }}'


class Beads(Element):
    """A view-independent "string of beads": ``count`` spheres sampled evenly
    along a :class:`LoopCurve` (or a :class:`PointPath`).  This is the simplest
    way to render a 3-D closed curve before the sweep engine (M4) exists."""

    def __init__(self, curve: Union[LoopCurve, PointPath], count: int,
                 radius, material: str) -> None:
        if isinstance(curve, PointPath):
            from .signals.core import Const
            curve = LoopCurve(curve, Const(0.0))
        self.curve = curve
        self.count = int(count)
        self.radius = radius
        self.material = material

    def roots(self) -> List:
        out: List = [self.curve]
        if isinstance(self.radius, Signal):
            out.append(self.radius)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        r = num(self.radius, ctx.clock, ctx.cache)
        lines: List[str] = []
        for k in range(self.count):
            p = self.curve.sample(k / self.count, ctx.clock, ctx.cache)
            lines.append(
                f'sphere {{ center {fmt3(p)}  radius {fmt(r)}  material "{self.material}" }}')
        return "\n".join(lines)


class Raw(Element):
    """Escape hatch: emit a fixed block of ftsl text verbatim (not animated)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return self.text


class SweptMesh(Element):
    """A profile swept along a spine curve into a triangle mesh (M4 sweep engine).

    The ``spine`` (a :class:`LoopCurve` or :class:`PointPath`) is sampled at
    ``count`` points *at the current clock*, oriented with a rotation-minimizing
    frame, scaled/twisted, and skinned into an OBJ that is written per-frame via
    ``ctx.asset_path``; the emitted ftsl is a ``mesh { file ... }`` reference.

    ``scale`` and ``twist`` may be plain numbers or :class:`Signal`\\ s (animated).
    ``turns`` adds a full ``turns * 2pi`` twist distributed along the spine.
    ``scale_profile`` is an optional ``f(u)->float`` multiplier (``u in [0,1)``)
    that swells/pinches the section along the spine (used by the ``blob`` preset).
    """

    def __init__(self, spine: Union[LoopCurve, PointPath], profile: Sequence[Tuple[float, float]],
                 *, count: int = 64, scale=1.0, twist=0.0, turns=0.0,
                 closed_spine: bool = True, closed_profile: bool = True,
                 material: str = "default", smooth: int = 1, name: str = "swept",
                 scale_profile: Optional[Callable[[float], float]] = None) -> None:
        if isinstance(spine, PointPath):
            spine = LoopCurve(spine, Const(0.0))
        self.spine = spine
        self.profile = [(float(a), float(b)) for (a, b) in profile]
        self.count = int(count)
        self.scale = scale
        self.twist = twist
        self.turns = turns
        self.closed_spine = closed_spine
        self.closed_profile = closed_profile
        self.material = material
        self.smooth = int(smooth)
        self.name = name
        self.scale_profile = scale_profile

    def roots(self) -> List:
        out: List = [self.spine]
        for v in (self.scale, self.twist, self.turns):
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        n = self.count
        pts = [self.spine.sample(k / n, ctx.clock, ctx.cache) for k in range(n)]
        base_sc = num(self.scale, ctx.clock, ctx.cache)
        base_tw = num(self.twist, ctx.clock, ctx.cache)
        turns = num(self.turns, ctx.clock, ctx.cache)
        scales: List[float] = []
        twists: List[float] = []
        for k in range(n):
            u = k / n
            mult = self.scale_profile(u) if self.scale_profile is not None else 1.0
            scales.append(base_sc * mult)
            twists.append(base_tw + turns * 2.0 * math.pi * u)
        rings = _sweep.sweep_rings(pts, self.profile, scales, twists, self.closed_spine)
        verts, faces = _sweep.skin_rings(rings, self.closed_spine, self.closed_profile)
        path = ctx.asset_path(self.name, "obj")
        _sweep.write_obj(path, verts, faces)
        return (f'mesh {{ file "{path.as_posix()}"  smooth {self.smooth}  '
                f'material "{self.material}" }}')


class IsoMesh(Element):
    """A scalar field **baked to a triangle mesh** per frame via marching cubes
    (M7), then referenced as ``mesh { file ... }``.

    ftrace root-finds isosurfaces directly, so most fields should be an
    :class:`~loom.iso.Isosurface` (emitted as a ``function { expr }`` string) —
    that is sharper and needs no baking.  Use ``IsoMesh`` only when a field must
    become geometry: a numpy-only field with no ftsl twin, a sampled volume, or a
    mesh destined for another tool.

    ``field`` is a :class:`~loom.spatial.SpatialExpr` (baked at the clock) or a
    vectorised ``f(X, Y, Z) -> ndarray``.  ``bounds``/``res``/``iso``/``adaptive``
    /``coarse`` pass straight through to :func:`loom.mcubes.mesh_field`.  The mesh
    is written per-frame via ``ctx.asset_path`` and re-baked every frame (so an
    animated field morphs); a **time-independent** field is baked once and cached.
    """

    def __init__(self, field, *, bounds=1.0, res=48, iso: float = 0.0,
                 adaptive: bool = False, coarse: int = 8,
                 material: str = "default", smooth: int = 1, name: str = "isomesh") -> None:
        self.field = field
        self.bounds = bounds
        self.res = res
        self.iso = float(iso)
        self.adaptive = bool(adaptive)
        self.coarse = int(coarse)
        self.material = material
        self.smooth = int(smooth)
        self.name = name
        self._cache_static: Optional[Tuple[list, list]] = None

    def roots(self) -> List:
        # A SpatialExpr exposes its temporal coefficients for cycle checking.
        if hasattr(self.field, "param_signals"):
            return list(self.field.param_signals())
        return []

    def _static(self) -> bool:
        return hasattr(self.field, "uses_time") and not self.field.uses_time()

    def emit(self, ctx: EmitCtx) -> str:
        from . import mcubes as _mc
        if self._static() and self._cache_static is not None:
            verts, faces = self._cache_static
        else:
            verts, faces = _mc.mesh_field(
                self.field, bounds=self.bounds, res=self.res, iso=self.iso,
                clock=ctx.clock, cache=ctx.cache,
                adaptive=self.adaptive, coarse=self.coarse)
            if self._static():
                self._cache_static = (verts, faces)
        path = ctx.asset_path(self.name, "obj")
        _sweep.write_obj(path, verts, faces)
        return (f'mesh {{ file "{path.as_posix()}"  smooth {self.smooth}  '
                f'material "{self.material}" }}')


def ribbon(spine, *, width: float = 0.3, material: str = "default", count: int = 64,
           twist=0.0, turns=0.0, closed_spine: bool = True, smooth: int = 0,
           name: str = "ribbon") -> SweptMesh:
    """A flat strip (open line profile) swept along the spine."""
    return SweptMesh(spine, _sweep.line_profile(width), count=count, scale=1.0,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=False, material=material, smooth=smooth, name=name)


def tube(spine, *, radius: float = 0.1, sides: int = 12, material: str = "default",
         count: int = 64, twist=0.0, turns=0.0, closed_spine: bool = True,
         smooth: int = 1, name: str = "tube") -> SweptMesh:
    """A closed circular tube swept along the spine."""
    return SweptMesh(spine, _sweep.circle_profile(sides, 1.0), count=count, scale=radius,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=True, material=material, smooth=smooth, name=name)


def blob(spine, *, radius: float = 0.15, sides: int = 16, bulge: float = 0.6,
         lobes: int = 2, material: str = "default", count: int = 96,
         twist=0.0, turns=0.0, closed_spine: bool = True, smooth: int = 1,
         name: str = "blob") -> SweptMesh:
    """A tube whose radius swells and pinches around the loop (``lobes`` bulges)."""
    def _prof(u: float) -> float:
        return 1.0 + bulge * math.sin(2.0 * math.pi * lobes * u)
    return SweptMesh(spine, _sweep.circle_profile(sides, 1.0), count=count, scale=radius,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=True, material=material, smooth=smooth, name=name,
                     scale_profile=_prof)


def fan(spine, *, width: float = 0.4, material: str = "default", count: int = 64,
        twist=0.0, turns=0.0, smooth: int = 0, name: str = "fan") -> SweptMesh:
    """An open ribbon swept along an *open* spine (fans out end to end)."""
    return SweptMesh(spine, _sweep.line_profile(width), count=count, scale=1.0,
                     twist=twist, turns=turns, closed_spine=False,
                     closed_profile=False, material=material, smooth=smooth, name=name)


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

class Light(Element):
    """Generic ``light <kind> { ...props... }``.  Props are animatable or strings
    (e.g. ``spd="preset:bb6500"``)."""

    def __init__(self, kind: str, **props) -> None:
        self.kind = kind
        self.props = props

    def roots(self) -> List:
        return [v for v in self.props.values() if isinstance(v, (Signal, VecSignal))]

    def emit(self, ctx: EmitCtx) -> str:
        parts = [f"{k} {value_token(v, ctx.clock, ctx.cache)}"
                 for k, v in self.props.items()]
        return f"light {self.kind} {{ " + "  ".join(parts) + " }"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera(Element):
    def __init__(self, eye, look_at, up=(0, 1, 0), fov_y=40.0,
                 mode: str = "R", res: Tuple[int, int] = (480, 480),
                 name: str = "cam") -> None:
        self.eye = VecSignal.of(eye) if not isinstance(eye, VecSignal) else eye
        self.look_at = VecSignal.of(look_at) if not isinstance(look_at, VecSignal) else look_at
        self.up = VecSignal.of(up) if not isinstance(up, VecSignal) else up
        self.fov_y = fov_y
        self.mode = mode
        self.res = (int(res[0]), int(res[1]))
        self.name = name

    def roots(self) -> List:
        out: List = [self.eye, self.look_at, self.up]
        if isinstance(self.fov_y, Signal):
            out.append(self.fov_y)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        e = vec3(self.eye, ctx.clock, ctx.cache)
        la = vec3(self.look_at, ctx.clock, ctx.cache)
        up = vec3(self.up, ctx.clock, ctx.cache)
        fov = num(self.fov_y, ctx.clock, ctx.cache)
        return (f'camera "{self.name}" {{\n'
                f'    eye {fmt3(e)}  look_at {fmt3(la)}  up {fmt3(up)}  fov_y {fmt(fov)}\n'
                f'    mode {self.mode}\n'
                f'    film {{ res {self.res[0]} {self.res[1]} }}\n'
                f'}}')


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, camera: Camera, *, units: str = "meters",
                 spectral: Tuple[float, float, float] = (360, 830, 1)) -> None:
        self.camera = camera
        self.units = units
        self.spectral = spectral
        self.textures: List[Element] = []
        self.patterns: List[Element] = []
        self.materials: List[Material] = []
        self.elements: List[Element] = []
        self.lights: List[Light] = []

    def add(self, *elems: Element) -> "Scene":
        for e in elems:
            # Textures/patterns are emitted before the materials that bind them
            # (ftrace resolves them in an earlier pass, but keep the text tidy).
            if isinstance(e, Texture):
                self.textures.append(e)
            elif isinstance(e, Pattern):
                self.patterns.append(e)
            elif isinstance(e, Material):
                self.materials.append(e)
            elif isinstance(e, Light):
                self.lights.append(e)
            else:
                self.elements.append(e)
        return self

    def _all_elements(self) -> List[Element]:
        return [*self.textures, *self.patterns, *self.materials, *self.elements,
                *self.lights, self.camera]

    def check_cycles(self) -> None:
        """Run the loop detector over every modulator in the scene."""
        for el in self._all_elements():
            for r in el.roots():
                detect_signal_cycle(r)

    def emit(self, clock: Clock, cache: Optional[Cache] = None, *,
             assets_dir: Optional["Path"] = None, tag: str = "") -> str:
        ctx = EmitCtx(clock=clock, cache=cache, assets_dir=assets_dir, tag=tag)
        lo, hi, step = self.spectral
        header = f"scene {{ units {self.units}  spectral {fmt(lo)} {fmt(hi)} {fmt(step)} }}"
        blocks = [header, ""]
        for tx in self.textures:
            blocks.append(tx.emit(ctx))
        if self.textures:
            blocks.append("")
        for p in self.patterns:
            blocks.append(p.emit(ctx))
        if self.patterns:
            blocks.append("")
        for m in self.materials:
            blocks.append(m.emit(ctx))
        blocks.append("")
        for e in self.elements:
            blocks.append(e.emit(ctx))
        blocks.append("")
        for lt in self.lights:
            blocks.append(lt.emit(ctx))
        blocks.append("")
        blocks.append(self.camera.emit(ctx))
        return "\n".join(blocks) + "\n"
