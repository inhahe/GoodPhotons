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
    """Base scene element.  Emits ftsl text and exposes its modulator roots.

    An element may carry an optional :class:`~loom.transform.Transform` in ``xf``
    (position / size / rotation / skew, each animatable).  The transform is applied
    by the *container* (the :class:`Scene` or an enclosing :class:`Group`) when it
    emits the element — see :func:`emit_element` / :func:`element_roots` — so
    ``emit()`` always returns the element's *own* untransformed block and nesting a
    transformed element inside a transformed :class:`Group` composes correctly.
    """

    xf = None  # Optional[Transform]; applied by the container on emit

    def roots(self) -> List:
        """Every Signal / VecSignal stored on this element (for cycle checking)."""
        out: List = []
        for v in vars(self).values():
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        raise NotImplementedError

    def transformed(self, transform=None, *, translate=None, rotate=None,
                    scale=None, skew=None) -> "Element":
        """Attach a :class:`~loom.transform.Transform` (position / size / rotation /
        skew, all signal-modulatable) and return ``self`` for chaining::

            scene.add(Sphere((0, 0, 0), 1, "m").transformed(translate=(2, 0, 0),
                                                             scale=1.5))

        Pass a ready ``transform=Transform(...)`` or the individual fields.  Meant for
        geometry (spheres/meshes/sweeps/volumes); ``skew`` needs ftrace's ``shear``
        and does not apply to analytic ``sphere{}`` (which would become an ellipsoid).
        """
        from .transform import Transform
        self.xf = transform if transform is not None else Transform(
            translate=translate, rotate=rotate, scale=scale, skew=skew)
        return self


def emit_element(e: "Element", ctx: EmitCtx) -> str:
    """Emit an element's block, wrapping it in its :class:`~loom.transform.Transform`
    (an ftsl ``group { … }``) when it carries one.  Containers use this instead of
    calling ``e.emit()`` directly so per-element transforms are honoured (and nest)."""
    body = e.emit(ctx)
    xf = getattr(e, "xf", None)
    return xf.wrap(body, ctx) if xf is not None else body


def element_roots(e: "Element") -> List:
    """An element's modulator roots, including its transform's, for cycle checking."""
    out = list(e.roots())
    xf = getattr(e, "xf", None)
    if xf is not None:
        out.extend(xf.roots())
    return out


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
        return (f'{self.name} = texture {{ file "{self.file}"  '
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
        return f'{self.name} = material {{ ' + "  ".join(parts) + " }"


class ProcTexture(Element):
    """A **procedural (function-defined) UV-space skin**: ``texture "name" { rgb
    "r(u,v)" "g(u,v)" "b(u,v)" res N … }``.

    Instead of a bitmap file, the albedo is three ftsl expressions of the surface
    UV coordinates ``u`` and ``v`` (and constants / ``pi``), using the ftsl pattern
    grammar (``sin cos sqrt min max clamp mix step smoothstep noise`` …).  ftrace
    bakes them once to a ``res``×``res`` **linear** RGB grid at load and then treats
    the result exactly like an image texture — so the same UV-wrap, Jakob-Hanika
    spectral upsampling, triplanar, GPU and raster paths apply, and a material binds
    it with the usual ``reflect texture:<name>``.  The expressions are functions of
    ``u, v`` only (the world-space pattern variables carry no value in UV space).
    Like :class:`Texture`, it holds no modulators and is emitted once.  Use
    :func:`func_skin` to make the texture *and* its material together.
    """

    def __init__(self, name: str, r: str, g: str, b: str, *, res: int = 512,
                 filter: str = "bilinear", wrap: str = "clamp") -> None:
        self.name = name
        self.r = str(r)
        self.g = str(g)
        self.b = str(b)
        res = int(res)
        if res < 1:
            raise ValueError("texture res must be >= 1")
        if filter not in ("bilinear", "nearest"):
            raise ValueError('texture filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('texture wrap must be "repeat", "clamp" or "mirror"')
        self.res = res
        self.filter = filter
        self.wrap = wrap

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return (f'{self.name} = texture {{ rgb "{self.r}" "{self.g}" "{self.b}"  '
                f'res {self.res}  filter {self.filter}  wrap {self.wrap} }}')


def func_skin(name: str, r: str, g: str, b: str, *, mtype: str = "diffuse",
              res: int = 512, filter: str = "bilinear", wrap: str = "clamp",
              **props) -> Tuple["ProcTexture", "Material"]:
    """Wrap a **procedural** UV-space skin over a surface: build the
    :class:`ProcTexture` (three ``r(u,v) g(u,v) b(u,v)`` ftsl expressions) **and** a
    :class:`Material` bound to it::

        scene.add(*func_skin("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)"),
                  Sphere((0, 0, 0), 1, "stripes"))

    The material's ``reflect`` is the baked skin (ftrace's ``reflect texture:<name>``);
    extra ``props`` (e.g. ``roughness=…``) pass through, and the texture and material
    share ``name``.
    """
    tex = ProcTexture(name, r, g, b, res=res, filter=filter, wrap=wrap)
    mat = Material(name, mtype, reflect=f"texture:{name}", **props)
    return tex, mat


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


class Group(Element):
    """Apply one :class:`~loom.transform.Transform` to several child elements at
    once, emitted as a single ftsl ``group { … <children> }``.

    Position / size / rotation / skew all animate (each field may be a
    :class:`~loom.signals.core.Signal` / :class:`~loom.signals.vector.VecSignal`).
    Children may themselves be transformed — a child's own ``xf`` composes *inside*
    this group's (nested ftsl groups), so::

        Group(Sphere(...).transformed(scale=2), Beads(...),
              translate=(0, 1, 0), rotate=(0, t*90, 0))

    rotates the whole cluster while the sphere keeps its local 2× size.  Give the
    transform via the individual fields or a ready ``transform=Transform(...)``.
    """

    def __init__(self, *children: "Element", translate=None, rotate=None,
                 scale=None, skew=None, transform=None) -> None:
        from .transform import Transform
        self.children = list(children)
        self.xf = transform if transform is not None else Transform(
            translate=translate, rotate=rotate, scale=scale, skew=skew)

    def roots(self) -> List:
        out: List = []
        for c in self.children:
            out.extend(element_roots(c))
        return out

    def emit(self, ctx: EmitCtx) -> str:
        return "\n".join(emit_element(c, ctx) for c in self.children)


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
# Participating media / volumes
# ---------------------------------------------------------------------------

class Volume(Element):
    """A participating-medium region emitted as ftrace's ``medium { … }`` block.

    ftrace already renders volumes richly — homogeneous fog, bounded
    heterogeneous blobs whose density is a formula, and imported NanoVDB grids.
    loom's strength is the *procedural* case: a ``density`` field is loom's bread
    and butter (it's how :class:`~loom.iso.Isosurface` works), so a ``Volume``
    lets you animate clouds/fog with the same signal machinery as everything
    else — the scattering coefficients and the density formula are all
    :class:`~loom.signals.core.Signal`-valued.

    ``sigma_t`` (extinction), ``albedo`` (single-scatter albedo) and ``g``
    (Henyey–Greenstein anisotropy) are animatable scalars; ``rayleigh`` swaps the
    HG phase for a Rayleigh one.

    ``density`` shapes a *heterogeneous* medium (``None`` = uniform):

    * a :class:`~loom.spatial.SpatialExpr` — the natural loom field, emitted as an
      inline ``density "<expr>"`` over world ``x y z r`` (animatable, seamless);
    * a ``str`` — either ``"pattern:<name>"`` (bind a named pattern) or a raw ftsl
      expression;
    * ``"vdb:<path>"`` — reference an existing NanoVDB grid (loom doesn't *generate*
      sparse voxels, but it can point at one).

    The region is bounded by exactly one of ``box=(min, max)``, ``sphere=(center,
    radius)`` or ``obj="name"`` (fill a named scene object's interior).  A
    heterogeneous medium needs a finite region for the delta-tracking majorant, so
    give a bound *or* an explicit ``density_max``; ``density_max`` overrides the
    engine's grid estimate when set.
    """

    def __init__(self, *, sigma_t=1.0, albedo: Union[Signal, float] = 0.8,
                 g: Union[Signal, float] = 0.0, rayleigh: bool = False,
                 density=None, density_max=None,
                 box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                 sphere: Optional[Tuple[Sequence[float], float]] = None,
                 obj: Optional[str] = None, name: str = "volume") -> None:
        n_bounds = sum(x is not None for x in (box, sphere, obj))
        if n_bounds > 1:
            raise ValueError("Volume: give at most one of box=, sphere=, obj=")
        self.sigma_t = sigma_t
        self.albedo = albedo
        self.g = g
        self.rayleigh = bool(rayleigh)
        self.density = density
        self.density_max = density_max
        self.box = (tuple(float(c) for c in box[0]),
                    tuple(float(c) for c in box[1])) if box is not None else None
        self.sphere = ((tuple(float(c) for c in sphere[0]), float(sphere[1]))
                       if sphere is not None else None)
        self.obj = obj
        self.name = name

    def roots(self) -> List:
        out: List = []
        for v in (self.sigma_t, self.albedo, self.g, self.density_max):
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        # A SpatialExpr density exposes its temporal coefficients for cycle checking.
        if hasattr(self.density, "param_signals"):
            out.extend(self.density.param_signals())
        return out

    def _density_token(self, ctx: EmitCtx) -> Optional[str]:
        d = self.density
        if d is None:
            return None
        if hasattr(d, "emit"):                       # a loom SpatialExpr field
            return 'density "' + d.emit(("x", "y", "z"), ctx) + '"'
        s = str(d)
        if s.startswith("pattern:") or s.startswith("vdb:"):
            return f"density {s}"
        return f'density "{s}"'                       # raw ftsl expression

    def emit(self, ctx: EmitCtx) -> str:
        clock, cache = ctx.clock, ctx.cache
        parts = [f"sigma_t {fmt(num(self.sigma_t, clock, cache))}",
                 f"albedo {fmt(num(self.albedo, clock, cache))}",
                 f"g {fmt(num(self.g, clock, cache))}"]
        if self.rayleigh:
            parts.append("rayleigh true")
        lines = ["medium {", "    " + "  ".join(parts)]
        if self.box is not None:
            mn, mx = self.box
            lines.append(f"    bounds {{ min {fmt3(mn)}  max {fmt3(mx)} }}")
        elif self.sphere is not None:
            c, rad = self.sphere
            lines.append(f"    bounds {{ center {fmt3(c)}  radius {fmt(rad)} }}")
        elif self.obj is not None:
            lines.append(f'    bounds {{ object "{self.obj}" }}')
        dtok = self._density_token(ctx)
        if dtok is not None:
            lines.append("    " + dtok)
        if self.density_max is not None:
            lines.append(f"    density_max {fmt(num(self.density_max, clock, cache))}")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

class Light(Element):
    """Generic ``light <kind> { ...props... }``.  Props are animatable or strings
    and must use ftrace's real light schema (``spd`` for emission, plus per-kind
    geometry: ``origin``/``u``/``v`` for an area light, ``center``/``radius`` for a
    sphere, etc. — see ftrace's ``addLight``).  loom does not invent light fields;
    the one convenience is ``color=(r, g, b)``, which is emitted as an ``spd rgb …``
    emission spectrum, since ftrace lights are spectral and have no ``color`` field.
    """

    def __init__(self, kind: str, **props) -> None:
        self.kind = kind
        self.props = props

    def roots(self) -> List:
        return [v for v in self.props.values() if isinstance(v, (Signal, VecSignal))]

    def emit(self, ctx: EmitCtx) -> str:
        # Unified header: anonymous light with the subtype carried as a `kind`
        # property (`light { kind point  ... }`) rather than a bareword after KIND.
        parts = [f"kind {self.kind}"]
        for k, v in self.props.items():
            tok = value_token(v, ctx.clock, ctx.cache)
            if k == "color":
                # ftrace lights carry their emission in a spectral `spd`; there is no
                # `color` field. Author an RGB colour, emit it as `spd rgb r g b` (the
                # Jakob-Hanika upsample turns the triple into an emission spectrum).
                parts.append(f"spd rgb {tok}")
            else:
                parts.append(f"{k} {tok}")
        return "light { " + "  ".join(parts) + " }"


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
        return (f'{self.name} = camera {{\n'
                f'    eye {fmt3(e)}  look_at {fmt3(la)}  up {fmt3(up)}  fov_y {fmt(fov)}\n'
                f'    mode {self.mode}\n'
                f'    film {{ res {self.res[0]} {self.res[1]} }}\n'
                f'}}')


class CameraCurve(Element):
    """A genuine ftrace ``camera_curve`` flypath (milestone M13).

    The eye rides a Catmull-Rom spline through ``points`` with arc-length (or
    ``density``-shaped) speed, animatable lens/orientation tracks, and ftrace's
    two-axis orientation model.  Unlike :class:`Camera` — which loom re-bakes to a
    static ``camera`` block every frame — a ``camera_curve`` is emitted **once** and
    *ftrace itself* expands the N frames.  So pass it in place of the camera
    (``Scene(camera=CameraCurve(...))``) and render the single emitted ``.ftsl`` with
    ftrace to get the whole flyby; loom's per-frame clock does not drive it.

    Orientation mirrors ftrace's grammar 1:1 (nothing here loom can't emit):

    * **forward** (pick one, else the path tangent): ``look_at=(x,y,z)`` fixed target,
      ``look_points=[(x,y,z), …]`` a second aim spline, or ``fwd_at=[(t,x,y,z), …]``
      direction keyframes.
    * **up**: ``up_at=[(t,x,y,z), …]`` vector keyframes, or ``roll``/``roll_at`` an
      angle (degrees) about the reference up.
    * **reference frame**: ``frame`` sets the default for both axes and
      ``fwd_frame`` / ``up_frame`` override per axis — each ``"world"`` (fixed world
      axes, the classic behavior) or ``"travel"`` (the curve's rotation-minimizing
      frame, so the shot banks into turns; closed loops close seamlessly).  A
      ``fwd_at``/``up_at`` vector is read in the travel basis (x=right, y=up,
      z=forward) when its axis is ``"travel"``, else as a world direction.

    Scalar tracks (``roll_at``/``fov_at``/``zoom_at``/``fstop_at``/``focus_at``) are
    ``[(t, value), …]``; vector tracks (``fwd_at``/``up_at``) are ``[(t, x, y, z), …]``,
    with ``t`` the normalized timeline in ``[0, 1]``.
    """

    _FRAMES = ("world", "travel")

    def __init__(self, points, *, up=(0, 1, 0), fov_y=40.0, mode: str = "R",
                 res: Tuple[int, int] = (480, 480), frames: Optional[int] = None,
                 density=None, density_at=None, closed: bool = False,
                 spline: Optional[str] = None, look_at=None, look_points=None,
                 roll=None, roll_at=None, fov_at=None, zoom_at=None, fstop_at=None,
                 focus_at=None, fwd_at=None, up_at=None, frame: Optional[str] = None,
                 fwd_frame: Optional[str] = None, up_frame: Optional[str] = None,
                 min_reach=None, look_smooth=None, exposure_lock: bool = False,
                 fps=None, name: str = "curve") -> None:
        pts = [tuple(float(c) for c in p) for p in points]
        if len(pts) < 2:
            raise ValueError("CameraCurve needs >= 2 control `points`")
        if frames is None and density is None and density_at is None:
            raise ValueError("CameraCurve needs `frames=` or a `density=`/`density_at=`")
        if look_at is not None and look_points is not None:
            raise ValueError("CameraCurve: give at most one of look_at= / look_points=")
        for label, fv in (("frame", frame), ("fwd_frame", fwd_frame), ("up_frame", up_frame)):
            if fv is not None and fv not in self._FRAMES:
                raise ValueError(f'CameraCurve {label} must be "world" or "travel"')
        self.points = pts
        self.up = tuple(float(c) for c in up)
        self.fov_y = float(fov_y)
        self.mode = mode
        self.res = (int(res[0]), int(res[1]))
        self.frames = None if frames is None else int(frames)
        self.density = None if density is None else float(density)
        self.density_at = None if density_at is None else [(float(t), float(r)) for t, r in density_at]
        self.closed = bool(closed)
        self.spline = spline
        self.look_at = None if look_at is None else tuple(float(c) for c in look_at)
        self.look_points = (None if look_points is None
                            else [tuple(float(c) for c in p) for p in look_points])
        self.roll = None if roll is None else float(roll)
        self._scalar_tracks = {
            "roll_at": self._norm_scalar(roll_at), "fov_at": self._norm_scalar(fov_at),
            "zoom_at": self._norm_scalar(zoom_at), "fstop_at": self._norm_scalar(fstop_at),
            "focus_at": self._norm_scalar(focus_at),
        }
        self._vector_tracks = {
            "fwd_at": self._norm_vector(fwd_at), "up_at": self._norm_vector(up_at),
        }
        self.frame = frame
        self.fwd_frame = fwd_frame
        self.up_frame = up_frame
        self.min_reach = None if min_reach is None else float(min_reach)
        self.look_smooth = None if look_smooth is None else float(look_smooth)
        self.exposure_lock = bool(exposure_lock)
        self.fps = None if fps is None else float(fps)
        self.name = name

    @staticmethod
    def _norm_scalar(track):
        if track is None:
            return None
        return [(float(t), float(v)) for t, v in track]

    @staticmethod
    def _norm_vector(track):
        if track is None:
            return None
        out = []
        for kf in track:
            t, x, y, z = kf
            out.append((float(t), float(x), float(y), float(z)))
        return out

    def roots(self) -> List:
        return []   # a camera_curve is a static authored flight (no per-frame signals)

    def emit(self, ctx: EmitCtx) -> str:
        L = [f'{self.name} = camera_curve {{']
        for p in self.points:
            L.append(f"    point {fmt3(p)}")
        if self.look_points:
            L.append(f"    look curve")
            for p in self.look_points:
                L.append(f"    look_point {fmt3(p)}")
        elif self.look_at is not None:
            L.append(f"    look_at {fmt3(self.look_at)}")
        L.append(f"    up {fmt3(self.up)}   fov_y {fmt(self.fov_y)}   mode {self.mode}")
        if self.spline is not None:
            L.append(f"    spline {self.spline}")
        if self.frames is not None:
            L.append(f"    frames {self.frames}")
        if self.density is not None:
            L.append(f"    density {fmt(self.density)}")
        if self.density_at:
            for t, r in self.density_at:
                L.append(f"    density_at {fmt(t)} {fmt(r)}")
        if self.closed:
            L.append(f"    closed")
        # Reference-frame keywords (only when set; absence == world == legacy behavior).
        if self.frame is not None:
            L.append(f"    frame {self.frame}")
        if self.fwd_frame is not None:
            L.append(f"    fwd_frame {self.fwd_frame}")
        if self.up_frame is not None:
            L.append(f"    up_frame {self.up_frame}")
        # Orientation vector tracks.
        for key, track in self._vector_tracks.items():
            if track:
                for t, x, y, z in track:
                    L.append(f"    {key} {fmt(t)} {fmt3((x, y, z))}")
        # Scalar constant + tracks.
        if self.roll is not None:
            L.append(f"    roll {fmt(self.roll)}")
        for key, track in self._scalar_tracks.items():
            if track:
                for t, v in track:
                    L.append(f"    {key} {fmt(t)} {fmt(v)}")
        if self.min_reach is not None:
            L.append(f"    min_reach {fmt(self.min_reach)}")
        if self.look_smooth is not None:
            L.append(f"    look_smooth {fmt(self.look_smooth)}")
        if self.exposure_lock:
            L.append(f"    exposure_lock")
        if self.fps is not None:
            L.append(f"    fps {fmt(self.fps)}")
        L.append(f"    film {{ res {self.res[0]} {self.res[1]} }}")
        L.append("}")
        return "\n".join(L)


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
        self.records: List[Element] = []
        self.materials: List[Material] = []
        self.elements: List[Element] = []
        self.lights: List[Light] = []

    def add(self, *elems: Element) -> "Scene":
        from .record import Record as _Record  # lazy: record.py imports scene.Element
        for e in elems:
            # Textures/patterns/records are emitted before the materials that bind
            # them (ftrace resolves them in an earlier pass, but keep the text tidy).
            if isinstance(e, (Texture, ProcTexture)):
                self.textures.append(e)
            elif isinstance(e, Pattern):
                self.patterns.append(e)
            elif isinstance(e, _Record):
                self.records.append(e)
            elif isinstance(e, Material):
                self.materials.append(e)
            elif isinstance(e, Light):
                self.lights.append(e)
            else:
                self.elements.append(e)
        return self

    def _all_elements(self) -> List[Element]:
        return [*self.textures, *self.patterns, *self.records, *self.materials,
                *self.elements, *self.lights, self.camera]

    def check_cycles(self) -> None:
        """Run the loop detector over every modulator in the scene."""
        for el in self._all_elements():
            for r in element_roots(el):
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
        for rec in self.records:
            blocks.append(rec.emit(ctx))
        if self.records:
            blocks.append("")
        for m in self.materials:
            blocks.append(m.emit(ctx))
        blocks.append("")
        for e in self.elements:
            blocks.append(emit_element(e, ctx))
        blocks.append("")
        for lt in self.lights:
            blocks.append(emit_element(lt, ctx))
        blocks.append("")
        blocks.append(self.camera.emit(ctx))
        return "\n".join(blocks) + "\n"
