"""
Mesh deformers from loom (ftrace ROADMAP 11 stage E): a mesh file whose vertices handles move.

ftrace evaluates the deformers itself, at every motion key on the same rest mesh, so a shutter
blurs the deformation exactly; loom animates the handles. ::

    tip = vec(0.0, Sine(1.0) * 0.3, 0.0)
    scene.add(DeformedMesh("meshes/bar.obj", [
        Elastic([Handle((0, 0, 0), radius=0.25),                 # the root, clamped
                 Handle((2, 0, 0), to=vec(2, 0, 0) + tip, radius=0.15)]),
    ], material="rubber", name="bar"))

A handle sits at ``at`` on the rest mesh, in the mesh's own coordinates (as its file gives
them, before any transform or group: a bend travels with the mesh), and is moved to ``to`` (a
tuple or a VecSignal; omitted, it holds still). With motion keys (``Scene.emit(motion=...)``) a
moving handle gets one ``to`` per key.

* :class:`Falloff` moves the vertices within each handle's ``radius`` by its move, weighted by a
  ``profile`` (smooth, linear, gaussian).
* :class:`Rbf` is the smooth field that takes every handle's point exactly to its target.
* :class:`Elastic` holds each handle's vertices (those within its ``radius``) at their targets and
  lets the rest follow as rigidly as it can (as-rigid-as-possible). Given ``dynamic=Dynamics(...)``
  it has inertia -- it lags, overshoots and settles -- and is simulated through the shot once, in
  frame order, into a cache each frame then reads: :func:`~loom.drive.render_range` (and so
  ``python -m loom.shot`` / ``ftrace -shot``) runs ``ftrace -bake-elastic`` over the shot's frames
  before rendering them. A dynamic elastic must be its mesh's first deformer.

Deformers stack in order, each moving the last one's output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .ftsl_emit import EmitCtx, fmt, fmt3, site_node, vec3
from .scene import Element, Group

Vec3 = Tuple[float, float, float]


def _moving(v) -> bool:
    """Whether ``v`` is a signal (something that may move), not a plain tuple."""
    return site_node(v) is not None


class Handle:
    """A point ``at`` on the rest mesh moved to ``to`` (a 3-tuple or a VecSignal; ``None``: held
    where it is). ``radius``: how far a falloff reaches, or which vertices an elastic holds."""

    def __init__(self, at: Sequence[float], to=None, radius: float = 0.0) -> None:
        self.at = tuple(float(x) for x in at)
        if len(self.at) != 3:
            raise ValueError(f"Handle: `at` needs three coordinates, got {at!r}")
        self.to = to
        self.radius = float(radius)

    def roots(self) -> List:
        n = site_node(self.to)
        return [n] if n is not None else []

    def emit(self, ctx: EmitCtx) -> str:
        r = f"  radius {fmt(self.radius)}" if self.radius > 0.0 else ""
        if self.to is None:
            return f"handle {{ at {fmt3(self.at)}{r} }}"
        if ctx.motion is not None and _moving(self.to):
            # Keys only where the handle moves over the interval, as Transform.wrap writes them:
            # identical keys would make a still mesh deform through the shutter for nothing.
            keys = [fmt3(vec3(self.to, kc, kcache)) for kc, kcache in ctx.key_clocks()]
            if any(k != keys[0] for k in keys):
                body = "  ".join(f"key {{ to {k} }}" for k in keys)
                return f"handle {{ at {fmt3(self.at)}{r}  motion {{ {body} }} }}"
        return f"handle {{ at {fmt3(self.at)}  to {fmt3(vec3(self.to, ctx.clock, ctx.cache))}{r} }}"


class Deformer:
    """A deformer: its ftrace kind, its handles and its options."""
    kind = ""

    def __init__(self, handles: Iterable[Handle]) -> None:
        self.handles = list(handles)
        if not self.handles:
            raise ValueError(f"{type(self).__name__}: give it at least one Handle")

    def roots(self) -> List:
        return [n for h in self.handles for n in h.roots()]

    def options(self, ctx: EmitCtx, mesh: str) -> str:
        return ""

    def emit(self, ctx: EmitCtx, mesh: str) -> str:
        opts = self.options(ctx, mesh)
        hs = "  ".join(h.emit(ctx) for h in self.handles)
        return f"deform {self.kind} {{ {opts + '  ' if opts else ''}{hs} }}"


class Falloff(Deformer):
    """Each handle moves the vertices within its radius by its move, weighted by ``profile``."""
    kind = "falloff"

    def __init__(self, handles: Iterable[Handle], profile: str = "smooth") -> None:
        super().__init__(handles)
        for h in self.handles:
            if not h.radius > 0.0:
                raise ValueError("Falloff: every Handle needs a radius (how far its pull reaches)")
        if profile not in ("smooth", "linear", "gaussian"):
            raise ValueError(f"Falloff: profile smooth, linear or gaussian, got {profile!r}")
        self.profile = profile

    def options(self, ctx: EmitCtx, mesh: str) -> str:
        return "" if self.profile == "smooth" else f"profile {self.profile}"


class Rbf(Deformer):
    """The smooth field taking every handle's point exactly to its target (``kernel``:
    biharmonic, triharmonic, gaussian or multiquadric; ``eps`` their width for the last two)."""
    kind = "rbf"

    def __init__(self, handles: Iterable[Handle], kernel: str = "biharmonic", eps: float = 1.0) -> None:
        super().__init__(handles)
        if kernel not in ("biharmonic", "triharmonic", "gaussian", "multiquadric"):
            raise ValueError(f"Rbf: kernel biharmonic, triharmonic, gaussian or multiquadric, got {kernel!r}")
        self.kernel, self.eps = kernel, float(eps)

    def options(self, ctx: EmitCtx, mesh: str) -> str:
        out = [] if self.kernel == "biharmonic" else [f"kernel {self.kernel}"]
        if self.eps != 1.0:
            out.append(f"eps {fmt(self.eps)}")
        return "  ".join(out)


class Dynamics:
    """An elastic's inertia: ``mass`` per unit area, ``stiffness``, ``damping`` (the share of
    velocity lost per second, 0-1), ``substeps`` a frame, ARAP ``iterations`` a step. ``cache``
    is the records' path with one frame number in it (``%04d``); by default
    ``<the frames' directory>/<mesh name>_%04d.ftec``."""

    def __init__(self, mass: float = 1.0, stiffness: float = 2.0e4, damping: float = 0.9,
                 substeps: int = 8, iterations: int = 8, cache: Optional[str] = None) -> None:
        if not (mass > 0 and stiffness > 0 and 0 <= damping <= 1 and substeps >= 1 and iterations >= 1):
            raise ValueError("Dynamics: mass and stiffness above 0, damping 0-1, substeps and iterations >= 1")
        self.mass, self.stiffness, self.damping = float(mass), float(stiffness), float(damping)
        self.substeps, self.iterations, self.cache = int(substeps), int(iterations), cache

    def cache_path(self, ctx: EmitCtx, mesh: str) -> str:
        if self.cache is not None:
            # Absolute, from where the shot runs, as its outdir is: the bake and the frames run
            # ftrace in the repo root, where a relative path would name another place.
            return Path(self.cache).resolve().as_posix()
        import tempfile
        d = Path(ctx.assets_dir) if ctx.assets_dir is not None else Path(tempfile.gettempdir())
        return (d.resolve() / f"{mesh}_%04d.ftec").as_posix()

    def emit(self, ctx: EmitCtx, mesh: str) -> str:
        c = ctx.clock
        path = self.cache_path(ctx, mesh)
        if path in ctx.elastic_caches:
            raise ValueError(f"Dynamics: two dynamic elastics would share the cache {path!r} -- give "
                             f"their meshes different names, or each Dynamics its own `cache`")
        ctx.elastic_caches.add(path)
        return (f'dynamic {{ cache "{path}"  frame {int(c.frame)}  mass {fmt(self.mass)}  '
                f"stiffness {fmt(self.stiffness)}  damping {fmt(self.damping)}  fps {fmt(c.fps)}  "
                f"substeps {self.substeps}  iterations {self.iterations} }}")


class Elastic(Deformer):
    """As-rigid-as-possible: each handle's vertices (within its radius) held at their targets,
    the rest following like rubber. ``dynamic``: a :class:`Dynamics` for inertia (baked)."""
    kind = "elastic"

    def __init__(self, handles: Iterable[Handle], iterations: Optional[int] = None,
                 dynamic: Optional[Dynamics] = None) -> None:
        super().__init__(handles)
        for h in self.handles:
            if not h.radius > 0.0:
                raise ValueError("Elastic: every Handle needs a radius (the vertices it holds)")
        if iterations is not None and dynamic is not None:
            raise ValueError("Elastic: a dynamic elastic takes its rounds per step from "
                             "Dynamics(iterations=...)")
        self.iterations, self.dynamic = iterations, dynamic

    def options(self, ctx: EmitCtx, mesh: str) -> str:
        out = [] if self.iterations is None else [f"iterations {int(self.iterations)}"]
        if self.dynamic is not None:
            out.append(self.dynamic.emit(ctx, mesh))
        return "  ".join(out)


class DeformedMesh(Element):
    """A mesh file moved by ``deformers`` (applied in order), as ftrace's ``deform`` blocks."""

    def __init__(self, file: Union[str, Path], deformers: Sequence[Deformer], *,
                 material: str = "default", name: str = "deformed") -> None:
        self.file = Path(file)
        self.deformers = list(deformers)
        if not self.deformers:
            raise ValueError("DeformedMesh: give it at least one deformer")
        for i, d in enumerate(self.deformers):
            if i > 0 and isinstance(d, Elastic) and d.dynamic is not None:
                raise ValueError("DeformedMesh: a dynamic Elastic must be the first deformer")
        self.material = material
        self.name = name

    def roots(self) -> List:
        return [n for d in self.deformers for n in d.roots()]

    def dynamic(self) -> bool:
        return any(isinstance(d, Elastic) and d.dynamic is not None for d in self.deformers)

    def emit(self, ctx: EmitCtx) -> str:
        body = "  ".join(d.emit(ctx, self.name) for d in self.deformers)
        return (f'mesh "{self.name}" {{ file "{self.file.as_posix()}"  material "{self.material}"\n'
                f"    {body} }}")


def needs_elastic_bake(elements: Iterable[Element]) -> bool:
    """Whether any of ``elements`` (groups searched) is a mesh with a dynamic elastic."""
    for e in elements:
        if isinstance(e, DeformedMesh) and e.dynamic():
            return True
        if isinstance(e, Group) and needs_elastic_bake(e.children):
            return True
    return False
