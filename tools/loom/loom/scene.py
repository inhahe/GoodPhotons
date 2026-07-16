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

from typing import Dict, List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, Cache, detect_signal_cycle
from .signals.vector import VecSignal
from .interp import LoopCurve
from .data import PointPath
from .ftsl_emit import num, vec3, fmt, fmt3, value_token


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

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class Material(Element):
    def __init__(self, name: str, mtype: str = "diffuse", **props) -> None:
        self.name = name
        self.mtype = mtype
        self.props = props

    def roots(self) -> List:
        return [v for v in self.props.values() if isinstance(v, (Signal, VecSignal))]

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        parts = [f"type {self.mtype}"]
        for k, v in self.props.items():
            parts.append(f"{k} {value_token(v, clock, cache)}")
        return f'material "{self.name}" {{ ' + "  ".join(parts) + " }"


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

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        c = vec3(self.center, clock, cache)
        r = num(self.radius, clock, cache)
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

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        r = num(self.radius, clock, cache)
        lines: List[str] = []
        for k in range(self.count):
            p = self.curve.sample(k / self.count, clock, cache)
            lines.append(
                f'sphere {{ center {fmt3(p)}  radius {fmt(r)}  material "{self.material}" }}')
        return "\n".join(lines)


class Raw(Element):
    """Escape hatch: emit a fixed block of ftsl text verbatim (not animated)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def roots(self) -> List:
        return []

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        return self.text


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

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        parts = [f"{k} {value_token(v, clock, cache)}" for k, v in self.props.items()]
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

    def emit(self, clock: Clock, cache: Optional[Cache]) -> str:
        e = vec3(self.eye, clock, cache)
        la = vec3(self.look_at, clock, cache)
        up = vec3(self.up, clock, cache)
        fov = num(self.fov_y, clock, cache)
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
        self.materials: List[Material] = []
        self.elements: List[Element] = []
        self.lights: List[Light] = []

    def add(self, *elems: Element) -> "Scene":
        for e in elems:
            if isinstance(e, Material):
                self.materials.append(e)
            elif isinstance(e, Light):
                self.lights.append(e)
            else:
                self.elements.append(e)
        return self

    def _all_elements(self) -> List[Element]:
        return [*self.materials, *self.elements, *self.lights, self.camera]

    def check_cycles(self) -> None:
        """Run the loop detector over every modulator in the scene."""
        for el in self._all_elements():
            for r in el.roots():
                detect_signal_cycle(r)

    def emit(self, clock: Clock, cache: Optional[Cache] = None) -> str:
        lo, hi, step = self.spectral
        header = f"scene {{ units {self.units}  spectral {fmt(lo)} {fmt(hi)} {fmt(step)} }}"
        blocks = [header, ""]
        for m in self.materials:
            blocks.append(m.emit(clock, cache))
        blocks.append("")
        for e in self.elements:
            blocks.append(e.emit(clock, cache))
        blocks.append("")
        for lt in self.lights:
            blocks.append(lt.emit(clock, cache))
        blocks.append("")
        blocks.append(self.camera.emit(clock, cache))
        return "\n".join(blocks) + "\n"
