"""
Loom isosurface emission (M5) — an animatable implicit surface ``f(x,y,z)=0``.

ftrace renders arbitrary-expression isosurfaces natively (``isosurface { function
{ expr "..." } contained_by { ... } }``, sphere-traced with a Lipschitz bound), so
loom does NOT mesh these: it *emits the formula* and lets the renderer root-find.

What loom animates is the **coordinate frame** the field is read in, exactly the
Layer-2 slicer result: a 3-input field (a gyroid, a Schwarz surface, ...) seen
through an N-D rotation + scale + drift is an **affine remap of (x,y,z)** — tilt,
shear, spatial frequency, and a phase drift.  Every one of those is
:class:`~loom.signals.core.Signal`-valued, so the surface tilts, breathes and
flows.  A *phase drift* that advances by an integer multiple of ``2*pi`` across the
loop returns the field bit-for-bit to its start (``sin`` is ``2*pi``-periodic), so
the animation is seamless — the same discipline as every other loom loop.

A genuinely morphing (topology-changing) surface needs a >=4-input field, which
ftrace cannot evaluate directly; that is the marching-cubes path (M7, deferred).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Const, Number, TimeFn
from .signals.vector import VecSignal
from .mathnd import Mat
from .ftsl_emit import EmitCtx, num, fmt
from .scene import Element

# A field template: given coordinate *expression strings* cx, cy, cz (each an ftsl
# infix formula in x/y/z), return the implicit field expression f(cx,cy,cz).
FieldFn = Callable[[str, str, str], str]


# ---------------------------------------------------------------------------
# Triply-periodic minimal-surface field templates (all 2*pi-periodic per axis)
# ---------------------------------------------------------------------------

def gyroid(cx: str, cy: str, cz: str) -> str:
    """Schoen gyroid: ``sin x cos y + sin y cos z + sin z cos x``."""
    return (f"sin({cx})*cos({cy})+sin({cy})*cos({cz})+sin({cz})*cos({cx})")


def schwarz_p(cx: str, cy: str, cz: str) -> str:
    """Schwarz P ("primitive"): ``cos x + cos y + cos z``."""
    return f"cos({cx})+cos({cy})+cos({cz})"


def schwarz_d(cx: str, cy: str, cz: str) -> str:
    """Schwarz D ("diamond")."""
    return (f"sin({cx})*sin({cy})*sin({cz})"
            f"+sin({cx})*cos({cy})*cos({cz})"
            f"+cos({cx})*sin({cy})*cos({cz})"
            f"+cos({cx})*cos({cy})*sin({cz})")


def neovius(cx: str, cy: str, cz: str) -> str:
    """Neovius surface: ``3(cos x + cos y + cos z) + 4 cos x cos y cos z``."""
    return (f"3*(cos({cx})+cos({cy})+cos({cz}))"
            f"+4*cos({cx})*cos({cy})*cos({cz})")


FIELDS = {
    "gyroid": gyroid,
    "schwarz_p": schwarz_p,
    "schwarz_d": schwarz_d,
    "neovius": neovius,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mat3_at(m: Optional[Mat], clock, cache) -> Tuple[Tuple[float, float, float], ...]:
    """Materialize a 3x3 rotation/transform to concrete floats at the clock."""
    if m is None:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if m.nrows != 3 or m.ncols != 3:
        raise ValueError("Isosurface rotation must be a 3x3 Mat")
    return tuple(tuple(m.rows[i][j].at(clock, cache) for j in range(3))
                 for i in range(3))  # type: ignore[return-value]


def _coord_expr(freq: float, row: Sequence[float], drift: float) -> str:
    """One transformed coordinate: ``freq*(a*x + b*y + c*z) + drift`` as ftsl text.

    Each numeric coefficient is parenthesized so no ``+-`` adjacency ever reaches
    the expression parser (``(1)*x+(-0.5)*y`` is well-formed; ``1*x+-0.5*y`` risks
    tripping the shunting-yard)."""
    a, b, c = row
    lin = f"({fmt(a)})*x+({fmt(b)})*y+({fmt(c)})*z"
    return f"({fmt(freq)}*({lin})+({fmt(drift)}))"


# ---------------------------------------------------------------------------
# The scene element
# ---------------------------------------------------------------------------

class Isosurface(Element):
    """An animatable implicit surface emitted as an ftsl ``isosurface`` block.

    ``field`` is a name in :data:`FIELDS` or a :data:`FieldFn`.  ``freq`` scales
    the spatial frequency (cells per unit), ``rotation`` is an animatable 3x3
    :class:`Mat` (build it with :func:`loom.rotations`) that tilts/shears the
    coordinate frame, ``drift`` is an animatable phase offset per axis (advance it
    by ``2*pi`` over the loop for a seamless flow), and ``threshold`` is the level
    set ``f = threshold``.  All of ``freq``/``threshold``/``drift`` may be Signals.

    The surface is clipped to ``container``: a box (``bounds`` = (min, max)) or a
    ``"sphere"`` (``center``/``radius``) — a sphere reads the unavoidable cut of a
    space-filling surface as a rounded edge instead of hard box facets.
    """

    def __init__(self, field: Union[str, FieldFn], *,
                 freq: Union[Signal, Number] = 1.0,
                 threshold: Union[Signal, Number] = 0.0,
                 drift: Union[VecSignal, Sequence] = (0.0, 0.0, 0.0),
                 rotation: Optional[Mat] = None,
                 material: str = "default",
                 container: str = "box",
                 bounds: Tuple[Sequence[float], Sequence[float]] =
                     ((-math.pi, -math.pi, -math.pi), (math.pi, math.pi, math.pi)),
                 center: Sequence[float] = (0.0, 0.0, 0.0),
                 radius: float = math.pi,
                 max_gradient: float = 0.0,
                 method: str = "adaptive",
                 open: bool = False,
                 name: str = "iso") -> None:
        self.field: FieldFn = FIELDS[field] if isinstance(field, str) else field
        self.freq = freq
        self.threshold = threshold
        self.drift = drift if isinstance(drift, VecSignal) else VecSignal.of(drift)
        self.rotation = rotation
        self.material = material
        self.container = container
        self.bounds = (tuple(float(c) for c in bounds[0]),
                       tuple(float(c) for c in bounds[1]))
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.max_gradient = float(max_gradient)
        self.method = method
        self.open = bool(open)
        self.name = name

    def roots(self):
        out = [self.drift]
        for v in (self.freq, self.threshold):
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        if self.rotation is not None:
            for r in self.rotation.rows:
                out.extend(r)
        # a param-animatable field template (e.g. a PovFn) contributes its params
        if hasattr(self.field, "param_signals"):
            out.extend(self.field.param_signals())
        return out

    def emit(self, ctx: EmitCtx) -> str:
        clock, cache = ctx.clock, ctx.cache
        M = _mat3_at(self.rotation, clock, cache)
        f = num(self.freq, clock, cache)
        thr = num(self.threshold, clock, cache)
        d = self.drift.at(clock, cache)
        cx = _coord_expr(f, M[0], d[0])
        cy = _coord_expr(f, M[1], d[1])
        cz = _coord_expr(f, M[2], d[2])
        # context-aware templates (PovFn) bake their params; plain FieldFns don't
        field_expr = (self.field.build(cx, cy, cz, ctx)
                      if hasattr(self.field, "build")
                      else self.field(cx, cy, cz))
        expr = f"{field_expr}-({fmt(thr)})"

        lines = [f'isosurface "{self.name}" {{']
        lines.append(f'    material "{self.material}"')
        lines.append(f'    function {{ expr "{expr}" }}')
        if self.container == "sphere":
            cc = self.center
            lines.append(f'    contained_by {{ sphere {{ center {fmt(cc[0])} '
                         f'{fmt(cc[1])} {fmt(cc[2])}  radius {fmt(self.radius)} }} }}')
        else:
            mn, mx = self.bounds
            lines.append(f'    contained_by {{ min {fmt(mn[0])} {fmt(mn[1])} {fmt(mn[2])}'
                         f'  max {fmt(mx[0])} {fmt(mx[1])} {fmt(mx[2])} }}')
        if self.max_gradient > 0.0:
            lines.append(f'    max_gradient {fmt(self.max_gradient)}')
        if self.method != "adaptive":
            lines.append(f'    method {self.method}')
        if self.open:
            lines.append('    open on')
        lines.append('}')
        return "\n".join(lines)


def gyroid_surface(**kw) -> Isosurface:
    """Convenience: an :class:`Isosurface` using the gyroid field."""
    kw.setdefault("name", "gyroid")
    return Isosurface("gyroid", **kw)


def phase_drift(turns: float = 1.0) -> Signal:
    """A seamless phase ramp ``2*pi*turns*t`` over one loop.

    Feed it into an :class:`Isosurface`'s ``drift`` (per axis) so the field flows
    steadily and returns bit-for-bit to its start at the wrap (``sin``/``cos`` are
    ``2*pi``-periodic, so an integer ``turns`` closes the loop exactly)."""
    k = 2.0 * math.pi * float(turns)
    return TimeFn(lambda t: k * t)
