"""
Loom function-driven materials (M6) — animated reflectance / colour / IOR /
roughness over **space and time**.

ftrace evaluates a procedural ``pattern { expr "f(x,y,z, nx,ny,nz, r)" }`` at every
hit and binds its scalar to a material knob (``roughness``/``ior``/
``film_thickness``/``density``/a mix ``weight_map``).  The pattern is a function of
*space* only — it has no clock — so loom animates it the same way it animates an
isosurface: it **re-emits the formula each frame with the time-varying constants
baked in** (spatial frequency, an N-D tilt of the sampling frame, and a phase
drift).  A ``phase_drift`` advancing by ``2*pi`` over the loop returns the pattern
bit-for-bit at the wrap, so the material animation is seamless.

Two building blocks:
- :class:`FuncPattern` — a named ``pattern`` block (bind it with e.g.
  ``Material("m", "metal", roughness="pattern:rough")`` or a mix ``weight_map``);
- :class:`MixMaterial` — a 2-layer A/B blend whose ``weight_map`` is a
  :class:`FuncPattern`, giving spatially/temporally varying **colour** (the knob
  ftrace can't drive from a scalar pattern directly).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Number
from .signals.vector import VecSignal
from .mathnd import Mat
from .ftsl_emit import EmitCtx, num, fmt
from .scene import Element, Material, Pattern
from .iso import _mat3_at, _coord_expr

# A pattern template: coordinate expression strings -> a scalar ftsl formula.
PatternFn = Callable[[str, str, str], str]


# ---------------------------------------------------------------------------
# scalar pattern templates (each returns a value, typically in [0, 1])
# ---------------------------------------------------------------------------

def waves(cx: str, cy: str, cz: str) -> str:
    """A 1-D sinusoid along the first coordinate, remapped to [0, 1]."""
    return f"0.5+0.5*sin({cx})"

def checker(cx: str, cy: str, cz: str) -> str:
    """A 3-D checkerboard in [0, 1] (``sign`` of the product of three sines)."""
    return f"0.5+0.5*sign(sin({cx})*sin({cy})*sin({cz}))"

def rings(cx: str, cy: str, cz: str) -> str:
    """Concentric shells: ``0.5 + 0.5 sin |c|`` in [0, 1]."""
    return f"0.5+0.5*sin(sqrt(({cx})*({cx})+({cy})*({cy})+({cz})*({cz})))"

def blobs(cx: str, cy: str, cz: str) -> str:
    """Value noise in [0, 1] (ftrace's 3-arg ``noise``)."""
    return f"noise({cx},{cy},{cz})"


PATTERNS = {"waves": waves, "checker": checker, "rings": rings, "blobs": blobs}


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

class FuncPattern(Pattern):
    """A named procedural ``pattern`` emitted as ``pattern "name" { expr "..." }``.

    ``template`` is a name in :data:`PATTERNS`, a :data:`PatternFn` (called with the
    baked coordinate strings), or a literal ftsl expression string (static, in terms
    of ``x``/``y``/``z``).  ``freq``/``drift``/``rotation`` bake a time-varying affine
    sampling frame into the formula exactly as :class:`loom.Isosurface` does.
    """

    def __init__(self, name: str, template: Union[str, PatternFn], *,
                 freq: Union[Signal, Number] = 1.0,
                 drift: Union[VecSignal, Sequence] = (0.0, 0.0, 0.0),
                 rotation: Optional[Mat] = None) -> None:
        self.name = name
        if isinstance(template, str) and template in PATTERNS:
            self.template: Union[str, PatternFn] = PATTERNS[template]
        else:
            self.template = template
        self.freq = freq
        self.drift = drift if isinstance(drift, VecSignal) else VecSignal.of(drift)
        self.rotation = rotation

    def roots(self):
        out = [self.drift]
        if isinstance(self.freq, (Signal, VecSignal)):
            out.append(self.freq)
        if self.rotation is not None:
            for r in self.rotation.rows:
                out.extend(r)
        # a param-animatable template (e.g. a PovFn) contributes its params
        if hasattr(self.template, "param_signals"):
            out.extend(self.template.param_signals())
        return out

    def _body(self, ctx: EmitCtx) -> str:
        if isinstance(self.template, str):
            return self.template  # literal static expression (in x/y/z)
        M = _mat3_at(self.rotation, ctx.clock, ctx.cache)
        f = num(self.freq, ctx.clock, ctx.cache)
        d = self.drift.at(ctx.clock, ctx.cache)
        cx = _coord_expr(f, M[0], d[0])
        cy = _coord_expr(f, M[1], d[1])
        cz = _coord_expr(f, M[2], d[2])
        # context-aware templates (PovFn) bake their params; plain PatternFns don't
        if hasattr(self.template, "build"):
            return self.template.build(cx, cy, cz, ctx)
        return self.template(cx, cy, cz)

    def emit(self, ctx: EmitCtx) -> str:
        return f'{self.name} = pattern {{ expr "{self._body(ctx)}" }}'


class MixMaterial(Material):
    """A 2-layer A/B material blend selected by a per-hit ``weight_map`` pattern.

    ``layers`` is ``[(mat_name, weight), (mat_name, weight)]``; the ``weight_map``
    (a :class:`FuncPattern` name) drives the selection of layer 0 (layer 1 gets
    ``1 - map``), so a spatial/temporal pattern paints **colour** across a surface.
    """

    def __init__(self, name: str, layers: Sequence[Tuple[str, float]],
                 *, weight_map: Optional[str] = None) -> None:
        self.name = name
        self.layers: List[Tuple[str, float]] = [(str(n), float(w)) for n, w in layers]
        if len(self.layers) != 2 and weight_map is not None:
            raise ValueError("a weight_map mix needs exactly 2 layers")
        self.weight_map = weight_map

    def roots(self):
        return []

    def emit(self, ctx: EmitCtx) -> str:
        lines = [f'{self.name} = material {{', '    type mix']
        for n, w in self.layers:
            lines.append(f'    layer "{n}" {fmt(w)}')
        if self.weight_map is not None:
            lines.append(f'    weight_map pattern:{self.weight_map}')
        lines.append('}')
        return "\n".join(lines)
