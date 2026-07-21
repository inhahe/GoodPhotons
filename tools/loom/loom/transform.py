"""
Loom per-object transform — animatable position / size / rotation / skew.

A :class:`Transform` bundles four independently *animatable* fields — ``translate``
(position), ``scale`` (size), ``rotate`` (Euler XYZ degrees) and ``skew`` (shear) —
and knows how to wrap any element's emitted ``.ftsl`` block in a ``group { … }`` so
the renderer bakes the composed affine into world space.  Every field is a plain
number / sequence / :class:`~loom.signals.core.Signal` / :class:`~loom.signals.vector.VecSignal`,
so position, size, rotation and skew all modulate over time like everything else in loom.

Transform order matches ftrace's ``group`` (see ``src/mesh.h`` ``MeshXform`` and
``src/ftsl.h`` ``addGroup``): ``world = translate + Rz·Ry·Rx·(scale ⊙ (shear · local))``
— shear applied first (in the object's local frame), then scale, then Euler rotate
X→Y→Z, then translate.

**Skew / shear convention.**  ``skew=(a, b, c)`` is the unit-diagonal upper-triangular
shear ::

    x' = x + a·y + b·z
    y' =     y + c·z
    z' =         z

so ``a`` skews X along Y, ``b`` skews X along Z, ``c`` skews Y along Z.  This needs
ftrace's ``shear`` statement (added alongside translate/rotate/scale in ``group``);
analytic ``sphere{}`` cannot be sheared (it would become an ellipsoid) — use a mesh /
sweep for skewed geometry.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from .signals.core import Signal, Number
from .signals.vector import VecSignal
from .ftsl_emit import EmitCtx, vec3, fmt3


def _as_vec3(v) -> VecSignal:
    """Normalise a 3-vector-ish (sequence / VecSignal) to a dim-3 VecSignal."""
    if isinstance(v, VecSignal):
        if v.dim < 3:
            raise ValueError("Transform vector fields need >= 3 components")
        return v
    if isinstance(v, (Signal, int, float)):
        raise ValueError("translate/rotate/skew need a 3-sequence or VecSignal, "
                         "not a scalar")
    vs = VecSignal.of(v)
    if vs.dim < 3:
        raise ValueError("Transform vector fields need >= 3 components")
    return vs


def _as_scale(v) -> VecSignal:
    """Normalise a scale (scalar broadcasts to uniform, else per-axis)."""
    if isinstance(v, VecSignal):
        if v.dim < 3:
            raise ValueError("Transform scale needs 1 (uniform) or >= 3 components")
        return v
    if isinstance(v, (Signal, int, float)):
        s = v  # uniform: broadcast the same (possibly animated) scalar to all axes
        return VecSignal([s, s, s])
    vs = VecSignal.of(v)
    if vs.dim == 1:
        c = vs.components[0]
        return VecSignal([c, c, c])
    if vs.dim < 3:
        raise ValueError("Transform scale needs 1 (uniform) or >= 3 components")
    return vs


class Transform:
    """An animatable position / size / rotation / skew for any scene element.

    All four fields default to identity (``None``) and each is optional::

        Transform(translate=(1, 0, 0), scale=2.0, rotate=(0, 45, 0), skew=(0.3, 0, 0))

    ``translate`` / ``rotate`` / ``skew`` take a 3-sequence or :class:`VecSignal`;
    ``scale`` also accepts a scalar (uniform).  Any component may be a
    :class:`~loom.signals.core.Signal`, so the whole transform animates.
    """

    def __init__(self, translate=None, rotate=None, scale=None, skew=None) -> None:
        self.translate: Optional[VecSignal] = None if translate is None else _as_vec3(translate)
        self.rotate: Optional[VecSignal] = None if rotate is None else _as_vec3(rotate)
        self.scale: Optional[VecSignal] = None if scale is None else _as_scale(scale)
        self.skew: Optional[VecSignal] = None if skew is None else _as_vec3(skew)

    def is_identity(self) -> bool:
        return (self.translate is None and self.rotate is None
                and self.scale is None and self.skew is None)

    def roots(self) -> List:
        """Every VecSignal stored on this transform (for cycle checking)."""
        return [v for v in (self.translate, self.rotate, self.scale, self.skew)
                if v is not None]

    # ---- ftsl emission (geometry) ------------------------------------------
    def wrap(self, inner: str, ctx: EmitCtx) -> str:
        """Wrap an element's emitted block(s) in an ftsl ``group { … }`` carrying
        this transform.  Returns ``inner`` unchanged when the transform is identity."""
        clock, cache = ctx.clock, ctx.cache
        stmts: List[str] = []
        if self.translate is not None:
            stmts.append(f"translate {fmt3(vec3(self.translate, clock, cache))}")
        if self.rotate is not None:
            stmts.append(f"rotate {fmt3(vec3(self.rotate, clock, cache))}")
        if self.scale is not None:
            stmts.append(f"scale {fmt3(vec3(self.scale, clock, cache))}")
        if self.skew is not None:
            stmts.append(f"shear {fmt3(vec3(self.skew, clock, cache))}")
        if not stmts:
            return inner
        lines = ["group {"]
        for s in stmts:
            lines.append("    " + s)
        for block in inner.split("\n"):
            lines.append("    " + block if block else block)
        lines.append("}")
        return "\n".join(lines)
