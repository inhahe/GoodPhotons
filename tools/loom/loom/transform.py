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
ftrace's ``shear`` statement (added alongside translate/rotate/scale in ``group``).
A ``sphere{}`` under a non-uniform scale or shear is auto-tessellated by ftrace into a
smooth-normal ellipsoid / sheared quadric mesh at load (a uniform-scaled sphere keeps
the fast analytic path), so squashed and skewed spheres just work.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

from .signals.core import Signal, Number, Sin, Cos, as_signal
from .signals.vector import VecSignal
from .ftsl_emit import EmitCtx, vec3, fmt3

_D2R = math.pi / 180.0


def _rad(angle: Signal) -> Signal:
    """Degrees→radians as a Signal expression (angle may itself be a Signal)."""
    return as_signal(angle) * _D2R


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

    # ---- inverse mapping (dataset sampling frame) --------------------------
    def inverse_apply(self, query) -> VecSignal:
        """Map a **world-space** query point back into this transform's *local* frame.

        The forward map is the same one :meth:`wrap` bakes into ftrace,
        ``world = T + Rz·Ry·Rx·(S ⊙ (Shear·local))``; this returns ``local`` as a
        VecSignal *expression*.  It is what decouples a sampling curve from a data
        object: a :class:`~loom.data.Grid` / :class:`~loom.data.Scatter` stores its
        values in a fixed local frame and carries a Transform placing it in world
        space; a curve that lives in world space is inverse-mapped through this before
        interpolation, so **moving / resizing / skewing the data object changes which
        local coordinate each world curve point lands on — and therefore the value it
        reads back.**  Every transform parameter may be a :class:`Signal`, so the whole
        remap animates and threads into the modulation DAG.

        Supports 2-D and 3-D queries.  In 2-D only the in-plane parameters act
        (``translate``/``scale`` XY, ``rotate`` about Z, ``skew`` X-along-Y); the
        out-of-plane components are ignored.
        """
        q = VecSignal.of(query)
        d = q.dim
        if d not in (2, 3):
            raise ValueError(
                "Transform.inverse_apply supports only 2-D or 3-D queries "
                f"(got dim {d})")
        c: List[Signal] = list(q.components)
        # Undo, outermost first: translate, then rotation, then scale, then shear.
        if self.translate is not None:
            t = self.translate.components
            c = [c[i] - t[i] for i in range(d)]
        if self.rotate is not None:
            c = self._inv_rotate(c, d)
        if self.scale is not None:
            s = self.scale.components
            c = [c[i] / s[i] for i in range(d)]
        if self.skew is not None:
            c = self._inv_shear(c, d)
        return VecSignal(c)

    def _inv_rotate(self, c: List[Signal], d: int) -> List[Signal]:
        r = self.rotate.components
        if d == 2:
            cz, sz = Cos(_rad(r[2])), Sin(_rad(r[2]))
            x, y = c[0], c[1]
            return [cz * x + sz * y, cz * y - sz * x]
        cx, sx = Cos(_rad(r[0])), Sin(_rad(r[0]))
        cy, sy = Cos(_rad(r[1])), Sin(_rad(r[1]))
        cz, sz = Cos(_rad(r[2])), Sin(_rad(r[2]))
        x, y, z = c
        # Rz^T
        x, y = cz * x + sz * y, cz * y - sz * x
        # Ry^T
        x, z = cy * x - sy * z, cy * z + sy * x
        # Rx^T
        y, z = cx * y + sx * z, cx * z - sx * y
        return [x, y, z]

    def _inv_shear(self, c: List[Signal], d: int) -> List[Signal]:
        sk = self.skew.components
        a = sk[0]
        if d == 2:
            # forward x' = x + a*y  ->  x = x' - a*y'
            return [c[0] - a * c[1], c[1]]
        b, cc = sk[1], sk[2]
        z = c[2]
        y = c[1] - cc * z
        x = c[0] - a * y - b * z
        return [x, y, z]

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
