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
from typing import List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Number, Sin, Cos, Const, as_signal
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
        return VecSignal(self._inverse_map(list(q.components), q.dim, lambda s: s))

    def inverse_apply_spatial(self, coords) -> List:
        """The :mod:`loom.spatial` twin of :meth:`inverse_apply`: map world-space
        **SpatialExpr** coordinates back into this transform's local frame.

        Identical formula, identical code path — only the algebra differs, because a
        transform's own parameters are temporal :class:`Signal`\\s in both cases and
        the spatial algebra coerces them (``spatial._Sig``).  This is what lets a
        placed :class:`~loom.data.Grid` be sampled from a *field* expression
        (``grid(X, Y)``) and still read the same values a world-space curve would
        (:func:`loom.interp._local_query`) — the placement is folded into the emitted
        coordinate arguments, since ftsl's ``grid`` block has no transform of its own.
        """
        from .spatial import sexpr
        c = [sexpr(x) for x in coords]
        return self._inverse_map(c, len(c), sexpr)

    def _inverse_map(self, c: List, d: int, wrap) -> List:
        """Shared body of :meth:`inverse_apply` / :meth:`inverse_apply_spatial`.

        ``c`` holds the query components in *some* algebra (temporal Signals or
        spatial expressions) and ``wrap`` lifts one of this transform's own scalar
        parameters into that same algebra, so the two backends can never drift.
        """
        if d not in (2, 3):
            raise ValueError(
                "Transform.inverse_apply supports only 2-D or 3-D queries "
                f"(got dim {d})")
        # Undo, outermost first: translate, then rotation, then scale, then shear.
        if self.translate is not None:
            t = self.translate.components
            c = [c[i] - wrap(t[i]) for i in range(d)]
        if self.rotate is not None:
            c = self._inv_rotate(c, d, wrap)
        if self.scale is not None:
            s = self.scale.components
            c = [c[i] / wrap(s[i]) for i in range(d)]
        if self.skew is not None:
            c = self._inv_shear(c, d, wrap)
        return c

    def _inv_rotate(self, c: List, d: int, wrap) -> List:
        r = self.rotate.components
        if d == 2:
            cz, sz = wrap(Cos(_rad(r[2]))), wrap(Sin(_rad(r[2])))
            x, y = c[0], c[1]
            return [cz * x + sz * y, cz * y - sz * x]
        cx, sx = wrap(Cos(_rad(r[0]))), wrap(Sin(_rad(r[0])))
        cy, sy = wrap(Cos(_rad(r[1]))), wrap(Sin(_rad(r[1])))
        cz, sz = wrap(Cos(_rad(r[2]))), wrap(Sin(_rad(r[2])))
        x, y, z = c
        # Rz^T
        x, y = cz * x + sz * y, cz * y - sz * x
        # Ry^T
        x, z = cy * x - sy * z, cy * z + sy * x
        # Rx^T
        y, z = cx * y + sx * z, cx * z - sx * y
        return [x, y, z]

    def _inv_shear(self, c: List, d: int, wrap) -> List:
        sk = self.skew.components
        a = wrap(sk[0])
        if d == 2:
            # forward x' = x + a*y  ->  x = x' - a*y'
            return [c[0] - a * c[1], c[1]]
        b, cc = wrap(sk[1]), wrap(sk[2])
        z = c[2]
        y = c[1] - cc * z
        x = c[0] - a * y - b * z
        return [x, y, z]

    # ---- ftsl emission (geometry) ------------------------------------------
    def _values(self, clock, cache) -> List[Tuple[str, Tuple[float, float, float]]]:
        """(ftsl keyword, value) for each field this transform sets, at ``clock``."""
        out: List[Tuple[str, Tuple[float, float, float]]] = []
        for kw, v in (("translate", self.translate), ("rotate", self.rotate),
                      ("scale", self.scale), ("shear", self.skew)):
            if v is not None:
                out.append((kw, vec3(v, clock, cache)))
        return out

    def wrap(self, inner: str, ctx: EmitCtx) -> str:
        """Wrap an element's emitted block(s) in an ftsl ``group { … }`` carrying
        this transform.  Returns ``inner`` unchanged when the transform is identity.

        With ``ctx.motion``, a transform whose value changes over the motion interval is
        written as a ``motion { key { … } … }`` block, one key per motion key; one that holds
        still is written as before. A rotation's keys follow the angle continuously from the
        instant's key outward (:meth:`_rotation_keys`), because ftrace turns through the angle
        between keys: an angle a periodic signal folds from 359 back to 1 would otherwise spin
        the long way round, and one that really turns 250 degrees between keys must not be read
        as a fold."""
        vals = self._values(ctx.clock, ctx.cache)
        if not vals:
            return inner
        head = ["    %s %s" % (kw, fmt3(v)) for kw, v in vals]
        if ctx.motion is not None:
            keys = [self._values(c, k) for c, k in ctx.key_clocks()]
            # Follow a rotation that varies before asking whether anything moves: one turning a
            # whole turn between keys reads the same angle at every key, and must still move.
            if self.rotate is not None and not all(isinstance(c, Const) for c in self.rotate.components):
                f = [kw for kw, _ in vals].index("rotate")
                for key, r in zip(keys, self._rotation_keys(ctx)):
                    key[f] = ("rotate", r)
            text = [[(kw, fmt3(v)) for kw, v in key] for key in keys]
            if any(t != text[0] for t in text):
                head = ["    motion {"] + [
                    "        key { %s }" % "  ".join("%s %s" % (kw, fmt3(v)) for kw, v in key)
                    for key in keys] + ["    }"]
        lines = ["group {"] + head
        for block in inner.split("\n"):
            lines.append("    " + block if block else block)
        lines.append("}")
        return "\n".join(lines)

    # Sub-steps per key interval over which a rotation is followed: a periodic signal's fold
    # (a jump of a whole turn) shows inside one sub-step, while a real turn spreads over all of
    # them, so each sub-step's change is folded to within half a turn and summed. A spin is read
    # right up to 180 * _ROT_SUBSTEPS degrees between keys.
    _ROT_SUBSTEPS = 16

    def _rotation_keys(self, ctx: EmitCtx) -> List[Tuple[float, float, float]]:
        """The rotation at each of ``ctx.motion``'s keys, the instant's as evaluated and each
        other followed continuously from its neighbour nearer the instant."""
        offs = ctx.motion.offsets()
        anchor = ctx.motion.instant_key()
        raw = lambda d: vec3(self.rotate, *ctx.offset_clock(d))
        out: List[Optional[List[float]]] = [None] * len(offs)
        out[anchor] = list(raw(offs[anchor]))
        for step in (1, -1):
            k = anchor
            while 0 <= k + step < len(offs):
                val, prev = list(out[k]), raw(offs[k])
                for j in range(1, self._ROT_SUBSTEPS + 1):
                    r = raw(offs[k] + (offs[k + step] - offs[k]) * j / self._ROT_SUBSTEPS)
                    for i in range(3):
                        dlt = r[i] - prev[i]
                        val[i] += dlt - 360.0 * round(dlt / 360.0)
                    prev = r
                out[k + step] = val
                k += step
        return [tuple(v) for v in out]
