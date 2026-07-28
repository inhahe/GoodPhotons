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
from .mathnd import Mat, Affine
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


def _shift(var: str, p: float) -> str:
    """``var`` shifted by placement ``p``: ``x`` when ``p==0`` (byte-identical to the
    un-placed form), else ``(x-(p))`` with the offset parenthesized."""
    return var if p == 0.0 else f"({var}-({fmt(p)}))"


def _coord_expr(freq: float, row: Sequence[float], drift: float,
                place: Sequence[float] = (0.0, 0.0, 0.0)) -> str:
    """One transformed coordinate: ``freq*(a*(x-px) + b*(y-py) + c*(z-pz)) + drift``.

    ``place`` offsets the coordinate frame so the pattern origin sits at ``place``
    (the J2 placement).  Each numeric coefficient is parenthesized so no ``+-``
    adjacency ever reaches the expression parser (``(1)*x+(-0.5)*y`` is well-formed;
    ``1*x+-0.5*y`` risks tripping the shunting-yard).  With ``place == (0,0,0)`` the
    output is byte-identical to the un-placed form."""
    a, b, c = row
    px, py, pz = place
    lin = (f"({fmt(a)})*{_shift('x', px)}+({fmt(b)})*{_shift('y', py)}"
           f"+({fmt(c)})*{_shift('z', pz)}")
    return f"({fmt(freq)}*({lin})+({fmt(drift)}))"


# --- small numeric 3x3 helpers (parent-frame composition at a baked frame) ---

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]


def _m3_transpose(A: Mat3) -> Mat3:
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))  # type: ignore


def _m3_mul(A: Mat3, B: Mat3) -> Mat3:
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))  # type: ignore


def _m3_apply(A: Mat3, v: Sequence[float]) -> Vec3:
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore


def _v3_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


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

    ``placement`` (animatable :class:`VecSignal`, default origin) positions the
    surface: the coordinate frame is read as ``M·(x − placement)`` **and** the
    ``contained_by`` box/sphere is translated by ``placement``, so the container
    tracks the pattern as the blob drifts/tumbles around a room over the loop.  A
    parent :class:`~loom.mathnd.Affine` frame (set by a :class:`Room`) composes on
    top: the frame's rotation folds into ``M`` and its translation into the world
    placement (the room frame should be **rigid** — rotations + translation, as
    :func:`loom.rotations` produces — since the fold assumes an orthonormal linear
    part).  A box container under a rotating room emits the conservative world
    axis-aligned bounding box of the rotated local box.
    """

    def __init__(self, field: Union[str, FieldFn], *,
                 freq: Union[Signal, Number] = 1.0,
                 threshold: Union[Signal, Number] = 0.0,
                 drift: Union[VecSignal, Sequence] = (0.0, 0.0, 0.0),
                 rotation: Optional[Mat] = None,
                 placement: Union[VecSignal, Sequence] = (0.0, 0.0, 0.0),
                 material: str = "default",
                 container: str = "box",
                 bounds: Tuple[Sequence[float], Sequence[float]] =
                     ((-math.pi, -math.pi, -math.pi), (math.pi, math.pi, math.pi)),
                 center: Sequence[float] = (0.0, 0.0, 0.0),
                 radius: float = math.pi,
                 max_gradient: float = 0.0,
                 method: str = "adaptive",
                 samples: Optional[int] = None,
                 accuracy: Optional[float] = None,
                 refine: Optional[str] = None,
                 uv: Optional[str] = None,
                 open: bool = False,
                 name: str = "iso") -> None:
        self.field: FieldFn = FIELDS[field] if isinstance(field, str) else field
        self.freq = freq
        self.threshold = threshold
        self.drift = drift if isinstance(drift, VecSignal) else VecSignal.of(drift)
        self.rotation = rotation
        self.placement = (placement if isinstance(placement, VecSignal)
                          else VecSignal.of(placement))
        self.material = material
        self.container = container
        self.bounds = (tuple(float(c) for c in bounds[0]),
                       tuple(float(c) for c in bounds[1]))
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.max_gradient = float(max_gradient)
        if method not in ("adaptive", "sample", "fixed"):
            raise ValueError('Isosurface method must be "adaptive", "sample" or "fixed"')
        self.method = method
        # Fixed-step march controls — only meaningful under method="sample"/"fixed",
        # where ftrace's step is box-diagonal/`samples`, else `accuracy` (a world
        # length), else a 256-sample default. Emitting them was previously impossible
        # from loom, so a sampled march was stuck on that default.
        self.samples = None if samples is None else int(samples)
        self.accuracy = None if accuracy is None else float(accuracy)
        if refine is not None and refine not in ("bisect", "regula_falsi", "falsi", "secant"):
            raise ValueError('Isosurface refine must be "bisect" or "regula_falsi"')
        self.refine = refine
        # Procedural UV wrap for pattern/expression materials on this native surface:
        # "planar" / "spherical" / "cylindrical", optionally with " axis=x|y|z".
        # The axis MUST be key=value — a bareword would parse as a stray statement
        # (see FTSL.md §2), which is why it is validated here rather than passed raw.
        if uv is not None:
            parts = str(uv).split()
            if not parts or parts[0] not in ("planar", "spherical", "cylindrical"):
                raise ValueError('Isosurface uv must start with "planar", "spherical" '
                                 f'or "cylindrical" (got {uv!r})')
            for extra in parts[1:]:
                if extra not in ("axis=x", "axis=y", "axis=z"):
                    raise ValueError('Isosurface uv option must be axis=x|y|z '
                                     f'(got {extra!r}); a bareword axis is silently dropped')
        self.uv = uv
        self.open = bool(open)
        self.name = name
        # a transient parent Affine frame, set by an enclosing Room during its emit.
        self._parent: Optional[Affine] = None

    def roots(self):
        out = [self.drift, self.placement]
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
        p = tuple(float(c) for c in self.placement.at(clock, cache))   # local placement
        mn, mx = self.bounds
        cc = self.center
        if self._parent is not None:                  # fold the room frame in
            Plin = _mat3_at(self._parent.linear, clock, cache)
            Poff = tuple(float(c) for c in self._parent.offset.at(clock, cache))
            M = _m3_mul(M, _m3_transpose(Plin))       # M_eff = M · Pᵀ
            # container corners are authored relative to local placement -> world via P
            if self.container == "sphere":
                cc = _v3_add(_m3_apply(Plin, _v3_add(cc, p)), Poff)
            else:
                # conservative world AABB of the rotated, placed local box
                xs, ys, zs = [], [], []  # type: ignore[var-annotated]
                for bx in (mn[0], mx[0]):
                    for by in (mn[1], mx[1]):
                        for bz in (mn[2], mx[2]):
                            wx, wy, wz = _v3_add(
                                _m3_apply(Plin, (bx + p[0], by + p[1], bz + p[2])), Poff)
                            xs.append(wx); ys.append(wy); zs.append(wz)
                mn = (min(xs), min(ys), min(zs))
                mx = (max(xs), max(ys), max(zs))
            p = _v3_add(_m3_apply(Plin, p), Poff)      # p_eff = P·p_local + T
        else:
            # no parent: container authored relative to placement -> translate by p
            mn = _v3_add(mn, p); mx = _v3_add(mx, p)
            cc = _v3_add(cc, p)
        cx = _coord_expr(f, M[0], d[0], p)
        cy = _coord_expr(f, M[1], d[1], p)
        cz = _coord_expr(f, M[2], d[2], p)
        # context-aware templates (PovFn) bake their params; plain FieldFns don't
        field_expr = (self.field.build(cx, cy, cz, ctx)
                      if hasattr(self.field, "build")
                      else self.field(cx, cy, cz))
        expr = f"{field_expr}-({fmt(thr)})"

        lines = [f'{self.name} = isosurface {{']
        lines.append(f'    material "{self.material}"')
        lines.append(f'    function {{ expr "{expr}" }}')
        if self.container == "sphere":
            lines.append(f'    contained_by {{ sphere {{ center {fmt(cc[0])} '
                         f'{fmt(cc[1])} {fmt(cc[2])}  radius {fmt(self.radius)} }} }}')
        else:
            lines.append(f'    contained_by {{ min {fmt(mn[0])} {fmt(mn[1])} {fmt(mn[2])}'
                         f'  max {fmt(mx[0])} {fmt(mx[1])} {fmt(mx[2])} }}')
        if self.max_gradient > 0.0:
            lines.append(f'    max_gradient {fmt(self.max_gradient)}')
        if self.method != "adaptive":
            lines.append(f'    method {self.method}')
        if self.samples is not None:
            lines.append(f'    samples {self.samples}')
        if self.accuracy is not None:
            lines.append(f'    accuracy {fmt(self.accuracy)}')
        if self.refine is not None:
            lines.append(f'    refine {self.refine}')
        if self.uv is not None:
            lines.append(f'    uv {self.uv}')
        if self.open:
            lines.append('    open on')
        lines.append('}')
        return "\n".join(lines)


class Room(Element):
    """A group of placed :class:`Isosurface` (or nested :class:`Room`) children under
    one animatable :class:`~loom.mathnd.Affine` frame.

    A ``Room`` owns a child list and a rigid ``frame`` (rotation + translation, e.g.
    from :func:`loom.affine` / :func:`loom.rotations`).  On :meth:`emit` it hands each
    child the composed frame as its transient ``_parent`` — so a child authored at a
    local ``placement`` lands at ``frame ∘ child_placement`` in the world, its
    coordinate field tilts with the room, and its ``contained_by`` box/sphere tracks
    along (see :class:`Isosurface`).  Child ``name``\\s are namespaced ``room/child`` so
    the emitted ``isosurface`` blocks never collide, and the whole subtree animates:
    ``frame`` is baked per frame, so the room can tumble while its blobs drift inside.

    Rooms nest: an enclosing room folds its frame into this one before passing it down
    (``outer ∘ inner``), and names stack (``outer/inner/gyroidA``).  The frame should
    stay **rigid** (orthonormal linear part) — the child fold assumes ``Pᵀ`` inverts
    ``P``.  For a seamless loop, translate on *closed* curves and rotate by integer
    turns so ``frame`` returns to its start at the wrap.
    """

    def __init__(self, name: str, *children: Element,
                 frame: Optional[Affine] = None) -> None:
        self.name = name
        self.children: list = list(children)
        if frame is not None and frame.dim != 3:
            raise ValueError("Room frame must be a 3-D Affine")
        self.frame = frame if frame is not None else Affine.identity(3)
        # transient parent frame, set by an enclosing Room during its emit.
        self._parent: Optional[Affine] = None

    def add(self, *children: Element) -> "Room":
        self.children.extend(children)
        return self

    def roots(self):
        out: list = []
        # the room frame's animatable parts (matrix entries + translation)
        for r in self.frame.linear.rows:
            out.extend(r)
        out.append(self.frame.offset)
        for c in self.children:
            out.extend(c.roots() if hasattr(c, "roots") else [])
        return out

    def _effective_frame(self) -> Affine:
        """This room's frame with any enclosing room folded in (``parent ∘ frame``)."""
        return self.frame if self._parent is None else self._parent.compose(self.frame)

    def emit(self, ctx: EmitCtx) -> str:
        eff = self._effective_frame()
        blocks = []
        for c in self.children:
            if not hasattr(c, "_parent"):
                # a plain element with no placement hook: emit as-is.
                blocks.append(c.emit(ctx))
                continue
            saved_parent = c._parent
            saved_name = getattr(c, "name", None)
            try:
                c._parent = eff if saved_parent is None else eff.compose(saved_parent)
                if saved_name is not None:
                    c.name = f"{self.name}/{saved_name}"
                blocks.append(c.emit(ctx))
            finally:
                c._parent = saved_parent
                if saved_name is not None:
                    c.name = saved_name
        return "\n\n".join(blocks)


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
