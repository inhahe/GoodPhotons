"""
Loom N-D linear algebra — animatable matrices, Givens rotations, and the slicer.

Everything here is *animatable*: matrix entries are Signals, so a rotation angle
(or a whole matrix) can be modulated over time.  Applying a :class:`Mat` to a
:class:`~loom.signals.vector.VecSignal` builds more graph — it does not evaluate.

The headline operator is :func:`slice3`, the general "rotate + take a 3-D slice"
map ``P = O + a*u + b*v + c*w``.  Feeding the first ``k`` components of ``P`` to a
``k``-input field is the honest behavior we derived:

- a **3-input** field (e.g. a 3-arg gyroid) rotated in N-D and sliced back to 3-D
  only ever sees an **affine** remap of ``(x, y, z)`` (tilt / shear / scale /
  drift) — no genuinely new structure appears;
- a genuinely **k>=4-input** field morphs, because the slice offset feeds a real
  extra input.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Union

from .signals.core import Signal, Const, Sin, Cos, as_signal, Number
from .signals.vector import VecSignal, Vecish, vec

Angle = Union[Signal, Number]


class Mat:
    """A matrix whose entries are Signals (so it can animate)."""

    def __init__(self, rows: Sequence[Sequence[Union[Signal, Number]]]) -> None:
        self.rows: List[List[Signal]] = [[as_signal(x) for x in r] for r in rows]
        self.nrows = len(self.rows)
        if self.nrows == 0:
            raise ValueError("Mat needs at least one row")
        self.ncols = len(self.rows[0])
        if any(len(r) != self.ncols for r in self.rows):
            raise ValueError("Mat rows must be equal length")

    @classmethod
    def identity(cls, n: int) -> "Mat":
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    def transpose(self) -> "Mat":
        return Mat([[self.rows[i][j] for i in range(self.nrows)]
                    for j in range(self.ncols)])

    def matmul(self, other: "Mat") -> "Mat":
        if self.ncols != other.nrows:
            raise ValueError(f"shape mismatch {self.nrows}x{self.ncols} @ "
                             f"{other.nrows}x{other.ncols}")
        out: List[List[Signal]] = []
        for i in range(self.nrows):
            row: List[Signal] = []
            for j in range(other.ncols):
                acc: Signal = self.rows[i][0] * other.rows[0][j]
                for k in range(1, self.ncols):
                    acc = acc + self.rows[i][k] * other.rows[k][j]
                row.append(acc)
            out.append(row)
        return Mat(out)

    def __matmul__(self, other: Union["Mat", VecSignal, Vecish]):
        if isinstance(other, Mat):
            return self.matmul(other)
        return self.apply(other)

    def apply(self, v: Vecish) -> VecSignal:
        """Return ``self @ v`` as a VecSignal."""
        vv = VecSignal.of(v)
        if vv.dim != self.ncols:
            raise ValueError(f"cannot apply {self.nrows}x{self.ncols} to dim {vv.dim}")
        comps: List[Signal] = []
        for i in range(self.nrows):
            acc: Signal = self.rows[i][0] * vv.components[0]
            for k in range(1, self.ncols):
                acc = acc + self.rows[i][k] * vv.components[k]
            comps.append(acc)
        return VecSignal(comps)


def rotation(n: int, i: int, j: int, angle: Angle) -> Mat:
    """An ``n x n`` Givens rotation by ``angle`` in the ``(i, j)`` coordinate plane.

    Compose several (via ``@``) to "rotate on any number of axes."  ``angle`` may
    be an animatable Signal.
    """
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError("need 0 <= i, j < n and i != j")
    a = as_signal(angle)
    c = Cos(a)
    s = Sin(a)
    rows: List[List[Union[Signal, Number]]] = [
        [1.0 if r == cc else 0.0 for cc in range(n)] for r in range(n)
    ]
    rows[i][i] = c
    rows[j][j] = c
    rows[i][j] = -s
    rows[j][i] = s
    return Mat(rows)


def rotations(n: int, planes: Sequence[tuple]) -> Mat:
    """Compose several Givens rotations.  ``planes`` = ``[(i, j, angle), ...]``,
    applied left-to-right (the first listed is applied first to a vector)."""
    m = Mat.identity(n)
    for (i, j, ang) in planes:
        m = rotation(n, i, j, ang).matmul(m)
    return m


class Affine:
    """An N-D affine map ``x -> linear @ x + offset`` (rotation/shear + translation).

    Both parts are animatable (matrix entries and offset components are Signals).
    The point of :class:`Affine` is **composition**: an arbitrarily long chain of
    N-D Givens rotations and translations folds into a single ``(linear, offset)``
    baked once per frame, so the emitted expression carries one affine instead of
    a sequential stack (fewer ops in ftrace's per-hit eval).

    Order convention (matches :func:`rotations`): ``a.compose(b)`` is ``a ∘ b`` —
    *apply ``b`` first*, then ``a``.  Algebra:
    ``a(b(x)) = a.linear @ (b.linear @ x + b.offset) + a.offset``
    ``       = (a.linear @ b.linear) @ x + (a.linear @ b.offset + a.offset)``.
    """

    def __init__(self, linear: Mat, offset: Vecish) -> None:
        self.linear = linear
        self.offset = VecSignal.of(offset)
        if linear.nrows != linear.ncols:
            raise ValueError("Affine linear part must be square")
        if self.offset.dim != linear.nrows:
            raise ValueError(f"offset dim {self.offset.dim} != linear size "
                             f"{linear.nrows}")

    @property
    def dim(self) -> int:
        return self.linear.nrows

    @classmethod
    def identity(cls, n: int) -> "Affine":
        return cls(Mat.identity(n), [0.0] * n)

    @classmethod
    def translation(cls, offset: Vecish) -> "Affine":
        off = VecSignal.of(offset)
        return cls(Mat.identity(off.dim), off)

    @classmethod
    def rotation(cls, n: int, i: int, j: int, angle: Angle) -> "Affine":
        return cls(rotation(n, i, j, angle), [0.0] * n)

    @classmethod
    def linear_map(cls, m: Mat) -> "Affine":
        return cls(m, [0.0] * m.nrows)

    def compose(self, other: "Affine") -> "Affine":
        """Return ``self ∘ other`` (apply ``other`` first, then ``self``)."""
        if self.dim != other.dim:
            raise ValueError(f"cannot compose dim {self.dim} with {other.dim}")
        linear = self.linear.matmul(other.linear)
        offset = self.linear.apply(other.offset) + self.offset
        return Affine(linear, offset)

    def apply(self, v: Vecish) -> VecSignal:
        """Return ``linear @ v + offset`` as a VecSignal."""
        return self.linear.apply(v) + self.offset

    def __matmul__(self, other: Union["Affine", VecSignal, Vecish]):
        if isinstance(other, Affine):
            return self.compose(other)
        return self.apply(other)


def affine(n: int, ops: Sequence[tuple]) -> Affine:
    """Fold a chain of N-D rotations/translations into one :class:`Affine`.

    ``ops`` is a list applied **left-to-right** (the first listed is applied first
    to a vector), each one of:

    - ``("rot", i, j, angle)`` — a Givens rotation in the ``(i, j)`` plane;
    - ``("move", offset)`` — a translation by an N-D ``offset`` (numbers/Signals);
    - a raw :class:`Mat` (used as a linear map) or a ready :class:`Affine`.

    Angles/offsets may be Signals, so the whole composite animates and is baked to
    a single ``(linear, offset)`` per frame.
    """
    acc = Affine.identity(n)
    for op in ops:
        if isinstance(op, Affine):
            step = op
        elif isinstance(op, Mat):
            step = Affine.linear_map(op)
        elif isinstance(op, tuple) and op and op[0] == "rot":
            _, i, j, ang = op
            step = Affine.rotation(n, i, j, ang)
        elif isinstance(op, tuple) and op and op[0] == "move":
            step = Affine.translation(op[1])
        else:
            raise ValueError(f"unrecognized affine op: {op!r}")
        if step.dim != n:
            raise ValueError(f"affine op dim {step.dim} != {n}")
        acc = step.compose(acc)  # first-listed applied first
    return acc


def slice3(O: Vecish, u: Vecish, v: Vecish, w: Vecish) -> Callable[..., VecSignal]:
    """Build the slicer ``(a, b, c) -> O + a*u + b*v + c*w`` (all N-D VecSignals).

    Returns a function mapping three scene coordinates (numbers or Signals) to
    the N-D query point.  Feed the first ``k`` components of that point to a
    ``k``-input field.
    """
    Ov = VecSignal.of(O)
    uv = VecSignal.of(u)
    vv = VecSignal.of(v)
    wv = VecSignal.of(w)
    dim = Ov.dim
    if not (uv.dim == vv.dim == wv.dim == dim):
        raise ValueError("O, u, v, w must share a dimension")

    def query(a: Union[Signal, Number],
              b: Union[Signal, Number],
              c: Union[Signal, Number]) -> VecSignal:
        return Ov + uv * as_signal(a) + vv * as_signal(b) + wv * as_signal(c)

    query.dim = dim  # type: ignore[attr-defined]
    return query
