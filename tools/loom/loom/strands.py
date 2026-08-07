"""
Control points for ftrace's native ``curve`` primitive (FTSL §8.6).

A loom spine (a :class:`~loom.interp.LoopCurve` over a
:class:`~loom.data.PointPath`) is *sampled* into points at emit time — loom's
"discretize LAST, per frame" rule.  ftrace's ``curve`` takes **control** points
under a chosen basis, so the question this module answers is: given the sampled
polyline, what control points and which basis reproduce it?

**Open spine → ``catmull_rom``.**  It interpolates every control point, so the
samples *are* the control points and the emitted strand passes exactly through
them.  Its end tangents come from clamp-duplicated phantoms (``curve.h``'s
``at(i) = pts[clamp(i)]``), which is the right answer at a free end.

**Closed spine → ``bspline``, wrapped by 3.**  This is the part that has to be
structural rather than patched (loom principle #4: *seamless loops are
structural*).  Catmull-Rom cannot close: its clamped ends mean the seam span's
tangent is computed from a duplicated phantom instead of the point that really
precedes it, so the loop is C1-broken at exactly one place — and no amount of
wrapping fixes it without also drawing some span twice.  A **uniform cubic
B-spline** has no phantoms at all: ``curve.h`` gives span ``sp`` the points
``pts[sp .. sp+3]`` and yields ``n-3`` spans.  So emitting

    Q = [C_0 … C_{N-1}, C_0, C_1, C_2]        (N+3 points → exactly N spans)

is the periodic B-spline over ``C``: every span is drawn once, no index is ever
clamped, and the seam is C2 like every other join.  There is nothing special
about the seam because the seam does not exist.

The price is that a B-spline *approximates* its control points, so ``C`` cannot
just be the samples — the strand would shrink inside the polygon.  Instead solve
for the ``C`` whose B-spline passes through the samples.  A uniform cubic
B-spline span starts at ``(Q[sp] + 4·Q[sp+1] + Q[sp+2]) / 6``, so requiring span
``sp`` to start at sample ``S[sp]`` gives the cyclic tridiagonal system

    C[i-1] + 4·C[i] + C[i+1] = 6·S[i-1]        (indices mod N)

solved here by Sherman–Morrison.  Its circulant eigenvalues are
``4 + 2cos(2πk/N) ∈ [2, 6]``, so it is well conditioned at any N.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, ...]


# ---------------------------------------------------------------------------
# Linear algebra: cyclic tridiagonal solve
# ---------------------------------------------------------------------------

def _tridiag(a: Sequence[float], b: Sequence[float], c: Sequence[float],
             r: Sequence[float]) -> List[float]:
    """Thomas algorithm for a plain tridiagonal system (``a`` sub-, ``b`` diagonal,
    ``c`` super-diagonal).  ``a[0]`` and ``c[n-1]`` are unused."""
    n = len(b)
    if n < 1:
        return []
    bet = b[0]
    if bet == 0.0:
        raise ZeroDivisionError("singular tridiagonal system")
    x = [0.0] * n
    gam = [0.0] * n
    x[0] = r[0] / bet
    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        if bet == 0.0:
            raise ZeroDivisionError("singular tridiagonal system")
        x[j] = (r[j] - a[j] * x[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        x[j] -= gam[j + 1] * x[j + 1]
    return x


def _cyclic(a: Sequence[float], b: Sequence[float], c: Sequence[float],
            alpha: float, beta: float, r: Sequence[float]) -> List[float]:
    """Solve a tridiagonal system that also has the two *corner* entries
    ``A[n-1][0] = alpha`` and ``A[0][n-1] = beta`` — i.e. a periodic band — by the
    Sherman–Morrison rank-1 correction of the non-cyclic solve.  Needs ``n >= 3``."""
    n = len(b)
    if n < 3:
        raise ValueError("cyclic solve needs at least 3 unknowns")
    gamma = -b[0]                      # any nonzero value; this one keeps bb well scaled
    bb = list(b)
    bb[0] = b[0] - gamma
    bb[n - 1] = b[n - 1] - alpha * beta / gamma
    x = _tridiag(a, bb, c, r)
    u = [0.0] * n
    u[0] = gamma
    u[n - 1] = alpha
    z = _tridiag(a, bb, c, u)
    fact = ((x[0] + beta * x[n - 1] / gamma)
            / (1.0 + z[0] + beta * z[n - 1] / gamma))
    return [x[j] - fact * z[j] for j in range(n)]


def closed_bspline_controls(samples: Sequence[Point]) -> List[Point]:
    """Control points whose **closed** uniform cubic B-spline passes exactly through
    ``samples`` (sample ``i`` at the start of span ``i``).

    Solves ``C[i-1] + 4·C[i] + C[i+1] = 6·S[i-1]`` (mod N) per component.  Needs at
    least 3 samples.
    """
    n = len(samples)
    if n < 3:
        raise ValueError(f"a closed strand needs at least 3 spine samples, got {n}")
    dim = len(samples[0])
    a = [1.0] * n
    b = [4.0] * n
    c = [1.0] * n
    cols: List[List[float]] = []
    for d in range(dim):
        rhs = [6.0 * samples[(i - 1) % n][d] for i in range(n)]
        cols.append(_cyclic(a, b, c, 1.0, 1.0, rhs))
    return [tuple(cols[d][i] for d in range(dim)) for i in range(n)]


# ---------------------------------------------------------------------------
# Samples -> (basis, control points, per-point strand parameter)
# ---------------------------------------------------------------------------

def strand_controls(samples: Sequence[Point], closed: bool
                    ) -> Tuple[str, List[Point], List[float]]:
    """Turn a sampled spine into what a ``curve { }`` block needs.

    Returns ``(basis, points, radius_u)``:

    ``basis``
        ``"bspline"`` when ``closed``, else ``"catmull_rom"``.
    ``points``
        the control points to emit, in order.
    ``radius_u``
        for each emitted point, the strand parameter ``u ∈ [0, 1]`` whose radius
        belongs on it.  ftrace attaches radii to control points and interpolates
        them **linearly along the span**, and each basis pins them differently
        (``curve.h``'s ``spanPoints``: Catmull-Rom span ``sp`` uses ``rat(sp)`` and
        ``rat(sp+1)``; B-spline span ``sp`` uses ``rat(sp+1)`` and ``rat(sp+2)``),
        so this mapping is what keeps a radius profile registered to the geometry
        instead of sliding by one point.
    """
    n = len(samples)
    if not closed:
        if n < 2:
            raise ValueError(f"an open strand needs at least 2 spine samples, got {n}")
        du = 1.0 / (n - 1)
        return "catmull_rom", list(samples), [k * du for k in range(n)]
    ctl = closed_bspline_controls(samples)
    pts = ctl + ctl[:3]                      # N+3 points -> exactly N spans
    # Span sp starts at sample sp (u = sp/N) and its start radius is rat(sp+1),
    # so control point j carries the radius of sample j-1.
    rad_u = [((j - 1) % n) / n for j in range(n + 3)]
    return "bspline", pts, rad_u


def sample_spine(spine, count: int, closed: bool, clock, cache) -> List[Point]:
    """Sample a loom spine into ``count`` points, at the parameters a strand wants.

    A **closed** spine is sampled at ``k/count`` for ``k in [0, count)`` — the
    half-open sweep that tiles the loop exactly once, so wrapping the control
    points closes it with no duplicated sample.  An **open** spine is sampled at
    ``k/(count-1)``, endpoint *inclusive*, so the strand actually reaches its tip
    (:func:`loom.interp.eval_curve` clamps rather than wraps on an open path).
    """
    if closed:
        return [spine.sample(k / count, clock, cache) for k in range(count)]
    du = 1.0 / (count - 1)
    return [spine.sample(k * du, clock, cache) for k in range(count)]
