"""Tight *active-band* Lipschitz bounds for the POV isosurface builtins (P3.3 S2, Option B).

ftrace's isosurface sphere-marcher advances along each ray by ``field_value / max_gradient``,
so ``max_gradient`` must be an upper bound on ``|grad f|`` everywhere the ray can travel while
the field is still nonzero — under-estimate it and the march oversteps a zero crossing and
punches holes in the solid; over-estimate it and every step shrinks and the render crawls.

Most POV algebraic builtins are ``f = clamp(P0 * r(x,y,z), -10, +10)`` for some polynomial
``r``.  Far from the surface the *unclamped* polynomial has an enormous gradient (a degree-6
form grows like the 5th power of the radius), but there ``f`` is **railed** — pinned to +/-10,
flat, gradient exactly 0.  A whole-box bound (Option A) is therefore dominated by those railed
tails and is uselessly loose.  The gradient only matters on the **active band** where the field
is un-railed (``|P0 r| < 10``): sphere-tracing's guarantee (a step of ``value/max_gradient``
never overshoots a zero when ``max_gradient >= |grad f|`` along the traversed segment) holds as
long as ``max_gradient`` bounds ``|grad f|`` over that band, because to cross from the rail
value +/-10 down to 0 the field must travel at least ``10 / band_bound`` — exactly the step
size.  So bounding ``|grad f|`` over the band alone is both **rigorous** (never an
under-estimate) and **tight** (ignores the high-gradient railed tails).

We enclose the bound with a **vectorised interval grid**: the box is cut into an ``N^3`` grid of
sub-boxes and natural interval arithmetic is evaluated over the whole grid at once (numpy arrays
of lower/upper endpoints), on the *factored* polynomial (not the expanded one) with exact
even-power handling (``[a,b]**2`` straddling 0 -> ``[0, max(a^2,b^2)]``) so the dependency blow-up
that wrecks a naive expanded enclosure never happens.  Sub-boxes that are provably fully railed
(``P0 r`` cannot reach ``[-10, 10]``) contribute zero and are masked out; the bound is the maximum
interval-sup of ``|grad f|`` over the surviving cells, times a small safety factor — a hard upper
bound on the true band maximum.  A second, finer pass is run only over the neighbourhood of the
surviving band to tighten the estimate cheaply.

Only pure-polynomial ``r`` (and a couple of closed-form fields) are handled here; a function
whose ``r`` involves noise / ``atan2`` / rotation is left to the caller's conservative default.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

# The POV clamp rail (source/vm/fnintern.cpp: fmin(10., fmax(P0*r, -10.))).
_CLAMP = 10.0
# Multiply the certified interval bound by this before returning: cheap insurance against
# floating-point error in the interval-endpoint -> float -> sqrt chain, so we never dip below
# the true maximum.  (The interval enclosure is already outward-rounded; this is belt-and-braces.)
_SAFETY = 1.02

# Closed-form |grad f| ceilings for fields that are exact/near signed-distance or a simple norm,
# where interval B&B would be overkill.  Each maps params -> bound.
_ANALYTIC: Dict[str, Callable[[Sequence[float]], float]] = {
    # -P0 + sqrt(x^2+y^2+z^2): exact SDF, |grad| == 1 for any radius.
    "f_sphere": lambda p: 1.0,
    # -P1 + sqrt((sqrt(x^2+z^2)-P0)^2 + y^2): |grad| == 1 (distance to the ring).
    "f_torus": lambda p: 1.0,
    # sqrt(x^2 P0^2 + y^2 P1^2 + z^2 P2^2): |grad|^2 is a Rayleigh quotient bounded by the
    # largest squared semi-axis, so |grad| <= max(|P0|,|P1|,|P2|).
    "f_ellipsoid": lambda p: max(abs(float(p[0])), abs(float(p[1])), abs(float(p[2]))),
}


# --- clamped-polynomial r builders -----------------------------------------------------------
# Each returns the sympy expression r(x,y,z) exactly as ported from src/pov_functions.h (the
# thing inside ``fmin(10, fmax(P0*r, -10))``); the clamp argument is ``params[0] * r``.  Kept as
# thin transcriptions so they can be eyeballed against the C one-to-one.
_R_BUILDERS: Dict[str, Callable] = {}


def _reg(name: str):
    def deco(fn):
        _R_BUILDERS[name] = fn
        return fn
    return deco


@_reg("f_heart")
def _r_heart(x, y, z, p):
    # r = -((2x^2+y^2+z^2-1)^3 - 0.1 x^2 z^3 - y^2 z^3)
    from sympy import Rational
    a = 2 * x * x + y * y + z * z - 1
    return -(a ** 3 - Rational(1, 10) * x * x * z ** 3 - y * y * z ** 3)


@_reg("f_hunt_surface")
def _r_hunt(x, y, z, p):
    # r = -(4 (x^2+y^2+z^2-13)^3 + 27 (3x^2+y^2-4z^2-12)^2)
    s = x * x + y * y + z * z
    a = s - 13
    b = 3 * x * x + y * y - 4 * z * z - 12
    return -(4 * a ** 3 + 27 * b ** 2)


@_reg("f_kummer_surface_v1")
def _r_kummer1(x, y, z, p):
    # r = -(x^4+y^4+z^4 - x^2-y^2-z^2 - x^2 y^2 - x^2 z^2 - y^2 z^2 + 1)
    x2, y2, z2 = x * x, y * y, z * z
    return -(x2 * x2 + y2 * y2 + z2 * z2 - x2 - y2 - z2
             - x2 * y2 - x2 * z2 - y2 * z2 + 1)


def analyzable(name: str) -> bool:
    """True if :func:`active_band_grad_bound` can produce a bound for ``name`` (a closed-form
    field or a transcribed clamped polynomial); False means the caller should keep its default."""
    return name in _ANALYTIC or name in _R_BUILDERS


_CACHE: Dict[Tuple, float] = {}


class _IV:
    """A vectorised interval: ``lo``/``hi`` are numpy arrays (one interval per grid cell).

    Implements just enough natural-interval arithmetic for the polynomial fields — add, sub,
    negate, multiply, integer power (with exact even-power handling), and scalar div — so a
    sympy-lambdified expression evaluates over an entire grid of sub-boxes at once.  Endpoints
    are plain float64; the caller's safety factor absorbs the (sub-ULP) rounding this ignores.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    @staticmethod
    def _as(o):
        return o if isinstance(o, _IV) else _IV(o, o)

    def __add__(self, o):
        o = _IV._as(o)
        return _IV(self.lo + o.lo, self.hi + o.hi)
    __radd__ = __add__

    def __sub__(self, o):
        o = _IV._as(o)
        return _IV(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        o = _IV._as(o)
        return _IV(o.lo - self.hi, o.hi - self.lo)

    def __neg__(self):
        return _IV(-self.hi, -self.lo)

    def __mul__(self, o):
        import numpy as np
        o = _IV._as(o)
        a, b, c, d = self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi
        return _IV(np.minimum(np.minimum(a, b), np.minimum(c, d)),
                   np.maximum(np.maximum(a, b), np.maximum(c, d)))
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, _IV):
            raise TypeError("interval/interval division not needed here")
        if o >= 0:
            return _IV(self.lo / o, self.hi / o)
        return _IV(self.hi / o, self.lo / o)

    def __pow__(self, n):
        import numpy as np
        if not isinstance(n, int) or n < 0:
            raise TypeError(f"only non-negative integer powers supported, got {n!r}")
        if n == 0:
            return _IV(np.ones_like(self.lo), np.ones_like(self.hi))
        if n == 1:
            return _IV(self.lo, self.hi)
        if n % 2 == 1:                          # odd -> monotonic increasing
            return _IV(self.lo ** n, self.hi ** n)
        # even -> depends on whether the interval straddles 0
        la, ha = np.abs(self.lo), np.abs(self.hi)
        mx = np.maximum(la, ha) ** n
        straddle = (self.lo <= 0.0) & (self.hi >= 0.0)
        mn = np.where(straddle, 0.0, np.minimum(la, ha) ** n)
        return _IV(mn, mx)


def _octasect(box):
    """Split every box in a vectorised set into its 8 octants (2x2x2 midpoint cut)."""
    import numpy as np
    xlo, xhi, ylo, yhi, zlo, zhi = box
    mx, my, mz = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
    XL, XH, YL, YH, ZL, ZH = [], [], [], [], [], []
    for xa, xb in ((xlo, mx), (mx, xhi)):
        for ya, yb in ((ylo, my), (my, yhi)):
            for za, zb in ((zlo, mz), (mz, zhi)):
                XL.append(xa); XH.append(xb)
                YL.append(ya); YH.append(yb)
                ZL.append(za); ZH.append(zb)
    return [np.concatenate(XL), np.concatenate(XH),
            np.concatenate(YL), np.concatenate(YH),
            np.concatenate(ZL), np.concatenate(ZH)]


def active_band_grad_bound(name: str, params: Sequence[float], box_half: float,
                           tol: float = 0.05, seed: int = 16,
                           max_rounds: int = 12, max_boxes: int = 500_000) -> Optional[float]:
    """A rigorous, tight upper bound on ``|grad f|`` over the active band of POV builtin
    ``name`` inside the cube ``[-box_half, box_half]^3``.

    ``params`` are the shape-param values in call order (``params[0]`` is the clamp scale).
    Returns ``None`` when ``name`` has no analyzable form (noise / atan2 / rotation fields), in
    which case the caller falls back to its conservative default.

    Method: a vectorised adaptive branch-and-bound.  Start from a ``seed^3`` grid; each round
    (a) encloses ``|grad f|`` over every sub-box with natural interval arithmetic (rigorous
    upper), (b) samples every sub-box centre that lies in the active band for a *certified lower
    bound* ``L`` on the true maximum, (c) discards every sub-box whose interval-sup is below
    ``L`` — such a box provably cannot contain the maximum — and (d) octasects the survivors.
    Because discarded boxes always sit below an attained value, the running ``max interval-sup``
    stays a hard global upper bound; refinement stops once it is within ``tol`` of ``L``.  The
    result is never an under-estimate (a small safety factor also covers the ignored float
    rounding), and it is tight because only the thin band near the surface is ever refined.
    """
    p = tuple(float(v) for v in params)
    if name in _ANALYTIC:
        return float(_ANALYTIC[name](p))
    if name not in _R_BUILDERS:
        return None

    box_half = float(box_half)
    key = (name, p, round(box_half, 9), round(tol, 6), seed)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    import numpy as np
    import sympy

    x, y, z = sympy.symbols("x y z", real=True)
    r = _R_BUILDERS[name](x, y, z, p)
    p0 = p[0]
    ap0 = abs(p0)
    # |grad f|^2 on the band = P0^2 (r_x^2 + r_y^2 + r_z^2).  Keep the *factored* derivatives
    # (no expand) so the natural interval extension stays tight — an expanded high-degree form
    # suffers catastrophic interval dependency.  lambdify to plain operators evaluable on both
    # our vectorised _IV intervals and on plain numpy float arrays (centre sampling).
    grad2 = sum(sympy.diff(r, v) ** 2 for v in (x, y, z))
    r_fn = sympy.lambdify((x, y, z), r, modules="math")
    g_fn = sympy.lambdify((x, y, z), grad2, modules="math")

    def _railed_mask_and_sup(box):
        xlo, xhi, ylo, yhi, zlo, zhi = box
        IX, IY, IZ = _IV(xlo, xhi), _IV(ylo, yhi), _IV(zlo, zhi)
        rv = r_fn(IX, IY, IZ)
        ca_lo, ca_hi = p0 * rv.lo, p0 * rv.hi
        if p0 < 0:
            ca_lo, ca_hi = ca_hi, ca_lo
        active = (ca_hi >= -_CLAMP) & (ca_lo <= _CLAMP)
        gv = g_fn(IX, IY, IZ)
        sup = ap0 * np.sqrt(np.maximum(gv.hi, 0.0))
        return active, sup

    def _certified_lower(box):
        xlo, xhi, ylo, yhi, zlo, zhi = box
        cx, cy, cz = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
        inband = np.abs(p0 * r_fn(cx, cy, cz)) <= _CLAMP
        if not inband.any():
            return 0.0
        gc = ap0 * np.sqrt(np.maximum(g_fn(cx, cy, cz), 0.0))
        return float(gc[inband].max())

    # seed grid
    e = np.linspace(-box_half, box_half, seed + 1)
    lo, hi = e[:-1], e[1:]
    LX, LY, LZ = np.meshgrid(lo, lo, lo, indexing="ij")
    HX, HY, HZ = np.meshgrid(hi, hi, hi, indexing="ij")
    box = [LX.ravel(), HX.ravel(), LY.ravel(), HY.ravel(), LZ.ravel(), HZ.ravel()]

    bound = None
    for _ in range(max_rounds):
        active, sup = _railed_mask_and_sup(box)
        if not active.any():
            break                               # no un-railed band inside the box
        box = [c[active] for c in box]
        sup = sup[active]
        L = _certified_lower(box)
        upper = float(sup.max())
        keep = sup >= L                         # boxes that could still contain the maximum
        upper_keep = float(sup[keep].max()) if keep.any() else L
        bound = max(L, upper_keep)
        if upper_keep <= L * (1.0 + tol):
            break
        box = [c[keep] for c in box]
        if box[0].size * 8 > max_boxes:
            break
        box = _octasect(box)

    if bound is None:
        return None
    bound = float(bound) * _SAFETY
    _CACHE[key] = bound
    return bound
