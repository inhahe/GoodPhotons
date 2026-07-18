"""True-N-D forms for the ``POV_ND_GENERALIZABLE`` isosurface builtins (P3.4).

Most POV builtins (``f_torus``, ``f_heart``, …) are intrinsically 3-D: slicing the
surface into an ``--dims>3`` isosurface can only *reorient* an affine remap of
``(x, y, z)`` (P3.3 S6 — see ``gyroid_nd._pov_affine``).  A small subset, listed in
:data:`loom.pov.POV_ND_GENERALIZABLE`, is instead built from **symmetric sums /
polynomials** and so has an honest N-D generalization: an ``N``-coordinate field
``F(xi_0, …, xi_{N-1})`` that at ``N = 3`` reduces *exactly* to the ``f_*`` call and
for ``N > 3`` folds in the extra slice axes.

This module supplies, for each of the nine functions,

* :func:`nd_field_expr` — the FTSL string ``F(xi_0, …, xi_{N-1})`` built from a list
  of coordinate expressions (one per N-D dim; each is a linear form in ``x/y/z``);
* :func:`nd_field_eval` — the same field as a Python number (for the "matches the
  ``f_*`` builtin at ``N = 3``" cross-check and numeric gradient tests);
* :func:`nd_grad_bound_xi` — a rigorous, conservative upper bound on ``|grad_xi F|``
  over ``|xi_i| <= Xi``, so the caller can make the sphere-marcher hole-safe (the
  emitted field is ``F(A.p)`` with ``A`` the N×3 slice Jacobian, whose gradient is
  ``A^T grad_xi F``; the caller scales this bound by ``sigma_max(A)``).

Coordinate conventions mirror the C bodies in ``src/pov_functions.h``:

* fully symmetric — ``f_sphere``, ``f_ellipsoid`` (per-axis weights; extra dims weight
  1), ``f_ovals_of_cassini``, ``f_isect_ellipsoids``, ``f_cross_ellipsoids``;
* one distinguished "height/axis" dim (index 1, the ``y`` axis) with the rest radial —
  ``f_paraboloid``, ``f_quartic_paraboloid``, ``f_poly4``;
* one distinguished "polar" dim (index 2, the ``z`` axis) with the rest equatorial —
  ``f_superellipsoid``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .ftsl_emit import fmt
from .pov import POV_ND_GENERALIZABLE

# The POV clamp rail (fmin(10., fmax(P0*r, -10.))) shared by the algebraic builtins.
_CLAMP = 10.0


# ---------------------------------------------------------------------------
# small FTSL builders
# ---------------------------------------------------------------------------
def _sq(c: str) -> str:
    return f"({c})*({c})"


def _p4(c: str) -> str:
    s = _sq(c)
    return f"({s})*({s})"


def _sum(parts: Sequence[str]) -> str:
    return "+".join(parts) if parts else "0"


def _nested(op: str, parts: Sequence[str]) -> str:
    """Fold a list into nested binary ``min(a, min(b, …))`` / ``max(...)`` calls."""
    parts = list(parts)
    acc = parts[-1]
    for p in reversed(parts[:-1]):
        acc = f"{op}({p},{acc})"
    return acc


def _clamp(expr: str) -> str:
    return f"clamp({expr},{fmt(-_CLAMP)},{fmt(_CLAMP)})"


# ---------------------------------------------------------------------------
# N-D field emission — FTSL string
# ---------------------------------------------------------------------------
def nd_field_expr(name: str, coords: List[str], values: Sequence[float]) -> str:
    """The FTSL expression for the N-D generalization of ``name`` over ``coords``.

    ``coords`` is the list of ``D`` coordinate expressions ``xi_0 … xi_{D-1}`` (each a
    linear form in ``x/y/z``); ``values`` the shape params (``arity - 3``).  At ``D = 3``
    with ``coords == ["x", "y", "z"]`` this is numerically identical to
    ``f_name(x, y, z, *values)``.
    """
    if name not in POV_ND_GENERALIZABLE:
        raise ValueError(f"{name!r} has no true-N-D form (not in POV_ND_GENERALIZABLE)")
    P = [float(v) for v in values]
    D = len(coords)

    if name == "f_sphere":
        r = _sum([_sq(c) for c in coords])
        return f"(({fmt(-P[0])})+sqrt({r}))"

    if name == "f_ellipsoid":
        w = [P[0], P[1], P[2]] + [1.0] * (D - 3)
        terms = [f"({fmt(w[i] * w[i])})*{_sq(coords[i])}" for i in range(D)]
        return f"sqrt({_sum(terms)})"

    if name == "f_paraboloid":
        rad = _sum([_sq(coords[i]) for i in range(D) if i != 1])
        inner = f"({coords[1]})-({rad})"
        return _clamp(f"({fmt(P[0])})*({inner})")

    if name == "f_quartic_paraboloid":
        rad = _sum([_p4(coords[i]) for i in range(D) if i != 1])
        inner = f"({coords[1]})-({rad})"
        return _clamp(f"({fmt(P[0])})*({inner})")

    if name == "f_ovals_of_cassini":
        R = _sum([_sq(c) for c in coords])
        off = _sum([_sq(coords[i]) for i in range(D) if i != 1])
        r2 = f"(({R})+({fmt(P[1] * P[1])}))"
        r = (f"(-(({r2})*({r2})-({fmt(P[3] * P[1] * P[1])})*({off})"
             f"-({fmt(P[2] * P[2])})))")
        return _clamp(f"({fmt(P[0])})*({r})")

    if name in ("f_isect_ellipsoids", "f_cross_ellipsoids"):
        R = _sum([_sq(c) for c in coords])
        terms = []
        for k in range(D):
            qk = f"({fmt(P[0])})*({R})+({fmt(1.0 - P[0])})*{_sq(coords[k])}"
            terms.append(f"exp(({fmt(-P[1])})*({qk}))")
        op = "min" if name == "f_isect_ellipsoids" else "max"
        r = _nested(op, terms)
        return f"(({fmt(P[3])})-({r})*({fmt(P[2])}))"

    if name == "f_poly4":
        c1 = coords[1]
        c1s = _sq(c1)
        poly = (f"({fmt(P[0])})+({fmt(P[1])})*({c1})+({fmt(P[2])})*({c1s})"
                f"+({fmt(P[3])})*({c1s})*({c1})+({fmt(P[4])})*({c1s})*({c1s})")
        temp = f"max(({poly}),{fmt(-5.0)})"
        rad = _sum([_sq(coords[i]) for i in range(D) if i != 1])
        return f"((-({temp}))+sqrt({rad}))"

    if name == "f_superellipsoid":
        p = 2.0 / P[0]
        n = 1.0 / P[1]
        equ = [f"pow(abs({coords[i]}),{fmt(p)})" for i in range(D) if i != 2]
        G = _sum(equ)
        H = f"pow(abs({coords[2]}),{fmt(2.0 * n)})"
        inner = f"pow(({G}),{fmt(P[0] * n)})+({H})"
        return f"(1-pow(({inner}),{fmt(P[1] * 0.5)}))"

    raise ValueError(f"no N-D form implemented for {name!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# N-D field — numeric evaluation (tests / cross-checks)
# ---------------------------------------------------------------------------
def nd_field_eval(name: str, xi: Sequence[float], values: Sequence[float]) -> float:
    """The N-D field of :func:`nd_field_expr` as a Python number at coordinates ``xi``."""
    if name not in POV_ND_GENERALIZABLE:
        raise ValueError(f"{name!r} has no true-N-D form")
    P = [float(v) for v in values]
    xi = [float(c) for c in xi]
    D = len(xi)

    if name == "f_sphere":
        return -P[0] + math.sqrt(sum(c * c for c in xi))
    if name == "f_ellipsoid":
        w = [P[0], P[1], P[2]] + [1.0] * (D - 3)
        return math.sqrt(sum(w[i] * w[i] * xi[i] * xi[i] for i in range(D)))
    if name == "f_paraboloid":
        rad = sum(xi[i] * xi[i] for i in range(D) if i != 1)
        return min(_CLAMP, max(P[0] * (xi[1] - rad), -_CLAMP))
    if name == "f_quartic_paraboloid":
        rad = sum((xi[i] * xi[i]) ** 2 for i in range(D) if i != 1)
        return min(_CLAMP, max(P[0] * (xi[1] - rad), -_CLAMP))
    if name == "f_ovals_of_cassini":
        R = sum(c * c for c in xi)
        off = sum(xi[i] * xi[i] for i in range(D) if i != 1)
        r2 = R + P[1] * P[1]
        r = -(r2 * r2 - P[3] * P[1] * P[1] * off - P[2] * P[2])
        return min(_CLAMP, max(P[0] * r, -_CLAMP))
    if name in ("f_isect_ellipsoids", "f_cross_ellipsoids"):
        R = sum(c * c for c in xi)
        terms = [math.exp(-P[1] * (P[0] * R + (1.0 - P[0]) * xi[k] * xi[k]))
                 for k in range(D)]
        r = min(terms) if name == "f_isect_ellipsoids" else max(terms)
        return P[3] - r * P[2]
    if name == "f_poly4":
        y = xi[1]
        temp = max(P[0] + P[1] * y + P[2] * y * y + P[3] * y ** 3 + P[4] * y ** 4, -5.0)
        rad = math.sqrt(sum(xi[i] * xi[i] for i in range(D) if i != 1))
        return -temp + rad
    if name == "f_superellipsoid":
        p = 2.0 / P[0]
        n = 1.0 / P[1]
        G = sum(abs(xi[i]) ** p for i in range(D) if i != 2)
        H = abs(xi[2]) ** (2.0 * n)
        return 1.0 - (G ** (P[0] * n) + H) ** (P[1] * 0.5)
    raise ValueError(f"no N-D eval for {name!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# rigorous conservative bound on |grad_xi F| over |xi_i| <= Xi
# ---------------------------------------------------------------------------
def nd_grad_bound_xi(name: str, values: Sequence[float], xi_max: float,
                     dims: int) -> Optional[float]:
    """A safe upper bound on ``|grad_xi F|`` for the N-D field over ``|xi_i| <= xi_max``.

    Conservative (an over-estimate only shrinks the marcher step, never punches holes).
    Returns ``None`` for ``f_superellipsoid`` (non-Lipschitz corners when the roundness
    exponents are ``< 1``), leaving the caller to fall back to its own default.
    """
    P = [float(v) for v in values]
    X = float(xi_max)
    D = int(dims)

    if name == "f_sphere":
        return 1.0                                   # exact unit-gradient SDF
    if name == "f_ellipsoid":
        return max(abs(P[0]), abs(P[1]), abs(P[2]), 1.0)
    if name == "f_paraboloid":
        return abs(P[0]) * math.sqrt(1.0 + 4.0 * (D - 1) * X * X)
    if name == "f_quartic_paraboloid":
        return abs(P[0]) * math.sqrt(1.0 + 16.0 * (D - 1) * (X ** 6))
    if name == "f_ovals_of_cassini":
        r2max = D * X * X + P[1] * P[1]
        per = (4.0 * r2max + 2.0 * abs(P[3]) * P[1] * P[1]) * X
        return abs(P[0]) * math.sqrt(D) * per
    if name in ("f_isect_ellipsoids", "f_cross_ellipsoids"):
        # exp(-P1*q_k) with q_k = P0*sum(xi_{!=k}^2) + xi_k^2 >= 0 for P0 >= 0, so the
        # exp factor is <= 1 when P0, P1 >= 0; otherwise bound it explicitly.
        if P[0] >= 0.0 and P[1] >= 0.0:
            emax = 1.0
        else:
            qspan = (abs(P[0]) * D + abs(1.0 - P[0])) * X * X
            emax = math.exp(abs(P[1]) * qspan)
        dq = 2.0 * X * max(abs(P[0]), 1.0)           # |d q_k / d xi_i| bound
        return abs(P[2]) * abs(P[1]) * math.sqrt(D) * dq * emax
    if name == "f_poly4":
        tp = abs(P[1]) + 2.0 * abs(P[2]) * X + 3.0 * abs(P[3]) * X * X \
            + 4.0 * abs(P[4]) * (X ** 3)
        return math.sqrt(tp * tp + 1.0)
    if name == "f_superellipsoid":
        return None                                  # non-Lipschitz corners; caller defaults
    return None
