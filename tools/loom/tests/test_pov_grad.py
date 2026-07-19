"""S2 tests: the active-band gradient bounder (loom.pov_grad, Option B).

The headline discipline mirrors the M9 POV table tests — a *rigor* guard and a *tightness*
guard.  For each transcribed algebraic builtin we compare the bounder's ceiling against a dense
numerical sample of ``|grad f|`` over the same box, restricted to the un-railed active band:
the bound must never fall below the sampled maximum (rigor), and must not exceed it by much
(tightness).  The rest check the closed-form fields, the ``None`` fallback for un-analyzable
builtins, the ``analyzable`` predicate, and caching.  Runnable directly or under pytest.

Skips cleanly if numpy or sympy is unavailable (the bounder's optional deps).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import sympy  # noqa: F401
    _HAVE_DEPS = True
except Exception:  # noqa: BLE001
    _HAVE_DEPS = False

from loom.pov_grad import (active_band_grad_bound, analyzable,  # noqa: E402
                           bbox_analyzable, surface_bbox)


# --- exact-C reference polynomials (for the numerical rigor/tightness cross-check) -----------

def _heart(x, y, z):
    a = 2 * x * x + y * y + z * z - 1
    return -(a ** 3 - 0.1 * x * x * z ** 3 - y * y * z ** 3)


def _hunt(x, y, z):
    s = x * x + y * y + z * z
    a = s - 13
    b = 3 * x * x + y * y - 4 * z * z - 12
    return -(4 * a ** 3 + 27 * b ** 2)


def _kummer1(x, y, z):
    x2, y2, z2 = x * x, y * y, z * z
    return -(x2 * x2 + y2 * y2 + z2 * z2 - x2 - y2 - z2
             - x2 * y2 - x2 * z2 - y2 * z2 + 1)


_REFS = {
    "f_heart": (_heart, (1.0,)),
    "f_hunt_surface": (_hunt, (1.0,)),
    "f_kummer_surface_v1": (_kummer1, (1.0,)),
}

_BOX = 1.365          # a representative render box: radius 1.3 * 1.05


def _numeric_band_max(f, p0, box, n=131):
    g = np.linspace(-box, box, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    F = f(X, Y, Z)
    gx, gy, gz = np.gradient(F, g, g, g)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz) * abs(p0)
    band = np.abs(p0 * F) <= 10.0
    return float(gm[band].max())


def test_bounder_is_rigorous_never_under_estimates():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    for name, (f, params) in _REFS.items():
        b = active_band_grad_bound(name, params, _BOX)
        nm = _numeric_band_max(f, params[0], _BOX)
        assert b is not None
        assert b >= nm, f"{name}: bound {b} < sampled max {nm} (UNDER-ESTIMATE)"


def test_bounder_is_tight():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    for name, (f, params) in _REFS.items():
        b = active_band_grad_bound(name, params, _BOX)
        nm = _numeric_band_max(f, params[0], _BOX)
        # a coarse numerical sample slightly *under*-reads the true max, so allow generous slack;
        # the bounder's own tol is 5% + a 2% safety factor over the true sup.
        assert b <= nm * 1.35, f"{name}: bound {b} loose vs sampled max {nm} (ratio {b / nm:.2f})"


def test_analytic_sphere_and_torus_are_unit():
    assert active_band_grad_bound("f_sphere", (1.0,), _BOX) == 1.0
    assert active_band_grad_bound("f_torus", (0.8, 0.25), _BOX) == 1.0


def test_analytic_ellipsoid_is_max_semixaxis():
    # |grad| ceiling for sqrt(x^2 a^2 + y^2 b^2 + z^2 c^2) is max(|a|,|b|,|c|)
    assert active_band_grad_bound("f_ellipsoid", (1.0, 1.0, 1.0), _BOX) == 1.0
    assert active_band_grad_bound("f_ellipsoid", (1.0, 2.0, 0.5), _BOX) == 2.0
    assert active_band_grad_bound("f_ellipsoid", (3.0, 0.5, 0.5), _BOX) == 3.0


def test_ellipsoid_analytic_bound_dominates_numeric():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    a, b, c = 1.0, 2.0, 0.5
    f = lambda x, y, z: np.sqrt(x * x * a * a + y * y * b * b + z * z * c * c)  # noqa: E731
    # ellipsoid is never railed; its "band" is the whole box, so sample everywhere > 0.
    g = np.linspace(-_BOX, _BOX, 121)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    F = f(X, Y, Z)
    gx, gy, gz = np.gradient(F, g, g, g)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz)
    nm = float(gm[F > 1e-6].max())
    # analytic ceiling is exactly max(a,b,c)=2; np.gradient's boundary stencil can overshoot the
    # true max by a few ULP, so compare with a tiny slack (not a real under-estimate).
    assert active_band_grad_bound("f_ellipsoid", (a, b, c), _BOX) >= nm - 1e-6


def test_unanalyzable_returns_none():
    # noise / atan2 / rotation / un-transcribed fields aren't handled -> None (caller defaults)
    for name in ("f_klein_bottle", "f_noise3d", "f_helical_torus"):
        assert active_band_grad_bound(name, (1.0,), _BOX) is None


def test_analyzable_predicate():
    assert analyzable("f_sphere")
    assert analyzable("f_heart")
    assert analyzable("f_ellipsoid")
    assert not analyzable("f_klein_bottle")
    assert not analyzable("f_noise3d")


def test_cache_returns_identical_value():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    a = active_band_grad_bound("f_kummer_surface_v1", (1.0,), _BOX)
    b = active_band_grad_bound("f_kummer_surface_v1", (1.0,), _BOX)
    assert a == b


def test_bigger_box_gives_bigger_bound_for_high_degree():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    # a degree-6 field's band gradient grows with the domain it is sliced over.
    small = active_band_grad_bound("f_hunt_surface", (1.0,), 1.0)
    big = active_band_grad_bound("f_hunt_surface", (1.0,), 2.0)
    assert big > small


# --- S3 container bbox sizing ----------------------------------------------------------------

def test_bbox_analyzable_predicate():
    assert bbox_analyzable("f_sphere")
    assert bbox_analyzable("f_torus")
    assert bbox_analyzable("f_ellipsoid")
    assert bbox_analyzable("f_heart")
    assert bbox_analyzable("f_hunt_surface")
    assert not bbox_analyzable("f_klein_bottle")
    assert not bbox_analyzable("f_noise3d")


def test_bbox_sphere_is_its_radius():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    ext, bounded = surface_bbox("f_sphere", (1.0,))
    assert bounded
    assert abs(ext - 1.0) < 0.05                      # grid granularity
    ext2, _ = surface_bbox("f_sphere", (2.5,))
    assert abs(ext2 - 2.5) < 0.1


def test_bbox_ellipsoid_reaches_long_semiaxis():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    # semi-axes are 1/P: params (1,2,0.5) -> longest is 1/0.5 = 2 along z, at level 1
    ext, bounded = surface_bbox("f_ellipsoid", (1.0, 2.0, 0.5), level=1.0)
    assert bounded
    assert 1.9 < ext < 2.2


def test_bbox_hunt_surface_is_large():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    # the motivating case: hunt's surface sits far out (~3.67), well beyond the old 1.3 default
    ext, bounded = surface_bbox("f_hunt_surface", (1.0,))
    assert bounded
    assert 3.0 < ext < 4.5


def test_bbox_never_clips_the_surface():
    # a container of the returned half-extent must actually contain the surface: sample the field
    # just OUTSIDE the box and confirm it is all on one side (no crossing beyond the box).
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    for name, params, f in (("f_sphere", (1.0,), _sphere_field),
                            ("f_heart", (1.0,), _heart)):
        ext, bounded = surface_bbox(name, params)
        assert bounded
        shell = ext * 1.15
        g = np.linspace(-shell, shell, 41)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        outside = np.maximum(np.maximum(np.abs(X), np.abs(Y)), np.abs(Z)) > ext * 1.02
        F = f(X, Y, Z)
        s = F[outside] < 0
        assert s.all() or (~s).all(), f"{name}: surface crosses beyond the derived box"


def _sphere_field(x, y, z):
    return -1.0 + np.sqrt(x * x + y * y + z * z)


def test_bbox_unbounded_surface_is_flagged():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    # kummer_v1's zero set has sheets that run to the search boundary -> not auto-sizable
    res = surface_bbox("f_kummer_surface_v1", (1.0,))
    assert res is not None
    _ext, bounded = res
    assert not bounded


def test_bbox_unanalyzable_returns_none():
    assert surface_bbox("f_klein_bottle", (1.0,)) is None
    assert surface_bbox("f_noise3d", (1.0,)) is None


def test_bbox_cache_returns_identical():
    if not _HAVE_DEPS:
        print("  (skip: numpy/sympy unavailable)")
        return
    a = surface_bbox("f_torus", (0.8, 0.25))
    b = surface_bbox("f_torus", (0.8, 0.25))
    assert a == b


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
