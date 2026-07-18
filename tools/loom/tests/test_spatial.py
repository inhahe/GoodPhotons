"""M10.5 tests: the shared spatial-expression tree (:mod:`loom.spatial`).

One definition, two backends: emitted as an ftsl string (3-D isosurface /
pattern) *and* evaluated numerically over numpy grids (2-D raster field).  Cover
deterministic emission, numeric semantics, time-dependence introspection
(``uses_time`` / ``time_signals``), the shared path driving both
:class:`FuncPattern` (string) and :class:`Canvas2D.field` (pixels), static-field
baking, and :class:`Isosurface` integration.  Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, Sine,
    SpatialExpr, sexpr, X, Y, Z, T, SPATIAL_PATTERNS,
    sin, cos, sqrt, sign, clamp,
    FuncPattern, Isosurface, Canvas2D,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(expr, coords=("x", "y", "z"), clock=None):
    ctx = EmitCtx(clock=clock or Clock(t=0.0), cache=Cache())
    return expr.emit(coords, ctx)


def test_emit_is_deterministic_and_parenthesized():
    expr = X * Y + 2.0
    assert _emit(expr) == "(((x)*(y))+(2))"


def test_emit_bakes_time_and_signal_coefficients():
    # T bakes to the clock phase; a Signal coefficient bakes to its value
    e = 0.5 + 0.5 * sin(sexpr(6.0) * X + T)
    txt = _emit(e, clock=Clock(t=0.25))
    assert "sin(" in txt and "(0.25)" in txt  # T baked in
    s = Sine(cycles=1.0) * X
    txt2 = _emit(s, clock=Clock.at_frame(3, 12))
    # Sine at t=0.25 is sin(2*pi*0.25)=1 -> coefficient baked to ~1
    assert "(1)" in txt2 or "1)" in txt2


def test_eval_np_matches_math():
    e = sin(X) * cos(Y) + sqrt(Z)
    xs = np.array([0.3, 1.1])
    ys = np.array([0.7, -0.2])
    zs = np.array([0.25, 4.0])
    got = e.eval_np((xs, ys, zs), Clock(t=0.0), Cache())
    exp = np.sin(xs) * np.cos(ys) + np.sqrt(zs)
    assert np.allclose(got, exp)


def test_sign_and_clamp_semantics_match_ftrace():
    e = sign(X)
    got = e.eval_np((np.array([-2.0, 0.0, 3.0]), 0.0, 0.0), Clock(t=0.0), Cache())
    assert list(got) == [-1.0, 0.0, 1.0]              # (x>0)-(x<0)
    c = clamp(X, -0.5, 0.5)
    got = c.eval_np((np.array([-9.0, 0.1, 9.0]), 0.0, 0.0), Clock(t=0.0), Cache())
    assert list(got) == [-0.5, 0.1, 0.5]


def test_uses_time_and_time_signals():
    assert not sin(X).uses_time()
    assert (X + T).uses_time()
    s = Sine(cycles=1.0)
    e = s * sin(X)
    assert e.uses_time()
    ts = e.time_signals()
    assert len(ts) == 1 and ts[0] is s
    assert sin(X).time_signals() == []


def test_same_expr_drives_pattern_string_and_2d_raster():
    expr = SPATIAL_PATTERNS["waves"](6.0)
    # 3-D: emitted as an ftsl pattern string
    fp = FuncPattern("shared", expr)
    txt = fp.emit(EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    assert txt.startswith('pattern "shared"') and "sin(" in txt
    # the spatial exprs's (no) time signals fold into the pattern's DAG roots
    assert fp.roots() is not None
    # 2-D: evaluated numerically into pixels
    c = Canvas2D(16, 16, view=(-1, -1, 1, 1))
    c.field(expr)
    img = c.rasterize(Clock.at_frame(0, 4))
    assert img.getpixel((0, 8)) != img.getpixel((15, 8))  # varies along X


def test_rgb_triple_field():
    c = Canvas2D(12, 12)
    c.field((0.5 + 0.5 * sin(4 * X), 0.5 + 0.5 * sin(4 * Y), sexpr(0.2)))
    img = c.rasterize(Clock.at_frame(0, 4))
    r, g, b = img.getpixel((6, 6))
    assert b == int(0.2 * 255 + 0.5)


def test_static_field_is_baked_once():
    c = Canvas2D(16, 16)
    c.field(sin(4 * X))                 # no time -> auto-static
    assert c._field[3] is True
    c.rasterize(Clock.at_frame(0, 4))
    assert c._static_raster is not None
    # identical output on a later frame (baked buffer reused)
    a = c.rasterize(Clock.at_frame(0, 4)).tobytes()
    b = c.rasterize(Clock.at_frame(3, 4)).tobytes()
    assert a == b


def test_time_varying_field_is_not_cached_and_animates():
    two_pi = 2.0 * math.pi
    c = Canvas2D(16, 16)
    c.field(0.5 + 0.5 * sin(4 * X + T * two_pi))
    assert c._field[3] is False
    a = c.rasterize(Clock.at_frame(0, 8)).tobytes()
    b = c.rasterize(Clock.at_frame(3, 8)).tobytes()
    assert c._static_raster is None
    assert a != b


def test_isosurface_accepts_spatial_expr():
    surf = SPATIAL_PATTERNS["gyroid"](1.0)
    iso = Isosurface(surf, container="sphere", radius=2.0, name="g",
                     material="skin")
    txt = iso.emit(EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    assert txt.startswith('isosurface "g" {')
    assert "sin(" in txt and "cos(" in txt
    assert txt.count("{") == txt.count("}")


def test_animated_isosurface_expr_exposes_roots():
    # a phase-drifting gyroid: the temporal Signal must reach the iso's roots
    s = Sine(cycles=1.0, amp=0.3, bias=1.0)
    surf = SPATIAL_PATTERNS["gyroid"](s)   # animated frequency
    iso = Isosurface(surf, name="g2")
    assert any(r is s for r in iso.roots())
    a = iso.emit(EmitCtx(clock=Clock.at_frame(0, 12), cache=Cache()))
    b = iso.emit(EmitCtx(clock=Clock.at_frame(3, 12), cache=Cache()))
    assert a != b


def test_coercion_errors():
    try:
        sexpr("nope")
    except TypeError:
        pass
    else:
        raise AssertionError("a string must not coerce into a spatial expr")


def test_rings_center_shifts_source_and_default_is_unchanged():
    from loom.spatial import rings
    # default-centred rings emits the plain x/y/z distance (byte-identical to old form)
    assert "sqrt(((((x)*(x))+((y)*(y)))+((z)*(z))))" in _emit(rings(2.0))
    # a shifted centre subtracts the offset -> the pattern moves
    shifted = rings(2.0, center=(0.5, 0.0, 0.0))
    assert "(x)-(0.5)" in _emit(shifted)
    # numerically: the crest that sat at the origin is displaced
    xs = np.array([0.0, 0.5])
    at = lambda e, x: e.eval_np((np.array([x]), 0.0, 0.0), Clock(t=0.0), Cache())[0]
    assert abs(at(rings(6.0), 0.0) - at(rings(6.0, center=(0.5, 0, 0)), 0.5)) < 1e-9


def test_interference_superposes_two_sources_in_range():
    from loom.spatial import interference
    e = interference(3.0)
    # two radial sources -> two sqrt distance terms in the emitted field
    assert _emit(e).count("sqrt") == 2
    # stays in [0, 1] across the plane
    xs, ys = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
    v = e.eval_np((xs, ys, np.zeros_like(xs)), Clock(t=0.0), Cache())
    assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9
    # on the perpendicular bisector (x=0) the two path lengths are equal -> full crest
    mid = e.eval_np((np.array([0.0]), np.array([0.0]), np.array([0.0])),
                    Clock(t=0.0), Cache())[0]
    assert mid > 0.99


def test_interference_animated_source_loops():
    from loom.spatial import interference
    # a moving emitter (Signal x-coordinate) makes the field time-varying but seamless
    sx = Sine(cycles=1.0, amp=0.4)
    e = interference(6.0, source_a=(sx, 0.0, 0.0))
    assert e.uses_time()
    a = _emit(e, clock=Clock.at_frame(0, 12))
    b = _emit(e, clock=Clock.at_frame(3, 12))
    assert a != b                                        # animates
    # frame N wraps back to frame 0 (closed loop)
    assert _emit(e, clock=Clock.at_frame(0, 12)) == _emit(e, clock=Clock.at_frame(12, 12))


def test_moire_two_gratings_and_animated_angle():
    from loom.spatial import moire
    e = moire(6.0, angle=0.2)
    assert _emit(e).count("sin(((6)*") == 2              # two overlaid freq-6 rulings
    xs, ys = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
    v = e.eval_np((xs, ys, np.zeros_like(xs)), Clock(t=0.0), Cache())
    assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9
    # an animated rotation angle makes the fringes crawl
    m = moire(6.0, angle=Sine(cycles=1.0, amp=0.3))
    assert m.uses_time()
    assert _emit(m, clock=Clock(t=0.0)) != _emit(m, clock=Clock(t=0.3))


def test_new_spatial_presets_registered():
    for name in ("interference", "moire"):
        assert name in SPATIAL_PATTERNS
        expr = SPATIAL_PATTERNS[name](4.0)
        assert isinstance(expr, SpatialExpr)


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
