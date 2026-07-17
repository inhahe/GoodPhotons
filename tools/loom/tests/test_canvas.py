"""M10 tests: the 2-D backend (:mod:`loom.canvas`).

Cover the user-facing primitive (``plot`` an RGB at an (x, y) at the current
time), both output formats (SVG vector + raster PNG), the y-up coordinate
mapping, DAG integration (roots exposed, cycles caught, per-frame animation),
seamless closed-loop vs distinct open-timeline endpoints, the per-pixel
``field`` (raster-only), strokes, and ``curve_points``.  Runnable directly or
under pytest.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Sine, Cosine, Ramp, vec, PointPath, LoopCurve,
    Canvas2D, curve_points, SignalCycleError,
)
from loom.signals.core import Const  # noqa: E402


def _svg(canvas, frame=0, frames=8, loop=True):
    return canvas.emit_svg(Clock.at_frame(frame, frames, loop=loop), Cache())


def test_plot_adds_marker_and_exposes_roots():
    c = Canvas2D(64, 64)
    c.plot(x=0.2, y=-0.3, color=vec(1.0, 0.5, 0.0), radius=5)
    assert len(c.markers) == 1
    roots = c.markers[0].roots()
    assert len(roots) == 5  # x, y, color, radius, opacity


def test_svg_has_background_and_circle():
    c = Canvas2D(100, 100, background=(0.0, 0.0, 0.0))
    c.plot(0.0, 0.0, vec(1.0, 0.0, 0.0), radius=4)
    svg = _svg(c)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert '<rect width="100" height="100" fill="#000000"' in svg
    assert "<circle" in svg and 'fill="#ff0000"' in svg


def test_coordinate_mapping_is_centred_and_y_up():
    c = Canvas2D(200, 100, view=(-1, -1, 1, 1))
    # origin maps to canvas centre
    px, py = c._to_px(0.0, 0.0)
    assert abs(px - 100.0) < 1e-9 and abs(py - 50.0) < 1e-9
    # +y world is up (smaller pixel-y)
    _, top = c._to_px(0.0, 1.0)
    _, bot = c._to_px(0.0, -1.0)
    assert top < bot


def test_marker_animates_across_frames():
    c = Canvas2D(128, 128)
    c.plot(x=Cosine(cycles=1.0) * 0.8, y=Sine(cycles=1.0) * 0.8,
           color=vec(1, 1, 1))
    a = _svg(c, frame=0, frames=8)
    b = _svg(c, frame=2, frames=8)
    assert a != b, "a Signal-driven marker must move across frames"


def test_closed_loop_marker_is_seamless():
    c = Canvas2D(64, 64)
    c.plot(x=Cosine(cycles=1.0) * 0.7, y=Sine(cycles=1.0) * 0.7, color=vec(1, 0, 0))
    frames = 12
    first = _svg(c, frame=0, frames=frames, loop=True)
    wrap = _svg(c, frame=frames, frames=frames, loop=True)  # t wraps to 0
    assert first == wrap, "closed-loop wrap frame must equal frame 0"


def test_open_timeline_endpoints_differ():
    c = Canvas2D(64, 64)
    c.plot(x=Ramp(-0.7, 0.7), y=0.0, color=vec(1, 1, 1))
    frames = 10
    a = _svg(c, frame=0, frames=frames, loop=False)
    z = _svg(c, frame=frames - 1, frames=frames, loop=False)
    assert a != z, "an open timeline must have distinct endpoints"


def test_rasterize_returns_image_of_right_size():
    c = Canvas2D(48, 32, background=(0.1, 0.2, 0.3))
    c.plot(0.0, 0.0, vec(1, 1, 1), radius=3)
    img = c.rasterize(Clock.at_frame(0, 4))
    assert img.size == (48, 32)
    assert img.mode == "RGB"


def test_field_paints_pixels_and_svg_ignores_it():
    c = Canvas2D(32, 32)

    def fld(X, Y, clock, cache):
        return (0.5 + 0.5 * X, 0.5 + 0.5 * Y, X * 0.0 + clock.t)

    c.field(fld)
    img = c.rasterize(Clock.at_frame(1, 4))
    # left/right edges differ (X varies) -> field really painted
    assert img.getpixel((0, 16)) != img.getpixel((31, 16))
    # SVG has no per-pixel surface: the field is omitted (only the bg rect)
    svg = _svg(c)
    assert "<rect" in svg and "<circle" not in svg


def test_stroke_emits_polyline_and_polygon():
    c = Canvas2D(100, 100)
    c.stroke([(-0.5, -0.5), (0.5, 0.0), (0.0, 0.5)], vec(0, 1, 0), width=3)
    c.stroke([(-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)], vec(0, 0, 1), closed=True)
    svg = _svg(c)
    assert "<polyline" in svg and "<polygon" in svg


def test_curve_points_from_loopcurve():
    path = PointPath([(-0.6, -0.6), (0.6, -0.4), (0.5, 0.5), (-0.5, 0.4)],
                     closed=True)
    curve = LoopCurve(path, 0.0)
    pts = curve_points(curve, 16)
    assert len(pts) == 16
    c = Canvas2D(128, 128)
    c.stroke(pts, vec(1, 1, 0), width=2, closed=True)
    # animate the path -> the stroke must move; static here just checks it emits
    svg = _svg(c)
    assert "<polygon" in svg


def test_cycle_is_detected():
    c = Canvas2D(32, 32)
    # build a self-referential scalar signal
    from loom import Add
    node = Add(Const(1.0), Const(0.0))
    node.a = node  # forge a cycle
    c.plot(x=node, y=0.0, color=vec(1, 1, 1))
    try:
        c.check_cycles()
    except SignalCycleError:
        return
    raise AssertionError("a cyclic signal must be caught by check_cycles")


def test_validation_errors():
    for bad in (
        lambda: Canvas2D(0, 10),
        lambda: Canvas2D(10, 10, view=(0, 0, 0, 1)),
        lambda: Canvas2D(10, 10).plot(0, 0, vec(1, 0)),        # 2-comp colour
        lambda: Canvas2D(10, 10).plot(0, 0, vec(1, 0, 0), shape="star"),
        lambda: Canvas2D(10, 10).stroke([(0, 0)], vec(1, 0, 0)),  # < 2 points
    ):
        try:
            bad()
        except (ValueError, Exception):
            continue
        raise AssertionError("expected a validation error")


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
