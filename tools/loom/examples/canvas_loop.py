"""
Loom example: a seamless-looping 2-D **motion graphic** (M10).

Everything here is the *same* modulation DAG that drives the 3-D ftrace scenes —
a 2-D animation is just Signals sampled at the current clock (DESIGN.md §12 M10),
rendered by loom's own :class:`loom.Canvas2D` (no ftrace).  Three layers, each
seamless because it is built from periodic leaves / whole-turn drifts:

- a full-canvas **field**: radial sine bands rotating a whole ``2*pi`` turn;
- a ring of orbiting **plot** markers (the user's core primitive — an RGB at an
  (x, y) at the current time), hue-cycling and radius-pulsing with 1-periodic
  :class:`loom.Sine`;
- a closed **stroke** scribble (a :class:`loom.LoopCurve` sampled to a polygon).

Run:
  python examples/canvas_loop.py          # print frame-0 SVG to stdout
  python examples/canvas_loop.py --render # render PNG+SVG frames + a seamless GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, Cosine, vec, PointPath, LoopCurve,
    Canvas2D, curve_points,
)


def rotating_bands(freq=6.0, turns=1.0):
    """A vectorized field: radial sine bands that rotate a whole turn over the loop."""
    import numpy as np

    def fn(X, Y, clock, cache):
        a = turns * 2.0 * math.pi * clock.t          # whole-turn phase -> seamless
        u = X * math.cos(a) + Y * math.sin(a)
        band = 0.5 + 0.5 * np.sin(freq * u + a)
        r = np.sqrt(X * X + Y * Y)
        glow = np.clip(1.0 - r, 0.0, 1.0)
        return (0.10 + 0.30 * band * glow,
                0.04 + 0.10 * band,
                0.18 + 0.55 * band * glow)
    return fn


def build_canvas(size=512) -> Canvas2D:
    c = Canvas2D(size, size, view=(-1.2, -1.2, 1.2, 1.2),
                 background=(0.02, 0.02, 0.05))
    c.field(rotating_bands(freq=7.0))

    # a closed scribble drawn as a polygon stroke
    path = PointPath([(-0.75, -0.55), (0.7, -0.7), (0.85, 0.5),
                      (0.0, 0.85), (-0.8, 0.55)], closed=True)
    c.stroke(curve_points(LoopCurve(path, 0.0), 40),
             vec(0.9, 0.9, 1.0), width=2, opacity=0.5, closed=True)

    # a ring of orbiting, hue-cycling, radius-pulsing markers
    ndots = 10
    for k in range(ndots):
        ph = k / ndots
        ang = Sine(cycles=1.0, phase=ph)             # 1-periodic -> seamless
        cang = Cosine(cycles=1.0, phase=ph)
        rad = 0.78
        c.plot(x=cang * rad, y=ang * rad,
               color=vec(0.5 + 0.5 * Sine(cycles=1.0, phase=ph),
                         0.5 + 0.5 * Sine(cycles=1.0, phase=ph + 0.33),
                         0.5 + 0.5 * Sine(cycles=1.0, phase=ph + 0.66)),
               radius=6.0 + 4.0 * (0.5 + 0.5 * math.cos(ph * math.tau)))
    return c


def main() -> int:
    render = "--render" in sys.argv
    c = build_canvas()
    if not render:
        from loom import Clock, Cache
        print(c.emit_svg(Clock.at_frame(0, 60), Cache()))
        return 0
    from loom import render_canvas
    render_canvas(c, frames=60, name="canvas_loop", fps=30, fmt="both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
