"""
Loom example: **time as a value** — freeze / delay / warp, and the 4-D time-shear.

A :class:`loom.Signal` is a *pure function of a clock*, so evaluating one at some
other phase was always well-defined; what was missing was a way to **write the
phase down**.  :class:`loom.Phase` is that value, and :func:`loom.retime` is the
node that samples a sub-graph at it.  Everything else is sugar:

- :func:`loom.freeze` — ``sig(t = at)``, the animation stopped (and ``at`` may
  itself be animated, which is a *scrubbable* hold).
- :func:`loom.delay` — ``sig(t - dt)``, an echo.  On a closed clock the phase
  wraps, so a delayed copy of a seamless loop is still seamless.
- :func:`loom.warp` — ``sig(g(t))``, an arbitrary reparameterization of time.

And the payoff, one tier up, where a *coordinate* is in scope:

- :class:`loom.SigAt` — sample a modulator at a phase that is itself a **field**.
  ``SigAt(sig, T - X / c)`` is a wave whose phase **lags with distance**: the left
  edge of the frame is showing you the signal's present, the right edge its past.
  That is a 4-D shear of the spacetime block, not a 3-D pattern that happens to
  move, and it is not something you can get by animating a coefficient — a plain
  ``Signal`` used as a spatial term bakes *one number per frame*, shared by the
  whole field.

``SigAt`` is deliberately **single-backend**: ftrace evaluates a pattern per hit
and has no access to loom's modulator DAG, so there is no honest ftsl spelling
(baking one number would silently drop the shear, which is the whole effect).
The workflow is discretise-then-render — evaluate it numerically (a 2-D canvas,
:func:`loom.mesh_field`, :func:`loom.vdbio.bake_field`) and render *that*.

Run:
  python examples/time_shear.py            # print the retime family, tabulated
  python examples/time_shear.py --render2d # render the sheared wave as a seamless loop
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, Phase, Clock, Cache,
    freeze, delay, warp, retime,
    SigAt, Canvas2D, X, Y, T, saturate,
)

FRAMES = 24
CYCLES = 3           # cycles of the signal per loop (its temporal frequency)
SPEED = 1.0          # view-widths the wave travels per loop


def sheared_wave(cycles=CYCLES, speed=SPEED):
    """A travelling wave: the same 1-D signal, seen later the further right you go.

    ``T - X/speed`` is the classic retarded time of a wave equation.  Note that the
    *spatial* frequency is not authored — it falls out as ``cycles/speed``, exactly
    as it does for a real wave (``k = w/c``).  Both come out whole here, so the
    pattern tiles across the view **and** the temporal loop still closes.
    """
    return SigAt(Sine(cycles=cycles), T - X / speed)


def build_canvas(size=512) -> Canvas2D:
    # a ring-shaped emitter would need a radial phase; keep it planar and honest:
    # brightness is the wave, tinted by height so the shear direction is obvious.
    w = sheared_wave()
    band = saturate(0.5 + 0.5 * w)
    tint = saturate(0.5 + 0.5 * Y)
    c = Canvas2D(size, size, view=(0.0, -1.0, 1.0, 1.0),
                 background=(0.02, 0.02, 0.05))
    c.field((band * (0.25 + 0.55 * tint), band * 0.45, 0.20 + band * 0.75))
    return c


def _tabulate() -> None:
    s = Sine()                                  # sin(2*pi*t), one cycle per loop
    rows = {
        "sig(t)          ": s,
        "freeze(sig, 0)  ": freeze(s, 0.0),
        "freeze(sig, .25)": freeze(s, 0.25),
        "delay(sig, .25) ": delay(s, 0.25),
        "delay(sig, -.25)": delay(s, -0.25),     # negative dt looks *ahead*
        "warp(sig, 2t)   ": warp(s, lambda t: 2.0 * t),
        "retime(sig, t^2)": retime(s, Phase() * Phase()),
    }
    hdr = "  ".join(f"f{f:02d}" for f in range(0, FRAMES, 3))
    print(f"{'':18}{hdr}")
    for name, sig in rows.items():
        cells = []
        for f in range(0, FRAMES, 3):
            cells.append(f"{sig.at(Clock.at_frame(f, FRAMES), Cache()):+.2f}")
        print(f"{name}  " + " ".join(cells))
    print()
    print("  freeze  is flat; delay is the same wave rotated (and still seamless:")
    print("          f00 and f24 agree, because the phase wraps on a closed clock);")
    print("          warp(2t) runs two cycles per loop, so the loop still closes.")


def _shear_profile() -> None:
    try:
        import numpy as np
    except ImportError:
        print("\n(numpy not installed — skipping the spatial shear profile)")
        return
    w = sheared_wave()
    flat = Sine(cycles=3) * (X * 0.0 + 1.0)     # the SAME signal, baked per frame
    xs = np.linspace(0.0, 1.0, 9)
    coords = (xs, np.zeros_like(xs), np.zeros_like(xs))
    print("\n# the shear, sampled across x at a few frames")
    print(f"{'':10}" + "  ".join(f"x={x:.2f}" for x in xs))
    for f in (0, 6, 12):
        c = Clock.at_frame(f, FRAMES)
        row = np.asarray(w.eval_np(coords, c, Cache()))
        print(f"  f{f:02d}    " + "  ".join(f"{v:+.2f}" for v in row))
    c = Clock.at_frame(0, FRAMES)
    row = np.asarray(flat.eval_np(coords, c, Cache()))
    print("  flat   " + "  ".join(f"{v:+.2f}" for v in row)
          + "   <- a plain Signal term: one number, no shear")


def main() -> int:
    if "--render2d" in sys.argv:
        from loom import render_canvas
        render_canvas(build_canvas(), frames=FRAMES * 2, name="time_shear",
                      fps=24, fmt="both")
        return 0
    _tabulate()
    _shear_profile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
