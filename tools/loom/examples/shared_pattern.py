"""
Loom example: **one spatial definition, two backends** (M10.5).

A single :class:`loom.SpatialExpr` — a phase-drifting Schoen gyroid — is defined
once and then used *both* ways:

- **3-D**: as an :class:`loom.Isosurface` field, where loom **emits it as an ftsl
  string** in x/y/z (with the ``T`` drift baked per frame) and ftrace root-finds it;
- **2-D**: as a :class:`loom.Canvas2D` ``field``, where loom **evaluates the same
  expression numerically** over the pixel grid (a z=0 slice) into a raster.

The drift is ``T * 2*pi`` over one closed loop, so both the surface flow and the
2-D slice return bit-for-bit at the wrap — seamless (``sin``/``cos`` are
``2*pi``-periodic).  This is the payoff of keeping coordinates in a shared spatial
algebra instead of the time-only DAG: the pattern isn't written twice.

Run:
  python examples/shared_pattern.py           # print the shared expr's ftsl + 2D SVG
  python examples/shared_pattern.py --render2d # render the seamless 2-D loop (+GIF)
  python examples/shared_pattern.py --render3d # render the 3-D isosurface loop (ftrace)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    X, Y, Z, T, sin, cos, saturate,
    Scene, Material, Camera, Raw, Light, Isosurface,
    Canvas2D,
)

TWO_PI = 2.0 * math.pi


def drifting_gyroid(freq=1.0, turns=1.0):
    """A Schoen gyroid whose phase drifts a whole ``turns*2*pi`` over the loop."""
    d = T * (TWO_PI * turns)
    fx, fy, fz = freq * X + d, freq * Y + d, freq * Z + d
    return sin(fx) * cos(fy) + sin(fy) * cos(fz) + sin(fz) * cos(fx)


def build_scene(res=(480, 480)) -> Scene:
    field = drifting_gyroid(freq=1.0, turns=1.0)
    scene = Scene(Camera(
        eye=(0.0, 0.0, 7.2), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("skin", "diffuse", reflect="rgb 0.30 0.55 0.95"),
        Material("wall", "diffuse", reflect="whitewall 0.8"),
        Isosurface(field, threshold=0.0, container="sphere", radius=math.pi,
                   max_gradient=4.0, name="gyroid", material="skin"),
        Raw('quad { origin -6 -6 -5  u 12 0 0  v 0 12 0  material "wall" }'),
        Light("area", origin="-2.5 3.0 3.0", u="2.5 0 0", v="0 2.5 0",
              normal="0.4 -0.7 -0.6", spd="preset:bb6500"),
    )
    return scene


def build_canvas(size=512) -> Canvas2D:
    # the SAME field, sliced at z=0 and mapped to a blue band (evaluated numerically)
    g = drifting_gyroid(freq=1.0, turns=1.0)
    band = saturate(0.5 + 0.5 * g)
    c = Canvas2D(size, size, view=(-math.pi, -math.pi, math.pi, math.pi),
                 background=(0.02, 0.02, 0.05))
    c.field((band * 0.30, band * 0.62, 0.25 + band * 0.70))
    return c


def main() -> int:
    if "--render2d" in sys.argv:
        from loom import render_canvas
        render_canvas(build_canvas(), frames=60, name="shared_2d", fps=30, fmt="both")
        return 0
    if "--render3d" in sys.argv:
        from loom import render_range, assemble_gif
        from loom.drive import default_outdir
        pngs = render_range(build_scene(), 48, name="shared_3d", fps=24,
                            noise=4.0, interval=4.0)
        assemble_gif(pngs, default_outdir("shared_3d") / "shared_3d.gif", fps=24)
        return 0

    from loom import Clock, Cache
    from loom.ftsl_emit import EmitCtx
    field = drifting_gyroid()
    ctx = EmitCtx(clock=Clock.at_frame(6, 48), cache=Cache())
    print("# shared field, emitted as ftsl (3-D isosurface), frame 6/48:")
    print("  " + field.emit(("x", "y", "z"), ctx))
    print("\n# same field, 2-D SVG has no per-pixel surface, so here's the raster")
    print("# path instead — run with --render2d / --render3d to see both loops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
