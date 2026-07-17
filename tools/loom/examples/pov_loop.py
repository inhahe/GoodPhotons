"""
Loom example: a seamless looping **POV-Ray function** isosurface (M9).

ftrace imports ~78 of POV-Ray's isosurface builtins (``f_torus``, ``f_heart``,
``f_sphere``, …); loom wraps them as :func:`loom.pov` templates whose shape
**params are Signals baked per frame**.  Here an ``f_torus`` (major/minor radius)
spins a full ``2*pi`` about its vertical axis while its **minor radius pulses**
with a periodic :class:`loom.Sine` — both motions return bit-for-bit at the wrap,
so the loop is seamless.

Honesty (DESIGN.md §11.7): ``f_torus`` is *not* periodic, so a phase *drift* would
NOT close the loop; seamless motion for a POV surface means a coordinate transform
that returns to itself — a whole-turn rotation — plus periodic params.

Run:
  python examples/pov_loop.py            # print frame-0 .ftsl to stdout
  python examples/pov_loop.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, rotations, phase_drift,
    Scene, Material, Camera, Raw, Light, Isosurface, pov,
)


def build_scene(res=(480, 480)) -> Scene:
    # a full 2*pi turn about the vertical (xz-plane) over the loop -> seamless
    spin = rotations(3, [(0, 2, phase_drift(1.0))])
    # minor radius pulses periodically (Sine is 1-periodic in t) -> seamless
    minor = Sine(cycles=1.0, amp=0.07, bias=0.27)

    scene = Scene(Camera(
        eye=(0.0, 1.4, 4.4), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=34, mode="R", res=res))
    scene.add(
        Material("skin", "diffuse", reflect="rgb 0.85 0.55 0.20"),
        Material("wall", "diffuse", reflect="whitewall 0.8"),
        Isosurface(pov("f_torus", 0.72, minor), rotation=spin,
                   container="sphere", radius=1.35, max_gradient=3.0,
                   name="torus", material="skin"),
        # open-fronted room so the surface is lit
        Raw('quad { origin -2.4 -1.7 -2.4  u 4.8 0 0  v 0 0 4.8  material "wall" }'),
        Raw('quad { origin -2.4  1.7 -2.4  u 4.8 0 0  v 0 0 4.8  material "wall" }'),
        Raw('quad { origin -2.4 -1.7 -2.4  u 4.8 0 0  v 0 3.4 0  material "wall" }'),
        Raw('quad { origin -2.4 -1.7 -2.4  u 0 0 4.8  v 0 3.4 0  material "wall" }'),
        Raw('quad { origin  2.4 -1.7 -2.4  u 0 0 4.8  v 0 3.4 0  material "wall" }'),
        Light("area",
              origin="-1.0 1.68 -1.0", u="2.0 0 0", v="0 0 2.0",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


def main() -> int:
    render = "--render" in sys.argv
    scene = build_scene()
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, 48), Cache()))
        return 0

    from loom import render_range, assemble_gif
    frames = 48
    pngs = render_range(scene, frames, name="pov_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("pov_loop") / "pov_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
