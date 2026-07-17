"""
Loom example: a **one-shot, non-looping** animation (M6.5).

Looping is a *choice* in loom, not a baked invariant.  This scene deliberately
does *not* loop: a sphere **rises** (its centre `y` is a linear :class:`Ramp`)
and **grows** (its radius is a smoothstep :class:`Ease`) once, from a distinct
start to a distinct end.  It is rendered on an **open** clock
(``render_range(..., loop=False)`` → ``t = k/(frames-1)``, endpoints inclusive),
so frame 0 and the final frame are genuinely different — there is no seam to
close, by design.

Contrast with ``material_loop.py`` / ``gyroid_loop.py``, which compose *periodic*
leaves on the default **closed** clock and therefore loop seamlessly.

Run:
  python examples/open_timeline.py            # print frame-0 .ftsl to stdout
  python examples/open_timeline.py --render   # render the one-shot (loop=False)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    vec, Ramp, Ease,
    Scene, Material, Camera, Sphere, Raw, Light,
)


def build_scene(res=(480, 480)) -> Scene:
    # non-periodic drivers: distinct at t=0 and t=1 (one-shot, no loop)
    rise = vec(0.0, Ramp(-0.7, 0.7), 0.0)          # centre climbs linearly
    grow = Ease(0.45, 1.05, mode="in-out")         # radius eases open→shut

    scene = Scene(Camera(
        eye=(0.0, 0.4, 4.8), look_at=(0, 0.2, 0), up=(0, 1, 0),
        fov_y=34, mode="R", res=res))
    scene.add(
        Material("ball", "diffuse", reflect="rgb 0.85 0.35 0.30"),
        Material("wall", "diffuse", reflect="whitewall 0.8"),
        Sphere(rise, grow, "ball"),
        # open-fronted room so the sphere is lit
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
        # open mapping: frame 0 of 48 is the start (t=0), NOT a loop wrap
        print(scene.emit(Clock.at_frame(0, 48, loop=False), Cache()))
        return 0

    from loom import render_range, assemble_gif
    frames = 48
    pngs = render_range(scene, frames, name="open_timeline", fps=24,
                        noise=4.0, interval=4.0, loop=False)  # <-- one-shot
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("open_timeline") / "open_timeline.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
