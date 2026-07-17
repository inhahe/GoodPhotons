"""
Loom example: a seamless looping 3-D "scribble".

The scribbles3 idea, lifted into 3-D and time:
  - a **closed space curve** (LoopCurve) through a ring of control anchors gives
    the shape — seamless by construction, no seam angle to pick;
  - each anchor takes its **own seamless time journey** (periodic LoopNoise per
    axis), so the whole curve breathes and the animation loops with no cut — a
    "loop of loops."

The curve is drawn as a string of beads (Beads = spheres sampled along it), a
view-independent geometry that needs no sweep engine.

Run:
  python examples/scribble_loop.py            # print frame-0 .ftsl to stdout
  python examples/scribble_loop.py --render    # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Const, LoopNoise, vec, PointPath, LoopCurve,
    Scene, Material, Beads, Light, Camera, Raw,
)


def build_scene(res=(480, 480)) -> Scene:
    n_anchors = 6
    ring_r = 0.42
    wobble = 0.16

    anchors = []
    for i in range(n_anchors):
        ang = 2.0 * math.pi * i / n_anchors
        bx, by, bz = ring_r * math.cos(ang), 0.0, ring_r * math.sin(ang)
        # each anchor wobbles seamlessly over the loop; distinct seeds per axis
        px = bx + LoopNoise(cells=5, seed=100 + i, freq=1, amp=wobble)
        py = by + LoopNoise(cells=5, seed=200 + i, freq=1, amp=wobble * 1.3)
        pz = bz + LoopNoise(cells=5, seed=300 + i, freq=1, amp=wobble)
        anchors.append(vec(px, py, pz))

    path = PointPath(anchors, closed=True)
    curve = LoopCurve(path, Const(0.0))   # u unused by Beads; it samples explicitly

    scene = Scene(Camera(
        eye=(0.0, 0.55, 1.9), look_at=(0.0, 0.0, 0.0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("wire", "diffuse", reflect=0.86),
        Material("floor", "diffuse", reflect=0.6),
        Beads(curve, count=160, radius=0.055, material="wire"),
        # a soft floor and a key light above/side
        Raw('quad { origin -2 -0.55 -2  u 4 0 0  v 0 0 4  material "floor" }'),
        Light("area",
              origin="-0.4 1.5 0.5", u="0.8 0 0", v="0 0 0.8",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


def main() -> int:
    render = "--render" in sys.argv
    scene = build_scene()
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock(t=0.0), Cache()))
        return 0

    from loom import render_range, assemble_gif
    frames = 48
    pngs = render_range(scene, frames, name="scribble_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("scribble_loop") / "scribble_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
