"""
Loom example: a seamless looping swept **ribbon** (and a twin tube).

The M4 sweep engine in action.  A closed space curve (LoopCurve) through a ring
of anchors is the *spine*; each anchor wobbles on its own seamless LoopNoise
journey so the spine breathes and the loop closes with no cut.  A flat strip
(``ribbon``) and a round ``tube`` are swept along that spine with a
rotation-minimizing frame — no roll, no flips, and the closed-spine twist is
distributed so the seam vanishes.  The ribbon also carries one full ``turns``
of twist along its length for a Mobius-ish flourish.

Unlike Beads (view-independent spheres), this emits a real per-frame OBJ mesh
that ftrace loads via ``mesh { file ... }``.

Run:
  python examples/ribbon_loop.py            # print frame-0 .ftsl to stdout
  python examples/ribbon_loop.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Const, LoopNoise, vec, PointPath, LoopCurve,
    Scene, Material, ribbon, tube, Light, Camera, Raw,
)


def build_scene(res=(480, 480)) -> Scene:
    n_anchors = 6
    ring_r = 0.42
    wobble = 0.16

    anchors = []
    for i in range(n_anchors):
        ang = 2.0 * math.pi * i / n_anchors
        bx, by, bz = ring_r * math.cos(ang), 0.0, ring_r * math.sin(ang)
        px = bx + LoopNoise(cells=5, seed=100 + i, freq=1, amp=wobble)
        py = by + LoopNoise(cells=5, seed=200 + i, freq=1, amp=wobble * 1.3)
        pz = bz + LoopNoise(cells=5, seed=300 + i, freq=1, amp=wobble)
        anchors.append(vec(px, py, pz))

    spine = LoopCurve(PointPath(anchors, closed=True), Const(0.0))

    scene = Scene(Camera(
        eye=(0.0, 0.75, 2.0), look_at=(0.0, 0.0, 0.0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("skin", "diffuse", reflect=0.82),
        Material("floor", "diffuse", reflect=0.6),
        # a flat strip with one full twist along the loop
        ribbon(spine, width=0.13, count=140, turns=1.0,
               material="skin", name="ribbon"),
        Raw('quad { origin -2 -0.55 -2  u 4 0 0  v 0 0 4  material "floor" }'),
        Light("area",
              origin="-0.4 1.6 0.6", u="0.9 0 0", v="0 0 0.9",
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
    pngs = render_range(scene, frames, name="ribbon_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("ribbon_loop") / "ribbon_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
