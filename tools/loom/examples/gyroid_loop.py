"""
Loom example: a seamless looping **gyroid** isosurface (M5).

A Schoen gyroid ``sin x cos y + sin y cos z + sin z cos x = threshold`` is a
3-input field, so animating it honestly means animating the *frame it is read in*
(the Layer-2 slicer result):

  - a **phase drift** advancing by one full ``2*pi`` over the loop makes the
    surface flow steadily along a diagonal and return bit-for-bit at the wrap
    (``sin``/``cos`` are ``2*pi``-periodic) — seamless by construction;
  - a **tilt** (an animated Givens rotation of the (x,y,z) frame) slowly turns the
    lattice; over one full loop it returns to its start;
  - the **threshold** breathes with a seamless ``Sine`` so the walls thicken and
    thin (the surface's genus pulses).

ftrace renders the isosurface natively (sphere-traced ``function`` field clipped to
a sphere container), so no meshing is involved — loom just emits the formula.

Run:
  python examples/gyroid_loop.py            # print frame-0 .ftsl to stdout
  python examples/gyroid_loop.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, vec, rotations,
    Scene, Material, Camera, gyroid_surface, phase_drift, Light, Raw,
)


def build_scene(res=(480, 480)) -> Scene:
    # one full 2*pi diagonal drift over the loop -> seamless flow (one cell of travel)
    drift = vec(phase_drift(1.0), phase_drift(1.0), phase_drift(1.0))
    # a gentle full-turn tilt in the (x,z) plane, seamless (returns to start)
    tilt = rotations(3, [(0, 2, Sine(cycles=1, amp=math.pi))])
    # walls breathe: threshold sweeps +-0.5 around 0
    thr = Sine(cycles=1, amp=0.5, bias=0.0)

    # A gyroid BALL: freq packs a few cells into a radius-R sphere. Enclosed in a
    # bright open-fronted room so the concave lattice is actually lit (a single
    # light in open void self-shadows to black).
    R = 1.3
    scene = Scene(Camera(
        eye=(0.0, 0.7, 5.2), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=34, mode="R", res=res))
    scene.add(
        Material("shell", "diffuse", reflect=0.85),
        Material("wall", "diffuse", reflect=0.78),
        gyroid_surface(freq=6.0, threshold=thr, drift=drift, rotation=tilt,
                       container="sphere", center=(0, 0, 0), radius=R,
                       material="shell", name="gyroid"),
        # open-fronted room centered on the ball (front at +z left open for the camera)
        Raw('quad { origin -2 -1.7 -2  u 4 0 0  v 0 0 4  material "wall" }'),   # floor
        Raw('quad { origin -2  1.7 -2  u 4 0 0  v 0 0 4  material "wall" }'),   # ceiling
        Raw('quad { origin -2 -1.7 -2  u 4 0 0  v 0 3.4 0  material "wall" }'), # back
        Raw('quad { origin -2 -1.7 -2  u 0 0 4  v 0 3.4 0  material "wall" }'), # left
        Raw('quad { origin  2 -1.7 -2  u 0 0 4  v 0 3.4 0  material "wall" }'), # right
        Light("area",
              origin="-0.9 1.68 -0.9", u="1.8 0 0", v="0 0 1.8",
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
    pngs = render_range(scene, frames, name="gyroid_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("gyroid_loop") / "gyroid_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
