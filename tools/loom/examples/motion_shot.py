"""
Loom example: a short **motion-blurred shot** for ``ftrace -shot`` / ``python -m loom.shot``.

A gold ball rolls across a room while a red block spins and a blue one rises and turns, over an
open 48-frame timeline. The camera's 180-degree shutter (``SHUTTER``) makes each frame a
motion-blurred one: loom writes every moving transform and sphere as ftrace motion keys around
the frame, and ftrace's mode R averages the frame over the shutter.

    ftrace -shot tools/loom/examples/motion_shot.py -mp4 -spp 256
    python -m loom.shot tools/loom/examples/motion_shot.py --range 0 3 -- -spp 64
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Scene, Camera, Material, Sphere, Group, Light, Raw, TimeFn, vec,
)

FRAMES = 48
FPS = 24.0
LOOP = False
SHUTTER = (-0.25, 0.25)

_CUBE = "\n".join([
    "quad { origin -0.25 -0.25  0.25  u 0.5 0 0  v 0 0.5 0  material %(m)s }",
    "quad { origin  0.25 -0.25 -0.25  u -0.5 0 0  v 0 0.5 0  material %(m)s }",
    "quad { origin -0.25 -0.25 -0.25  u 0 0 0.5  v 0 0.5 0  material %(m)s }",
    "quad { origin  0.25 -0.25  0.25  u 0 0 -0.5  v 0 0.5 0  material %(m)s }",
    "quad { origin -0.25  0.25  0.25  u 0.5 0 0  v 0 0 -0.5  material %(m)s }",
    "quad { origin -0.25 -0.25 -0.25  u 0.5 0 0  v 0 0 0.5  material %(m)s }",
])


def build(clock=None, res=(480, 300)) -> Scene:
    """The shot. Time is the open timeline's phase t in [0, 1]."""
    s = Scene(Camera(eye=(0, 1.0, 4.2), look_at=(0, 0.45, 0), fov_y=38, mode="R", res=res))
    roll = TimeFn(lambda t: -1.4 + 2.8 * t, periodic=False)            # 2.8 m over the shot
    s.add(
        Material("floor", "diffuse", reflect=0.6),
        Material("wall", "diffuse", reflect=0.7),
        Material("red", "diffuse", reflect="rgb 0.75 0.12 0.08"),
        Material("blue", "diffuse", reflect="rgb 0.10 0.22 0.75"),
        Material("gold", "glossy", reflect="rgb 0.95 0.72 0.30", roughness=0.12),
        Raw("quad { origin -3 0 -2  u 0 0 5  v 6 0 0  material floor }\n"
            "quad { origin -3 0 -2  u 6 0 0  v 0 3 0  material wall }"),
        Sphere(vec(roll, 0.3, 0.4), 0.3, "gold"),
        Group(Raw(_CUBE % {"m": "red"}), translate=(1.05, 0.3, -0.5),
              rotate=vec(0.0, TimeFn(lambda t: 720.0 * t, periodic=False), 0.0)),
        Group(Group(Raw(_CUBE.replace("0.25", "0.18") % {"m": "blue"}), translate=(0, 0.3, 0),
                    rotate=vec(0.0, 0.0, TimeFn(lambda t: 540.0 * t, periodic=False))),
              translate=vec(-1.1, TimeFn(lambda t: 0.6 * t, periodic=False), -0.4)),
        Light("area", origin="-0.6 2.9 -0.2", u="1.2 0 0", v="0 0 1.2", normal="0 -1 0",
              spd="preset:bb6500", power=400),
    )
    return s
