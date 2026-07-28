"""
Loom example: a **Room of gyroids** — many placed isosurfaces under one frame (J2).

This exercises the J2 pipeline: a :func:`make_gyroid` factory that returns a
placed :class:`~loom.iso.Isosurface`, and a :class:`~loom.iso.Room` that gathers
several instances under one animatable :class:`~loom.mathnd.Affine` frame.

  - Each blob is a *different* triply-periodic minimal surface (gyroid / Schwarz-P /
    Schwarz-D / Neovius) with its own spatial frequency, material, and container
    radius, so the room reads as a little museum of implicit surfaces.
  - Each blob **orbits** on a *closed* circle ``(R cos, y, R sin)`` — a closed curve
    returns bit-for-bit at the wrap, so the placement animation is seamless.  Its
    ``contained_by`` sphere tracks along (the whole point of J2 placement).
  - The **room itself** slowly tumbles: an integer-turn Givens rotation of the frame,
    which folds into every child's coordinate field *and* AABB, and returns to its
    start at the loop wrap.
  - Each blob's field also drifts by ``2*pi`` over the loop, so the lattice flows
    inside each ball while the balls orbit and the room turns — three seamless
    motions composed, all emitted as one ``.ftsl`` per frame.

Run:
  python examples/room_of_gyroids.py            # print frame-0 .ftsl to stdout
  python examples/room_of_gyroids.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, Cosine, vec, rotations, affine,
    Scene, Material, Camera, Isosurface, Room, phase_drift, Light, Raw,
)


def make_gyroid(surface: str = "gyroid", *, freq: float = 6.0, radius: float = 0.9,
                material: str = "shell", threshold=0.0, drift_turns: float = 1.0,
                placement=(0.0, 0.0, 0.0), name: str = "blob") -> Isosurface:
    """Factory: one seamless, drifting minimal-surface **ball** clipped to a sphere.

    ``surface`` picks the field (``gyroid`` / ``schwarz_p`` / ``schwarz_d`` /
    ``neovius``); ``freq`` packs that many cells into the ``radius`` sphere; the field
    drifts diagonally by ``drift_turns`` full ``2*pi`` cycles over the loop (integer =
    seamless).  ``placement`` (a static point or animatable :class:`VecSignal`) sets
    where the ball sits — the container sphere follows it.
    """
    drift = vec(phase_drift(drift_turns), phase_drift(drift_turns),
                phase_drift(drift_turns))
    return Isosurface(surface, freq=freq, threshold=threshold, drift=drift,
                      container="sphere", center=(0, 0, 0), radius=radius,
                      placement=placement, material=material, name=name)


def _orbit(R: float, y: float, phase: float):
    """A closed circular orbit placement ``(R cos, y, R sin)`` — seamless over a loop.

    ``phase`` (in turns) spaces instances evenly around the ring."""
    return vec(Cosine(cycles=1, phase=phase, amp=R),
               y,
               Sine(cycles=1, phase=phase, amp=R))


# the four blobs on the ring: (surface, freq, radius, material, drift turns)
_BLOBS = [
    ("gyroid",    6.0, 0.85, "shell_a", 1),
    ("schwarz_p", 5.0, 0.85, "shell_b", 1),
    ("schwarz_d", 5.0, 0.85, "shell_c", 1),
    ("neovius",   4.0, 0.85, "shell_d", 1),
]


def build_scene(res=(480, 480)) -> Scene:
    n = len(_BLOBS)
    R = 1.35  # orbit radius

    # the room turns one full seamless turn in the (x,z) plane over the loop.
    room = Room("hall",
                frame=affine(3, [("rot", 0, 2, Sine(cycles=1, amp=math.pi))]))
    for i, (surf, freq, rad, mat, turns) in enumerate(_BLOBS):
        room.add(make_gyroid(surf, freq=freq, radius=rad, material=mat,
                             drift_turns=turns, name=f"g{i}",
                             placement=_orbit(R, 0.0, i / n)))

    scene = Scene(Camera(
        eye=(0.0, 1.1, 6.4), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("shell_a", "diffuse", reflect="rgb 0.85 0.55 0.30"),
        Material("shell_b", "diffuse", reflect="rgb 0.35 0.65 0.85"),
        Material("shell_c", "diffuse", reflect="rgb 0.55 0.85 0.45"),
        Material("shell_d", "diffuse", reflect="rgb 0.80 0.45 0.75"),
        Material("wall", "diffuse", reflect=0.72),
        room,
        # a bright open-fronted, open-topped-lit box so the orbiting balls are lit
        # (front left open for the camera; a broad area light on the ceiling).
        Raw('quad { origin -3 -1.6 -3  u 6 0 0  v 0 0 6  material "wall" }'),   # floor
        Raw('quad { origin -3  2.0 -3  u 6 0 0  v 0 0 6  material "wall" }'),   # ceiling
        Raw('quad { origin -3 -1.6 -3  u 6 0 0  v 0 3.6 0  material "wall" }'), # back
        Raw('quad { origin -3 -1.6 -3  u 0 0 6  v 0 3.6 0  material "wall" }'), # left
        Raw('quad { origin  3 -1.6 -3  u 0 0 6  v 0 3.6 0  material "wall" }'), # right
        Light("area",
              origin="-2.2 1.98 -2.2", u="4.4 0 0", v="0 0 4.4",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


def main() -> int:
    render = "--render" in sys.argv
    scene = build_scene()
    scene.check_cycles()
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, 48), Cache()))
        return 0

    from loom import render_range, assemble_gif
    from loom.drive import default_outdir
    frames = 48
    pngs = render_range(scene, frames, name="room_of_gyroids", fps=24,
                        noise=4.0, interval=4.0)
    assemble_gif(pngs, default_outdir("room_of_gyroids") / "room_of_gyroids.gif",
                 fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
