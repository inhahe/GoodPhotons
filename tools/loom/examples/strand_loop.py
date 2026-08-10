"""
Loom example: seamless looping **fibers** on ftrace's native ``curve`` primitive.

Compare ``ribbon_loop.py``, which sweeps a profile into a per-frame OBJ.  Here the
same kind of breathing closed spine is emitted as a ``curve { }`` block — control
points and a radius, no mesh file at all — and ftrace flattens it into a watertight
chain of round cones at load time.

Three things this shows that a swept mesh cannot do cheaply:

* **A tangle is free.**  Eight closed fibers is eight short text blocks; as tubes it
  would be eight OBJs re-tessellated and re-written every frame.
* **The loop closes structurally.**  A closed spine is emitted as a wrapped periodic
  B-spline, so the seam is C2 like every other joint — there is no seam to hide.
  (Catmull-Rom's ends are clamp-duplicated and *cannot* close; see ``loom/strands.py``.)
* **The radius is a channel.**  Each fiber carries a *periodic* ``radius_profile``, so
  it swells and pinches around the loop and still meets itself at the seam, plus a
  ``radius`` Signal so the whole tangle breathes over the loop's period.

A tuft of tapered ``hair()`` strands on the floor shows the open case: interpolating
Catmull-Rom, thick root, near-zero tip.

Run:
  python examples/strand_loop.py            # print frame-0 .ftsl to stdout
  python examples/strand_loop.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Const, LoopNoise, vec, PointPath, LoopCurve,
    Scene, Material, strand, hair, Light, Camera, Raw,
)

N_FIBERS = 8
N_ANCHORS = 7
RING_R = 0.46


def _fiber_spine(f: int) -> LoopCurve:
    """A closed spine: a ring of anchors, each on its own seamless LoopNoise walk, so
    the shape is different per fiber and returns exactly to itself over the loop."""
    anchors = []
    for i in range(N_ANCHORS):
        ang = 2.0 * math.pi * i / N_ANCHORS + 0.35 * f
        r = RING_R * (1.0 + 0.10 * math.sin(3 * ang + f))
        bx, by, bz = r * math.cos(ang), 0.13 * math.sin(2 * ang + f), r * math.sin(ang)
        anchors.append(vec(
            bx + LoopNoise(cells=5, seed=100 + 17 * f + i, freq=1, amp=0.10),
            by + LoopNoise(cells=5, seed=200 + 17 * f + i, freq=1, amp=0.13),
            bz + LoopNoise(cells=5, seed=300 + 17 * f + i, freq=1, amp=0.10),
        ))
    return LoopCurve(PointPath(anchors, closed=True), Const(0.0))


def _bulge(lobes: int):
    """A *periodic* radius profile: f(1) == f(0), so the fiber's thickness closes at
    the seam.  A non-periodic one (say ``1 + u``) is a step in the fiber exactly there,
    and Strand warns about it rather than silently snapping."""
    def prof(u: float) -> float:
        return 1.0 + 0.55 * math.sin(2.0 * math.pi * lobes * u)
    return prof


def build_scene(res=(480, 480)) -> Scene:
    scene = Scene(Camera(
        eye=(0.0, 0.80, 2.25), look_at=(0.0, -0.05, 0.0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("fibre", "diffuse", reflect="rgb 0.82 0.46 0.20"),
        Material("bristle", "diffuse", reflect="rgb 0.30 0.55 0.85"),
        Material("floor", "diffuse", reflect=0.6),
    )

    for f in range(N_FIBERS):
        # `radius` is a Signal, so the whole tangle breathes; the profile is periodic,
        # so each fiber still closes on itself.
        r = 0.016 + LoopNoise(cells=3, seed=900 + f, freq=1, amp=0.006)
        scene.add(strand(_fiber_spine(f), radius=r, count=96, segments=2,
                         radius_profile=_bulge(2 + f % 3),
                         material="fibre", name=f"fiber{f}"))

    # The open case: a ring of tapered guide hairs standing off the floor, set OUTSIDE
    # the tangle so their taper reads against the backdrop. Thick root, near-zero tip —
    # and no seam question at all, because an open strand has two ends.
    n_hair = 22
    for h in range(n_hair):
        a = 2.0 * math.pi * h / n_hair
        cx, cz = math.cos(a), math.sin(a)
        bx, bz = 0.70 * cx, 0.70 * cz
        lean, height = 0.26, 0.36
        pts = []
        for k in range(4):
            u = k / 3.0
            # sway grows toward the tip (u**2), so the root stays planted on the floor
            sway = LoopNoise(cells=4, seed=40 + 7 * h + k, freq=1, amp=0.10 * u * u)
            pts.append(vec(bx + lean * u * u * cx + sway,
                           -0.55 + height * u,
                           bz + lean * u * u * cz + sway))
        scene.add(hair(PointPath(pts, closed=False), radius=0.012, tip=0.0006,
                       count=16, material="bristle", name=f"bristle{h}"))

    scene.add(
        Raw('quad { origin -2 -0.56 -2  u 4 0 0  v 0 0 4  material "floor" }'),
        Light("area", origin="-0.5 1.7 0.7", u="1.0 0 0", v="0 0 1.0",
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
    pngs = render_range(scene, frames, name="strand_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("strand_loop") / "strand_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
