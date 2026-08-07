"""
Loom example: a **§F1 viewable build** for the native viewer's live channel (§F4 item 2).

This is the smallest honest demonstration of the distinction the whole live channel
turns on:

  * The three spatial dims the viewer *shows* can be re-projected for free — orbiting
    with the left mouse button just rotates the camera over geometry that is already
    baked into the sidecar.
  * A **parameter dimension** is not in that geometry at all.  ``freq`` and
    ``threshold`` below change what the surface *is*, so moving along one of them
    cannot be a re-projection: loom has to re-derive the scene and re-run marching
    cubes.  That is what the viewer's right-drag gesture asks for, and what the
    latest-wins job queue in ``viewer_gui.cpp`` throttles.

So this file deliberately carries **both** kinds of geometry:

  * an :class:`~loom.scene.IsoMesh` — a gyroid ball baked to triangles, i.e. the
    thing that must be *re-tessellated*; and
  * a :class:`~loom.scene.SweptMesh` tube whose radius is another parameter, so the
    Curves tab has a spine to draw and the Meshes tab has a skin.

Both are cheap at the default resolution (a bake is well under a second), which is
what makes a sweep drag feel like a drag rather than a series of stalls.

Open it live:

  ftrace -viewer <sidecar.json> -loom examples/viewer_live.py

or, since ``save_sidecar`` records this file's path under the sidecar's ``build``
key, write the sidecar first and let it name its own build::

  python examples/viewer_live.py --sidecar out/live.json
  ftrace -viewer out/live.json

(``python -m loom.viewer <scene>`` is a *different* thing — the resident
re-introspection server the GUI drives. It takes no ``--sidecar``.)

Run standalone:

  python examples/viewer_live.py                  # print frame-0 .ftsl to stdout
  python examples/viewer_live.py --sidecar [path] # write the sidecar
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Camera, IsoMesh, Light, Material, PointPath, Raw, Scene, Sine, tube,
)
# NB: the top-level `loom.gyroid` is loom.iso's raw `gyroid(cx, cy, cz)` helper. The
# one wanted here is the SpatialExpr pattern, which IsoMesh can bake with numpy.
from loom.spatial import gyroid  # noqa: E402


# --------------------------------------------------------------------------
# The §F1 contract: `build(clock=None, **params) -> Scene`, side-effect-free at
# import, returning a FRESH Scene every call (the viewer calls it once per sweep).
#
# Every keyword here becomes a control row in the viewer's Live panel, and its
# DEFAULT's Python type is what loom reports as the param's type tag — so `res`
# stays an int across the JSON round trip rather than arriving as `48.0` and
# breaking `range()`.
# --------------------------------------------------------------------------
def build(clock=None,
          *,
          freq: float = 6.0,
          threshold: float = 0.0,
          radius: float = 1.15,
          res: int = 40,
          tube_radius: float = 0.09,
          twist: bool = True) -> Scene:
    """A gyroid ball (baked to a mesh) orbited by a swept tube.

    ``freq`` packs more or fewer lattice cells into the same ball and ``threshold``
    thickens or thins its walls — both change the *topology* of the surface, so both
    force a re-bake.  ``radius``/``res`` resize and re-sample the bake volume,
    ``tube_radius`` re-skins the tube, and ``twist`` toggles the tube's animated
    roll (the DAG panel gains an edge when it is on).
    """
    # The field is a SpatialExpr, so IsoMesh bakes it with numpy and the sidecar
    # carries real vertices/faces — which is exactly what has to be recomputed when
    # a parameter moves.
    field = gyroid(freq) - threshold

    ball = IsoMesh(field, bounds=radius, res=int(res), iso=0.0,
                   material="shell", name="gyroid_ball")

    spine = PointPath([(1.7, 0.0, 0.0), (0.0, 1.1, 1.7),
                       (-1.7, 0.0, 0.0), (0.0, -1.1, -1.7)], closed=True)
    ring = tube(spine, radius=tube_radius, material="wire", name="orbit")
    if twist:
        ring.twist = Sine(cycles=1)          # seamless over the loop

    scene = Scene(Camera(eye=(0.0, 0.8, 5.4), look_at=(0, 0, 0), up=(0, 1, 0),
                         fov_y=36, mode="R", res=(480, 480)))
    scene.add(
        Material("shell", "diffuse", reflect=0.85),
        Material("wire", "diffuse", reflect=0.55),
        Material("wall", "diffuse", reflect=0.78),
        ball,
        ring,
        # open-fronted room, so the concave lattice is lit rather than self-shadowed
        Raw('quad { origin -2.4 -2.0 -2.4  u 4.8 0 0  v 0 0 4.8  material "wall" }'),
        Raw('quad { origin -2.4  2.0 -2.4  u 4.8 0 0  v 0 0 4.8  material "wall" }'),
        Raw('quad { origin -2.4 -2.0 -2.4  u 4.8 0 0  v 0 4.0 0  material "wall" }'),
        Raw('quad { origin -2.4 -2.0 -2.4  u 0 0 4.8  v 0 4.0 0  material "wall" }'),
        Raw('quad { origin  2.4 -2.0 -2.4  u 0 0 4.8  v 0 4.0 0  material "wall" }'),
        Light("area", origin="-1.1 1.96 -1.1", u="2.2 0 0", v="0 0 2.2",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


FRAMES = 48          # the loop length this scene is authored for


def main() -> int:
    from loom import Cache, Clock
    if "--sidecar" in sys.argv:
        from loom.viewer import ViewerModel
        i = sys.argv.index("--sidecar")
        out = (sys.argv[i + 1] if len(sys.argv) > i + 1
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "viewer_live.json"))
        # Save WITH a clock: the sidecar's `frame` block is where the viewer gets its
        # `frames` from, and a sidecar saved without one advertises frames = 1 — which
        # leaves the viewer's transport with nowhere to advance to.
        ViewerModel(build).save_sidecar(out, Clock.at_frame(0, FRAMES))
        print(out)
        return 0
    print(build().emit(Clock.at_frame(0, FRAMES), Cache()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
