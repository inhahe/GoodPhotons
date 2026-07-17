"""
Loom example: **bake a scalar field to a mesh** with marching cubes (M7).

ftrace root-finds isosurfaces directly, so a field you *can* write as ftsl should
stay an :class:`loom.Isosurface` (sharper, no baking).  ``IsoMesh`` is for the
case where a field must become **geometry** — a numpy-only field with no ftsl
twin, a sampled volume, or a mesh you'll hand to another tool.  Here we bake a
breathing "blobby" field (a smooth-min union of three moving spheres) to an OBJ
per frame and let ftrace render the triangles.

The field morphs over the loop (``T``), so a fresh mesh is baked each frame; at
the wrap it returns bit-for-bit (``sin``/``cos`` are ``2*pi``-periodic), so the
mesh sequence is a seamless loop.

Run:
  python examples/mesh_bake.py            # print the baked mesh's stats (frame 0)
  python examples/mesh_bake.py --still    # render one frame (held live window)
  python examples/mesh_bake.py --render   # render the seamless baked-mesh loop
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    X, Y, Z, T, sin, cos, sqrt, smin,
    Scene, Material, Camera, Raw, Light, IsoMesh, mesh_field,
)

TWO_PI = 2.0 * math.pi


def _ball(cx, cy, cz, r):
    """Signed-ish sphere field: negative inside, zero on the surface."""
    return sqrt((X - cx) * (X - cx) + (Y - cy) * (Y - cy) + (Z - cz) * (Z - cz)) + (-r)


def blobby(k=0.55):
    """Three orbiting spheres fused by a smooth-min (``smin``) union — a classic
    metaball look.  The orbit phase drifts a whole turn over the loop."""
    a = T * TWO_PI
    b1 = _ball(0.9 * cos(a), 0.9 * sin(a), 0.0, 0.7)
    b2 = _ball(0.9 * cos(a + TWO_PI / 3), 0.0, 0.9 * sin(a + TWO_PI / 3), 0.7)
    b3 = _ball(0.0, 0.9 * cos(a + 2 * TWO_PI / 3), 0.9 * sin(a + 2 * TWO_PI / 3), 0.7)
    return smin(smin(b1, b2), b3)


def build_scene(res=(480, 480)) -> Scene:
    field = blobby()
    scene = Scene(Camera(
        eye=(0.0, 0.0, 6.0), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("skin", "diffuse", reflect="rgb 0.95 0.45 0.30"),
        Material("wall", "diffuse", reflect="whitewall 0.8"),
        IsoMesh(field, bounds=2.2, res=64, iso=0.0, adaptive=True, coarse=10,
                material="skin", smooth=1, name="blob"),
        Raw('quad { origin -6 -6 -4  u 12 0 0  v 0 12 0  material "wall" }'),
        Light("area", origin="-2.5 3.0 3.0", u="2.5 0 0", v="0 2.5 0",
              normal="0.4 -0.7 -0.6", spd="preset:bb6500"),
    )
    return scene


def main() -> int:
    if "--still" in sys.argv:
        from loom import render_still
        render_still(build_scene(), t=0.15, name="mesh_bake", noise=4.0, interval=4.0)
        return 0
    if "--render" in sys.argv:
        from loom import render_range, assemble_gif
        from loom.drive import default_outdir
        pngs = render_range(build_scene(), 48, name="mesh_bake", fps=24,
                            noise=4.0, interval=4.0)
        assemble_gif(pngs, default_outdir("mesh_bake") / "mesh_bake.gif", fps=24)
        return 0

    # default: bake frame 0 and report mesh stats + the emitted ftsl line
    from loom import Clock, Cache
    from loom.ftsl_emit import EmitCtx
    import tempfile
    from pathlib import Path
    field = blobby()
    v, f = mesh_field(field, bounds=2.2, res=64, iso=0.0,
                      clock=Clock.at_frame(0, 48), cache=Cache(),
                      adaptive=True, coarse=10)
    print(f"# baked blobby field @ frame 0/48 (res 64, adaptive narrow-band):")
    print(f"  {len(v)} vertices, {len(f)} triangles")
    d = Path(tempfile.mkdtemp())
    im = IsoMesh(field, bounds=2.2, res=64, adaptive=True, coarse=10,
                 material="skin", name="blob")
    print("# emitted ftsl (mesh reference):")
    print("  " + im.emit(EmitCtx(clock=Clock.at_frame(0, 48), cache=Cache(), assets_dir=d)))
    print("\n# run with --still / --render to see ftrace render the baked mesh loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
