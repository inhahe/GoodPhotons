"""
Loom example: a seamless looping **function-driven material** (M6).

A sphere painted by a 2-layer mix of a warm and a cool diffuse, selected per hit
by a procedural ``waves`` pattern.  The pattern is a function of *space* (x,y,z),
and loom bakes a time-varying **phase drift** into it each frame: over one loop the
drift advances by a full ``2*pi``, so the colour bands flow steadily around the
sphere and return bit-for-bit at the wrap (``sin`` is ``2*pi``-periodic) — seamless.
A second drifting pattern drives the sphere's **roughness**, so the highlight
crawls too.

This is the honest reach of ftrace's material system: a scalar ``pattern`` binds to
roughness / ior / a mix ``weight_map``; colour comes from blending two coloured
layers by that weight.

Run:
  python examples/material_loop.py            # print frame-0 .ftsl to stdout
  python examples/material_loop.py --render   # render a seamless looping GIF
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    vec, phase_drift,
    Scene, Material, Camera, Sphere, Raw, Light,
    FuncPattern, MixMaterial, waves, rings,
)


def build_scene(res=(480, 480)) -> Scene:
    # colour bands flow one full cell around the sphere over the loop
    band_drift = vec(phase_drift(1.0), phase_drift(1.0), 0.0)
    rough_drift = vec(0.0, phase_drift(1.0), phase_drift(1.0))

    scene = Scene(Camera(
        eye=(0.0, 0.6, 4.6), look_at=(0, 0, 0), up=(0, 1, 0),
        fov_y=34, mode="R", res=res))
    scene.add(
        # two coloured layers + the drifting selector between them
        FuncPattern("bands", waves, freq=6.0, drift=band_drift),
        FuncPattern("rough", rings, freq=5.0, drift=rough_drift),
        Material("warm", "diffuse", reflect="rgb 0.92 0.42 0.14"),
        # cool layer is glossy; the drifting `rough` pattern crawls its highlight
        Material("cool", "glossy", reflect="rgb 0.16 0.44 0.92",
                 roughness="pattern:rough"),
        MixMaterial("skin", [("warm", 0.5), ("cool", 0.5)], weight_map="bands"),
        Material("wall", "diffuse", reflect="whitewall 0.8"),
        Sphere((0, 0, 0), 1.2, "skin"),
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
        print(scene.emit(Clock.at_frame(0, 48), Cache()))
        return 0

    from loom import render_range, assemble_gif
    frames = 48
    pngs = render_range(scene, frames, name="material_loop", fps=24,
                        noise=4.0, interval=4.0)
    from loom.drive import default_outdir
    assemble_gif(pngs, default_outdir("material_loop") / "material_loop.gif", fps=24)
    return 0


if __name__ == "__main__":
    sys.exit(main())
