"""
Loom example: the :mod:`jumping_jack` tumble in the **mode-W gyroid-showcase idiom**.

Same body, same motion, same room, same world-static gyroid carve — see
``jumping_jack.py`` for all of that, which this module imports rather than copies.
What changes is the surfacing, which is lifted verbatim from
``scenes/_room_of_gyroids_f12.ftsl`` (the scene rendered as
``png/w_gyroid_showcase.png``):

* ``jumping_jack`` gives three arms glossy ``metal:gold`` and three SF10 glass — a
  high-contrast picture built out of specular highlights and dispersive caustics.
* here both triples are plain **matte pastel diffuse**, two of the showcase's four
  shell colours: purple on one axis triple, green on the other.  No gloss, no
  emission, no dispersion; the form is carried entirely by soft Lambertian shading
  across the gyroid carve.

Keeping the two triples different colours does the same job the gold/glass split did
in the original: the ±y spin axis carries **one of each**, which labels the jack's top
and bottom so the precession stays readable instead of looking like generic wobble.

Run:
  python examples/pastel_jack.py            # print frame-0 .ftsl to stdout
  python examples/pastel_jack.py --still    # one held still (look check)
  python examples/pastel_jack.py --render   # render the looping MP4 + GIF

Knobs: ``--res N``, ``--gi N`` (gather rays), plus ``--t PHASE`` / ``--name NAME``
for ``--still``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Material, Camera, Light  # noqa: E402
from loom.ftsl_emit import fmt  # noqa: E402

import jumping_jack as jj  # noqa: E402


# ---------------------------------------------------------------------------
# Surfacing — the showcase's shell palette
# ---------------------------------------------------------------------------
# Two of the four shells of `scenes/_room_of_gyroids_f12.ftsl`, byte for byte.  They
# are ordinary diffuse albedos in the 0-1 range, NOT radiance: this scene is in
# RELATIVE exposure mode (see LUMENS below), so there is no absolute level to hit.
PURPLE = "rgb 0.80 0.45 0.75"   # shell_d — the "gold" triple
GREEN = "rgb 0.55 0.85 0.45"    # shell_c — the "glass" triple

# `jumping_jack` puts `lumens 60000` on its ceiling panel, which switches the film into
# ABSOLUTE exposure mode (fixed sensor gain, `-ev` as a real stop compensation).  The
# showcase does not, and that accounts for as much of its look as the materials do: with
# no flux authored the film runs on **relative auto-exposure**, so the panel, the walls
# and the shells are balanced against each other per image rather than against a physical
# cd/m^2 scale.  Passing lumens=None here reproduces that; a number switches to absolute.
LUMENS = None


def build_scene(res=(480, 480), *, purple=PURPLE, green=GREEN, lumens=LUMENS,
                spin: float = 1.0) -> Scene:
    """The jumping-jack scene surfaced as two matte pastel gyroid shells."""
    # Same standing height and same camera framing as the original — the composition
    # was tuned there and none of it depends on the materials.
    cy = jj.rest_height(jj.Y0, arm=jj.ARM, ball=jj.BALL, tilt=jj.TILT)
    scene = Scene(Camera(eye=(0.5, 0.7 + cy, 5.5), look_at=(0.0, 0.05 + cy, 0.0),
                         up=(0, 1, 0), fov_y=38, mode="W", res=res))
    light = dict(origin=f"-1.6 {fmt(jj.Y1 - 0.02)} -1.6", u="3.2 0 0", v="0 0 3.2",
                 normal="0 -1 0", spd="preset:bb6500")
    if lumens:
        light["lumens"] = lumens
    scene.add(
        Material("purple", "diffuse", reflect=purple),
        Material("green", "diffuse", reflect=green),
        Material("floor", "diffuse", reflect=0.30),
        Material("ceil", "diffuse", reflect=0.78),
        Material("wall", "diffuse", reflect=0.58),
        Material("left", "diffuse", reflect="rgb 0.52 0.30 0.26"),
        Material("right", "diffuse", reflect="rgb 0.28 0.40 0.52"),
        # `gold`/`glass` here are only the Jack element's two material SLOTS (the
        # names of its two arm triples); both are carved by the same gyroid.
        jj.Jack(gyroid_glass=True, spin=spin, centre=(0.0, cy, 0.0),
                gold="purple", glass="green"),
        *jj._room(),
        Light("area", **light),
    )
    return scene


# The showcase itself was rendered with a bare `-mode W -spp 8` — no gather, no ambient
# (verified: that reproduces `png/w_gyroid_showcase.png` to 0.33/255 mean abs error).
# That is why its gyroid interiors are pure black; Whitted direct lighting has no
# indirect term whatsoever, so a cavity the panel cannot see stays at zero.
#
# Here they are lifted with the one-bounce gather instead of with `-ambient`.  The two
# are not equivalent: `-ambient` adds a FLAT floor to every surface in the scene, and
# because relative auto-exposure then renormalises the whole frame, the image as a whole
# washes out and desaturates (at `-ambient 0.04` visibly so) to buy the cavities a lift.
# `-gi` puts *actual bounce light* only where light can actually bounce, so the cavities
# fill with colour bled off the surrounding shell while the contrast and saturation of
# everything else are untouched.
#
# 24 rays rather than 8 for the same reason `jumping_jack` uses 24: the gather is
# cacheless and stochastic, so the ray count is what sets how much its residual noise
# shimmers between frames.  No `-ev` — that is an absolute-mode control and this scene
# is relative.
SHOWCASE = ["-spp", "8", "-gi", "24", "-gi-clamp", "0.15", "-whitted-grid", "3"]


def _args():
    gi = int(jj._opt("--gi", 24))
    a = list(SHOWCASE)
    a[a.index("-gi") + 1] = str(gi)
    return a


def main() -> int:
    still = "--still" in sys.argv
    render = "--render" in sys.argv
    r = int(jj._opt("--res", 480))
    scene = build_scene(res=(r, r))

    if still:
        from loom.drive import render_still
        render_still(scene, t=jj._opt("--t", 0.0),
                     name=jj._sopt("--name", "pastel_jack"),
                     n=1, interval=8.0, extra_args=_args())
        return 0
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, jj.FRAMES), Cache()))
        return 0

    from loom import render_range
    from loom.drive import assemble_gif_ffmpeg, assemble_mp4, default_outdir
    pngs = render_range(scene, jj.FRAMES, name="pastel_jack", fps=jj.FPS, n=1,
                        interval=8.0, skip_existing=True, extra_args=_args())
    out = default_outdir("pastel_jack")
    assemble_mp4(pngs, out / "pastel_jack.mp4", fps=jj.FPS)
    assemble_gif_ffmpeg(pngs[::jj.GIF_STRIDE], out / "pastel_jack.gif", fps=jj.GIF_FPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
