"""
Loom example: the :mod:`jumping_jack` tumble, but the jack is **self-luminous**.

Same body, same motion, same room, same world-static gyroid carve — see
``jumping_jack.py`` for all of that, which this module imports rather than copies.
The one change is the pair of materials on the six arms:

* ``jumping_jack`` gives three arms glossy ``metal:gold`` and three SF10 glass, so
  the jack is lit *by* the room.
* here both triples are **coloured glossy surfaces that also carry an ``emit``
  spectrum** — a warm amber lattice on one axis triple and a cool cyan one on the
  other.  The room's ceiling panel stays (dimmed from 60000 to 36000 lm), because it
  is what still draws the filigree; the emission is what makes the lattice read as
  lit from within.

Why an emissive *material* (rather than six ``light`` blocks): a material's ``emit``
is consumed generically by every renderer via ``m.isLight``/``m.emit``, so the
gyroid-carved isosurface glows over its whole marched surface — every filigree wall,
including the ones deep inside the lattice.  No emitter geometry has to be
tessellated or matched to the carve, and the glow tracks the carve exactly as the
pose changes.  (Registering an *emitter* for importance sampling is a mesh-only
feature; an isosurface picks up emission-on-hit only, which is what a Whitted render
uses anyway.)

Keeping the two triples different colours does the same job the gold/glass split did
in the original: the ±y spin axis carries **one of each**, which labels the jack's top
and bottom so the precession stays readable instead of looking like generic wobble.

Run:
  python examples/glowing_jack.py            # print frame-0 .ftsl to stdout
  python examples/glowing_jack.py --still    # one held still (look check)
  python examples/glowing_jack.py --render   # render the looping MP4 + GIF

Knobs: ``--res N``, ``--warm E``, ``--cool E`` (emission scales), ``--lumens L``
(ceiling panel), plus ``--t PHASE`` / ``--name NAME`` for ``--still``.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))      # the loom package
# ...and this directory, so the sibling `jumping_jack` import below still resolves when
# this file is imported BY PATH rather than run as a script (`loom.viewer.load_build`,
# which the live viewer and `python -m loom.anim` both use, does not add it).
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loom import Scene, Material, Camera, Light  # noqa: E402
from loom.ftsl_emit import fmt  # noqa: E402

import jumping_jack as jj  # noqa: E402


# ---------------------------------------------------------------------------
# Emission levels — and why they are NOT cranked
# ---------------------------------------------------------------------------
# `emit` is spectral radiance (per nm), and the scene is in ABSOLUTE exposure mode
# because the ceiling panel carries `lumens` — so these numbers are physical, not
# arbitrary, and they have to be chosen against the room rather than by eye-balling a
# 0-1 range.  They were calibrated by rendering a strip of emissive spheres at
# E = 3e-2 / 1e-2 / 3e-3 / 1e-3 / 3e-4 under a 6000 lm panel at `-ev 11`: 3e-2 washes
# out to pale yellow, 3e-3 falls to a dull brown, ~1e-2 is a saturated glowing body.
#
# The trap is that "saturated glowing body" is exactly what *ruins* this subject.
# Emission is orientation-independent: it adds the SAME radiance to every visible
# point regardless of which way that point faces.  Push it above the reflected term
# and the gyroid filigree — whose entire legibility comes from shading varying across
# a curved, deeply self-occluding surface — collapses into a flat coloured silhouette
# with a few holes punched in it.  (Rendered and confirmed: at E = 1.35e-2, with the
# panel at 12000 lm, the six balls read as paper cut-outs.)
#
# Worse, the lattice can't rescue itself by lighting its own cavities: a mode-W gather
# ray arrives with specularArrival = false, so it takes emission only through NEE, and
# NEE needs a registered emitter — which a marched isosurface never has.  A brighter
# jack therefore does not brighten anything, not even itself.
#
# So the emission is deliberately set BELOW the reflected response, at roughly an
# eighth of the "glowing body" level.  It works as a floor: it lifts the deep gyroid
# cavities — which the panel cannot reach at all — off black and tints them, so the
# lattice reads as lit from *within* while the surface shading still carries the form.
WARM = (0.00630, 0.00240, 0.00075)   # amber — the "gold" triple
COOL = (0.00075, 0.00300, 0.00555)   # cyan  — the "glass" triple

# The bodies are GLOSSY, not the near-black diffuse emitters a pure light source would
# be.  Diffuse + emit was tried first and reads as flat coloured plastic: a Lambertian
# lobe under one broad ceiling panel varies slowly, so it barely draws the filigree,
# and whatever the emission then adds only washes it out further.  A tight glossy lobe
# instead throws a sharp highlight off every gyroid wall and edge, which is precisely
# the detail that has to survive — it plays the role the original's gold and dispersive
# glass played.  Hue matches the emission on each triple, so glow and shading agree.
WARM_REFL = "rgb 0.62 0.30 0.10"
COOL_REFL = "rgb 0.10 0.36 0.62"
ROUGHNESS = 0.08

# The panel therefore stays near the original 60000 lm — it is doing real work again
# (it is the only thing that shades the lattice), just pulled down enough that the
# emissive floor is still visible in the cavities.
LUMENS = 36000


def build_scene(res=(480, 480), *, warm=WARM, cool=COOL, lumens: float = LUMENS,
                spin: float = 1.0) -> Scene:
    """The jumping-jack scene with both arm triples emissive."""
    # Same standing height and same camera framing as the original — the composition
    # was tuned there and none of it depends on the materials.
    cy = jj.rest_height(jj.Y0, arm=jj.ARM, ball=jj.BALL, tilt=jj.TILT)
    scene = Scene(Camera(eye=(0.5, 0.7 + cy, 5.5), look_at=(0.0, 0.05 + cy, 0.0),
                         up=(0, 1, 0), fov_y=38, mode="W", res=res))
    rgb = lambda c: f"rgb {fmt(c[0])} {fmt(c[1])} {fmt(c[2])}"  # noqa: E731
    scene.add(
        # Glossy bodies that ALSO emit — see the note on WARM/COOL above for why the
        # reflected term has to stay dominant, and why it is glossy rather than diffuse.
        Material("warm", "glossy", reflect=WARM_REFL, roughness=ROUGHNESS, emit=rgb(warm)),
        Material("cool", "glossy", reflect=COOL_REFL, roughness=ROUGHNESS, emit=rgb(cool)),
        Material("floor", "diffuse", reflect=0.30),
        Material("ceil", "diffuse", reflect=0.78),
        Material("wall", "diffuse", reflect=0.58),
        Material("left", "diffuse", reflect="rgb 0.52 0.30 0.26"),
        Material("right", "diffuse", reflect="rgb 0.28 0.40 0.52"),
        # `gold`/`glass` here are only the Jack element's two material SLOTS (the
        # names of its two arm triples); both are carved by the gyroid.
        jj.Jack(gyroid_glass=True, spin=spin, centre=(0.0, cy, 0.0),
                gold="warm", glass="cool"),
        *jj._room(),
        Light("area", origin=f"-1.6 {fmt(jj.Y1 - 0.02)} -1.6", u="3.2 0 0", v="0 0 3.2",
              normal="0 -1 0", spd="preset:bb6500", lumens=lumens),
    )
    return scene


def main() -> int:
    still = "--still" in sys.argv
    render = "--render" in sys.argv
    r = int(jj._opt("--res", 480))
    scale = lambda base, flag: tuple(  # noqa: E731
        c * jj._opt(flag, 1.0) for c in base)
    scene = build_scene(res=(r, r), warm=scale(WARM, "--warm"),
                        cool=scale(COOL, "--cool"),
                        lumens=jj._opt("--lumens", LUMENS))

    if still:
        from loom.drive import render_still
        render_still(scene, t=jj._opt("--t", 0.0),
                     name=jj._sopt("--name", "glowing_jack"),
                     n=1, interval=8.0, extra_args=jj.WHITTED)
        return 0
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, jj.FRAMES), Cache()))
        return 0

    from loom import render_range
    from loom.drive import assemble_gif_gifski, assemble_mp4, default_outdir
    pngs = render_range(scene, jj.FRAMES, name="glowing_jack", fps=jj.FPS, n=1,
                        interval=8.0, skip_existing=True, extra_args=jj.WHITTED)
    out = default_outdir("glowing_jack")
    assemble_mp4(pngs, out / "glowing_jack.mp4", fps=jj.FPS)
    assemble_gif_gifski(pngs[::jj.GIF_STRIDE], out / "glowing_jack.gif", fps=jj.GIF_FPS,
                        width=jj.GIF_WIDTH, quality=jj.GIF_QUALITY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
