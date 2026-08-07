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

One thing is *not* pastel: a thin **gold ring** (:class:`Ring`) orbits the jack, tipped
45° off horizontal and spinning about the room's vertical like a coin caught halfway
between falling flat and standing on edge.  It is the scene's only specular element, and
it is deliberately the one that isn't matte — it draws the eye round the jack instead of
competing with it, and it puts a moving highlight in a picture that otherwise has none.

Run:
  python examples/pastel_jack.py            # print frame-0 .ftsl to stdout
  python examples/pastel_jack.py --still    # one held still (look check)
  python examples/pastel_jack.py --render   # render the looping MP4 + GIF

Knobs: ``--res N``, ``--gi N`` (gather rays), ``--t PHASE`` (``--still`` only), and
``--name NAME``, which names the output directory ``png/<NAME>/`` for *either* mode —
give a changed scene its own name rather than resuming on top of an old sequence, since
a resume can detect a half-finished frame but not a stale one.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Material, Camera, Element, Light  # noqa: E402
from loom.ftsl_emit import EmitCtx, fmt, fmt3  # noqa: E402

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


# ---------------------------------------------------------------------------
# The gold ring
# ---------------------------------------------------------------------------

RING_MINOR = 0.07     # tube radius — "thin-ish" against a ~1.8 ring radius
RING_TILT = 45.0      # degrees off horizontal: a coin exactly halfway through falling
# Whole turns of the contact point per loop, and the sign is the whole point.  The jack's
# `precess` walks its lean around world +y once per loop — the SAME axis and the SAME
# sense the ring's azimuth turns on — so a positive ring rate is an exact integer
# multiple of the jack's own rotation, and +3 read as the two being geared together
# rather than as two things happening at once.  Negative counter-rotates it: the eye gets
# opposing motion, which no shared axis can disguise, and the relative rate goes from
# 3 − 1 = 2 turns per loop to |−3 − 1| = 4.  It stays a whole number because a seamless
# loop needs every rotation to close, and −3 keeps the ring's own speed exactly what it
# was — only its direction changes.
RING_TURNS = -3.0


class Ring(Element):
    """A thin torus tipped ``tilt`` degrees off horizontal, spun about world +y.

    The motion is a spinning coin held at one instant of its collapse: the ring keeps a
    **constant** tilt, and what goes round is the *azimuth* of that tilt, so the point
    touching the floor walks a circle at floor level while the diametrically opposite
    point walks the same circle at the top of the ring's travel.  ``turns`` is that
    azimuth's whole turns per loop, and may be **negative** to counter-rotate against the
    jack's precession, which shares this axis (see :data:`RING_TURNS`).  In ftsl that is
    one leaf::

        torus { major R  minor r  rotate <tilt> <azimuth> 0  translate 0 <cy> 0 }

    because a leaf transform is TRS regardless of statement order (``src/ftsl.h``
    ``fieldXf`` → ``affineFromTRS``) and ``rotate rx ry rz`` composes as
    ``Rz·Ry·Rx`` — so ``rotate tilt az 0`` is exactly "tip about x, then carry the tilt
    around y", with the translate applied outside, i.e. the ring rotates about its own
    centre and is then lifted into place.

    **Sizing is forced, not chosen.**  ``reach`` is the half-height the ring must span:
    the caller passes the jack's own ``arm·cos(tilt) + ball`` (see
    :func:`jumping_jack.rest_height`), which is simultaneously the drop from the jack's
    centre to the floor and the rise from its centre to its top.  Put the ring's centre
    at the jack's centre and the two conditions the brief asks for — lowest point on the
    floor, highest point level with the top of the jack — become the single equation

        R·sin(tilt) + r = reach

    solved here for ``R``.  The ``+ r`` is not a fudge: at the extreme point of the tilted
    centreline the tangent is horizontal, so the tube's circular cross-section stands
    vertically there and contributes its full radius to the height.  Because the tilt is
    constant, both extremes are **constant in time** too — the ring touches down at every
    instant of the loop, never hovering and never clipping through the floor.

    The clearance is also fixed by that equation: with ``reach = 1.339`` and ``r = 0.07``
    the ring radius is 1.795, so its inner surface sits 1.725 from the jack's centre
    against the jack's 1.52 reach — a 0.20 margin the tumble can never close, since the
    jack only ever rotates about that same centre.

    A ``torus`` leaf is an exact, unit-Lipschitz SDF with analytic bounds
    (``src/implicit.h``), so unlike the jack's carved ``function`` field it needs neither
    ``contained_by`` nor ``max_gradient``.
    """

    def __init__(self, centre, reach, *, minor: float = RING_MINOR,
                 tilt: float = RING_TILT, turns: float = RING_TURNS,
                 material: str = "ring", name: str = "gold_ring") -> None:
        self.centre = tuple(float(c) for c in centre)
        self.reach = float(reach)
        self.minor = float(minor)
        self.tilt = float(tilt)
        self.turns = float(turns)
        self.material = material
        self.name = name

    @property
    def major(self) -> float:
        """Ring radius that puts the low point on the floor and the high point at the
        top of the jack — see the class docstring."""
        return (self.reach - self.minor) / math.sin(math.radians(self.tilt))

    def roots(self):
        return []       # every field is a plain float; nothing to cycle-check

    def emit(self, ctx: EmitCtx) -> str:
        # Wrapped into [0, 360) purely for tidy scene text — a rotation is periodic, so
        # this changes nothing, but it keeps a counter-rotating ring from emitting a
        # growing pile of negatives (and a literal `-0` at t=0).
        az = (360.0 * self.turns * ctx.clock.t) % 360.0
        return "\n".join([
            f'{self.name} = isosurface {{',
            f'    material "{self.material}"',
            f'    torus {{ major {fmt(self.major)}  minor {fmt(self.minor)}  '
            f'rotate {fmt(self.tilt)} {fmt(az)} 0  translate {fmt3(self.centre)} }}',
            '}',
        ])


def loop_frames(cycles: int = 1) -> int:
    """Frames in a loop holding ``cycles`` of the jack's own 7.2 s tumble."""
    return jj.FRAMES * int(cycles)


def build_scene(res=(480, 480), *, purple=PURPLE, green=GREEN, lumens=LUMENS,
                spin: float = 1.0, cycles: int = 1,
                ring_turns: float = RING_TURNS) -> Scene:
    """The jumping-jack scene surfaced as two matte pastel gyroid shells, ringed.

    ``ring_turns`` is the ring's rate in whole turns per *one jack cycle* (the base
    7.2 s / 432 frames), signed — negative counter-rotates against the precession.
    ``cycles`` lengthens the loop to that many jack cycles while the jack keeps its
    real-world speed (its ``spin``/``precess`` scale with it), which is the only way
    to run the ring at a **fractional** rate and still close the loop.

    Both rotations have to come back to their start at the end of the loop or the
    video pops when it repeats, so ``ring_turns * cycles`` must be a whole number.
    A half-speed ring (``ring_turns=1.5``) therefore needs ``cycles=2``: 3 whole ring
    turns against 2 jack cycles, a 14.4 s loop.  This is checked rather than trusted,
    because the failure mode is a one-frame jump at the wrap that is easy to miss when
    you watch the render go by frame at a time and obvious the moment it loops.
    """
    total_turns = ring_turns * cycles
    if abs(total_turns - round(total_turns)) > 1e-9:
        raise ValueError(
            f"ring_turns={ring_turns} over cycles={cycles} gives {total_turns} turns, "
            f"which does not close the loop (the ring would jump "
            f"{360.0 * (total_turns - int(total_turns)):.4g}° at the wrap). "
            f"Use a cycles that makes ring_turns*cycles a whole number — e.g. "
            f"cycles=2 for a half-speed ring.")
    # Same standing height as the original — that is pure geometry and none of it
    # depends on the materials.
    cy = jj.rest_height(jj.Y0, arm=jj.ARM, ball=jj.BALL, tilt=jj.TILT)
    reach = cy - jj.Y0                       # == arm·cos(tilt) + ball, both ways
    # The framing, though, HAS to change: `jumping_jack`'s eye/fov were tuned around a
    # 1.5 m-reach jack, and the ring is a 1.79 m-radius object that swings its nearest
    # arc 1.27 m out of the jack's plane and towards the eye — 4.7 m away instead of the
    # jack's 6.0, which is where perspective magnifies it most.  At the
    # original (z 5.5, fov 38) the ring's near top/bottom projects to 1.43× the half
    # frame — i.e. cut off for a third of every turn, which is worse than any framing
    # compromise.  `scraps/ring_fit.py` projects the whole torus surface at every azimuth
    # to find what does fit: back off 0.5 m, open up to fov 44, and lift eye + aim so the
    # ring's near arc is centred rather than riding the bottom edge.  Worst-case
    # projection is then 0.81 across and 0.94 down, so the ring clears the frame at every
    # azimuth with margin to spare.  Two notes on why not tighter: the eye must stay
    # inside the room (front wall at z = 6.6), so the extra room has to come from the
    # field of view rather than from backing off further; and an exhaustive sweep of the
    # same script does bottom out slightly tighter, at fov 41 with the eye at z 6.2, but
    # that buys only 8 % of subject size for 0.2 m less wall clearance.
    scene = Scene(Camera(eye=(0.5, 1.0 + cy, 6.0), look_at=(0.0, 0.3 + cy, 0.0),
                         up=(0, 1, 0), fov_y=44, mode="W", res=res))
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
        # The one specular surface in the scene — `jumping_jack`'s own gold, verbatim
        # (which is what `preset gold` expands to in src/materials.h).
        Material("ring", "glossy", reflect="metal:gold", roughness=0.05),
        # `gold`/`glass` here are only the Jack element's two material SLOTS (the
        # names of its two arm triples); both are carved by the same gyroid.
        # `spin`/`precess` are counted per LOOP, not per second, so lengthening the loop
        # to `cycles` jack tumbles means multiplying both by `cycles` — that is what keeps
        # the jack moving at exactly the speed it moves at in the 432-frame cut instead of
        # slowing down alongside the ring.
        jj.Jack(gyroid_glass=True, spin=spin * cycles, precess=float(cycles),
                centre=(0.0, cy, 0.0), gold="purple", glass="green"),
        # Likewise the ring: `ring_turns` is per jack cycle, so over `cycles` of them it
        # makes `ring_turns * cycles` whole turns — the quantity checked above.
        Ring((0.0, cy, 0.0), reach, turns=ring_turns * cycles),
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
    # `--name` applies to a sequence as well as to a still.  A sequence resumes on
    # `skip_existing`, which can spot a *partial* frame (its `-checkpoint` sidecar is
    # short) but has no way to spot a **stale** one — a finished frame of a different
    # scene looks identical to a finished frame of this one.  So when the scene changes,
    # the run needs a directory of its own rather than a clean-up you have to remember;
    # `--name` is how you give it one, and it keeps each set self-contained (frames,
    # `.ftsl`, checkpoints, MP4 and GIF all under `png/<name>/`).
    name = jj._sopt("--name", "pastel_jack")
    # `--ring-turns` is signed whole-or-half turns of the ring per jack cycle, and
    # `--cycles` is how many jack cycles the loop holds.  They are separate knobs because
    # only their PRODUCT has to be a whole number: a ring rate that is not itself whole
    # (the half-speed cut, 1.5) is legal as soon as the loop is long enough to close it.
    cycles = max(1, int(jj._opt("--cycles", 1)))
    ring_turns = float(jj._opt("--ring-turns", RING_TURNS))
    frames = loop_frames(cycles)
    scene = build_scene(res=(r, r), cycles=cycles, ring_turns=ring_turns)

    if still:
        from loom.drive import render_still
        render_still(scene, t=jj._opt("--t", 0.0), name=name,
                     n=1, interval=8.0, extra_args=_args())
        return 0
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, frames), Cache()))
        return 0

    from loom import render_range
    from loom.drive import assemble_gif_ffmpeg, assemble_mp4, default_outdir
    pngs = render_range(scene, frames, name=name, fps=jj.FPS, n=1,
                        interval=8.0, skip_existing=True, extra_args=_args())
    out = default_outdir(name)
    assemble_mp4(pngs, out / f"{name}.mp4", fps=jj.FPS)
    assemble_gif_ffmpeg(pngs[::jj.GIF_STRIDE], out / f"{name}.gif", fps=jj.GIF_FPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
