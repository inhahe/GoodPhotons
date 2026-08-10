"""
Loom example: a **jumping jack** (jackstone) tumbling inside a closed room, its
six arms carved out of a *world-static* gyroid field.

The visual idea — and the whole reason this is a CSG scene — is that the gyroid is
**not** attached to the jack.  ``sin x cos y + sin y cos z + sin z cos x = 0`` is
evaluated in **world** space and never moves; what moves is the *arm volume* that
selects a piece of it::

    isosurface {                       # one march for all three gold arms
        intersect {
            union { sphere ×3  cylinder ×3 }     <- ONLY these carry the pose
            function { expr "gyroid(x,y,z)" }    <- NO transform: world-static
        }
    }

So as the jack spins, gold filigree flows *through* the arms: new lattice walls
appear at the leading face and dissolve at the trailing one.  Put the ``rotate``
on the ``isosurface`` (or on the ``function`` leaf) instead and the effect dies —
the pattern would just ride along rigidly.

Motion is a real jack's tumble.  The **±y arm pair is the spin axis**: the jack
turns about it, that axis leans ``tilt`` degrees off vertical, and the lean walks
around a cone.  So the top ball rides a circle and the bottom ball rides the same
circle 180° out of phase — automatic, since the two are antipodal about a centre
that never moves — while the four equatorial arms whirl about the leaning axis.
A jack whose three rods are half glass and half gold is *not* an inertially
isotropic body, so a real one does precess like this.

An earlier version pre-rotated the body to keep the spin axis deliberately *off*
any arm, reasoning that spinning exactly about a rod leaves two of the six arms
"stationary".  That was wrong twice over: those two arms are not stationary (they
trace the precession circles, which is the motion one actually wants to see), and
without an arm on the axis there is no visible top or bottom at all, so the tumble
reads as arbitrary wobble rather than precession.

Both rates are whole turns per loop, and the field is static, so the loop closes
the instant the **pose** repeats — no phase drift needed anywhere.

The jack **stands on the floor** rather than floating: because the ±y pair is the spin
axis, the bottom ball's centre sits at a constant height ``−arm·cos(tilt)`` for the whole
loop, so the jack's depth below its own centre is the closed form ``arm·cos(tilt) + ball``
with no dependence on ``t``.  See :func:`rest_height`.

Three arms are gold (glossy ``metal:gold``) and three are SF10 glass.  The ±x axis is
all gold, the ±z axis is all glass, and the **spin axis carries one of each**, which
labels the jack's top and bottom and so keeps the precession readable.

Why the intersection is cheap: a raw gyroid has ``|grad| <= 2*sqrt(3)*freq``, which
would force the sphere-tracer into steps of ``d/(2*freq)``.  Dividing the
expression by ``2*freq`` renormalises it to ``|grad| <= sqrt(3)``, so one honest
``max_gradient 2`` covers it and the march runs ~freq times faster — and the
CSG partner (a true SDF union) stays unit-Lipschitz either way.

Rendered in **mode W** (deterministic Whitted) with the ``-gi`` one-bounce gather,
which is noise-free at low ``-spp`` and — having no irradiance cache — cannot
flicker across an animation.  The light carries ``lumens``, which puts the scene
in **absolute** exposure mode: without it ftrace's p99 auto-exposure re-anchors
every frame and the GIF pumps in brightness.

Run:
  python examples/jumping_jack.py            # print frame-0 .ftsl to stdout
  python examples/jumping_jack.py --still    # one held still (pose check)
  python examples/jumping_jack.py --render   # render the looping GIF

Knobs: ``--res N``, ``--freq F``, ``--solid F``, ``--counter F``, ``--ball R``,
``--rod R``, ``--fill F``,
``--t PHASE`` and ``--name NAME`` (both for ``--still``), ``--plain-glass``.
"""

from __future__ import annotations

import bisect
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Material, Camera, Light, Raw  # noqa: E402
from loom import Clamp, Cos, Phase, Signal, Slot, as_signal  # noqa: E402
from loom.scene import Element  # noqa: E402
from loom.ftsl_emit import EmitCtx, fmt, fmt3  # noqa: E402


# ---------------------------------------------------------------------------
# 3x3 rotation helpers (rows), matching ftrace's convention exactly
# ---------------------------------------------------------------------------
# ftrace builds a leaf/group transform as R = Rz(rz)·Ry(ry)·Rx(rx) from Euler
# angles in DEGREES, each a standard right-handed rotation (src/mesh.h,
# MeshXform::applyLinear).  We mirror that here so a pose computed in Python and
# a `rotate` emitted into the .ftsl agree.

Mat3 = Tuple[Tuple[float, float, float], ...]


def _rx(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _ry(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rz(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mm(A: Mat3, B: Mat3) -> Mat3:
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _mv(M: Mat3, v: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(sum(M[i][k] * v[k] for k in range(3)) for i in range(3))


# ---------------------------------------------------------------------------
# The gyroid's own value distribution, for picking level sets by volume
# ---------------------------------------------------------------------------
# `solid` / `fill` are volume fractions, so turning one into a level `g <= c` means
# inverting the CDF of the raw gyroid.  Two shortcuts fail here and both were tried:
#
#   * `volfrac(|g| <= g0) ~ 0.647*g0` is the fraction in a BAND around zero.  The
#     one-sided fraction is `(1 - 0.647*g0)/2`, i.e. HALF that slope — using the band
#     slope for a one-sided level silently delivered 0.378 when 0.26 was asked for, which
#     is why an entire sweep of "sparse" settings all came back looking half-filled.
#   * Any linear fit is only good near g = 0 anyway.  Sparse lattices need levels far out
#     in the tail (`solid = 0.15` is `c = -1.06` against a range of +/-1.5), where a
#     linear extrapolation is off by a factor of two.
#
# So invert the real thing.  `g` is sampled on a fixed grid over one full period, which is
# deterministic (no RNG, so the scene is reproducible) and independent of `freq` — the
# distribution of `sin u cos v + ...` does not care how fast u, v, w sweep.  110k samples
# is plenty for a volume fraction and takes a fraction of a second, once, cached.

_G_SORTED: Optional[List[float]] = None


def _gyroid_samples(n: int = 48) -> List[float]:
    """Sorted samples of the raw gyroid over one period. Cached; ``freq``-independent."""
    global _G_SORTED
    if _G_SORTED is None:
        tau = 2.0 * math.pi
        vals: List[float] = []
        trig = [(math.sin(tau * (i + 0.5) / n), math.cos(tau * (i + 0.5) / n))
                for i in range(n)]
        for sx, cx in trig:
            for sy, cy in trig:
                for sz, cz in trig:
                    vals.append(sx * cy + sy * cz + sz * cx)
        vals.sort()
        _G_SORTED = vals
    return _G_SORTED


def gyroid_quantile(frac: float) -> float:
    """Level ``c`` such that ``g <= c`` occupies volume fraction ``frac`` of space."""
    v = _gyroid_samples()
    p = min(max(float(frac), 0.0), 1.0)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


def gyroid_cdf(level: float) -> float:
    """Volume fraction of ``g <= level`` — the inverse of :func:`gyroid_quantile`."""
    v = _gyroid_samples()
    return bisect.bisect_left(v, float(level)) / len(v)


def aim_y_euler(d: Sequence[float]) -> Tuple[float, float, float]:
    """Euler angles (deg, ftrace's XYZ order) that rotate local **+y** onto unit ``d``.

    With ``ry = 0``, ``R·(0,1,0) = (−sin rz·cos rx, cos rz·cos rx, sin rx)``, so
    ``rx = asin(d.z)`` and ``rz = atan2(−d.x, d.y)``.  A cylinder is symmetric about
    its own axis, so leaving the third degree of freedom at zero costs nothing.
    """
    dz = max(-1.0, min(1.0, float(d[2])))
    rx = math.degrees(math.asin(dz))
    rz = math.degrees(math.atan2(-float(d[0]), float(d[1])))
    return (rx, 0.0, rz)


# ---------------------------------------------------------------------------
# Geometry / gyroid tuning
# ---------------------------------------------------------------------------
# These live here, above the class, so `Jack`'s defaults and the CLI defaults are the
# SAME constants — two copies of a tuned number is how a scene ends up rendering with
# settings nobody chose.
#
# `solid` is the setting that decides whether a part looks openwork or merely *dimpled*.
# It is the **volume fraction** of the gyroid solid used to carve the arms, and getting
# here took ruling out three plausible-sounding answers — all measured, all wrong:
#
# 1. `freq` is not the lever.  Rendered at 12 / 20 / 26 / 34 with a half-filled lattice:
#    all four give a golf ball whose dimples merely get finer.  A CSG intersection keeps
#    every piece of the ball's OWN outer surface where the field is negative — half of it
#    — and those broad smooth patches are what the eye reads as solid metal, at any
#    frequency.
# 2. Nor is a shell `|g| <= w` specified as a *thickness in metres*, which is a
#    well-disguised trap.  The gyroid surface is so convoluted (area ~3.09 per unit
#    volume per unit period) that a slab just 13.7% of a cell thick already occupies 43%
#    of space, so the first "thin shell" rendered indistinguishably from the half-space.
#    Thickness carries no intuition here; volume does.
# 3. Even a correctly-thin shell only half-works.  At 24% by volume the envelope survives
#    as narrow ribbons instead of broad caps — a real improvement, visible in
#    `png/jack_tune/` — but with under 3 cells across a ball there is still no clear line
#    of sight through the part, so it reads as knobbly rather than see-through.
#
# What the reference images (`png/gold_gyroids`, `png/gyroid_nd`) actually do is not CSG
# at all: a lone `function` field with `contained_by { sphere }`, where `contained_by`
# CLIPS rather than intersects, so the bare gyroid sheet is simply cut off at the sphere
# and there is no envelope surface anywhere.  That cannot be reproduced through CSG,
# whose whole job is to bound a solid — and it cannot be had per-arm either, since
# `contained_by` takes one axis-aligned box or sphere, not a rotating ball-and-rod.
#
# So the lever that remains is to make the carving solid *sparse*: at `solid = 0.5` the
# gyroid's two labyrinths are equal and the carve is chunky, while well below that the
# solid thins to an open network with gaps you can see the room through — and the
# envelope patches shrink with it, since they only survive where the field is negative.
# `solid` is converted to a level set by the measured mapping in `Jack.level`.
#
# That works, and among single-level carves `solid` is the only lever that does.  Swept at
# 720x720: 0.16 gives broad windows you can see the room through, 0.24 keeps the windows but
# on visibly heavier walls, 0.30+ is the golf ball again.  0.24 was the default until the
# counter-network below made 0.20 the better base; 0.16 is the openwork end of the range.
#
# `freq` then controls scale alone: cell period `2*pi/freq`, so a ball spans
# `ball*freq/pi` cells.  The intuition that *more* cells would look more like the lacy
# reference stills is exactly backwards, and was measured to be: at freq 22 (3.6 cells
# across a ball) and freq 30 (5.0) the dimpled golf ball comes straight back.  The
# reference gets away with six small cells only because it is a clipped sheet with no
# envelope at all; here the envelope survives every carve, so an opening only reads as an
# opening while it is large compared with the part.  ~2.5 cells across a ball is the
# balance — few enough for big windows, many enough to still read as a lattice.
#
# The glass is insensitive to all of this: swept over 0.16/0.20/0.24 the three renders are
# near-indistinguishable, because refraction through a faceted shell dominates whatever
# the shell's exact thickness is.  So `solid` is chosen on the gold alone.
#
# Sanity-checked for structural integrity rather than trusted: sampling every rod's cross
# section over 60 frames x 6 arms x 24 slices, the retained fraction never reaches zero at
# 0.16 (worst 2%), so no ball is ever carved free of the hub.
#
# ---------------------------------------------------------------------------
# `counter`: closing the sight-lines without closing the surface
# ---------------------------------------------------------------------------
# `solid` alone cannot separate "looks lacy" from "you can see the room through it",
# because for a plain one-level carve **the fraction of the ball's outer surface that
# survives is exactly the volume fraction** — so raising `solid` darkens the windows and
# fattens the skin in lockstep.  Two quantities, one knob.  Measured over 32 random ball
# placements x 4000 rays each (`scraps/see_through.py`, paired sampling so every candidate
# sees identical rays):
#
#     solid   see-through %   envelope %      see-through = rays crossing the ball that
#      0.24        14.9          23.4         meet NO material at all; envelope = share of
#      0.30        10.6          29.4         the ball's own smooth surface left behind
#      0.36         7.4          35.5
#      0.44         4.7          43.5
#
# Halving see-through costs ~12 points of envelope, i.e. the golf ball comes back.  Five
# other mechanisms were measured against that baseline, and the ranking was not the
# expected one:
#
#   * A 4-D gyroid sliced at a generic angle — quasiperiodic, so no exact straight
#     channels, and the obvious reading of "rotate it in a higher dimension".  It is
#     WORSE: 18.4% see-through at 24.3% envelope, against 14.9/23.4.  Quasiperiodicity
#     removes the *periodic* channels but replaces them with wider irregular voids, and it
#     is void width, not channel straightness, that a 1-metre ray cares about.
#   * Unioning a copy shifted half a period in all three axes: a no-op.  Shifting x,y,z by
#     pi each maps sin(x+pi)cos(y+pi) -> sin(x)cos(y), so the "shifted" gyroid is the
#     original.  (An unpaired first measurement made this identity look like a 1.7-point
#     win, which is why the sweep is paired now.)
#   * A denser core inside each arm ball: works (8.7% at unchanged 23.4% envelope) but
#     leaves a hard spherical seam mid-hole.
#   * A finer lattice unioned on top: the strongest blocker of all (1.6%) but it crusts the
#     skin over at 35.9% envelope — the golf ball by another route.
#   * A rotated incommensurate copy: 6.5% at 30.9% — good, but it pays full envelope for
#     the privilege since both copies lie on the skin.
#
# What actually works is to spend volume where the *sight-lines* are rather than where the
# surface is.  The voids of `g <= c` are `g > c`, and the OTHER labyrinth's core, `g >= t`,
# runs right down the middle of them.  So union a sparse counter-network onto the carve:
#
#     solid  counter   see-through %   envelope %   runs   patches
#      0.24     —           14.9          23.4      1.59     18.2
#      0.20    0.06          7.5          25.7      2.10     48.5
#      0.24    0.06          5.9          29.6      2.18     46.7
#
# `runs` is the mean number of separate solid segments a ray passes through and `patches`
# the number of disconnected islands the surviving envelope breaks into on a ball.  Both
# matter as much as the headline numbers: `runs` ~2.2 means you look *into* a hole and find
# more structure behind it (a pocket, not a window), and tripling the patch count means the
# same envelope percentage is spread over many small islands instead of a few broad caps —
# so 0.20+0.06 reads *lacier* than plain 0.24 despite covering 2 points more of the skin,
# while seeing through only half as much of it.
#
# The counter-network is a single connected structure, not dust: labelled over a 2x2x2
# block of cells it is 98.4% one component and spans all three axes even at 4% by volume
# (`scraps/see_through2.py`).  The combined solid is likewise connected and spanning.
#
# The symmetric version of the same idea — the double gyroid `|g| >= t`, both labyrinth
# cores and no sheet — measures slightly better still (7.6% at 24.1%).  It is not used,
# because its two networks are provably disjoint (largest component 49.2% of the solid, one
# per labyrinth): the arms would be two interlocked but unconnected lattices, and the point
# of the piece is six balls *held together* by six rods.
FREQ = 15.0
SOLID = 0.20
COUNTER = 0.06
FILL = 0.0
ARM = 1.0
BALL = 0.52
ROD = 0.32
TILT = 35.0     # precession half-angle; a module constant because `rest_height` needs it

# Bounds on a *driven* lean (see :func:`lean_signal`).  Both ends are real limits, not
# taste.  Below ~5° the cone collapses and the precession stops being visible at all,
# and `rest_height`'s proof that the bottom axis ball is always the lowest part of the
# jack holds only for `tilt < 45°` (it turns on `cos(tilt) > sin(tilt)`) — past that the
# equatorial arms swing lower, the solved standing height is wrong, and the jack cuts
# through the floor.  So the upper clamp is exactly where that proof expires.
LEAN_MIN = 5.0
LEAN_MAX = 44.0

_TAU = 2.0 * math.pi
_RAD = math.pi / 180.0          # degrees -> radians, as a multiplier for a Signal


# ---------------------------------------------------------------------------
# The element
# ---------------------------------------------------------------------------

# Body-frame half-axes: (direction, "gold" | "glass").  One whole axis is gold, one whole
# axis is glass, and the third carries one of each — three arms per material either way.
#
# Which axis gets the mixed pair is not arbitrary: it is **±y, the spin axis**.  Those two
# arms are the jack's top and bottom, the pair that rides the antiphase precession circles,
# and they are the only two whose motion you can actually track by eye.  Making them
# *different materials* labels the two ends of the axis, so the lean and its walk around
# the cone stay readable; a matched pair there would be visually interchangeable and the
# precession would again read as generic wobble.  The two same-material axes are the
# equatorial ones, which whirl too fast to identify individually anyway.
_ARMS: Tuple[Tuple[Tuple[int, int, int], str], ...] = (
    ((1, 0, 0), "gold"), ((-1, 0, 0), "gold"),        # equatorial axis, both gold
    ((0, 0, 1), "glass"), ((0, 0, -1), "glass"),      # equatorial axis, both glass
    ((0, 1, 0), "gold"), ((0, -1, 0), "glass"),       # spin axis, one of each
)


class Jack(Element):
    """Six ball-tipped arms on the body half-axes, tumbling, carved by a static gyroid.

    Emits **two** ``isosurface`` blocks — one per material — because an ftrace
    isosurface carries exactly one material, and folding each material's three arms
    into a single ``union`` means one sphere-trace instead of six.

    ``spin`` / ``precess`` are whole turns per loop (integers keep the loop seamless);
    ``tilt`` is the half-angle of the precession cone in degrees.  There is no body
    pre-rotation: the spin axis is exactly the ±y arm pair, which is what makes the
    precession legible (see the module docstring).

    ``gyroid_glass=False`` renders the glass arms as plain solid ball-and-rod — the
    fallback if intersecting a *dielectric* with a marched field proves too slow.

    **The pose is a Signal graph, and its three angles carry named**
    :class:`~loom.anim.Slot`\\ **s**, so the tumble is drivable from the curve editor
    without editing this file:

    ``jack_spin`` / ``jack_precess``
        Extra spin / precession **in degrees**, authored at 0 and *added* to the
        constant ``spin``/``precess`` rate terms.  As pure offsets they re-time the
        tumble against anything else in the scene without disturbing the whole-turn
        rates that keep the loop seamless.
    ``jack_lean``
        The half-angle of the precession cone, ``tilt`` by default.  Unlike the two
        phases this one changes the jack's *shape in space*, so the standing height has
        to follow it — pass the same signal through :func:`rest_height` and the jack
        stays exactly on the floor at every value (that is what ``tilt=lean_signal(…)``
        plus ``centre=(0, rest_height(Y0, tilt=that_same_signal), 0)`` is for, and
        :mod:`pastel_jack` does it).  Clamped to ``[LEAN_MIN, LEAN_MAX]``.

    ``centre`` accepts plain floats (the ordinary static case) *or* Signals per
    component; ``tilt`` likewise takes a float — wrapped into a fresh ``jack_lean``
    slot — or an existing Signal, which is how two elements share one lean.  Because a
    slot's value is pushed in from outside the clock, everything derived from one is
    resolved **per frame** in :meth:`emit` rather than cached on the instance.
    """

    def __init__(self, *, arm: float = ARM, ball: float = BALL, rod: float = ROD,
                 spin: float = 1.0, precess: float = 1.0,
                 tilt: Union[float, Signal] = TILT,
                 freq: float = FREQ, solid: float = SOLID, fill: float = FILL,
                 counter: float = COUNTER,
                 centre: Tuple[Union[float, Signal], ...] = (0.0, 0.0, 0.0),
                 gold: str = "gold", glass: str = "glass",
                 gyroid_glass: bool = True, name: str = "jack") -> None:
        self.centre = tuple(as_signal(c) for c in centre)
        self.arm = float(arm)
        self.ball = float(ball)
        self.rod = float(rod)
        self.fill = float(fill)
        self.spin = float(spin)
        self.precess = float(precess)
        self.freq = float(freq)
        self.solid = float(solid)
        self.counter = float(counter)
        self.gold = gold
        self.glass = glass
        self.gyroid_glass = bool(gyroid_glass)
        self.name = name
        # ---- the pose graph -------------------------------------------------
        # A float `tilt` gets its own slot; a Signal is taken as-is so a caller can hand
        # the SAME lean to the jack and to whatever else must track it.
        self.tilt = lean_signal(tilt) if not isinstance(tilt, Signal) else tilt
        self.spin_phase = Slot("jack_spin", 0.0)
        self.precess_phase = Slot("jack_precess", 0.0)
        # The offsets are ADDED to the rate terms rather than folded inside them, which
        # is deliberate: `(2*pi*rate)*t + (pi/180)*0.0` is bit-for-bit the closed form
        # this replaced, so switching to Signals is provably a refactor and not a
        # re-derivation. (Verified: all 432 frames of `pastel_jack` emit byte-identical.)
        self.spin_angle = _TAU * self.spin * Phase() + _RAD * self.spin_phase
        self.precess_angle = _TAU * self.precess * Phase() + _RAD * self.precess_phase

    def roots(self) -> List:
        # Every animatable value-site, so `detect_signal_cycle` can walk the graph and
        # `collect_slots` can find `jack_spin` / `jack_precess` / `jack_lean` by name.
        return [self.spin_angle, self.precess_angle, self.tilt, *self.centre]

    @property
    def reach(self) -> float:
        """Radius of the smallest origin-centred sphere containing the jack."""
        return self.arm + self.ball

    def pose(self, spin_rad: float, precess_rad: float, tilt_deg: float) -> Mat3:
        """Body->world rotation from three resolved angles.

        ``Ry(precession) · Rz(tilt) · Ry(spin)``, read right to left: the jack spins
        about its own ±y arm pair, that axis is laid over by ``tilt``, and the
        precession walks the lean around world +y.

        Because ``Ry(spin)`` fixes ŷ, the axis direction reduces to
        ``Ry(2π·precess·t) · Rz(tilt) · ŷ`` = ``(−sin tilt·cos θ, cos tilt, sin tilt·sin θ)``
        — independent of the spin, i.e. an exact cone of half-angle ``tilt`` about
        world +y.  That is what puts the two axis balls on antiphase circles of radius
        ``arm·sin(tilt)`` at heights ``±arm·cos(tilt)``.
        """
        return _mm(_ry(precess_rad),
                   _mm(_rz(math.radians(tilt_deg)),
                       _ry(spin_rad)))

    def gyroid_fields(self) -> List[str]:
        """The base carve: the ``function`` field(s) whose INTERSECTION is the solid.

        Renormalised to ~unit Lipschitz (see the module docstring), and returned as a
        list because the interesting case needs **two** leaves.

        With ``fill = 0`` this is the single field ``g``, whose solid is the half-space
        ``g <= 0``.  That is *not* the lacy look: intersecting a ball with it keeps about
        half of the ball's own smooth outer surface, so the part reads as a solid golf
        ball with dimples no matter how high ``freq`` goes.

        With ``fill > 0`` it is the **shell** ``|g| <= w``, i.e. the pair
        ``[g - w, -g - w]`` whose intersection is a slab hugging the zero level set.
        Now the ball's outer surface survives only as a narrow rim where the slab
        crosses it, and what's left is see-through gyroid filigree — the same surface the
        plain gyroid examples show, just clipped to the arm.

        ``fill`` is the **fraction of space the slab occupies**, which is the only
        scale-free way to ask for this.  Thickness-in-metres is a trap: the gyroid
        surface is so convoluted (area ~3.09/period per unit volume) that a slab merely
        13.7% of a cell thick already fills 43% of space — visually indistinguishable
        from the 50% half-space, which is exactly how a first attempt at this shell
        failed.  Measured (400k samples, and frequency-independent because the
        distribution of ``g`` does not depend on ``freq``):

            volfrac(|g_raw| <= g0) ~ 0.647 * g0   for g0 up to ~0.3
            |grad g_raw| ~ 1.529 * freq           near the zero set

        so ``g0 = fill / 0.647`` and the field-unit half-width is ``w = g0 / (2*freq)``.
        The implied world thickness ``2*g0 / (1.529*freq)`` is reported by
        :meth:`shell_report` — it matters only because it must stay a few pixels wide.
        """
        s = self._scale()
        raw = self._raw()
        c = self.level()
        inner = raw if c == 0.0 else f"{raw}-({fmt(c)})"
        core = f"({inner})*{fmt(s)}"
        if self.fill <= 0.0:
            return [core]
        # A shell of volume fraction `fill` straddling level c, again by quantile rather
        # than by any linear width formula.
        p = gyroid_cdf(c)
        lo = gyroid_quantile(p - 0.5 * self.fill)
        hi = gyroid_quantile(p + 0.5 * self.fill)
        return [f"({raw}-({fmt(hi)}))*{fmt(s)}", f"({fmt(lo)}-({raw}))*{fmt(s)}"]

    def counter_field(self) -> Optional[str]:
        """The counter-network ``g >= t``, or ``None`` when ``counter`` is off.

        The gyroid has two interpenetrating labyrinths, and the base carve ``g <= c`` is
        drawn entirely from one of them.  Its voids are therefore ``g > c`` — and the
        *other* labyrinth's core, the high-field region ``g >= t``, threads right down the
        middle of exactly those voids.  Unioning it in spends volume where the sight-lines
        are instead of where the surface is, which is the whole trick: see-through halves
        for ~2 points of envelope, where buying the same reduction out of ``solid`` would
        cost ~12 (module docstring for the measurements).

        ``counter`` is the volume fraction the network occupies, so the level is the
        ``1 - counter`` quantile.  (The gyroid is odd under point inversion, so this is
        just ``-gyroid_quantile(counter)``; it is written the direct way so it stays
        correct without leaning on that symmetry.)
        """
        if self.counter <= 0.0:
            return None
        t = self.counter_level()
        return f"(({fmt(t)})-({self._raw()}))*{fmt(self._scale())}"

    def fused_field(self) -> Optional[str]:
        """Base carve UNION counter-network as ONE leaf, evaluating the gyroid once.

        The obvious emission is a ``union`` of two ``function`` leaves, and it is correct —
        but ftrace evaluates every leaf at every sphere-trace step, so it costs *twice* the
        trigonometry (12 sin/cos per step instead of 6) on the hottest loop in the scene.
        The two leaves are ``a = (g-c)*s`` and ``b = (t-g)*s`` and their union is
        ``min(a, b)``, so use the identity

            min(a, b) = ((a + b) - |a - b|) / 2

        Here ``a + b = (t - c)*s`` is a **constant** — the ``g`` terms cancel — and
        ``a - b = (2g - c - t)*s``, so ``g`` survives in exactly one place:

            ((t - c) - |2g - c - t|) * s/2

        which is one leaf and six trig calls.  Exact, not an approximation: verified to
        agree with the two-leaf union sign-for-sign over 2M samples.

        The Lipschitz bound is unchanged, which is what keeps ``max_gradient 2`` honest:
        d/dg of the above is -/+ s, the same as the plain leaf's, so |grad| <= sqrt(3)
        still.

        Only the ``fill = 0`` case fuses.  A shell is ``max`` of two leaves that each carry
        their own ``g``, so unioning the counter onto it leaves ``g`` in two places whatever
        the algebra; that path stays an explicit tree.
        """
        if self.counter <= 0.0 or self.fill > 0.0:
            return None
        c, t, s = self.level(), self.counter_level(), self._scale()
        return (f"({fmt(t - c)}-abs(2*({self._raw()})-({fmt(c + t)})))"
                f"*{fmt(0.5 * s)}")

    def carve_lines(self, ind: str) -> List[str]:
        """The carving field as ONE field-tree node, indented by ``ind``.

        Base alone is a bare ``function``; a shell is an ``intersect`` of two; adding the
        counter-network unions one on.  Emitting a single node (rather than leaning on
        ``max`` being associative to splice extra children into the caller's ``intersect``)
        is what lets the counter-network be a *union* here — the two operations do not
        commute, so the nesting has to be explicit.
        """
        fused = self.fused_field()
        if fused is not None:
            return [f'{ind}function {{ expr "{fused}" }}']
        base = self.gyroid_fields()
        if len(base) == 1:
            node = [f'{ind}function {{ expr "{base[0]}" }}']
        else:
            node = ([f'{ind}intersect {{']
                    + [f'{ind}    function {{ expr "{e}" }}' for e in base]
                    + [f'{ind}}}'])
        cf = self.counter_field()
        if cf is None:
            return node
        return ([f'{ind}union {{']
                + ['    ' + s for s in node]
                + [f'{ind}    function {{ expr "{cf}" }}',
                   f'{ind}}}'])

    def _raw(self) -> str:
        f = self.freq
        return (f"sin({fmt(f)}*x)*cos({fmt(f)}*y)"
                f"+sin({fmt(f)}*y)*cos({fmt(f)}*z)"
                f"+sin({fmt(f)}*z)*cos({fmt(f)}*x)")

    def _scale(self) -> float:
        """Lipschitz renormaliser: |grad g_raw| <= 2*sqrt(3)*freq, so divide by 2*freq."""
        return 1.0 / (2.0 * self.freq)

    # |grad g_raw| ~ 1.529*freq near the zero set (measured); used only to report a wall
    # thickness in metres, never to pick a level.
    GRAD_PER_FREQ = 1.529

    def level(self) -> float:
        """The raw-gyroid level ``c`` whose solid ``g <= c`` has volume fraction ``solid``."""
        return gyroid_quantile(self.solid)

    def counter_level(self) -> float:
        """The level ``t`` whose solid ``g >= t`` has volume fraction ``counter``."""
        return gyroid_quantile(1.0 - self.counter)

    def shell_report(self) -> str:
        """One line of human-checkable geometry, for tuning by eye against numbers."""
        period = 2.0 * math.pi / self.freq
        out = (f"freq={self.freq:g} period={period:.4f} solid={self.solid:g} "
               f"level={self.level():+.3f}  "
               f"cells across ball={self.ball * self.freq / math.pi:.2f} "
               f"rod={self.rod * self.freq / math.pi:.2f}")
        if self.counter > 0.0:
            out += (f"  COUNTER={self.counter:g} t={self.counter_level():+.3f}"
                    f" total={self.solid + self.counter:.2f}")
        if self.fill > 0.0:
            p = gyroid_cdf(self.level())
            g0 = 0.5 * (gyroid_quantile(p + 0.5 * self.fill)
                        - gyroid_quantile(p - 0.5 * self.fill))
            thick = 2.0 * g0 / (self.GRAD_PER_FREQ * self.freq)
            out += (f"  SHELL fill={self.fill:g} wall={thick:.4f}"
                    f" ({thick / period * 100:.1f}% of a cell)")
        return out

    def _arm_leaves(self, M: Mat3, o: Tuple[float, float, float], kind: str) -> List[str]:
        out: List[str] = []
        for d_body, k in _ARMS:
            if k != kind:
                continue
            d = _mv(M, d_body)
            tip = tuple(self.arm * c + o[i] for i, c in enumerate(d))
            mid = tuple(0.5 * self.arm * c + o[i] for i, c in enumerate(d))
            rx, ry, rz = aim_y_euler(d)
            out.append(f'            sphere {{ center {fmt3(tip)}  '
                       f'radius {fmt(self.ball)} }}')
            out.append(f'            cylinder {{ translate {fmt3(mid)}  '
                       f'rotate {fmt(rx)} {fmt(ry)} {fmt(rz)}  '
                       f'radius {fmt(self.rod)}  height {fmt(self.arm)} }}')
        return out

    def _block(self, M: Mat3, o: Tuple[float, float, float], kind: str,
               material: str, carve: bool) -> str:
        leaves = self._arm_leaves(M, o, kind)
        lines = [f'{self.name}_{kind} = isosurface {{',
                 f'    material "{material}"']
        if carve:
            lines.append('    intersect {')
            lines.append('        union {')
            lines.extend('    ' + s for s in leaves)
            lines.append('        }')
            lines.extend(self.carve_lines('        '))
            lines.append('    }')
            # A `function` field always needs a container; a sphere on the jack's own
            # centre is pose-independent (the jack only ever rotates about it), so it
            # never has to be recomputed as the jack turns.  Note the container moves
            # with the jack but the `function` leaf does NOT — the field stays in world
            # space, which is the whole point.
            lines.append(f'    contained_by {{ sphere {{ center {fmt3(o)}  '
                         f'radius {fmt(self.reach)} }} }}')
            # |grad| of the renormalised gyroid is <= sqrt(3); the SDF partner is
            # unit-Lipschitz, and max(.,.) cannot exceed either bound.
            lines.append('    max_gradient 2')
        else:
            lines.append('    union {')
            lines.extend(leaves)
            lines.append('    }')
        lines.append('}')
        return "\n".join(lines)

    def emit(self, ctx: EmitCtx) -> str:
        # Resolve every animatable value ONCE per frame, through the emit cache, then
        # build the pose from plain floats — the matrix maths below has no business
        # knowing about the graph.
        M = self.pose(self.spin_angle.at(ctx.clock, ctx.cache),
                      self.precess_angle.at(ctx.clock, ctx.cache),
                      self.tilt.at(ctx.clock, ctx.cache))
        o = tuple(c.at(ctx.clock, ctx.cache) for c in self.centre)
        return "\n".join([self._block(M, o, "gold", self.gold, True),
                          self._block(M, o, "glass", self.glass, self.gyroid_glass)])


def lean_signal(default: float = TILT, name: str = "jack_lean") -> Signal:
    """A clamped, named :class:`~loom.anim.Slot` for the precession half-angle.

    Split out of :class:`Jack` so a caller can build the lean *before* the jack and hand
    the same node to both the pose and :func:`rest_height` — one channel, two consumers,
    which is what keeps a driven lean standing on the floor instead of hovering over it.
    """
    return Clamp(Slot(name, float(default)), LEAN_MIN, LEAN_MAX)


def rest_height(floor_y: float, *, arm: float = ARM, ball: float = BALL,
                tilt: Union[float, Signal] = TILT) -> Union[float, Signal]:
    """Centre height that stands the jack exactly on ``floor_y`` — a closed form.

    Pass a **Signal** ``tilt`` (from :func:`lean_signal`) and you get a Signal back, with
    the same expression built out of graph nodes: the height then tracks a driven lean
    frame by frame, so the jack keeps touching down no matter what the channel does.  A
    float ``tilt`` returns a float, unchanged, which is the ordinary static case.

    The jack's lowest point is not something to search for numerically; it falls out of
    the pose.  Of the six arms, the ±y pair *is* the spin axis, so those two ball centres
    sit at constant heights ``±arm·cos(tilt)`` (they ride circles, which is the whole
    precession story), while the four equatorial arms are perpendicular to that axis and
    so their ball centres only ever reach ``±arm·sin(tilt)``.  For any ``tilt < 45°``,
    ``cos(tilt) > sin(tilt)``, so the **bottom axis ball is always the lowest part of the
    jack** — and, being at a constant height, it is lowest by the same amount at every
    instant of the loop.  Hence

        depth below centre = arm·cos(tilt) + ball

    with no dependence on ``t`` at all.  (The rods never compete: an arm of direction
    ``d`` bottoms out at ``min(0, arm·d_y) − rod·√(1−d_y²)``, which for these proportions
    is 1.00 for the axis rod and 0.84 for an equatorial one, against the ball's 1.34.)

    Two things worth stating because they are easy to get wrong:

    * **The carve does not raise this.** One might expect the gyroid to shave the very
      bottom off the ball and leave the jack visually hovering, and worse, the field is
      world-static so the answer would depend on the height being solved for — a fixed
      point rather than a formula.  It does not, because the ball's lowest point traces a
      circle of radius ``arm·sin(tilt)`` through the field, ≈8.6 gyroid cells around per
      loop, so it samples the field's whole range many times over.  Measured at the height
      returned here: that point is solid material in 103 of 432 frames at ``solid=0.24``
      (58 at 0.16, 143 at 0.30).  The jack really does touch down, repeatedly.
    * **Touching at every instant is impossible** for a rigid tumbling jack — only a
      sphere could manage that.  "On the floor" therefore means tangent at its lowest
      moments and never penetrating, which is exactly what this height gives, and here the
      lowest moment happens to recur throughout the loop rather than once.
    """
    if isinstance(tilt, Signal):
        # Same expression, same association order, so the default value of the slot
        # reproduces the float branch bit for bit.
        return floor_y + arm * Cos(_RAD * tilt) + ball
    return floor_y + arm * math.cos(math.radians(tilt)) + ball


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

# Mode-W render settings shared by the still and the sequence.  `-ev` is a
# *compensation multiplier* on the absolute sensor gain (not a stop count), and it is
# fixed here rather than auto-anchored so every frame develops identically.
#
# `-spp 8` is not about noise — mode W is deterministic at any spp — but about the
# gyroid-carved *glass*: a labyrinth of dielectric facets sends each of the 8 hero
# wavelengths somewhere different, so at 1-2 spp the crystal reads as saturated
# confetti and only settles into dispersion-coloured facets once the pixel averages
# several sub-positions.
#
# `-gi-clamp 0.15` is the firefly ceiling on one gather ray — a jack made of dielectric
# filigree is exactly the case it exists for.  It stays well above `-ambient 0.05`,
# because the clamp also caps the far-field tail an escaping gather ray returns.
WHITTED = ["-spp", "8", "-gi", "24", "-ambient", "0.05", "-gi-clamp", "0.15",
           "-whitted-grid", "3", "-ev", "11"]

# 432 frames at 60 fps = a 7.2 s loop carrying ONE turn of spin and one of precession:
# 0.8333 deg per frame, ~50 deg/s.  The frame count is set by *angular resolution*, not by
# duration — the gyroid-carved glass throws a different refraction pattern at every
# orientation, so too few frames per degree turns continuous caustic motion into a
# flickering slideshow.  90 frames (4 deg/frame) read that way badly and 180 (2 deg) still
# a little; at 60 fps the motion is finally continuous.  The loop duration is held at 7.2 s
# throughout so the jack's actual rotation *speed* never changes — only how finely it is
# sampled.
#
# The GIF cannot be 60 fps: a GIF frame delay is an integer number of centiseconds, so the
# only exact rates are divisors of 100 and 60 is not one (100/60 = 1.667 cs, which players
# silently round to 2 cs = 50 fps, retiming the loop by 20%).  So the GIF is built from
# every GIF_STRIDE-th frame at GIF_FPS, chosen to land on the identical 7.2 s: 432/3 = 144
# frames at 20 fps, and 20 divides 100 exactly.  Same motion, same duration, same seam —
# just the coarser sampling the format can actually represent.  The MP4 is the 60 fps one.
FRAMES = 432
FPS = 60
GIF_STRIDE = 3
GIF_FPS = 20
# The GIF is also the README embed, and GitHub is the only viewer that matters for it, so
# it is encoded with gifski rather than ffmpeg's palettegen (measurably smaller per unit of
# quality on smooth shading — 2.7 MB vs 5.3 MB on this clip) and downscaled: a 480² GIF is
# ~4x the bytes for detail the README column width throws away anyway.
GIF_WIDTH = 320
GIF_QUALITY = 65

# Closed room, camera inside it (so glass has a whole interior to refract).  It is
# deep in +z because the camera stands *inside* and still has to clear the jack's
# 1.4 m reach at a flattering focal length.
X0, X1 = -3.4, 3.4
Y0, Y1 = -2.2, 2.6
Z0, Z1 = -3.4, 6.6
WX, WY, WZ = X1 - X0, Y1 - Y0, Z1 - Z0


def _room() -> List[Raw]:
    q = lambda o, u, v, m: Raw(  # noqa: E731
        f'quad {{ origin {fmt3(o)}  u {fmt3(u)}  v {fmt3(v)}  material "{m}" }}')
    return [
        q((X0, Y0, Z0), (WX, 0, 0), (0, 0, WZ), "floor"),    # floor
        q((X0, Y1, Z0), (WX, 0, 0), (0, 0, WZ), "ceil"),     # ceiling
        q((X0, Y0, Z0), (WX, 0, 0), (0, WY, 0), "wall"),     # back
        q((X0, Y0, Z1), (WX, 0, 0), (0, WY, 0), "wall"),     # front (behind camera)
        q((X0, Y0, Z0), (0, 0, WZ), (0, WY, 0), "left"),     # left
        q((X1, Y0, Z0), (0, 0, WZ), (0, WY, 0), "right"),    # right
    ]


def build_scene(res=(480, 480), *, gyroid_glass: bool = True, spin: float = 1.0,
                freq: float = FREQ, solid: float = SOLID, counter: float = COUNTER,
                ball: float = BALL, rod: float = ROD, fill: float = FILL) -> Scene:
    # Stand the jack on the floor rather than floating it at the origin.
    cy = rest_height(Y0, arm=ARM, ball=BALL, tilt=TILT)
    # The camera framing was tuned around a jack at the origin, so drop the eye and the
    # aim point by the same amount the jack dropped.  That preserves the composition
    # exactly (same relative geometry, same slight downward tilt) instead of re-tuning it,
    # and the eye stays 2.04 m above the floor, well inside the room.
    scene = Scene(Camera(eye=(0.5, 0.7 + cy, 5.5), look_at=(0.0, 0.05 + cy, 0.0),
                         up=(0, 1, 0), fov_y=38, mode="W", res=res))
    scene.add(
        # `preset gold` / `preset glass:SF10` expand to exactly these (src/materials.h).
        Material("gold", "glossy", reflect="metal:gold", roughness=0.05),
        Material("glass", "dielectric", ior="glass:SF10"),
        Material("floor", "diffuse", reflect=0.30),
        Material("ceil", "diffuse", reflect=0.78),
        Material("wall", "diffuse", reflect=0.58),
        # Gentle warm/cool side walls rather than Cornell's saturated red/green: the
        # `-gi` gather really does bleed a wall's colour onto the metal, and at full
        # Cornell saturation that reads as green camouflage on the gold instead of
        # modelling it.
        Material("left", "diffuse", reflect="rgb 0.52 0.30 0.26"),
        Material("right", "diffuse", reflect="rgb 0.28 0.40 0.52"),
        Jack(gyroid_glass=gyroid_glass, spin=spin, freq=freq, solid=solid,
             counter=counter, ball=ball, rod=rod, fill=fill, centre=(0.0, cy, 0.0)),
        *_room(),
        # `lumens` pins the exposure (absolute mode) so the loop cannot pump — see
        # the module docstring.
        Light("area", origin=f"-1.6 {fmt(Y1 - 0.02)} -1.6", u="3.2 0 0", v="0 0 3.2",
              normal="0 -1 0", spd="preset:bb6500", lumens=60000),
    )
    return scene


def _opt(flag: str, default: float) -> float:
    return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _sopt(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    still = "--still" in sys.argv
    render = "--render" in sys.argv
    plain = "--plain-glass" in sys.argv
    r = int(_opt("--res", 480))
    scene = build_scene(res=(r, r), gyroid_glass=not plain,
                        freq=_opt("--freq", FREQ), solid=_opt("--solid", SOLID),
                        counter=_opt("--counter", COUNTER),
                        ball=_opt("--ball", BALL), rod=_opt("--rod", ROD),
                        fill=_opt("--fill", FILL))

    if still:
        from loom.drive import render_still
        render_still(scene, t=_opt("--t", 0.0), name=_sopt("--name", "jumping_jack"),
                     n=1, interval=8.0, extra_args=WHITTED)
        return 0
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, FRAMES), Cache()))
        return 0

    from loom import render_range
    from loom.drive import assemble_gif_gifski, assemble_mp4, default_outdir
    pngs = render_range(scene, FRAMES, name="jumping_jack", fps=FPS, n=1,
                        interval=8.0, skip_existing=True, extra_args=WHITTED)
    out = default_outdir("jumping_jack")
    # The MP4 is the deliverable: full 60 fps, every frame, and about 5x smaller than the
    # GIF because it is not restricted to a 256-entry palette.
    assemble_mp4(pngs, out / "jumping_jack.mp4", fps=FPS)
    # The GIF is the compatibility copy — see the GIF_FPS comment for why it cannot be 60.
    assemble_gif_gifski(pngs[::GIF_STRIDE], out / "jumping_jack.gif", fps=GIF_FPS,
                        width=GIF_WIDTH, quality=GIF_QUALITY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
