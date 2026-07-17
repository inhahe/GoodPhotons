"""
Loom tool: **higher-dimensional gyroid slices** (`gyroid_nd.py`).

The Schoen gyroid is the level set of

    G(x,y,z) = sin x cos y + sin y cos z + sin z cos x
             = sum over the cyclic pairs of (x,y,z) of  sin(u_i) * cos(u_j)

i.e. a sum, over consecutive coordinate axes taken cyclically, of
``sin(this axis) * cos(next axis)``.  That structure generalizes verbatim to *N*
dimensions: keep the same cyclic ``sin*cos`` sum but over an arbitrary ordered set
of axes.  ftrace only evaluates ``f(x,y,z)``, so a higher dimension is realized as
an extra **plane-wave direction** — a generic unit direction in the rendered 3-space
(the projection of that N-D axis into the 3-D slice we actually see).  With D = 3 and
all three axes at harmonic 1 this reproduces the ordinary gyroid *exactly*.

Each **dimension** has:
  * a **direction** in (x,y,z) — the corresponding row of the N x 3 slice-embedding
    matrix.  By default (``--pin-axes``) dims 0/1/2 are pinned to the x/y/z axes so the
    slice always contains the ordinary xyz volume, and dims >= 3 get a generic random
    unit direction (a higher-D axis seen edge-on in the slice).  With ``--no-pin-axes``
    *every* dimension gets a random direction — a freely-oriented N-D slice, so even the
    base gyroid is rendered tilted.  (Each dim's random **phase** is that axis's offset,
    i.e. where the slice sits along it.)
  * an **oscillate** flag — an inert dimension contributes no term (the surface is
    invariant along it);
  * a **harmonic** — a positive-integer spatial-frequency multiplier.  The **main**
    dimension (the lowest-indexed oscillating one) is the fundamental (harmonic 1);
    a dimension is a *harmonic of the main* when its harmonic is an integer >= 2 (an
    overtone).  This is what "make some of them harmonics of others" means here.
  * a random **phase**.

Each dimension's argument is  ``u_d = harmonic_d * freq * (dir_d . (x,y,z)) + phase_d``
and the emitted field is the cyclic sum ``sum_i sin(u_{o_i}) * cos(u_{o_{i+1}})`` over
the oscillating dims ``o_0 < o_1 < ...`` (indices taken mod the oscillating count).

The rendered subject is just the gyroid from ``scenes/showcase.ftsl`` — a **thickened
gyroid sheet** (``abs(g) - t``), CSG-clipped to a ball — on its own, with no Cornell box or
glass sphere.  ``--material`` picks its surface: **gold** (default) — a conductor/mirror, so
it shows whatever surrounds it; or **glass** — a clear BK7 dielectric, where the lattice reads
through refraction and internal reflection instead.  Either way a flat uniform light would
wash the lattice out, so the subject is lit like a product shot by a procedural **studio
environment** — a dark neutral base plus a few bright soft "softbox" lights, written once as
an equirectangular ``studio_env.pfm`` beside the outputs and fed to ftrace's image-based
``light env`` — so the surface picks up crisp highlights that trace every facet while deep
shadows give depth, over a clean neutral background.  (The full gold/glass look only develops
under path tracing, ``--no-raster``; the fast rasterizer previews the geometry flat-shaded, and
for a clear ``--material`` renders it *see-through* — dimmed + milky-hazed via ftrace's
``-see-through`` pass, no refraction — so glass reads as glass in the preview too.)

This script **randomly picks** the field parameters.  For each of ``--count N`` variants it
**renders a seamless morphing video** in which the higher dimensions move the visible slice.
Every non-main oscillating dimension gets an integer rate (a "winding"); the ``--transform``
choice decides what that motion *is*, and both start from the exact current gyroid at frame 0
and loop seamlessly:

  * ``drift`` (default) — advance each dim's phase a whole number of cycles over the loop,
    translating the slice *through* that dimension.  The pattern slides.
  * ``rotate`` — rotate each dim's wavevector out of the 3-D slice into its own hidden axis
    (a whole number of turns).  The in-slice frequency waxes and wanes as the wave turns
    edge-on and back, so the lattice genuinely reshapes — the higher-D analogue of *turning*
    the object rather than sliding it.
  * ``bloom`` — pin frame 0 (and frame 1) to the *exact classic gyroid* from
    ``scenes/showcase.ftsl`` (``sin(f x)cos(f y) + sin(f y)cos(f z) + sin(f z)cos(f x)``),
    then swell one or more of its scalar **parameters** out and back with an envelope
    ``w = sin^2(pi t)`` (0 at the loop ends, 1 at the midpoint), so the clip always opens
    and closes on the recognizable showcase gyroid.  ``--bloom`` picks *which* parameter(s)
    bloom (comma-separated, default ``dims``):

      - ``dims`` — cross-blend the full N-D gyroid in and back out; the lattice *unfolds*
        into higher-D structure at the midpoint (the original bloom).
      - ``freq`` (aliases ``complexity``/``intricacy``) — hold the classic gyroid but pulse
        its spatial frequency up at mid-loop, so the pattern gets finer/more intricate and
        relaxes back.
      - ``threshold`` — swell the level-set value, breathing the surface off its zero set.
      - ``thickness`` — swell the sheet's half-thickness so the walls fatten and thin.

    Parameters combine (e.g. ``--bloom dims,freq``), and ``--bloom-amp`` scales every
    chosen parameter's peak swing (default 1).  The base frequency defaults to the showcase
    density (freq 40 at radius 0.32) unless ``--freq`` is given.

The assembled animated ``.gif`` (or ``.mp4`` via ``--format mp4``) and a ``.txt`` listing
every chosen value collect together in the output directory.  By default each run gets its
own fresh ``runNNN/`` subdirectory (``run001``, ``run002``, ... one past the highest existing,
never reusing a number) so successive runs never overwrite each other; pass ``--no-run-subdir``
to write straight into the base output directory instead.  Each variant's per-frame
``.ftsl``/``.png`` files live in their own subdir ``<outdir>/<variant>/``.  Frames render
with ftrace's fast headless rasterizer by default.  Use ``--no-video`` to instead emit a
single static ``.ftsl`` per variant (with a full comment header).  Any choice can be
**locked** from the CLI (see ``--help``): the dimension count, how many dims oscillate, how
many are harmonics of the main, the base frequency, and — per axis — whether it oscillates
and at what harmonic.

Examples::

    # 10 fully random variants, each a morphing video, into png/gyroid_nd/<variant>/
    python examples/gyroid_nd.py --count 10

    # reproducible; lock 6 dims, 4 oscillating, 2 of them harmonics of the main
    python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 --oscillating 4 --harmonics 2

    # start on the exact showcase gyroid, then bloom into higher-D structure and back
    python examples/gyroid_nd.py --dims 6 --transform bloom

    # stay the classic gyroid but pulse its complexity (frequency) up at mid-loop
    python examples/gyroid_nd.py --transform bloom --bloom complexity

    # bloom the higher-D unfold *and* an extra-intense frequency pulse together
    python examples/gyroid_nd.py --dims 6 --transform bloom --bloom dims,freq --bloom-amp 1.5

    # the classic gyroid, animated: x,y,z on at harmonic 1, 90 frames as an mp4
    python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 --axis 2:on:1 \
        --frames 90 --format mp4

    # start from the current gyroid and rotate it through the extra dimensions
    python examples/gyroid_nd.py --dims 6 --transform rotate

    # render the lattice as clear glass instead of gold (path-traced for real refraction)
    python examples/gyroid_nd.py --count 1 --material glass --no-raster --render-noise 3

    # watch each frame render live in one preview window (title tracks the variant)
    python examples/gyroid_nd.py --count 3 --preview

    # a 1280x720 video (preview window matches the size)
    python examples/gyroid_nd.py --count 2 --size 1280x720 --preview

    # just one static .ftsl scene file per variant, no video
    python examples/gyroid_nd.py --count 3 --no-video

    # high-quality path-traced frames instead of the rasterizer (far slower)
    python examples/gyroid_nd.py --count 1 --no-raster --render-noise 3
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Camera, Raw  # noqa: E402
from loom.ftsl_emit import fmt  # noqa: E402


# ---------------------------------------------------------------------------
# per-variant data
# ---------------------------------------------------------------------------

@dataclass
class Dim:
    index: int
    oscillate: bool
    harmonic: int                       # spatial-frequency multiplier (>=1)
    direction: Tuple[float, float, float]
    phase: float
    role: str                           # main | harmonic | independent | inert
    winding: int = 0                    # animation rate: integer cycles/turns over one
    #                                     video loop (0 = fixed; the main dim anchors at 0,
    #                                     others move).  In the ``drift`` transform this is
    #                                     the phase-advance count (slice translates through
    #                                     the dimension); in ``rotate`` it is the number of
    #                                     full turns the dim's wavevector rotates out of the
    #                                     3-D slice into its hidden axis.
    hidden_offset: float = 0.0          # rotate transform only: the slice's offset along
    #                                     this dim's hidden axis, i.e. the phase the wave
    #                                     picks up once it has rotated fully out of view


@dataclass
class Variant:
    seed: int
    dims: int
    freq: float
    threshold: float
    thickness: float = 0.5              # half-width of the thickened gold sheet
    pinned: bool = True                 # True: dims 0/1/2 pinned to world X/Y/Z (the slice
    #                                     contains the xyz volume); False: every dim gets a
    #                                     random direction (a freely-oriented N-D slice)
    bloom_params: Tuple[str, ...] = ()  # bloom transform only: which scalar parameters
    #                                     oscillate over the loop (subset of BLOOM_PARAMS:
    #                                     'dims' = crossfade in the full N-D field, 'freq' =
    #                                     pulse the spatial frequency / complexity, 'threshold'
    #                                     = shift the level set, 'thickness' = swell the sheet).
    #                                     Empty for the drift/rotate transforms.
    bloom_amp: float = 1.0              # scales every bloom parameter's peak swing
    dim_list: List[Dim] = dc_field(default_factory=list)

    @property
    def oscillating(self) -> List[int]:
        return [d.index for d in self.dim_list if d.oscillate]

    @property
    def main(self) -> Optional[int]:
        osc = self.oscillating
        return min(osc) if osc else None

    @property
    def harmonic_dims(self) -> List[int]:
        m = self.main
        return [d.index for d in self.dim_list
                if d.oscillate and d.index != m and d.harmonic > 1]


# ---------------------------------------------------------------------------
# axis-lock parsing:  d:on | d:off | d:on:h
# ---------------------------------------------------------------------------

@dataclass
class AxisLock:
    on: Optional[bool] = None
    harmonic: Optional[int] = None


def parse_axis_lock(spec: str, locks: Dict[int, AxisLock]) -> None:
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--axis '{spec}': expected INDEX:on|off[:HARMONIC] (e.g. 4:on:3)")
    try:
        idx = int(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(f"--axis '{spec}': axis index must be an integer")
    if idx < 0:
        raise argparse.ArgumentTypeError(f"--axis '{spec}': axis index must be >= 0")
    state = parts[1].strip().lower()
    if state in ("on", "osc", "oscillate", "true", "1", "yes"):
        on = True
    elif state in ("off", "no", "false", "0", "static", "inert"):
        on = False
    else:
        raise argparse.ArgumentTypeError(
            f"--axis '{spec}': state must be 'on' or 'off', got '{parts[1]}'")
    harmonic = None
    if len(parts) >= 3 and parts[2] != "":
        try:
            harmonic = int(parts[2])
        except ValueError:
            raise argparse.ArgumentTypeError(f"--axis '{spec}': harmonic must be an integer")
        if harmonic < 1:
            raise argparse.ArgumentTypeError(f"--axis '{spec}': harmonic must be >= 1")
        if not on:
            raise argparse.ArgumentTypeError(
                f"--axis '{spec}': can't set a harmonic on an 'off' axis")
    lk = locks.setdefault(idx, AxisLock())
    if lk.on is not None and lk.on != on:
        raise argparse.ArgumentTypeError(
            f"--axis: axis {idx} locked both on and off")
    lk.on = on
    if harmonic is not None:
        if lk.harmonic is not None and lk.harmonic != harmonic:
            raise argparse.ArgumentTypeError(
                f"--axis: axis {idx} locked to two different harmonics "
                f"({lk.harmonic} and {harmonic})")
        lk.harmonic = harmonic


# ---------------------------------------------------------------------------
# the picker
# ---------------------------------------------------------------------------

def _rand_unit(rng: random.Random) -> Tuple[float, float, float]:
    while True:
        v = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
        n = math.sqrt(sum(c * c for c in v))
        if n > 1e-6:
            return (v[0] / n, v[1] / n, v[2] / n)


def pick_variant(seed: int, args: argparse.Namespace,
                 axis_locks: Dict[int, AxisLock]) -> Variant:
    rng = random.Random(seed)

    transform = getattr(args, "transform", "drift")
    bloom_params = (_parse_bloom_params(getattr(args, "bloom", None))
                    if transform == "bloom" else ())
    bloom_dims = "dims" in bloom_params    # crossfade the full N-D field in (vs. only pulsing
    #                                        scalar params around the fixed classic gyroid)

    # 1) total dimension count -------------------------------------------------
    max_forced_axis = max(axis_locks) if axis_locks else -1
    if args.dims is not None:
        D = args.dims
        if max_forced_axis >= D:
            raise SystemExit(f"error: --axis references axis {max_forced_axis} but "
                             f"--dims is {D} (axis index must be < dims)")
    else:
        lo = max(args.dims_range[0], max_forced_axis + 1, 3)
        hi = max(args.dims_range[1], lo)
        D = rng.randint(lo, hi)

    # 2) forced on/off + forced harmonics -------------------------------------
    forced_on = {d for d, lk in axis_locks.items() if lk.on is True}
    forced_off = {d for d, lk in axis_locks.items() if lk.on is False}
    forced_harm = {d: lk.harmonic for d, lk in axis_locks.items()
                   if lk.harmonic is not None}
    for d in forced_harm:                       # a forced harmonic implies oscillating
        forced_on.add(d)
    conflict = forced_on & forced_off
    if conflict:
        raise SystemExit(f"error: axis {sorted(conflict)} locked both on and off")
    must_on = sorted(forced_on)
    must_off = sorted(forced_off)

    # 3) how many oscillate ----------------------------------------------------
    lo_m = max(2, len(must_on))
    hi_m = D - len(must_off)
    if hi_m < lo_m:
        raise SystemExit(f"error: cannot satisfy oscillation locks — need at least "
                         f"{lo_m} oscillating dims but only {hi_m} are available")
    if args.oscillating is not None:
        M = args.oscillating
        if M < lo_m or M > hi_m:
            raise SystemExit(f"error: --oscillating {M} is out of range [{lo_m}, {hi_m}] "
                             f"given the current locks / dims")
    elif bloom_dims:
        # bloom starts as the 3-term classic gyroid and unfolds into the full N-D
        # field, so we want that full field to be as rich as possible: every
        # available dimension oscillates (D terms >> the 3 classic terms).
        M = hi_m
    else:
        M = rng.randint(lo_m, hi_m)
    candidates = [d for d in range(D) if d not in forced_on and d not in forced_off]
    rng.shuffle(candidates)
    osc = set(must_on) | set(candidates[:max(0, M - len(must_on))])

    # 4) main + harmonics-of-main ---------------------------------------------
    main = min(osc)
    non_main = sorted(osc - {main})
    forced_gt1 = [d for d in non_main if forced_harm.get(d, 0) > 1]
    forced_one = [d for d in non_main if forced_harm.get(d, 0) == 1]
    free = [d for d in non_main if d not in forced_harm]
    lo_h = len(forced_gt1)
    hi_h = len(forced_gt1) + len(free)
    if args.harmonics is not None:
        H = args.harmonics
        if H < lo_h or H > hi_h:
            raise SystemExit(f"error: --harmonics {H} is out of range [{lo_h}, {hi_h}] "
                             f"given the current locks / oscillating dims")
    elif bloom_dims:
        # bloom keeps every extra dim at the fundamental (h1) so it unfolds into a
        # clean pure N-D gyroid; complexity comes from the dimensions, not harmonics.
        H = lo_h
    else:
        H = rng.randint(lo_h, hi_h) if hi_h >= lo_h else lo_h
    rng.shuffle(free)
    chosen_harm = set(forced_gt1) | set(free[:max(0, H - len(forced_gt1))])

    # 5) assemble every dimension ---------------------------------------------
    dims: List[Dim] = []
    for d in range(D):
        pin = getattr(args, "pin_axes", True)
        direction = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))[d] \
            if (pin and d < 3) else _rand_unit(rng)
        oscillate = d in osc
        if not oscillate:
            dims.append(Dim(d, False, 0, direction, 0.0, "inert"))
            continue
        if d in forced_harm:
            harmonic = forced_harm[d]
        elif d == main:
            harmonic = 1
        elif d in chosen_harm:
            harmonic = rng.randint(2, max(2, args.max_harmonic))
        else:
            harmonic = 1
        phase = 0.0 if args.phase0 else rng.uniform(0.0, 2.0 * math.pi)
        if d == main:
            role = "main"
        elif harmonic > 1:
            role = "harmonic"
        else:
            role = "independent"
        dims.append(Dim(d, True, harmonic, direction, phase, role))

    # Animation drift: the main dim anchors (winding 0); every other oscillating dim
    # advances its phase by an integer number of cycles over one loop, so the slice
    # translates *through* that dimension and the interference pattern morphs.  Rates
    # are distinct-ish (1,2,..,max,1,2,..) so no two dims move in lockstep.
    max_w = max(1, args.max_winding)
    non_main_osc = [d for d in sorted(osc) if d != main]
    by_index = {dm.index: dm for dm in dims}
    for i, d in enumerate(non_main_osc):
        by_index[d].winding = (i % max_w) + 1

    if args.freq is not None:
        freq = args.freq
    elif transform == "bloom":
        # Bloom's frame 0 IS the showcase gyroid; default its density to match showcase
        # (freq 40 at radius 0.32) for whatever container radius is in use.
        freq = SHOWCASE_RF / max(1e-6, args.radius)
    else:
        freq = rng.uniform(*args.freq_range)

    # Per-dim hidden-axis offset for the `rotate` transform: the slice sits this far along
    # each oscillating dim's hidden axis, so as the wavevector rotates out of the 3-D slice
    # the term picks up a real phase (not just a frequency fade) — distinct offsets keep the
    # dims from morphing in lockstep.  Drawn last so it never perturbs the drift-mode stream
    # (identical seeds still reproduce the same field for the default transform).
    for dm in dims:
        if dm.oscillate:
            dm.hidden_offset = rng.uniform(0.5, 2.0)

    return Variant(seed=seed, dims=D, freq=freq, threshold=args.threshold,
                   thickness=args.thickness, pinned=getattr(args, "pin_axes", True),
                   bloom_params=bloom_params, bloom_amp=getattr(args, "bloom_amp", 1.0),
                   dim_list=dims)


# ---------------------------------------------------------------------------
# field expression
# ---------------------------------------------------------------------------

def _arg_expr(direction: Tuple[float, float, float], coeff: float, phase: float) -> str:
    """The per-dimension argument ``coeff*(dir . (x,y,z)) + phase`` as an ftsl expression."""
    parts = []
    for c, var in zip(direction, ("x", "y", "z")):
        if abs(c) < 1e-9:
            continue
        parts.append(f"({fmt(c)})*{var}")
    lin = "+".join(parts) if parts else "0"
    if abs(phase) < 1e-9:
        return f"({fmt(coeff)}*({lin}))"
    return f"({fmt(coeff)}*({lin})+({fmt(phase)}))"


def _u_expr(dim: Dim, freq: float, phase: float) -> str:
    """The per-dimension argument u_d = harmonic*freq*(dir . (x,y,z)) + phase."""
    return _arg_expr(dim.direction, dim.harmonic * freq, phase)


def _classic_gyroid_expr(freq: float) -> str:
    """The plain 3-D Schoen gyroid on the world X/Y/Z axes at ``freq`` — exactly the
    field in ``scenes/showcase.ftsl`` (``sin(f x)cos(f y) + sin(f y)cos(f z) +
    sin(f z)cos(f x)``).  This is the fixed frame-0 subject for the ``bloom`` transform."""
    f = fmt(freq)
    ux, uy, uz = f"({f}*x)", f"({f}*y)", f"({f}*z)"
    return f"sin({ux})*cos({uy})+sin({uy})*cos({uz})+sin({uz})*cos({ux})"


# The showcase gyroid's density as a radius*frequency product (radius 0.32, freq 40):
# matching it at any container radius needs freq = SHOWCASE_RF / radius, so the bloom
# base reads as *the showcase gyroid* regardless of the ball size (freq 40 at r=0.32).
SHOWCASE_RF = 0.32 * 40.0

# Supported ways the higher dimensions animate the slice over one loop.
TRANSFORMS = ("drift", "rotate", "bloom")

# Scalar gyroid parameters the ``bloom`` transform can oscillate over the loop (each
# starts and ends at its base value, so frame 0 is always the recognizable base gyroid).
#   dims      — cross-blend the full N-D field in and back out (the original bloom)
#   freq      — pulse the spatial frequency, i.e. the pattern's intricacy / complexity
#   threshold — shift the isosurface level set (channels open and close)
#   thickness — swell and thin the gold sheet's half-width
BLOOM_PARAMS = ("dims", "freq", "threshold", "thickness")

# Each bloomable parameter's natural peak swing at the mid-loop (scaled by --bloom-amp):
#   freq      * (1 + swing*w)   -> at amp 1, w 1: 2x frequency (twice as many cells)
#   threshold + swing*w         -> shift the level set (g roughly spans +/-1.5)
#   thickness * (1 + swing*w)   -> at amp 1, w 1: 2x sheet half-width
_BLOOM_SWING = {"freq": 1.0, "threshold": 0.6, "thickness": 1.0}


def _parse_bloom_params(spec: Optional[str]) -> Tuple[str, ...]:
    """Parse a ``--bloom`` spec (comma-separated) into a validated tuple of parameters.
    Accepts the friendly aliases 'complexity'/'intricacy' for 'freq'.  Empty -> ('dims',)."""
    items = [x.strip().lower() for x in (spec or "").split(",") if x.strip()]
    out: List[str] = []
    for it in items:
        if it in ("complexity", "intricacy"):
            it = "freq"
        if it not in BLOOM_PARAMS:
            raise SystemExit(f"error: --bloom '{it}' is not one of "
                             f"{', '.join(BLOOM_PARAMS)} (alias 'complexity' -> freq)")
        if it not in out:
            out.append(it)
    return tuple(out) if out else ("dims",)


def _bloom_env(t: float) -> float:
    """The bloom envelope w(t) = sin^2(pi t): 0 at t=0 and t=1 (so both loop ends are
    *exactly* the base gyroid and the loop is seamless), 1 at the mid-loop peak."""
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t))


def bloom_freq(v: "Variant", t: float) -> float:
    """The (possibly time-varying) base frequency at loop phase ``t``.  Equals ``v.freq``
    unless 'freq' is a bloom target, in which case it swells to its peak at mid-loop."""
    if "freq" in v.bloom_params:
        return v.freq * (1.0 + v.bloom_amp * _BLOOM_SWING["freq"] * _bloom_env(t))
    return v.freq


def bloom_threshold(v: "Variant", t: float) -> float:
    """The isosurface level set at ``t`` (shifted from ``v.threshold`` when 'threshold' blooms)."""
    if "threshold" in v.bloom_params:
        return v.threshold + v.bloom_amp * _BLOOM_SWING["threshold"] * _bloom_env(t)
    return v.threshold


def bloom_thickness_scale(v: "Variant", t: float) -> float:
    """Multiplier on the sheet half-width at ``t`` (1 unless 'thickness' blooms)."""
    if "thickness" in v.bloom_params:
        return 1.0 + v.bloom_amp * _BLOOM_SWING["thickness"] * _bloom_env(t)
    return 1.0


def field_expr(v: Variant, t: float = 0.0, transform: str = "drift",
               freq: Optional[float] = None) -> str:
    """Emit the gyroid field at loop phase ``t`` in [0,1) under the chosen ``transform``.

    Both transforms reproduce the exact static field at ``t=0`` (and loop seamlessly, so
    ``t=1`` matches ``t=0``), then move the higher dimensions in between:

    * ``drift`` — translate the slice *through* each dimension: each dim's phase advances
      by ``2*pi*winding*t`` (an integer number of whole cycles over the loop).  The pattern
      slides.
    * ``rotate`` — rotate each dim's wavevector *out of* the 3-D slice into its own hidden
      axis by angle ``2*pi*winding*t``.  The in-slice frequency scales by ``cos`` (the
      stripes widen, vanish, and re-form as the wave turns edge-on and back) while the
      out-of-slice tilt adds a ``sin``-weighted phase from the slice's ``hidden_offset``.
      The lattice genuinely reshapes — the higher-D analogue of turning the object — rather
      than merely sliding.  The main dim (winding 0) stays put and anchors the pattern so it
      never fully dissolves.
    * ``bloom`` — frame 0 (and frame 1) is the *exact classic 3-D gyroid* (the
      ``scenes/showcase.ftsl`` field on X/Y/Z); over the loop the full N-D gyroid is
      cross-blended in and back out by an envelope ``w(t) = sin^2(pi t)`` (0 at the ends,
      1 at the midpoint).  The video therefore begins as the recognizable showcase gyroid
      and *unfolds* into its higher-dimensional structure at mid-loop, then folds back —
      a seamless "bloom".  The higher dimensions still drift while blended in.
    """
    if transform == "bloom":
        # ``bloom`` pins frame 0 (and frame 1) to the base gyroid, then oscillates the
        # selected parameters over the loop with the envelope w = sin^2(pi t).  The
        # frequency swing (if 'freq' blooms) applies to *both* the classic base and the
        # full field, so the whole pattern pulses in intricacy together.
        w = _bloom_env(t)
        fr = bloom_freq(v, t)
        g_classic = _classic_gyroid_expr(fr)
        if "dims" not in v.bloom_params:
            # No dimensional bloom: the loop is the recognizable classic gyroid the whole
            # time (frame 0 = showcase); only the scalar parameters pulse around it.
            return g_classic
        # Dimensional bloom: cross-fade the classic gyroid with the full N-D field.  The
        # envelope is 0 at t=0,1 (both ends exactly the classic gyroid, seamless loop) and
        # 1 at t=0.5 (the full higher-D gyroid at its peak).
        if w <= 1e-9:
            return g_classic
        g_full = field_expr(v, t, "drift", freq=fr)         # higher dims drift while blended
        if w >= 1.0 - 1e-9:
            return g_full
        return f"({fmt(1.0 - w)})*({g_classic})+({fmt(w)})*({g_full})"
    fr = v.freq if freq is None else freq
    osc = sorted(v.oscillating)
    m = len(osc)
    by_index = {d.index: d for d in v.dim_list}
    u = {}
    two_pi = 2.0 * math.pi
    for d in osc:
        dim = by_index[d]
        if transform == "rotate":
            # Rotate the wavevector into the hidden axis: in-slice frequency k*cos(alpha),
            # plus a k*sin(alpha)*hidden_offset phase.  alpha is a whole number of turns
            # over the loop, so t=0 and t=1 both give alpha ≡ 0 -> the exact static field.
            k = dim.harmonic * fr
            alpha = two_pi * dim.winding * t
            coeff = k * math.cos(alpha)
            phase = (dim.phase + k * dim.hidden_offset * math.sin(alpha)) % two_pi
            u[d] = _arg_expr(dim.direction, coeff, phase)
        else:
            # Reduce the drifted phase modulo 2*pi so t=0 and t=1 emit the *same* constant
            # (a whole-cycle advance) -> a perfectly seamless loop despite float rounding.
            phase = (dim.phase + two_pi * dim.winding * t) % two_pi
            u[d] = _u_expr(dim, fr, phase)
    terms = []
    for i in range(m):
        a = osc[i]
        b = osc[(i + 1) % m]
        terms.append(f"sin({u[a]})*cos({u[b]})")
    return "+".join(terms)


# ---------------------------------------------------------------------------
# scene + header
# ---------------------------------------------------------------------------

# Just the gyroid from scenes/showcase.ftsl — the gold *thickened* gyroid sheet
# (|g| - t < 0), CSG-clipped to a ball — on its own, no Cornell box / glass sphere / room.
# Lighting is what makes gold read as gold: a conductor is a mirror, so it shows whatever
# surrounds it.  A single *uniform* environment gives it nothing but flat grey to reflect
# and the intricate lattice washes out.  Instead we light it like a product shot with a
# procedural *studio* environment (``studio_env_pfm`` below): a dark neutral base plus a
# few bright soft "softbox" lights, written once as an equirectangular .pfm and fed to
# ftrace's image-based ``light env { file ... }``.  The gold then picks up crisp highlights
# that trace every facet while deep shadows give depth — and the env is also the clean
# neutral background.  The lattice itself is the morphing higher-D field at loop phase ``t``.

_ENV_SPD = 1.5      # fallback uniform env radiance (used only if no studio map is supplied)

# Subject material presets (the isosurface's surface).  ``gold`` is the showcase
# conductor (a mirror that reveals the lattice by reflecting the studio lights);
# ``glass`` renders the same thickened sheet as a clear dielectric (BK7 crown glass)
# — the lattice reads through refraction/reflection instead of reflection alone.
# The rasterizer now shades clear dielectrics too, so glass previews meaningfully.
MATERIALS = {
    "gold":  'material "surf" { preset gold }',
    "glass": 'material "surf" { type dielectric ior glass:BK7 }',
}
# Materials that read as clear/transparent — the rasterizer previews these with ftrace's
# ``-see-through`` pass (dim + milky haze) instead of a solid pale ghost.
CLEAR_MATERIALS = {"glass"}

# Studio environment map (equirectangular, written as a Radiance-style .pfm).  Direction
# convention matches src/envmap.h: row 0 = +y (straight up), v=row/H -> theta=v*pi from +y;
# col -> phi=(u-0.5)*2pi, dir=(sinT cosP, cosT, sinT sinP).  The camera sits on +z looking
# at the origin, so the subject front faces +z and the lights below are placed accordingly.
_STUDIO_W, _STUDIO_H = 1024, 512
_STUDIO_BASE = (0.18, 0.185, 0.20)      # dark neutral ambient floor (clean background)


def _studio_lights() -> "List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]":
    """Soft key/fill/rim/top lights: (unit direction, angular radius rad, peak RGB)."""
    def n(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
        m = math.sqrt(sum(c * c for c in vec)) or 1.0
        return tuple(c / m for c in vec)  # type: ignore[return-value]
    return [
        (n((0.55, 0.65, 0.55)), 0.55, (7.0, 6.4, 5.2)),    # key   — warm, upper-right-front
        (n((-0.7, 0.15, 0.5)),  0.70, (2.2, 2.4, 2.8)),    # fill  — cool, left-front, soft
        (n((-0.2, 0.5, -0.85)), 0.35, (5.0, 5.0, 5.2)),    # rim   — behind-above, tight/hot
        (n((0.0, 1.0, 0.0)),    0.90, (0.9, 0.95, 1.1)),   # top   — gentle skylight
    ]


def studio_env_pfm(path: Path) -> Path:
    """Write (once) the procedural studio-lighting environment as an equirectangular .pfm.

    Returns ``path``.  If the file already exists it is left as-is (the map is fixed, so a
    whole batch shares one write).  Pure stdlib (``struct``); no image dependency.
    """
    if path.exists():
        return path
    import struct
    lights = _studio_lights()
    W, H = _STUDIO_W, _STUDIO_H
    rows: List[List[Tuple[float, float, float]]] = []
    for row in range(H):
        theta = (row + 0.5) / H * math.pi
        st, ct = math.sin(theta), math.cos(theta)
        line: List[Tuple[float, float, float]] = []
        for col in range(W):
            phi = ((col + 0.5) / W - 0.5) * 2.0 * math.pi
            d = (st * math.cos(phi), ct, st * math.sin(phi))
            r, g, b = _STUDIO_BASE
            for ldir, rad, peak in lights:
                cosang = max(-1.0, min(1.0, sum(a * c for a, c in zip(d, ldir))))
                w = math.exp(-(math.acos(cosang) / rad) ** 2 * 2.0)   # soft gaussian disc
                r += peak[0] * w; g += peak[1] * w; b += peak[2] * w
            line.append((r, g, b))
        rows.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"PF\n")
        f.write(f"{W} {H}\n".encode())
        f.write(b"-1.0\n")                              # little-endian, unit scale
        for row in range(H - 1, -1, -1):                # PFM scanlines run bottom-to-top
            for (r, g, b) in rows[row]:
                f.write(struct.pack("<fff", r, g, b))
    return path


def build_scene(v: Variant, *, t: float = 0.0, res=(480, 480), radius=1.3,
                env_file: Optional[str] = None, transform: str = "drift",
                material: str = "gold") -> Scene:
    """The lone thickened gyroid ball whose lattice is the morphing higher-D field.

    ``transform`` selects how the higher dimensions animate over the loop (see
    :func:`field_expr`): ``drift`` slides the slice through them, ``rotate`` turns their
    wavevectors out of the 3-D slice.  ``env_file`` is a path to an equirectangular
    environment map (see ``studio_env_pfm``) used for image-based lighting; with it the
    subject picks up the studio highlights that reveal the lattice.  Without it the scene
    falls back to a plain uniform env (flatter).  ``material`` picks the surface (see
    :data:`MATERIALS`): ``gold`` (a conductor/mirror) or ``glass`` (a clear BK7 dielectric).
    The gold/glass look develops under path tracing (``mode R``, i.e. ``--no-raster``); the
    fast rasterizer previews the geometry (now clear dielectrics too) flat-shaded.
    """
    mat_def = MATERIALS.get(material)
    if mat_def is None:
        raise SystemExit(f"error: --material '{material}' is not one of "
                         f"{', '.join(sorted(MATERIALS))}")
    expr = field_expr(v, t, transform)
    # Thicken the surface into a solid sheet (showcase's abs(g) - 0.5).  Scale the
    # half-width by sqrt(M/3) so walls stay visible as extra oscillating dims add
    # amplitude (M=3 reproduces the classic 0.5).
    m = max(1, len(v.oscillating))
    if transform == "bloom":
        w = _bloom_env(t)
        if "dims" in v.bloom_params:
            # Match the field cross-fade: at t=0,1 the sheet is exactly showcase's (half =
            # thickness); at the bloom peak it thickens to the full-field half.
            half = v.thickness * (1.0 + w * (math.sqrt(m / 3.0) - 1.0))
        else:
            half = v.thickness                              # classic base, fixed amplitude
        half *= bloom_thickness_scale(v, t)                 # optional 'thickness' bloom
        thr = bloom_threshold(v, t)                         # optional 'threshold' bloom
    else:
        half = v.thickness * math.sqrt(m / 3.0)
        thr = v.threshold
    inner = f"({expr})-({fmt(thr)})" if abs(thr) > 1e-9 else f"({expr})"
    sheet = f"abs({inner})-({fmt(half)})"
    # Lipschitz bound for the sphere-marcher: |grad f| <= 2*freq*sum(harmonic_d).  In bloom
    # mode the classic base always contributes its 3 unit terms, so floor the sum at 3; and
    # use the peak (possibly 'freq'-bloomed) frequency at this frame so the bound stays valid.
    sum_h = sum(d.harmonic for d in v.dim_list if d.oscillate)
    fr = v.freq
    if transform == "bloom":
        sum_h = max(sum_h, 3)
        fr = bloom_freq(v, t)
    grad_bound = 2.2 * fr * max(1, sum_h)
    box = radius * 1.05                                  # contained_by half-extent
    r = radius

    scene = Scene(Camera(eye=(0.0, 0.7 * r, 4.0 * r), look_at=(0, 0, 0),
                         up=(0, 1, 0), fov_y=36, mode="R", res=res))

    iso = Raw(
        "isosurface {\n"
        '    material "surf"\n'
        "    intersect {\n"
        f'        function {{ expr "{sheet}" }}\n'
        f"        sphere {{ center 0 0 0  radius {fmt(radius)} }}\n"
        "    }\n"
        f"    contained_by {{ min {fmt(-box)} {fmt(-box)} {fmt(-box)}"
        f"  max {fmt(box)} {fmt(box)} {fmt(box)} }}\n"
        f"    max_gradient {fmt(grad_bound)}\n"
        "}")

    if env_file is not None:
        # Image-based studio lighting: crisp highlights that reveal the lattice + clean bg.
        light = Raw(f'light env {{ file "{env_file}" rotate 0 intensity 1.0 }}')
    else:
        light = Raw(f'light env {{ spd {fmt(_ENV_SPD)} }}')   # flat fallback
    scene.add(
        Raw(mat_def),
        iso,
        light,
    )
    return scene


# ---------------------------------------------------------------------------
# human-readable axis / variant labels
# ---------------------------------------------------------------------------

def axis_name(i: int) -> str:
    """Dimension index -> axis letter.  0/1/2 -> X/Y/Z (the real rendered axes),
    3 -> W, and any higher hidden dimension -> ``d<index>`` (e.g. d4, d5)."""
    return "XYZW"[i] if 0 <= i < 4 else f"d{i}"


def axis_list(indices: List[int]) -> str:
    """Comma-joined axis letters for a list of dimension indices, or 'none'."""
    return ",".join(axis_name(i) for i in indices) if indices else "none"


def dims_desc(v: Variant) -> str:
    """e.g. 'D=3 (X,Y,Z)' or 'D=5 (X,Y,Z + 2 hidden)'."""
    extra = v.dims - 3
    base = "D={} (X,Y,Z".format(v.dims)
    return base + (")" if extra <= 0 else f" + {extra} hidden)")


def osc_harm_list(v: Variant) -> str:
    """The dimensions the gyroid actually oscillates (rotates) in, each tagged with
    its harmonic: e.g. 'X(h1), Y(h1), Z(h1), W(h2), d4(h3)'.  This is the compact
    'list of dimensions + harmonics' view of the same data the matrix carries."""
    parts = [f"{axis_name(d.index)}(h{d.harmonic})" for d in v.dim_list if d.oscillate]
    return ", ".join(parts) if parts else "none"


def matrix_lines(v: Variant) -> List[str]:
    """Comment-header block: the N×3 slice-embedding matrix A and its offsets — each
    N-D gyroid axis written as a unit *direction* in the rendered (x,y,z) volume (a
    row of A), next to its *offset* (the phase = where the slice sits along that axis)
    and its harmonic.  This is the 'matrix + offsets' view; osc_harm_list is the same
    data as a flat list.  Inert dimensions carry no term and are marked."""
    L = ["# slice-embedding matrix A (rows) + offsets — each N-D axis as a direction in",
         "#   the rendered (x,y,z) volume; 'offset' is the phase (slice position along that",
         "#   axis); 'harm' multiplies its spatial frequency:",
         "#",
         "#   axis  harm  |        A row: direction (x  y  z)        |  offset",
         "#   ----  ----  |  -------------------------------------  |  ---------"]
    for d in v.dim_list:
        name = axis_name(d.index)
        row = " ".join(f"{c:>+9.4f}" for c in d.direction)
        if d.oscillate:
            L.append(f"#   {name:>4}  h{d.harmonic:<3}  |  {row}  |  {d.phase:>8.4f}")
        else:
            L.append(f"#   {name:>4}  {'-':>4}  |  {row}  |  {'(inert)':>8}")
    return L


def orientation_desc(v: Variant, *, short: bool = False) -> str:
    """How the 3-D slice sits in the N-D space.  'world-aligned' = dims X/Y/Z point
    along the world axes (the --pin-axes default); 'tilted' = every axis direction is
    random.  This is about slice *orientation*, NOT about which axes oscillate."""
    if short:
        return "world-aligned" if v.pinned else "tilted"
    return ("world-aligned (the X/Y/Z axis directions point along the world axes)"
            if v.pinned else "tilted (every axis direction is random)")


def variant_banner(v: Variant, index: int, count: int) -> str:
    """Multi-line console summary printed once (committed, scrolled) when a variant
    begins — it carries the wide data that will not fit the in-place progress line:
    the oscillation list, the slice orientation, and the embedding matrix + offsets."""
    lines = [
        f"[gyroid_nd] gyroid {index + 1}/{count}  {dims_desc(v)}  "
        f"freq={fmt(v.freq)}  seed={v.seed}",
        f"    oscillates in : {osc_harm_list(v)}",
        f"    orientation   : {orientation_desc(v)}",
        "    matrix A (each axis's direction in x,y,z) + offset (phase):",
    ]
    for d in v.dim_list:
        name = axis_name(d.index)
        row = " ".join(f"{c:>+8.4f}" for c in d.direction)
        if d.oscillate:
            lines.append(f"      {name:>4}  h{d.harmonic:<2}  [ {row} ]  offset {d.phase:>8.4f}")
        else:
            lines.append(f"      {name:>4}       [ {row} ]  (inert, no term)")
    return "\n".join(lines)


def bloom_params_desc(v: Variant) -> str:
    """Human-readable list of what the bloom oscillates, e.g. 'higher-D structure,
    frequency (complexity)'."""
    names = {"dims": "higher-D structure", "freq": "frequency (complexity)",
             "threshold": "level set", "thickness": "sheet thickness"}
    parts = [names.get(p, p) for p in v.bloom_params]
    return ", ".join(parts) if parts else "higher-D structure"


def header(v: Variant, index: int, count: int, *,
           frames: Optional[int] = None, fps: Optional[float] = None,
           transform: str = "drift", material: str = "gold") -> str:
    osc = v.oscillating
    moving = [d.index for d in v.dim_list if d.oscillate and d.winding > 0]
    L = ["#" + "=" * 74,
         f"# Higher-dimensional gyroid slice — variant {index + 1}/{count}",
         f"# generated by gyroid_nd.py",
         "#",
         f"# variant seed          : {v.seed}   (regenerate: --variant-seed {v.seed} + the same locks)",
         f"# dimensions (D)        : {v.dims}   (higher/extra dims beyond x,y,z: {max(0, v.dims - 3)})",
         f"# oscillating dims      : {len(osc)}  -> {axis_list(osc)}  (indices {osc})",
         f"# oscillates in         : {osc_harm_list(v)}   (dim(harmonic), the axes that wave)",
         f"# main dimension        : {axis_name(v.main) if v.main is not None else '-'}   (fundamental, harmonic 1)",
         f"# harmonics of the main : {len(v.harmonic_dims)}  -> {axis_list(v.harmonic_dims)}",
         f"# slice orientation     : {orientation_desc(v)}",
         f"# base spatial frequency: {fmt(v.freq)}",
         f"# level set (threshold) : {fmt(v.threshold)}",
         f"# surface material      : {material}"
         + ("   (conductor / mirror — reflects the studio lights)" if material == "gold"
            else "   (clear BK7 dielectric — lattice reads through refraction)" if material == "glass"
            else "")]
    verb = {"rotate": "rotating", "bloom": "blooming"}.get(transform, "drifting")
    if frames is not None:
        secs = frames / fps if fps else 0.0
        motion = (f"frame 0 = classic showcase gyroid; blooms {bloom_params_desc(v)} at mid-loop"
                  if transform == "bloom" else f"{verb} dims -> {moving}")
        L.append(f"# animation             : {frames} frames @ {fmt(fps or 30.0)} fps "
                 f"(~{fmt(secs)}s seamless loop); transform '{transform}'; {motion}")
    # The matrix + offsets view (directions = rows of A, phases = offsets, harmonics).
    L.append("#")
    L += matrix_lines(v)
    # Animation-only detail: which oscillating dims move and how fast (winding), plus role.
    if frames is not None:
        rate_col = "turns" if transform == "rotate" else "drift"
        L += ["#",
              f"# animation per dim — {rate_col} = integer cycles/turns over one loop:",
              f"#   axis  {rate_col:<5}  role",
              "#   ----  -----  -----------"]
        for d in v.dim_list:
            name = axis_name(d.index)
            if d.oscillate:
                rate = f"{d.winding}" if d.winding > 0 else "-"
                L.append(f"#   {name:>4}  {rate:>5}  {d.role}")
            else:
                L.append(f"#   {name:>4}  {'-':>5}  inert")
    if transform == "bloom":
        L += ["#",
              f"# bloom parameters      : {', '.join(v.bloom_params)}  (amp {fmt(v.bloom_amp)})",
              f"#   oscillated over the loop by w(t) = sin^2(pi t)  (0 at t=0,1; 1 at t=0.5),",
              "#   so frame 0 (and frame 1) is exactly the base showcase gyroid — seamless."]
        if "dims" in v.bloom_params:
            L += ["#   dims:      F(t) = (1-w)*G_classic + w*G_full",
                  "#     G_classic = sin(f x)cos(f y) + sin(f y)cos(f z) + sin(f z)cos(f x)",
                  "#                 (the showcase gyroid; f = base frequency, shown at frame 0)",
                  "#     G_full    = the full N-D gyroid below (drifting), blended in 0->1->0"]
        else:
            L += ["#   (no 'dims' bloom: the field stays the classic showcase gyroid all loop;",
                  "#    only the scalar parameters below pulse around it)"]
        if "freq" in v.bloom_params:
            L.append(f"#   freq:      f(t) = {fmt(v.freq)} * (1 + {fmt(v.bloom_amp * _BLOOM_SWING['freq'])}*w)   (intricacy/complexity pulse)")
        if "threshold" in v.bloom_params:
            L.append(f"#   threshold: thr(t) = {fmt(v.threshold)} + {fmt(v.bloom_amp * _BLOOM_SWING['threshold'])}*w   (channels open/close)")
        if "thickness" in v.bloom_params:
            L.append(f"#   thickness: half(t) *= (1 + {fmt(v.bloom_amp * _BLOOM_SWING['thickness'])}*w)   (sheet swells/thins)")
        L.append("#")
    L += ["#",
          "# field:  sum over cyclic oscillating pairs (i, i+1) of  sin(u_i) * cos(u_j)"]
    if transform == "rotate":
        L += ["#   u_d = harmonic_d * freq * cos(a_d) * (dir_d . (x, y, z))",
              "#         + phase_d + harmonic_d * freq * hidden_offset_d * sin(a_d)",
              "#   a_d = 2*pi * winding_d * t   (the dim's wavevector rotates out of the",
              "#   3-D slice into its hidden axis; t runs 0->1, 'turns' column = winding_d)"]
    else:
        tail = "  (winding is the per-dim 'drift' rate above)" if frames is not None else ""
        L += ["#   u_d = harmonic_d * freq * (dir_d . (x, y, z)) + phase_d + 2*pi*winding_d*t",
              f"#   (t runs 0->1 over the loop.{tail})"]
    L += ["#" + "=" * 74, ""]
    return "\n".join(L)


def sidecar_text(v: Variant, index: int, count: int, *,
                 ftsl_name: str = "", video_name: str = "",
                 frames: Optional[int] = None, fps: Optional[float] = None,
                 transform: str = "drift", material: str = "gold") -> str:
    """Plain-text (non-comment) dump of every chosen value, saved beside each video.

    Reuses :func:`header` verbatim (stripped of its ``#`` comment prefixes) so the
    ``.txt`` sidecar and the ``.ftsl`` header can never drift apart.
    """
    lines: List[str] = []
    if ftsl_name or video_name:
        if video_name:
            lines.append(f"video file : {video_name}")
        if ftsl_name:
            lines.append(f"frames like: {ftsl_name}")
        lines.append("")
    for line in header(v, index, count, frames=frames, fps=fps,
                       transform=transform, material=material).splitlines():
        if line.startswith("# "):
            lines.append(line[2:])
        elif line == "#":
            lines.append("")
        elif line.startswith("#"):
            lines.append(line[1:])
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# in-place status line
# ---------------------------------------------------------------------------

_status_len = 0


def _status_reset() -> None:
    """Forget the live-line width (call after printing committed multi-line output so
    the next in-place status doesn't pad against a stale, longer previous line)."""
    global _status_len
    _status_len = 0


def _status(msg: str) -> None:
    """Overwrite the current status line in place (carriage-return, no newline)."""
    global _status_len
    pad = max(0, _status_len - len(msg))
    sys.stdout.write("\r" + msg + " " * pad)
    sys.stdout.flush()
    _status_len = len(msg)


def _status_commit(msg: str) -> None:
    """Overwrite the live line with ``msg`` and finalize it (newline)."""
    global _status_len
    pad = max(0, _status_len - len(msg))
    sys.stdout.write("\r" + msg + " " * pad + "\n")
    sys.stdout.flush()
    _status_len = 0


# ---------------------------------------------------------------------------
# optional single-window live preview (tkinter)
# ---------------------------------------------------------------------------

class _PreviewWindow:
    """One reusable window that displays each freshly-rendered frame in place.

    The window title carries the running context (which gyroid, its values, and the
    current frame).  It updates synchronously between frames (``update`` rather than
    ``mainloop``); if the toolkit is unavailable or the user closes the window the
    preview quietly disables itself and generation continues headless.
    """

    def __init__(self) -> None:
        self._root = None
        self._label = None
        self._photo = None            # keep a ref so Tk doesn't GC the image
        self._tk = None
        self._alive = True

    def _ensure(self) -> bool:
        if self._root is not None:
            return True
        if not self._alive:
            return False
        try:
            import tkinter as tk
            from PIL import ImageTk  # noqa: F401  (import checked here, used in show)
        except Exception as e:  # pragma: no cover
            print(f"\n[gyroid_nd] preview unavailable ({e}); continuing headless",
                  flush=True)
            self._alive = False
            return False
        self._tk = tk
        self._root = tk.Tk()
        self._root.title("gyroid_nd preview")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label = tk.Label(self._root)
        self._label.pack()
        return True

    def show(self, png: Path, title: str) -> None:
        if not self._ensure():
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(str(png)).convert("RGB")
            self._photo = ImageTk.PhotoImage(img)
            self._label.configure(image=self._photo)
            self._root.title(title)
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            # Window closed or display error: disable and keep rendering headless.
            self.close()

    def pump(self) -> None:
        """Service the window's event loop once (repaint, taskbar, drag).

        Called repeatedly while a frame renders so the window stays responsive
        instead of freezing for the whole (possibly multi-second) render.  A no-op
        if the preview never opened or was closed; errors disable it quietly.
        """
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:
            self.close()

    def close(self) -> None:
        self._alive = False
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None


# ---------------------------------------------------------------------------
# per-variant video pipeline
# ---------------------------------------------------------------------------

def _render_frame(ftrace: Path, root: Path, fp: Path, png: Path, *,
                  size: Tuple[int, int], raster: bool, noise: float,
                  see_through: bool = False, clarity: Optional[float] = None,
                  pump: Optional["Callable[[], None]"] = None) -> None:
    """Render one frame ``.ftsl`` -> ``.png``, headless and non-blocking.

    ``raster`` uses ftrace ``-raster`` (fast solid-shaded z-buffer preview); otherwise
    a path-traced render to a per-frame noise budget.  ``size`` is the ``(W, H)`` film
    resolution, passed as ``-r W H``.  No ``-window`` is passed so the process writes
    the PNG and exits, letting the whole frame range run unattended.  The renderer's
    own console chatter is captured (kept off the status line) and only surfaced if the
    frame fails.

    ``see_through`` (raster only) passes ftrace ``-see-through`` so clear dielectrics
    (glass) render as dimmed + milky-hazed rather than a solid pale ghost (no refraction —
    that needs path tracing); ``clarity`` (0..1) sets the per-surface transmittance via
    ``-glass-clarity`` (higher = clearer; ftrace's default is 0.85).

    ``pump`` is an optional callback invoked repeatedly *while* ftrace runs (used to
    keep the preview window's event loop serviced — otherwise it would freeze for the
    whole render, since a path-traced frame can take seconds).  With it the child is
    launched via ``Popen`` and polled instead of blocking in ``subprocess.run``.
    """
    import subprocess
    w, h = size
    if raster:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-raster",
               "-r", str(w), str(h)]
        if clarity is not None:
            cmd += ["-glass-clarity", f"{clarity:g}"]   # implies -see-through
        elif see_through:
            cmd.append("-see-through")
    else:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-r", str(w), str(h),
               "-interval", "8", "-checkpoint", "-noise", f"{noise:g}"]
    if pump is None:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        rc, out, err = r.returncode, r.stdout, r.stderr
    else:
        # Poll so the caller can pump its GUI event loop (~30 Hz) while ftrace works,
        # keeping the preview window responsive instead of frozen for the whole frame.
        import time
        proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        while proc.poll() is None:
            pump()
            time.sleep(0.03)
        out, err = proc.communicate()
        rc = proc.returncode
    if rc != 0:
        sys.stdout.write("\n")
        sys.stdout.write((out or "") + (err or ""))
        raise SystemExit(f"ftrace failed on {fp.name} (exit {rc})")


def _assemble_video(pngs: List[Path], out: Path, *, fps: float, pattern: str,
                    cwd: Path) -> Path:
    """Assemble frames into ``out`` — mp4 via ffmpeg, or a Pillow GIF (default).

    ``pattern`` is the ffmpeg input glob resolved relative to ``cwd`` (the frames may
    live in a subdirectory of the video's directory).  Both paths keep their tool output
    off the status line (ffmpeg captured; the GIF is built inline with Pillow rather than
    via the chatty ``loom.drive.assemble_gif``).
    """
    import shutil
    import subprocess
    if out.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit("ffmpeg not found for mp4 output (use --format gif)")
        cmd = [ffmpeg, "-y", "-framerate", f"{fps:g}", "-i", pattern,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out.name]
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if r.returncode != 0:
            sys.stdout.write("\n" + (r.stdout or "") + (r.stderr or ""))
            raise SystemExit(f"ffmpeg failed assembling {out.name} (exit {r.returncode})")
    else:
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise SystemExit("gif output needs Pillow (pip install pillow)") from e
        imgs = [Image.open(str(p)).convert("RGB") for p in pngs]
        duration_ms = max(1, int(round(1000.0 / fps)))
        imgs[0].save(str(out), save_all=True, append_images=imgs[1:],
                     duration=duration_ms, loop=0, optimize=True)
    return out


def _video_ext(fmt: str) -> str:
    """Resolve the ``--format`` choice to a concrete extension."""
    if fmt == "gif":
        return "gif"
    if fmt == "mp4":
        return "mp4"
    import shutil
    return "mp4" if shutil.which("ffmpeg") else "gif"   # auto


def make_video(frames_dir: Path, out_dir: Path, base: str, v: Variant, *, label: str,
               frames: int, fps: float, size: Tuple[int, int], radius: float,
               raster: bool, noise: float, fmt: str, env_file: Optional[str] = None,
               transform: str = "drift", material: str = "gold",
               clarity: Optional[float] = None,
               preview: Optional["_PreviewWindow"] = None) -> Path:
    """Emit ``frames`` morphing scene files, render them, and assemble one video.

    The per-frame ``.ftsl``/``.png`` files land in ``frames_dir`` (its own subdirectory),
    while the assembled video is written to ``out_dir`` (the shared collection directory).
    The gyroid drifts through its higher dimensions over a seamless loop (frame ``frames``
    == frame 0).  Frames render at ``size`` = ``(W, H)`` pixels, headless with the
    rasterizer by default.  Progress is shown on a single in-place status line built from
    ``label``; if a ``preview`` window is given, each rendered frame is shown in it in
    place (at the same ``size``).
    """
    from loom import Clock, Cache
    from loom.drive import find_ftrace, repo_root
    ftrace = find_ftrace()
    root = repo_root()
    verb = "raster" if raster else "trace"
    title_base = label.replace("[gyroid_nd] ", "")
    fw = max(3, len(str(frames - 1)))                   # frame-number field width
    clear = material in CLEAR_MATERIALS                 # raster see-through only for clear mats
    frame_clarity = clarity if clear else None
    pngs: List[Path] = []
    for k in range(frames):
        t = k / frames                                  # seamless loop: t in [0,1)
        _status(f"{label} | emit ftsl  frame {k + 1}/{frames}")
        scene = build_scene(v, t=t, res=size, radius=radius, env_file=env_file,
                            transform=transform, material=material)
        body = scene.emit(Clock(t=t, frame=k, frames=frames, fps=fps),
                          Cache(), assets_dir=frames_dir, tag=f"{k:0{fw}d}")
        fp = frames_dir / f"{base}_{k:0{fw}d}.ftsl"
        fp.write_text(body, encoding="utf-8")
        png = fp.with_suffix(".png")
        _status(f"{label} | {verb} frame {k + 1}/{frames}")
        _render_frame(ftrace, root, fp, png, size=size, raster=raster, noise=noise,
                      see_through=clear, clarity=frame_clarity,
                      pump=(preview.pump if preview is not None else None))
        pngs.append(png)
        if preview is not None:
            preview.show(png, f"{title_base}  |  frame {k + 1}/{frames}")
    out = out_dir / f"{base}.{_video_ext(fmt)}"
    # ffmpeg reads the frames from the per-variant subdir, relative to out_dir.
    pattern = f"{frames_dir.name}/{base}_%0{fw}d.png"
    _status(f"{label} | assembling {out.suffix.lstrip('.')}")
    _assemble_video(pngs, out, fps=fps, pattern=pattern, cwd=out_dir)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_size(spec: str) -> Tuple[int, int]:
    """Parse a ``--size`` value: ``N`` (square NxN) or ``WxH`` -> ``(W, H)``."""
    s = spec.strip().lower().replace(" ", "")
    try:
        if "x" in s:
            w_str, h_str = s.split("x", 1)
            w, h = int(w_str), int(h_str)
        else:
            w = h = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--size '{spec}': expected a number (square) or WIDTHxHEIGHT (e.g. 1280x720)")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"--size '{spec}': dimensions must be >= 1")
    return (w, h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gyroid_nd.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Generate N higher-dimensional gyroid-slice .ftsl scene files with "
                     "randomized\ndimension counts, oscillating axes and harmonics - every "
                     "choice lockable, and\nrecorded in a comment header in each output file."),
        epilog=("axis-lock format for --axis (repeatable):\n"
                "  INDEX:on          force this axis to oscillate (random harmonic role)\n"
                "  INDEX:off         force this axis inert (no term; surface invariant along it)\n"
                "  INDEX:on:H        force it to oscillate at integer harmonic H "
                "(H>=2 = overtone of the main)\n\n"
                "examples:\n"
                "  python examples/gyroid_nd.py --count 10\n"
                "  python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 "
                "--oscillating 4 --harmonics 2\n"
                "  python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 "
                "--axis 2:on:1   # classic gyroid\n"
                "  python examples/gyroid_nd.py --dims 6 --axis 4:on:3 --axis 1:off\n"
                "  python examples/gyroid_nd.py --dims 3 --no-pin-axes   # freely-tilted gyroid slice\n"
                "  python examples/gyroid_nd.py --dims 6 --transform bloom   # opens on the showcase gyroid"))

    g = p.add_argument_group("output")
    g.add_argument("-n", "--count", type=int, default=1,
                   help="number of variant .ftsl files to generate (default 1)")
    g.add_argument("--out", type=str, default=None,
                   help="output directory (default: <repo>/png/gyroid_nd)")
    g.add_argument("--name", type=str, default="gyroid_nd",
                   help="base filename for the outputs (default gyroid_nd)")
    g.add_argument("--run-subdir", action=argparse.BooleanOptionalAction, default=True,
                   help="put each run in a fresh numbered subdirectory (run001, run002, ...) "
                        "under the output dir so runs never overwrite each other (default). "
                        "--no-run-subdir writes straight into the output dir (old behavior, "
                        "overwrites same-named files from prior runs).")
    g.add_argument("--seed", type=int, default=None,
                   help="master RNG seed for a reproducible batch (default: random; the "
                        "chosen value is printed so you can reproduce the run)")
    g.add_argument("--variant-seed", type=int, action="append", default=[], metavar="S",
                   help="generate exactly the variant(s) with this seed, bypassing the master "
                        "(repeatable). Reproduces a single variant when combined with the same "
                        "locks it was made with.")

    g = p.add_argument_group("locks (fix a value instead of randomizing it)")
    g.add_argument("--dims", type=int, default=None,
                   help="lock the total number of dimensions D (>=3)")
    g.add_argument("--dims-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                   default=(3, 8), help="range for a random D when --dims is unset (default 3 8)")
    g.add_argument("--oscillating", type=int, default=None,
                   help="lock how many dimensions oscillate (>=2)")
    g.add_argument("--harmonics", type=int, default=None,
                   help="lock how many oscillating dims are harmonics (overtones) of the main dim")
    g.add_argument("--max-harmonic", type=int, default=5,
                   help="largest integer harmonic drawn for an overtone dim (default 5)")
    g.add_argument("--max-winding", type=int, default=2,
                   help="fastest per-dim drift rate: integer cycles a dimension advances "
                        "over one video loop as the slice moves through it (default 2)")
    g.add_argument("--axis", action="append", default=[], metavar="SPEC",
                   help="force one axis on/off and optionally its harmonic; repeatable "
                        "(see epilog)")
    g.add_argument("--freq", type=float, default=None,
                   help="lock the base spatial frequency (cells packed into the ball)")
    g.add_argument("--freq-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=(3.0, 7.0), help="range for a random freq when --freq is unset "
                                            "(default 3 7)")
    g.add_argument("--phase0", action="store_true",
                   help="set every phase to 0 (deterministic pattern position) instead of random")
    g.add_argument("--pin-axes", action=argparse.BooleanOptionalAction, default=True,
                   help="pin the first three dimensions to the world X/Y/Z axes so the slice "
                        "always contains the ordinary xyz volume (default; this is what lets "
                        "D=3 all-on reproduce the exact classic gyroid). --no-pin-axes instead "
                        "gives every dimension a random direction — a freely-oriented N-D slice, "
                        "so even the base gyroid comes out tilted.")

    g = p.add_argument_group("scene")
    g.add_argument("--threshold", type=float, default=0.0,
                   help="isosurface level set f = threshold (default 0; ~+/-0.7 thins the walls)")
    g.add_argument("--radius", type=float, default=1.3,
                   help="radius of the spherical container the lattice fills (default 1.3)")
    g.add_argument("--thickness", type=float, default=0.5,
                   help="half-width of the thickened sheet (showcase abs(g)-t style; "
                        "default 0.5; auto-scaled up with the oscillating-dim count)")
    g.add_argument("--material", choices=sorted(MATERIALS), default="gold",
                   help="surface of the gyroid sheet: 'gold' (default; a conductor/mirror "
                        "that reveals the lattice by reflecting the studio lights) or 'glass' "
                        "(a clear BK7 dielectric — the lattice reads through refraction). Both "
                        "develop fully under path tracing (--no-raster); with a clear material "
                        "the rasterizer previews it see-through (dim + milky haze) too.")
    g.add_argument("--glass-clarity", type=float, default=None, metavar="0..1",
                   help="for a clear --material under the rasterizer: per-surface see-through "
                        "transmittance passed to ftrace -glass-clarity (higher = clearer; "
                        "ftrace default 0.85). Ignored for gold or path-traced (--no-raster) "
                        "renders.")
    g.add_argument("--size", type=_parse_size, default=None, metavar="N|WxH",
                   help="video/frame size in pixels: a single number for square (e.g. 720) "
                        "or WIDTHxHEIGHT (e.g. 1280x720); the preview window matches it "
                        "(overrides --res)")
    g.add_argument("--res", type=int, default=480,
                   help="square render size when --size is unset (default 480)")

    g = p.add_argument_group("video (per variant)")
    g.add_argument("--transform", choices=TRANSFORMS, default="drift",
                   help="how the higher dimensions animate the loop: 'drift' (default) "
                        "translates the slice through them (the pattern slides); 'rotate' "
                        "turns each dim's wavevector out of the 3-D slice into a hidden axis "
                        "(the lattice reshapes — a higher-D 'rotation'); 'bloom' pins frame 0 "
                        "to the exact classic showcase gyroid and cross-blends the full N-D "
                        "field in and back out (w=sin^2(pi t)), so the clip opens as the "
                        "showcase gyroid and unfolds into higher-D structure. All start from "
                        "a seamless frame 0 and loop.")
    g.add_argument("--bloom", type=str, default=None, metavar="P[,P...]",
                   help="for --transform bloom: which parameter(s) oscillate over the loop "
                        "(comma-separated; default 'dims'). Choices: 'dims' (cross-blend the "
                        "full N-D field in and out — the original bloom), 'freq' (pulse the "
                        "spatial frequency = pattern intricacy/complexity; alias 'complexity'), "
                        "'threshold' (shift the level set so channels open/close), 'thickness' "
                        "(swell and thin the gold sheet). Frame 0 is always the base showcase "
                        "gyroid. e.g. --bloom freq  or  --bloom dims,freq")
    g.add_argument("--bloom-amp", type=float, default=1.0,
                   help="scale the peak swing of every bloomed parameter (default 1.0; at 1.0 "
                        "'freq'/'thickness' reach 2x at mid-loop). Only used with --transform bloom.")
    g.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                   help="render a seamless morphing video per variant (the gyroid drifting "
                        "through its higher dimensions); the videos + .txt sidecars collect "
                        "in the output dir, per-frame files in a subdir each (default on; "
                        "--no-video emits just one static .ftsl per variant)")
    g.add_argument("--frames", type=int, default=60,
                   help="frames per video (default 60)")
    g.add_argument("--fps", type=float, default=30.0,
                   help="video frame rate (default 30)")
    g.add_argument("--format", choices=("gif", "mp4", "auto"), default="gif",
                   help="video format (default gif; mp4 needs ffmpeg; auto: mp4 if "
                        "ffmpeg is present, else gif)")
    g.add_argument("--raster", action=argparse.BooleanOptionalAction, default=True,
                   help="render each frame with the fast headless rasterizer (default); "
                        "--no-raster path-traces every frame instead (far slower)")
    g.add_argument("--preview", action="store_true",
                   help="show each frame as it is rendered in one reusable preview window "
                        "whose title tracks the current gyroid / values / frame")
    g.add_argument("--render-noise", type=float, default=4.0,
                   help="per-frame noise-floor budget when path-tracing frames (--no-raster; "
                        "default 4%%)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dims is not None and args.dims < 3:
        raise SystemExit("error: --dims must be >= 3")
    if args.oscillating is not None and args.oscillating < 2:
        raise SystemExit("error: --oscillating must be >= 2")
    if args.count < 1:
        raise SystemExit("error: --count must be >= 1")
    if args.bloom is not None and args.transform != "bloom":
        raise SystemExit("error: --bloom only applies to --transform bloom")
    _parse_bloom_params(args.bloom)     # validate early (raises on a bad parameter name)

    # Video/frame pixel size: explicit --size (N or WxH) wins, else square --res.
    size = args.size if args.size is not None else (args.res, args.res)

    axis_locks: Dict[int, AxisLock] = {}
    for spec in args.axis:
        try:
            parse_axis_lock(spec, axis_locks)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    base_outdir = Path(args.out) if args.out else _default_outdir(args.name)
    if args.run_subdir:
        outdir = _next_run_dir(base_outdir)
        print(f"[gyroid_nd] run dir: {outdir}  (--no-run-subdir to write into {base_outdir})")
    else:
        outdir = base_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # One procedural studio-lighting environment map shared by the whole batch (see
    # studio_env_pfm).  Embed an absolute forward-slash path so ftrace resolves it from
    # its working dir (the repo root) regardless of where --out points.
    env_file = studio_env_pfm(outdir / "studio_env.pfm").resolve().as_posix()

    # Exact variant seeds (--variant-seed) bypass the master; otherwise derive `count`
    # of them from a concrete master seed we print, so the whole batch is reproducible.
    master_seed = args.seed if args.seed is not None else random.randrange(1, 2 ** 31 - 1)
    if args.variant_seed:
        seeds = list(args.variant_seed)
        print(f"[gyroid_nd] using {len(seeds)} explicit --variant-seed value(s)")
    else:
        master = random.Random(master_seed)
        seeds = [master.randrange(1, 2 ** 31 - 1) for _ in range(args.count)]
        print(f"[gyroid_nd] master seed {master_seed} "
              f"(reproduce this batch with --seed {master_seed})")

    if args.video and args.frames < 2:
        raise SystemExit("error: --frames must be >= 2 to make a video")

    count = len(seeds)
    width = max(3, len(str(count - 1)))
    made: List[Path] = []
    from loom import Clock, Cache
    preview = _PreviewWindow() if (args.video and args.preview) else None
    for k, vseed in enumerate(seeds):
        v = pick_variant(vseed, args, axis_locks)
        base = f"{args.name}{k:0{width}d}"
        # Full multi-line detail (orientation + matrix + offsets) is printed once,
        # committed/scrolled; the live per-frame line below stays short so it can
        # update in place without wrapping past the terminal width.
        print(variant_banner(v, k, count))
        _status_reset()
        label = f"[gyroid_nd] gyroid {k + 1}/{count}  {dims_desc(v)}"
        if args.video:
            # Videos and their .txt sidecars collect in the shared outdir; each video's
            # per-frame .ftsl/.png files live in their own subdir <outdir>/<base>/.
            frames_dir = outdir / base
            frames_dir.mkdir(parents=True, exist_ok=True)
            ext = _video_ext(args.format)
            (outdir / f"{base}.txt").write_text(
                sidecar_text(v, k, count, ftsl_name=f"{base}/{base}_NNN.ftsl",
                             video_name=f"{base}.{ext}",
                             frames=args.frames, fps=args.fps,
                             transform=args.transform, material=args.material),
                encoding="utf-8")
            video = make_video(frames_dir, outdir, base, v, label=label,
                               frames=args.frames, fps=args.fps, size=size,
                               radius=args.radius, raster=args.raster,
                               noise=args.render_noise, fmt=args.format,
                               env_file=env_file, transform=args.transform,
                               material=args.material, clarity=args.glass_clarity,
                               preview=preview)
            made.append(video)
            _status_commit(f"{label} | done -> {video.name} ({args.frames} frames)")
        else:
            # No video: one static scene file (t=0) with the full comment header.
            scene = build_scene(v, t=0.0, res=size, radius=args.radius, env_file=env_file,
                                transform=args.transform, material=args.material)
            body = scene.emit(Clock(t=0.0), Cache(), assets_dir=outdir,
                              tag=f"{k:0{width}d}")
            fp = outdir / f"{base}.ftsl"
            fp.write_text(header(v, k, count, transform=args.transform,
                                 material=args.material) + body,
                          encoding="utf-8")
            made.append(fp)
            _status_commit(f"{label} | wrote {fp.name}")

    if preview is not None:
        preview.close()
    kind = "video(s)" if args.video else "scene file(s)"
    print(f"[gyroid_nd] wrote {len(made)} {kind} to {outdir}")
    return 0


def _default_outdir(name: str) -> Path:
    try:
        from loom.drive import repo_root
        return repo_root() / "png" / name
    except Exception:
        return Path.cwd() / name


def _next_run_dir(base: Path) -> Path:
    """A fresh ``runNNN`` subdirectory under ``base`` that does not yet exist, one past
    the highest existing ``run<number>`` (so runs never overwrite even if some were
    deleted).  The directory is not created here — the caller mkdirs it."""
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    for p in base.iterdir():
        if p.is_dir():
            mt = re.fullmatch(r"run(\d+)", p.name)
            if mt:
                n = max(n, int(mt.group(1)) + 1)
    return base / f"run{n:03d}"


if __name__ == "__main__":
    sys.exit(main())
