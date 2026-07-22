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
the oscillating dims ``o_0 < o_1 < ...`` (indices taken mod the oscillating count) — the
standard way the Schoen gyroid generalizes, giving **m terms** for ``m`` oscillating dims.
``--coupling all`` instead sums over **every unordered pair** ``i<j`` of oscillating dims
(``sin(u_i)*cos(u_j)``), i.e. ``C(m,2)`` terms — e.g. 15 for 6 dims instead of 6 — a denser,
more interwoven lattice (the two schemes match in count only for ``m <= 3``, and even at
``m = 3`` cover different pairings, so ``--coupling all`` is not the exact classic gyroid).
``--coupling none`` gives an **empty** base graph, so the coupling is built up entirely from
``--pair I,J:on`` chords (below).  Think of the field as a **coupling graph** whose nodes are
dimensions and whose edges are the ``sin*cos`` terms: ``--axis`` edits *nodes* (turn a dim
on/off, set its harmonic) while ``--pair I,J:on|off`` edits individual *edges* on top of the
chosen base graph — delete a single term (``I,J:off``) or add an extra chord (``I,J:on``, which
forces both endpoints to oscillate).  The comma names an edge's two endpoints, distinct from
``--axis``'s hyphen node range (``LO-HI``).  Both the node and edge sets follow a **base
polarity + selective override** model: ``--axis-default {random,on,off}`` sets whether
un-named axes oscillate by default (then flip individuals with ``--axis``), and ``--coupling``
picks the base edge set (``cyclic``/``all``/``none``) that ``--pair`` then edits.

``--surface`` chooses which triply-periodic minimal surface to slice.  **gyroid** (default) is
the pairwise Schoen gyroid described above.  **primitive** is the Schwarz P surface, a
*per-node* field ``sum_d cos(u_d)`` (one cosine per oscillating dim, no edges).  Both share the
entire N-D slice machinery — the same ``u_d = harmonic_d * freq * (dir_d . xyz) + phase_d``
arguments and the same ``--dims``/``--oscillating``/``--harmonics``/``--oscillate`` (incl.
``bloom``) animation — and differ **only** in how those arguments combine into the scalar
field.  Because Schwarz P has no coupling edges, ``--coupling``/``--pair`` do not apply to it
(they only shape the gyroid's pairwise graph; a note is printed if given with ``primitive``).

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
Every non-main oscillating dimension gets an integer rate (a "winding"); the ``--oscillate``
axis you name decides what that motion *is*, and all start from the exact current gyroid at
frame 0 and loop seamlessly:

  * ``drift`` (default) — advance each dim's phase a whole number of cycles over the loop,
    translating the slice *through* that dimension.  The pattern slides.
  * ``rotate`` — rotate each dim's wavevector out of the 3-D slice into its own hidden axis
    (a whole number of turns).  The in-slice frequency waxes and wanes as the wave turns
    edge-on and back, so the lattice genuinely reshapes — the higher-D analogue of *turning*
    the object rather than sliding it.  (Each dim turns *independently* in its own private
    plane — the wavevectors tilt, but the slice's orientation as a whole is unchanged.)
  * ``tumble`` — rotate the **3-D slice itself** through the N-D space: a rigid N-D rotation
    of the whole slice basis (built from disjoint Givens rotations that couple the visible
    x/y/z axes with the hidden dimensions), a whole number of turns over the loop.  Unlike
    ``rotate`` (which tilts each wavevector on its own), this reorients the *entire* slice
    coherently — the higher-D axes swing into view and the visible ones swing out, so the
    lattice is genuinely re-sliced from a turning viewpoint.  With >= 4 oscillating dims this
    produces real morphing (new structure appears); with exactly 3 it reduces to spinning the
    gyroid rigidly (a plain 3-D rotation).
  * ``bloom`` — pin frame 0 (and frame 1) to the *exact classic gyroid* from
    ``scenes/showcase.ftsl`` (``sin(f x)cos(f y) + sin(f y)cos(f z) + sin(f z)cos(f x)``),
    then swell one or more of its scalar **parameters** out and back with an envelope
    ``w = sin^2(pi t)`` (0 at the loop ends, 1 at the midpoint), so the clip always opens
    and closes on the recognizable showcase gyroid.  You name *which* parameter(s) bloom as
    separate ``--oscillate`` swinger axes (``bloom`` is the dims-crossfade envelope itself):

      - ``bloom`` — cross-blend the full N-D gyroid in and back out; the lattice *unfolds*
        into higher-D structure at the midpoint (the original bloom).
      - ``freq`` — hold the classic gyroid but pulse its spatial frequency up at mid-loop,
        so the pattern gets finer/more intricate and relaxes back.
      - ``threshold`` — swell the level-set value, breathing the surface off its zero set.
      - ``thickness`` — swell the sheet's half-thickness so the walls fatten and thin.

    Swingers combine in one composite group (e.g. ``--oscillate bloom,freq``), and a per-item
    amplitude scales that parameter's peak swing (e.g. ``1.5*freq``; default 1).  The base
    frequency defaults to the showcase density (freq 40 at radius 0.32) unless ``--freq`` is given.

These transforms **layer**: ``--oscillate`` joins several axes into one composite group
with commas (e.g. ``--oscillate drift,tumble`` or ``--oscillate drift,rotate,tumble,bloom``)
and the field composes them.  The three *motions* stack on each oscillating dim's argument — ``tumble``
rotates the whole slice basis, ``rotate`` turns each wavevector out of the slice, ``drift``
advances the phase — while ``bloom`` wraps the composed field in its classic->full cross-fade
envelope (the motions then animate the full field it reveals).  Because every layer is the
identity at ``t=0`` and ``t=1``, any combination still starts on a seamless frame 0 and loops.

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
and at what harmonic.  The numeric locks (``--dims``, ``--oscillating``, ``--harmonics``,
``--freq``) each take not just a fixed value but also a **range** (``4-8``, a random pick in
the inclusive span) or a **set** (``4,6,8``, a random pick among the listed values), the same
shapes ``--axis``'s index field accepts — so you can pin a value, narrow it to a band, or
enumerate the allowed choices.  A fixed value never draws from the RNG, so pinning one leaves
every other seed-driven choice bit-for-bit reproducible.

Examples::

    # 10 fully random variants, each a morphing video, into png/gyroid_nd/<variant>/
    python examples/gyroid_nd.py --count 10

    # reproducible; lock 6 dims, 4 oscillating, 2 of them harmonics of the main
    python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 --oscillating 4 --harmonics 2

    # value locks take ranges/sets too: 5-8 dims, an even oscillating count, freq in 3-5
    python examples/gyroid_nd.py --count 5 --dims 5-8 --oscillating 4,6,8 --freq 3-5

    # base polarity + override: every axis oscillates except 2 and 5; hand-built coupling
    python examples/gyroid_nd.py --dims 6 --axis-default on --axis 2:off --axis 5:off
    python examples/gyroid_nd.py --dims 6 --coupling none --pair 0,1:on --pair 1,2:on --pair 2,0:on

    # start on the exact showcase gyroid, then bloom into higher-D structure and back
    python examples/gyroid_nd.py --dims 6 --oscillate bloom

    # stay the classic gyroid but pulse its complexity (frequency) up at mid-loop
    python examples/gyroid_nd.py --oscillate freq

    # bloom the higher-D unfold *and* an extra-intense frequency pulse together
    python examples/gyroid_nd.py --dims 6 --oscillate bloom,1.5*freq

    # the classic gyroid, animated: x,y,z on at harmonic 1, 90 frames as an mp4
    python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 --axis 2:on:1 \
        --frames 90 --format mp4

    # start from the current gyroid and rotate it through the extra dimensions
    python examples/gyroid_nd.py --dims 6 --oscillate rotate

    # LAYER several motions in one composite group: drift + tumble + a bloom cross-fade
    python examples/gyroid_nd.py --dims 6 --oscillate drift,tumble,bloom

    # tumble the whole 3-D slice through N-D space (the viewpoint turns, re-slicing it)
    python examples/gyroid_nd.py --dims 6 --oscillate tumble

    # tumble in 'slide' mode: an amplitude on tumble rocks the slice between two extremes
    # so the lattice breathes smaller/larger (bigger amp = more dramatic scale swing)
    python examples/gyroid_nd.py --dims 6 --oscillate 0.3*tumble

    # tumble but keep world X and Y pinned (only the other axes reorient the slice)
    python examples/gyroid_nd.py --dims 6 --oscillate tumble --lock 0,1

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
from loom import POV_FUNCS, POV_ND_GENERALIZABLE, pov_params  # noqa: E402
from loom import nd_field_expr, nd_grad_bound_xi  # noqa: E402
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
    bloom_amp: float = 1.0              # scales every bloom parameter's peak swing (the
    #                                     shared --bloom-amp scalar; the default for any
    #                                     swinger without a per-axis override in bloom_amps).
    bloom_amps: Dict[str, float] = dc_field(default_factory=dict)
    #                                     per-axis swinger amplitude overrides, keyed by
    #                                     parameter ('freq'/'threshold'/'thickness'). Set by
    #                                     the unified --oscillate grammar (e.g. 2*freq,0.5*
    #                                     threshold), where each swinger carries its own
    #                                     amplitude. Empty (the legacy --bloom/--bloom-amp
    #                                     path) => every swinger falls back to bloom_amp.
    bloom_rates: Dict[str, float] = dc_field(default_factory=dict)
    #                                     per-swinger envelope rate: full cycles of the sin^2
    #                                     bump over the loop (keyed like bloom_amps, plus 'dims'
    #                                     for the dimensional cross-fade). From a --oscillate
    #                                     swinger group's `rate`. Empty/absent => 1 (the single
    #                                     mid-loop bump; byte-identical to the legacy path). An
    #                                     *integer* rate still loops seamlessly; a non-integer
    #                                     rate pulses faster but the last frame != the first.
    bloom_phases: Dict[str, float] = dc_field(default_factory=dict)
    #                                     per-swinger envelope phase offset in radians (2*pi =
    #                                     one full bump-cycle, matching the winder clock), same
    #                                     keys. From a swinger group's `phase`. Absent => 0.
    #                                     With an integer rate any phase still loops seamlessly.
    tumble_planes: List[Tuple[int, int, int]] = dc_field(default_factory=list)
    #                                     tumble transform only: disjoint (i, j, winding)
    #                                     Givens rotations composing the per-frame N-D
    #                                     rotation of the whole slice basis.  Each dim index
    #                                     appears in at most one plane, so a rotated
    #                                     direction row can grow to at most |dir| <= sqrt(2)
    #                                     (keeps the sphere-marcher's Lipschitz bound valid).
    tumble_mode: str = "rotate"         # tumble transform only: 'rotate' spins the slice
    #                                     basis through full turns (winding * t); 'slide' rocks
    #                                     it back and forth between two extremes by
    #                                     theta(t) = (2*pi*tumble_amp) * sin(2*pi*winding*t),
    #                                     which makes the lattice appear to breathe smaller /
    #                                     larger as the projected frequency swells and shrinks.
    tumble_amp: float = 0.25            # tumble 'slide' mode only: peak swing in turns
    #                                     (0.25 = a quarter-turn rock each way).
    tumble_locked: Tuple[int, ...] = () # tumble transform only: axis indices excluded from the
    #                                     slice-orientation rotation (they stay fixed while the
    #                                     other axes tumble).
    osc_phase: float = 0.0              # constant radians offset added to the shared winding
    #                                     clock (2*pi = one turn); from a --oscillate winder
    #                                     group's `phase`.  0 (legacy) => t=0 is the base field.
    surface: str = "gyroid"             # which surface the field belongs to.  'gyroid'
    #                                     (default): the pairwise Schoen gyroid, sum over
    #                                     coupling edges (a,b) of sin(u_a)*cos(u_b) — uses the
    #                                     --coupling/--pair graph.  'primitive': the Schwarz P
    #                                     surface, sum over each oscillating node d of cos(u_d)
    #                                     — per-node, no edges, so --coupling/--pair do not
    #                                     apply.  Or any POV builtin (f_sphere, f_torus, ...):
    #                                     a solid isosurface sliced from src/pov_functions.h
    #                                     with shape params from pov_values (P3.3).
    pov_values: Tuple[float, ...] = ()   # POV-surface shape-param values in call order (arity
    #                                     - 3 of them), used only when `surface` is a POV
    #                                     builtin.  Empty for the TPMS families and for a
    #                                     0-parameter POV helper.  These are the *base* values
    #                                     (defaults, with any --lock NAME=VALUE pins applied);
    #                                     an animated param (pov_swing) sweeps around its base.
    pov_swing: Dict[str, float] = dc_field(default_factory=dict)
    #                                     S5: POV shape params made --oscillate swinger axes,
    #                                     {param_name: amplitude}.  Per frame the param value is
    #                                     clamp(base + amp*(hi-lo)*env(t), lo, hi) with env the
    #                                     shared sin^2 bump (its own clock in bloom_rates/phases);
    #                                     amp<0 sweeps downward.  Empty => static pov_values.
    pov_motion: bool = False            # S6: the user explicitly asked for a slice MOTION
    #                                     (drift/rotate/tumble via --oscillate or --transform),
    #                                     so a POV surface's (x,y,z) are remapped by the per-frame
    #                                     affine _pov_affine(v,t,transform) before the f_* call —
    #                                     tumble tilts/shears the slice out of 3-space, rotate
    #                                     foreshortens each axis, drift pans it (non-seamless).
    #                                     False (default, and for a pov_swing-only spec) keeps the
    #                                     shape a static f(x,y,z) — the pre-S6 behavior.
    coupling: str = "cyclic"            # which sin*cos pairs the field sums over:
    #                                     'cyclic' (default) = the m consecutive pairs
    #                                     (o_i, o_{i+1}) wrapping around, i.e. the standard
    #                                     Schoen-gyroid generalization (m terms for m oscillating
    #                                     dims); 'all' = every unordered pair i<j, sin(u_i)cos(u_j)
    #                                     (C(m,2) terms — a denser, more interwoven lattice for
    #                                     m>3; identical term *count* to cyclic only at m<=3).
    pair_on: frozenset = frozenset()    # individual coupling *edges* forced on (extra chords):
    #                                     each a frozenset({i, j}); both endpoints oscillate.
    pair_off: frozenset = frozenset()   # individual coupling edges deleted from the field:
    #                                     each a frozenset({i, j}); the sin(u_i)cos(u_j) term is
    #                                     dropped (edits on top of the 'cyclic'/'all' base graph).
    couple_clusters: tuple = ()         # explicit spatial-coupling clusters from --couple: a
    #                                     tuple of (dims_tuple_sorted, scheme) pairs, each an
    #                                     independent group whose members share sin*cos terms
    #                                     among themselves (scheme 'cyclic' ring or 'full' clique).
    #                                     Empty (legacy) => coupling_pairs() uses the --coupling/
    #                                     --pair base-graph path instead.  When set it REPLACES
    #                                     that base graph (disjoint clusters, base+override like
    #                                     --axis); dims in no cluster contribute no gyroid term.
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
# axis-lock parsing:  d:on | d:off | d:on:h   (d may be a range  lo-hi)
# ---------------------------------------------------------------------------

@dataclass
class AxisLock:
    on: Optional[bool] = None
    harmonic: Optional[int] = None


def _parse_axis_indices(token: str, spec: str) -> List[int]:
    """Parse the index field of an --axis spec into a list of axis indices.

    Accepts a single index (``4``) or an inclusive range (``3-6`` -> 3,4,5,6),
    so a run of dimensions can be locked in one flag instead of one per axis."""
    token = token.strip()
    if "-" in token:
        lo_s, _, hi_s = token.partition("-")
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--axis '{spec}': range must be INT-INT (e.g. 3-6)")
        if lo < 0 or hi < 0:
            raise argparse.ArgumentTypeError(f"--axis '{spec}': axis index must be >= 0")
        if hi < lo:
            raise argparse.ArgumentTypeError(
                f"--axis '{spec}': range end {hi} is before start {lo}")
        return list(range(lo, hi + 1))
    try:
        idx = int(token)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--axis '{spec}': axis index must be an integer")
    if idx < 0:
        raise argparse.ArgumentTypeError(f"--axis '{spec}': axis index must be >= 0")
    return [idx]


# ---------------------------------------------------------------------------
# value-lock spec grammar:  V | LO-HI | A,B,C
# ---------------------------------------------------------------------------
#
# The numeric locks (--dims, --oscillating, --harmonics, --freq) accept, like the
# index field of --axis, three shapes:
#   * a bare value  V      -> fixed (never randomized; draws nothing from the RNG)
#   * a range      LO-HI    -> a uniform random pick in [LO, HI] (inclusive)
#   * a set        A,B,C    -> a uniform random pick among the listed values
# A fixed value takes no RNG draw (so it can never perturb reproducibility); a range
# draws exactly as the legacy --*-range fallback did (randint for ints, uniform for
# freq), so existing seeds still reproduce bit-for-bit.  Parsed into a small tuple:
#   ("fixed", v) | ("range", lo, hi) | ("set", [v0, v1, ...])

def _parse_value_spec(token: str, name: str, integral: bool = True,
                      minimum=None):
    """Parse a value-lock spec (``V`` / ``LO-HI`` / ``A,B,C``) into a tuple.

    ``integral`` selects int vs float parsing; ``minimum`` (if given) is an
    inclusive lower bound enforced on every value.  Raises ArgumentTypeError on a
    malformed spec so argparse reports it cleanly."""
    token = str(token).strip()
    conv = int if integral else float
    kind = "integer" if integral else "number"

    def _num(s):
        try:
            return conv(s.strip())
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(
                f"{name}: '{token}' — expected a {kind}, a range LO-HI, or a set A,B,C")

    def _check(v):
        if minimum is not None and v < minimum:
            raise argparse.ArgumentTypeError(f"{name}: value {v} must be >= {minimum}")
        return v

    if "," in token:
        parts = [p for p in token.split(",") if p.strip() != ""]
        if not parts:
            raise argparse.ArgumentTypeError(f"{name}: empty set '{token}'")
        return ("set", [_check(_num(p)) for p in parts])
    hy = token.find("-", 1)                 # a hyphen that isn't a leading sign
    if hy != -1:
        lo = _check(_num(token[:hy]))
        hi = _check(_num(token[hy + 1:]))
        if hi < lo:
            raise argparse.ArgumentTypeError(
                f"{name}: range end {hi} is before start {lo} in '{token}'")
        return ("range", lo, hi)
    return ("fixed", _check(_num(token)))


def _dims_spec(token):
    return _parse_value_spec(token, "--dims", integral=True, minimum=3)


def _oscillating_spec(token):
    return _parse_value_spec(token, "--oscillating", integral=True, minimum=2)


def _harmonics_spec(token):
    return _parse_value_spec(token, "--harmonics", integral=True, minimum=0)


def _freq_spec(token):
    spec = _parse_value_spec(token, "--freq", integral=False, minimum=None)
    # freq must be strictly positive (a zero/negative frequency is degenerate)
    for v in (spec[1:] if spec[0] != "set" else spec[1]):
        if v <= 0:
            raise argparse.ArgumentTypeError(f"--freq: value {v} must be > 0")
    return spec


def parse_axis_lock(spec: str, locks: Dict[int, AxisLock]) -> None:
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--axis '{spec}': expected INDEX:on|off[:HARMONIC] (e.g. 4:on:3), "
            f"where INDEX is a number or a range (e.g. 3-6)")
    indices = _parse_axis_indices(parts[0], spec)
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
    for idx in indices:
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
# pair-lock parsing:  i,j:on | i,j:off   (an individual coupling *edge*)
# ---------------------------------------------------------------------------
#
# Where --axis operates on *nodes* (dimensions) of the coupling graph, --pair
# operates on individual *edges* — one sin(u_i)*cos(u_j) coupling term.  The
# comma (i,j) names the two endpoints of a single edge, distinct from the hyphen
# range (lo-hi) that --axis uses to name a run of nodes.  ':off' deletes that
# edge from the field; ':on' adds it as an extra chord (both endpoints must
# oscillate, so an 'on' edge forces its endpoints on, mirroring how a forced
# harmonic implies an oscillating axis).

def parse_pair_lock(spec: str, on_set: set, off_set: set) -> None:
    """Parse a single ``--pair i,j:on|off`` spec into the on/off edge sets.

    Each edge is stored as a ``frozenset({i, j})`` (unordered — an edge has no
    intrinsic direction; the field emits sin of the lower index, cos of the
    higher).  Raises on a malformed spec or an edge locked both on and off."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--pair '{spec}': expected I,J:on|off (e.g. 3,5:off), where I and J "
            f"are the two dimension indices of one coupling edge")
    ends = [e.strip() for e in parts[0].split(",")]
    if len(ends) != 2:
        raise argparse.ArgumentTypeError(
            f"--pair '{spec}': an edge needs exactly two endpoints separated by a "
            f"comma (e.g. 3,5); got '{parts[0]}'")
    try:
        i, j = int(ends[0]), int(ends[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--pair '{spec}': endpoints must be integers (e.g. 3,5)")
    if i < 0 or j < 0:
        raise argparse.ArgumentTypeError(f"--pair '{spec}': axis index must be >= 0")
    if i == j:
        raise argparse.ArgumentTypeError(
            f"--pair '{spec}': an edge needs two distinct endpoints (got {i},{j})")
    state = parts[1].strip().lower()
    if state in ("on", "osc", "oscillate", "true", "1", "yes"):
        on = True
    elif state in ("off", "no", "false", "0", "static", "inert"):
        on = False
    else:
        raise argparse.ArgumentTypeError(
            f"--pair '{spec}': state must be 'on' or 'off', got '{parts[1]}'")
    edge = frozenset((i, j))
    if on:
        if edge in off_set:
            raise argparse.ArgumentTypeError(
                f"--pair: edge {i},{j} locked both on and off")
        on_set.add(edge)
    else:
        if edge in on_set:
            raise argparse.ArgumentTypeError(
                f"--pair: edge {i},{j} locked both on and off")
        off_set.add(edge)


# ---------------------------------------------------------------------------
# --couple: spatial-coupling clusters (see examples/OSCILLATE_GRAMMAR.md §6)
# ---------------------------------------------------------------------------
#
# --couple is the field (space) counterpart to --oscillate (time): each CLUSTER is
# a comma-joined set of dim indices that share sin*cos terms *among themselves*;
# space-separated clusters are DISJOINT coupling groups (a dim in at most one).  A
# cluster's internal scheme is the ring (`cyclic`, default) or clique (`full`), set
# globally by --couple-scheme or per-cluster with a trailing `:full`/`:cyclic` tag.

def parse_couple(tokens, default_scheme: str = "cyclic"):
    """Parse the ``--couple`` token stream into a tuple of ``(dims_tuple, scheme)``
    clusters.  Each token is one CLUSTER ``d,d,…`` with an optional ``:full``/``:cyclic``
    tag (else ``default_scheme``).  Dims are non-negative ints, sorted and de-duplicated
    within a cluster; clusters must be pairwise disjoint (a dim belongs to one cluster).
    Raises :class:`argparse.ArgumentTypeError` on any malformed / overlapping input."""
    clusters = []
    seen: Dict[int, int] = {}                   # dim -> index of the cluster that owns it
    for tok in (tokens or []):
        spec = tok.strip()
        if spec == "":
            raise argparse.ArgumentTypeError("--couple: empty cluster (stray space?)")
        body, sep, tag = spec.partition(":")
        scheme = default_scheme
        if sep:
            scheme = tag.strip().lower()
            if scheme not in ("cyclic", "full"):
                raise argparse.ArgumentTypeError(
                    f"--couple '{spec}': scheme tag must be ':cyclic' or ':full', "
                    f"got ':{tag}'")
        dims = []
        for part in body.split(","):
            part = part.strip()
            if part == "":
                raise argparse.ArgumentTypeError(
                    f"--couple '{spec}': empty dim index (stray comma?)")
            try:
                d = int(part)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"--couple '{spec}': dim index must be an integer, got '{part}'")
            if d < 0:
                raise argparse.ArgumentTypeError(
                    f"--couple '{spec}': dim index must be >= 0, got {d}")
            if d in seen and seen[d] != len(clusters):
                raise argparse.ArgumentTypeError(
                    f"--couple: dim {d} appears in two clusters — clusters are "
                    f"disjoint (a dim belongs to at most one)")
            if d not in dims:
                dims.append(d)
            seen[d] = len(clusters)
        clusters.append((tuple(sorted(dims)), scheme))
    return tuple(clusters)


def resolve_couple(args: argparse.Namespace) -> None:
    """Normalize ``--couple`` onto the fields the picker reads: ``couple_clusters`` (the
    parsed clusters) and ``couple_axes`` (the flat set of all clustered dims, which the
    picker forces to oscillate, like a ``--pair …:on`` endpoint).  Idempotent.

    ``--couple`` is the primary spatial-coupling surface; it is mutually exclusive with a
    non-default legacy ``--coupling`` or any ``--pair`` (those still work on their own as
    the older base-graph path).  On the legacy path this leaves ``couple_clusters=()`` /
    ``couple_axes=set()`` — no-ops, so those variants stay bit-identical."""
    if getattr(args, "_couple_resolved", False):
        return
    args._couple_resolved = True
    args.couple_clusters = ()
    args.couple_axes = set()

    couple = getattr(args, "couple", None)
    if couple is None:
        return
    if getattr(args, "coupling", "cyclic") != "cyclic":
        raise SystemExit("error: choose the coupling with either --couple or --coupling, "
                         "not both (--couple is the cluster grammar; --coupling is the "
                         "legacy whole-graph scheme)")
    if getattr(args, "pair", None):
        raise SystemExit("error: --pair edits the legacy --coupling base graph; with "
                         "--couple put the two dims in the same cluster instead")
    scheme = getattr(args, "couple_scheme", "cyclic")
    clusters = parse_couple(couple, scheme)
    args.couple_clusters = clusters
    args.couple_axes = {d for members, _ in clusters for d in members}
# ---------------------------------------------------------------------------
#
# One namespace of animatable "change-axes" (spatial dim indices AND named
# motions/parameters), acted on by two verbs that share this grammar:
#
#     --oscillate  <group>  <group>  ...      (motion: these axes move)
#     --lock       <group>  <group>  ...      (held fixed; amp/rate/phase ignored)
#
#     group  =  item , item , ...  [ rate <expr> ] [ phase <expr> ]
#     item   =  [ amp * ] axisname
#
# * comma joins axes into ONE composite oscillator (shared clock, one degree of
#   freedom along the diagonal of its member axes);
# * space separates INDEPENDENT oscillators (each its own clock).
# * amp (per item) = that axis's amplitude / slope of the composite direction.
# * rate/phase (per group) = the shared clock; `rate`/`phase` are RESERVED words
#   greedily absorbed after a group until the next non-keyword token (a new group).
#
# This layer is the pure *parser + model*: it turns the token stream into a list
# of `OscGroup`s and validates the grammar (grouping, amplitudes, rate/phase,
# reserved words, malformed items). It does NOT resolve an axis to winder-vs-
# swinger semantics or check it exists for the chosen surface — that binding is a
# later wiring step (the axis namespace is surface-dependent). So an axis token is
# accepted if it is *syntactically* an axis: a non-negative integer (a spatial dim
# index) or a lowercase identifier — anything but the reserved `rate`/`phase`.

_OSC_RESERVED = ("rate", "phase")
_OSC_AXIS_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class OscGroup:
    """One composite oscillator: weighted axes sharing a clock.

    ``items`` is a list of ``(amplitude, axis_name)`` (the comma-joined members);
    ``rate`` = full cycles of the shared clock over the loop (default 1); ``phase``
    = starting offset in radians (default 0). For a ``--lock`` group the amplitudes
    and rate/phase are parsed but semantically ignored (a lock is just "held")."""
    items: List[Tuple[float, str]]
    rate: float = 1.0
    phase: float = 0.0
    rate_set: bool = False              # True iff `rate <expr>` was given explicitly
    phase_set: bool = False             # True iff `phase <expr>` was given explicitly

    def axes(self) -> List[str]:
        return [ax for _, ax in self.items]


def _osc_eval_num(expr: str, what: str) -> float:
    """Safely evaluate a small arithmetic expression (numbers, ``pi``/``tau``/``e``,
    ``+ - * / ** %``, parentheses, unary sign). Used for amplitudes and rate/phase
    so specs like ``pi/2``, ``2*pi``, ``1/3`` work. Rejects anything else."""
    import ast

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name):
            return {"pi": math.pi, "tau": math.tau, "e": math.e}[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return +v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = n.op
            if isinstance(op, ast.Add): return a + b
            if isinstance(op, ast.Sub): return a - b
            if isinstance(op, ast.Mult): return a * b
            if isinstance(op, ast.Div): return a / b
            if isinstance(op, ast.Pow): return a ** b
            if isinstance(op, ast.Mod): return a % b
        raise ValueError("unsupported expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return float(ev(tree.body))
    except (SyntaxError, ValueError, KeyError, ZeroDivisionError, TypeError,
            OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            f"--oscillate/--lock: bad {what} expression {expr!r} "
            f"(allowed: numbers, pi/tau/e, + - * / ** %, parentheses)") from exc


def _osc_axis(tok: str) -> str:
    """Validate one axis name (a non-negative integer or a lowercase identifier);
    normalize case; reject the reserved words ``rate``/``phase`` and bad forms."""
    ax = tok.strip().lower()
    if ax == "":
        raise argparse.ArgumentTypeError("--oscillate/--lock: empty axis name")
    if ax in _OSC_RESERVED:
        raise argparse.ArgumentTypeError(
            f"--oscillate/--lock: {ax!r} is a reserved word, not an axis name")
    if ax.isdigit():
        return ax                                       # spatial dim index
    if not _OSC_AXIS_RE.match(ax):
        raise argparse.ArgumentTypeError(
            f"--oscillate/--lock: {tok!r} is not a valid axis name "
            f"(expected a non-negative integer or a name like 'tumble')")
    return ax


def _osc_item(part: str) -> Tuple[float, str]:
    """Parse one ``[amp*]axis`` item into ``(amplitude, axis_name)``. The amplitude
    (default 1) may itself be an arithmetic expression; it is split on the LAST
    ``*`` so ``2*pi*tumble`` reads as amp ``2*pi`` on axis ``tumble``."""
    part = part.strip()
    if part == "":
        raise argparse.ArgumentTypeError(
            "--oscillate/--lock: empty item (stray comma?)")
    amp_expr, star, axis_tok = part.rpartition("*")
    if star:
        amp = _osc_eval_num(amp_expr, "amplitude")
    else:
        amp, axis_tok = 1.0, part
    return amp, _osc_axis(axis_tok)


def parse_oscillate(tokens) -> List[OscGroup]:
    """Parse the ``--oscillate`` token stream into a list of :class:`OscGroup`.

    Each group begins with a comma-joined item token, optionally followed by
    ``rate <expr>`` and/or ``phase <expr>`` (either order, each at most once),
    greedily absorbed until the next item token (which starts a new group)."""
    toks = list(tokens or [])
    groups: List[OscGroup] = []
    i = 0
    while i < len(toks):
        head = toks[i]; i += 1
        if head.lower() in _OSC_RESERVED:
            raise argparse.ArgumentTypeError(
                f"--oscillate/--lock: {head!r} must follow a group, not begin one")
        items = [_osc_item(p) for p in head.split(",")]
        rate, phase = 1.0, 0.0
        rate_set = phase_set = False
        seen = set()
        while i < len(toks) and toks[i].lower() in _OSC_RESERVED:
            kw = toks[i].lower(); i += 1
            if kw in seen:
                raise argparse.ArgumentTypeError(
                    f"--oscillate/--lock: {kw!r} given twice for one group")
            seen.add(kw)
            if i >= len(toks):
                raise argparse.ArgumentTypeError(
                    f"--oscillate/--lock: {kw!r} needs an expression after it")
            val = _osc_eval_num(toks[i], kw); i += 1
            if kw == "rate":
                rate = val; rate_set = True
            else:
                phase = val; phase_set = True
        groups.append(OscGroup(items, rate, phase, rate_set, phase_set))
    return groups


def parse_lock_axes(tokens) -> List[str]:
    """Parse the ``--lock`` token stream (same grammar as ``--oscillate``) into a
    flat, de-duplicated, order-preserving list of the axis names to hold fixed.
    Amplitudes and rate/phase are accepted for grammar symmetry but ignored."""
    seen: Dict[str, None] = {}
    for grp in parse_oscillate(tokens):
        for ax in grp.axes():
            seen.setdefault(ax, None)
    return list(seen.keys())


def transform_to_oscillate(transform: str, *, bloom_params=("dims",),
                           bloom_amp: float = 1.0, tumble_mode: str = "rotate",
                           tumble_amp: float = 0.25) -> List[OscGroup]:
    """Desugar today's ``--transform`` (+ its satellite ``--bloom`` / ``--bloom-amp``
    / ``--tumble-mode`` / ``--tumble-amp`` flags) into the canonical ``--oscillate``
    group model (OSCILLATE_GRAMMAR.md §3 migration map).

    This is the behavior-preserving bridge for staging step 2: the *same* semantics
    the tool implements today, re-expressed as a single composite :class:`OscGroup`.
    Layered transforms and bloom parameters all ride one shared clock (rate 1, phase
    0), matching the current single-loop cadence — the migration table maps e.g.
    ``--transform drift,tumble`` -> ``--oscillate drift,tumble`` (one composite) and
    ``--transform bloom --bloom freq,threshold`` -> ``--oscillate bloom,freq,threshold``.

    Item amplitudes capture today's magnitude knobs:

    * winders (``drift``/``rotate``/``tumble``) — amplitude 1 (winding is carried by
      ``rate``, left RNG-random per dim), except ``tumble`` under ``--tumble-mode
      slide`` which becomes ``tumble_amp*tumble`` (its bounded rock in turns);
    * ``bloom`` (the dimensional cross-fade) — amplitude 1, present iff ``dims`` is a
      bloom target (``--bloom dims``; the envelope itself carries no ``--bloom-amp``);
    * scalar swingers (``freq``/``threshold``/``thickness``) — amplitude ``bloom_amp``,
      present iff both the ``bloom`` transform is active and the parameter is a
      ``--bloom`` target.

    Returns ``[]`` for an empty/pinned transform (no motion). The result is a list for
    forward-compatibility with independent oscillators, but today's flags only ever
    produce a single group."""
    items: List[Tuple[float, str]] = []
    has_bloom = _has(transform, "bloom")
    # winder motions in canonical TRANSFORMS order (drift, rotate, tumble)
    for m in _motions(transform):
        if m == "tumble" and tumble_mode == "slide":
            items.append((float(tumble_amp), "tumble"))
        else:
            items.append((1.0, m))
    if has_bloom:
        bp = tuple(bloom_params or ())
        if "dims" in bp:
            items.append((1.0, "bloom"))
        # scalar swingers keep BLOOM_PARAMS order (freq, threshold, thickness)
        for p in ("freq", "threshold", "thickness"):
            if p in bp:
                items.append((float(bloom_amp), p))
    return [OscGroup(items)] if items else []


def oscillate_spec(groups: List[OscGroup]) -> str:
    """Render a list of :class:`OscGroup` back to a canonical ``--oscillate`` spec
    string (inverse of :func:`parse_oscillate` up to formatting). Groups are joined
    by spaces, items within a group by commas; an amplitude of 1 is elided, ``rate``
    and ``phase`` are printed only when non-default. Useful for the ``--transform``
    deprecation notice and for equivalence tests."""
    def fmt_num(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else repr(x)

    out: List[str] = []
    for grp in groups:
        parts = []
        for amp, ax in grp.items:
            parts.append(ax if amp == 1.0 else f"{fmt_num(amp)}*{ax}")
        chunk = ",".join(parts)
        if grp.rate != 1.0:
            chunk += f" rate {fmt_num(grp.rate)}"
        if grp.phase != 0.0:
            chunk += f" phase {fmt_num(grp.phase)}"
        out.append(chunk)
    return " ".join(out)


# Axis kinds for the unified grammar (see OSCILLATE_GRAMMAR.md §2). Winder *motions*
# drive the slice's orientation over the loop; scalar *swingers* pulse a surface
# parameter. `bloom` is the dimensional-crossfade envelope (a swinger with no amp of
# its own). Bare spatial-dim indices are winders too but need per-dim `rate` (P1.4).
_WINDER_MOTIONS = ("drift", "rotate", "tumble")
_SCALAR_SWINGERS = ("freq", "threshold", "thickness")


def resolve_oscillate(args: argparse.Namespace) -> None:
    """Normalize the loop-motion inputs on ``args`` into the canonical fields the rest
    of the generator reads (``transform`` / ``bloom`` / ``bloom_amps`` / ``tumble_mode``
    / ``tumble_amp`` / ``tumble_lock``). Idempotent (guarded by a sentinel) so it can be
    called from both ``main`` and ``pick_variant``.

    Two input paths converge here:

    * **legacy** — ``--transform`` (+ ``--bloom``/``--bloom-amp``/``--tumble-*``): left
      as-is (only defaulting ``transform`` to ``drift`` when unset). Byte-identical to
      before this layer existed.
    * **unified** — ``--oscillate``/``--lock`` (OSCILLATE_GRAMMAR.md): the parsed group
      model is mapped onto those same legacy fields, so ``pick_variant`` needs no new
      code path. Motions (``drift``/``rotate``/``tumble``) become the transform string;
      ``bloom`` and the scalar swingers (``freq``/``threshold``/``thickness``) become the
      ``--bloom`` target set, with each swinger's per-item ``amp`` recorded in
      ``bloom_amps``; ``amp*tumble`` selects slide mode with that amplitude; ``--lock``
      dim indices become ``tumble_lock``. This is the exact inverse of
      :func:`transform_to_oscillate`, so an ``--oscillate`` spec and its equivalent
      ``--transform`` spec produce identical variants.

    Winder ``rate``/``phase`` and bare dim indices (P1.4) resolve to three extra
    outputs the picker honors on top of the legacy fields, all no-ops on the legacy
    path:

    * ``args.osc_dim_windings`` — ``{dim: winding}`` from bare dim-index axes (e.g.
      ``--oscillate 3 rate 2`` -> ``{3: 2}``): each names one winder dim and pins it to
      that **exact** integer winding (``round(amp*rate)``), forcing it to oscillate.
    * ``args.osc_max_winding`` — the ceiling of the RNG-varied ``1..N`` winding cycle,
      taken from an explicit ``rate`` on a **motion** group (``drift``/``rotate``/
      ``tumble``); it overrides ``--max-winding`` while keeping the distinct-rate spread
      (so ``rate`` on a motion is "how fast, at most", consistent with a lone dim index
      whose ``rate`` is its exact winding).
    * ``args.osc_phase`` — a constant radians offset added to the shared winding clock
      (``2*pi`` = one turn), from a winder group's ``phase``.

    The engine has a single shared winding clock, so conflicting motion rates/phases
    across independent groups are rejected.  A swinger axis (``freq``/``threshold``/
    ``thickness``/``bloom``), by contrast, is an independent scalar, so it *does* take its
    own ``rate``/``phase`` (P3.2b): they generalize its ``sin^2`` bloom bump to
    ``0.5*(1-cos(2*pi*rate*t + phase))`` via ``args.bloom_rates`` / ``args.bloom_phases``.
    ``rate 1, phase 0`` (the default) is the legacy single mid-loop bump; an integer rate
    still loops seamlessly, a non-integer one pulses faster but won't close the loop."""
    if getattr(args, "_oscillate_resolved", False):
        return
    args._oscillate_resolved = True
    if getattr(args, "bloom_amps", None) is None:
        args.bloom_amps = {}
    # Per-swinger envelope clocks (P3.2b). Empty => rate 1 / phase 0 everywhere, i.e. the
    # legacy fixed sin^2 bump; the --transform path returns below with these intact.
    if getattr(args, "bloom_rates", None) is None:
        args.bloom_rates = {}
    if getattr(args, "bloom_phases", None) is None:
        args.bloom_phases = {}
    # S5: POV shape-param swingers ({param: amplitude}); populated below when an --oscillate
    # axis names one of the surface's shape params.  Always set so the legacy/early-return
    # paths leave it empty (static pov_values).
    if getattr(args, "pov_swing", None) is None:
        args.pov_swing = {}
    # Winder-clock outputs (P1.4). Always set so the picker's getattr fallbacks are exact;
    # the legacy --transform path returns below with these no-op defaults intact.
    args.osc_dim_windings = {}
    args.osc_max_winding = None
    args.osc_phase = 0.0

    # S6: whether the user explicitly requested a slice MOTION (drift/rotate/tumble).  A POV
    # surface only turns its default static f(x,y,z) into an animated affine remap when this
    # is True; the default "drift" the early-out installs below, and the benign filler "drift"
    # a pov_swing-only spec needs, both leave it False so a plain POV render stays static.
    # Recomputed unconditionally each call (pick_variant re-runs resolve_oscillate).
    args.pov_motion = False

    osc = getattr(args, "oscillate", None)
    lock = getattr(args, "lock", None)
    raw_transform = getattr(args, "transform", None)

    if osc is None and lock is None:
        if raw_transform is None:               # --transform default is None (see parser)
            args.transform = "drift"
        else:
            args.pov_motion = True              # an explicit --transform is a real motion request
        return

    if raw_transform is not None:
        raise SystemExit("error: specify the loop motion with either --transform or "
                         "--oscillate, not both")
    # the legacy satellite flags are subsumed by the grammar; reject them here so an
    # --oscillate spec is never silently overridden by a stray --bloom/--tumble-* flag.
    _legacy = (("--bloom", getattr(args, "bloom", None), None),
               ("--bloom-amp", getattr(args, "bloom_amp", 1.0), 1.0),
               ("--tumble-mode", getattr(args, "tumble_mode", "rotate"), "rotate"),
               ("--tumble-amp", getattr(args, "tumble_amp", 0.25), 0.25),
               ("--tumble-lock", getattr(args, "tumble_lock", None), None))
    for flag, val, default in _legacy:
        if val != default:
            raise SystemExit(f"error: {flag} is a --transform option; with --oscillate use "
                             f"the axis grammar instead (e.g. amplitudes like '2*freq')")

    groups = parse_oscillate(osc or [])
    lock_axes = parse_lock_axes(lock or [])

    motions: List[str] = []
    bloom_active = False
    bloom_params: List[str] = []
    bloom_amps: Dict[str, float] = {}
    bloom_rates: Dict[str, float] = {}      # per-swinger envelope rate (P3.2b)
    bloom_phases: Dict[str, float] = {}     # per-swinger envelope phase (radians)
    tumble_mode, tumble_amp = "rotate", 0.25
    dim_windings: Dict[int, int] = {}       # bare dim-index -> exact winding
    motion_ceilings: set = set()            # distinct explicit rate ceilings from motion groups
    winder_phases: set = set()              # distinct explicit phases from winder groups
    pov_swing: Dict[str, float] = {}        # S5: POV shape-param swingers {param: amp}
    # The surface's named shape-param axes (S5): an --oscillate token matching one of these
    # swings that param rather than being an unknown axis.  Empty for a non-POV surface.
    _surface = getattr(args, "surface", "gyroid")
    pov_axis_map = _pov_param_axis_map(_surface) if _is_pov_surface(_surface) else {}

    for grp in groups:
        g_axes = [ax for _, ax in grp.items]
        has_winder = any(ax in _WINDER_MOTIONS or ax.isdigit() for ax in g_axes)
        # A group's rate/phase applies to *all* its items uniformly (one shared clock): a
        # winder reads it as the winding ceiling / phase, and each swinger records it as its
        # own envelope clock (P3.2b — the sin^2 bump now runs at that rate/phase instead of
        # the fixed single mid-loop bump).  Independent swingers don't share a basis, so
        # their clocks never conflict (unlike the winders, checked below).
        if grp.phase_set and has_winder:
            winder_phases.add(grp.phase)
        for amp, ax in grp.items:
            if ax in _WINDER_MOTIONS:
                if ax == "tumble" and amp != 1.0:
                    tumble_mode, tumble_amp = "slide", float(amp)
                if ax not in motions:
                    motions.append(ax)
                # rate on a motion caps the varied winding cycle (option 2: "how fast, at
                # most"); amp on drift/rotate is unusual and amp on tumble already means the
                # slide amplitude, so the ceiling reads the group rate only.
                if grp.rate_set:
                    motion_ceilings.add(int(round(grp.rate)))
            elif ax == "bloom" or ax in _SCALAR_SWINGERS:
                bloom_active = True
                key = "dims" if ax == "bloom" else ax
                if key not in bloom_params:
                    bloom_params.append(key)
                if ax in _SCALAR_SWINGERS:
                    bloom_amps[ax] = float(amp)     # 'bloom'/dims carries no amp of its own
                if grp.rate_set:
                    bloom_rates[key] = grp.rate
                if grp.phase_set:
                    bloom_phases[key] = grp.phase
            elif ax.isdigit():
                # A lone dim index is the atomic winder: its amp*rate (turns) is that dim's
                # exact winding, and naming it pins it on (base+override, Q3).
                d = int(ax)
                w = int(round(amp * grp.rate))
                if w < 1:
                    raise SystemExit(f"error: --oscillate: dim {d} resolves to winding {w} "
                                     f"(amp {amp:g} * rate {grp.rate:g}); a winder needs at "
                                     f"least 1 whole turn")
                dim_windings[d] = w
            elif ax in pov_axis_map:
                # S5: a POV surface's named shape param swings over the loop like freq/threshold,
                # but range-aware: value(t) = clamp(base + amp*(hi-lo)*env(t), lo, hi).  Recorded
                # apart from the gyroid bloom swingers (it drives pov_values per frame, not the
                # dims cross-fade), sharing the same sin^2 envelope with its own clock in
                # bloom_rates/bloom_phases.  amp<0 sweeps the param downward from its base.
                pov_swing[ax] = float(amp)
                if grp.rate_set:
                    bloom_rates[ax] = grp.rate
                if grp.phase_set:
                    bloom_phases[ax] = grp.phase
            else:
                # A named param of *some other* POV surface (or a plain typo) lands here.  Give a
                # surface-aware hint when the current surface has shape params to offer.
                if pov_axis_map:
                    valid = ", ".join(pov_axis_map.keys())
                    raise SystemExit(f"error: --oscillate: unknown axis {ax!r} (--surface "
                                     f"{_surface} shape params: {valid}; or a motion/dims axis)")
                raise SystemExit(f"error: --oscillate: unknown axis {ax!r}")

    # One shared winding clock in the engine: independent motion groups can't carry
    # different rates or phases, so reject a conflict rather than silently dropping one.
    if len(motion_ceilings) > 1:
        raise SystemExit(f"error: --oscillate: conflicting winder rates {sorted(motion_ceilings)} "
                         f"— the slice has a single shared winding clock, so all motion groups "
                         f"must agree on one rate")
    if len(winder_phases) > 1:
        raise SystemExit(f"error: --oscillate: conflicting winder phases {sorted(winder_phases)} "
                         f"— one shared clock allows only one phase")
    if motion_ceilings:
        args.osc_max_winding = next(iter(motion_ceilings))
        if args.osc_max_winding < 1:
            raise SystemExit("error: --oscillate: a motion 'rate' must be >= 1 (whole turns)")
    if winder_phases:
        args.osc_phase = next(iter(winder_phases))
    args.osc_dim_windings = dim_windings

    canon = [m for m in _WINDER_MOTIONS if m in motions]
    if dim_windings and not canon:
        # Bare dim indices with no named motion animate via drift (the phase-advance that
        # consumes each dim's winding), so make that explicit.
        canon = ["drift"]
    # S6: real slice motion (a named drift/rotate/tumble, or bare dim windings) drives the POV
    # affine remap.  The pov_swing filler "drift" added below is NOT motion, so record this now.
    args.pov_motion = bool(canon)
    if bloom_active:
        canon.append("bloom")
    if not canon and pov_swing:
        # S5: a POV surface animated only by a shape-param swing carries no winder/dims motion,
        # but it *is* animated (pov_values sweep per frame).  field_expr ignores `transform` for a
        # POV surface, so name a benign 'drift' to satisfy the pipeline without a real winder.
        canon = ["drift"]
    if not canon:
        raise SystemExit("error: --oscillate names no motion axes")
    args.transform = "+".join(canon)
    if bloom_active:
        args.bloom = ",".join(p for p in ("dims",) + _SCALAR_SWINGERS if p in bloom_params)
    args.bloom_amps = bloom_amps
    args.bloom_rates = bloom_rates
    args.bloom_phases = bloom_phases
    args.pov_swing = pov_swing          # S5: {param: amp} POV shape-param swingers
    args.tumble_mode = tumble_mode
    args.tumble_amp = tumble_amp
    lock_dims = sorted({int(a) for a in lock_axes if a.isdigit()})
    if lock_dims:
        args.tumble_lock = ",".join(str(d) for d in lock_dims)


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

    resolve_oscillate(args)             # normalize --oscillate/--lock -> canonical fields
    resolve_couple(args)                # normalize --couple -> couple_clusters / couple_axes
    transform = _parse_transforms(getattr(args, "transform", "drift") or "drift")
    bloom_params = (_parse_bloom_params(getattr(args, "bloom", None))
                    if _has(transform, "bloom") else ())
    bloom_dims = "dims" in bloom_params    # crossfade the full N-D field in (vs. only pulsing
    #                                        scalar params around the fixed classic gyroid)

    # Winder-clock outputs from --oscillate (P1.4). No-ops on the legacy --transform path
    # ({}/None/0.0), so those variants stay bit-identical.
    osc_dim_windings = {int(d): int(w) for d, w in
                        (getattr(args, "osc_dim_windings", {}) or {}).items()}
    osc_max_winding = getattr(args, "osc_max_winding", None)
    osc_phase = float(getattr(args, "osc_phase", 0.0) or 0.0)
    eff_max_w = max(1, osc_max_winding if osc_max_winding is not None else args.max_winding)

    # coupling-edge edits (--pair): endpoints of every referenced edge must be valid
    # dims, and an 'on' edge forces both its endpoints to oscillate (an edge needs two
    # waving axes to couple), mirroring how a forced harmonic implies an oscillating axis.
    pair_on = frozenset(getattr(args, "pair_on", frozenset()))
    pair_off = frozenset(getattr(args, "pair_off", frozenset()))
    pair_on_axes = set().union(*pair_on) if pair_on else set()
    pair_ref_axes = pair_on_axes | (set().union(*pair_off) if pair_off else set())

    # --couple clusters (P2.1): every clustered dim is forced to oscillate (a coupling
    # cluster is only meaningful for waving dims), exactly like a --pair …:on endpoint.
    couple_clusters = tuple(getattr(args, "couple_clusters", ()) or ())
    couple_axes = {int(d) for d in (getattr(args, "couple_axes", set()) or set())}

    # 1) total dimension count -------------------------------------------------
    # bare dim-index axes in --oscillate are forced-on winders, so they raise the dim floor
    # just like an --axis/--pair reference; so do --couple cluster members.
    max_forced_axis = max([*axis_locks, *pair_ref_axes, *osc_dim_windings, *couple_axes],
                          default=-1)
    dims_spec = getattr(args, "dims", None)
    floor = max(max_forced_axis + 1, 3)         # smallest legal D given forced axes
    if dims_spec is not None:
        if dims_spec[0] == "fixed":
            D = dims_spec[1]
            if D < floor:
                src = "--axis/--pair" if pair_ref_axes else "--axis"
                raise SystemExit(f"error: {src} references axis {max_forced_axis} but "
                                 f"--dims is {D} (axis index must be < dims)")
        elif dims_spec[0] == "range":
            lo = max(dims_spec[1], floor)
            hi = max(dims_spec[2], lo)
            D = rng.randint(lo, hi)
        else:                                   # set: keep only values leaving room for locks
            opts = [v for v in dims_spec[1] if v >= floor]
            if not opts:
                src = "--axis/--pair" if pair_ref_axes else "--axis"
                raise SystemExit(f"error: --dims set {dims_spec[1]} has no value >= {floor} "
                                 f"(needed to hold the forced {src} axes)")
            D = rng.choice(opts)
    else:
        lo = max(args.dims_range[0], floor)
        hi = max(args.dims_range[1], lo)
        D = rng.randint(lo, hi)

    # 2) forced on/off + forced harmonics -------------------------------------
    forced_on = {d for d, lk in axis_locks.items() if lk.on is True}
    forced_off = {d for d, lk in axis_locks.items() if lk.on is False}
    forced_harm = {d: lk.harmonic for d, lk in axis_locks.items()
                   if lk.harmonic is not None}
    for d in forced_harm:                       # a forced harmonic implies oscillating
        forced_on.add(d)
    forced_on |= pair_on_axes                   # an 'on' coupling edge implies both endpoints wave
    forced_on |= set(osc_dim_windings)          # a bare dim-index axis pins that dim on (Q3)
    forced_on |= couple_axes                    # a --couple cluster member must oscillate (like :on)
    osc_win_conflict = set(osc_dim_windings) & forced_off
    if osc_win_conflict:
        raise SystemExit(f"error: axis {sorted(osc_win_conflict)} is both named in --oscillate "
                         f"(forced on) and locked off (--axis d:off / --lock)")
    couple_conflict = couple_axes & forced_off
    if couple_conflict:
        raise SystemExit(f"error: axis {sorted(couple_conflict)} is in a --couple cluster "
                         f"(forced on) and locked off (--axis d:off / --lock)")
    conflict = forced_on & forced_off
    if conflict:
        raise SystemExit(f"error: axis {sorted(conflict)} locked both on and off "
                         f"(check --axis / --pair …:on)")
    # axis polarity default: fill in every axis not named by an explicit lock.  'on'
    # forces all remaining axes to oscillate (override individuals with --axis d:off);
    # 'off' forces them inert (override with --axis d:on); 'random' (default) leaves the
    # picker free.  forced_off / forced_on (and pair-on endpoints) always win, so the
    # default only touches the still-free axes — mirroring the coupling base polarity.
    axis_default = getattr(args, "axis_default", "random")
    if axis_default == "on":
        for d in range(D):
            if d not in forced_off:
                forced_on.add(d)
    elif axis_default == "off":
        for d in range(D):
            if d not in forced_on:
                forced_off.add(d)
    must_on = sorted(forced_on)
    must_off = sorted(forced_off)

    # 3) how many oscillate ----------------------------------------------------
    lo_m = max(2, len(must_on))
    hi_m = D - len(must_off)
    if hi_m < lo_m:
        raise SystemExit(f"error: cannot satisfy oscillation locks — need at least "
                         f"{lo_m} oscillating dims but only {hi_m} are available")
    osc_spec = getattr(args, "oscillating", None)
    if osc_spec is not None:
        if osc_spec[0] == "fixed":
            M = osc_spec[1]
            if M < lo_m or M > hi_m:
                raise SystemExit(f"error: --oscillating {M} is out of range [{lo_m}, {hi_m}] "
                                 f"given the current locks / dims")
        elif osc_spec[0] == "range":
            lo = max(osc_spec[1], lo_m)
            hi = min(osc_spec[2], hi_m)
            if hi < lo:
                raise SystemExit(f"error: --oscillating range {osc_spec[1]}-{osc_spec[2]} "
                                 f"doesn't intersect the feasible [{lo_m}, {hi_m}]")
            M = rng.randint(lo, hi)
        else:
            opts = [v for v in osc_spec[1] if lo_m <= v <= hi_m]
            if not opts:
                raise SystemExit(f"error: --oscillating set {osc_spec[1]} has no value in the "
                                 f"feasible [{lo_m}, {hi_m}]")
            M = rng.choice(opts)
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
    harm_spec = getattr(args, "harmonics", None)
    if harm_spec is not None:
        if harm_spec[0] == "fixed":
            H = harm_spec[1]
            if H < lo_h or H > hi_h:
                raise SystemExit(f"error: --harmonics {H} is out of range [{lo_h}, {hi_h}] "
                                 f"given the current locks / oscillating dims")
        elif harm_spec[0] == "range":
            lo = max(harm_spec[1], lo_h)
            hi = min(harm_spec[2], hi_h)
            if hi < lo:
                raise SystemExit(f"error: --harmonics range {harm_spec[1]}-{harm_spec[2]} "
                                 f"doesn't intersect the feasible [{lo_h}, {hi_h}]")
            H = rng.randint(lo, hi)
        else:
            opts = [v for v in harm_spec[1] if lo_h <= v <= hi_h]
            if not opts:
                raise SystemExit(f"error: --harmonics set {harm_spec[1]} has no value in the "
                                 f"feasible [{lo_h}, {hi_h}]")
            H = rng.choice(opts)
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
    # are distinct-ish (1,2,..,max,1,2,..) so no two dims move in lockstep.  A --oscillate
    # 'rate' on a motion group raises the ceiling (eff_max_w) but keeps that spread.
    max_w = eff_max_w
    non_main_osc = [d for d in sorted(osc) if d != main]
    by_index = {dm.index: dm for dm in dims}
    for i, d in enumerate(non_main_osc):
        by_index[d].winding = (i % max_w) + 1
    # Bare dim-index axes (--oscillate 3 rate 2) pin an exact per-dim winding, overriding the
    # varied cycle above.  Applied after (no RNG drawn) so unspecified dims keep their stream.
    for d, w in osc_dim_windings.items():
        if d in by_index and by_index[d].oscillate:
            by_index[d].winding = w

    freq_spec = getattr(args, "freq", None)
    if freq_spec is not None:
        if freq_spec[0] == "fixed":
            freq = freq_spec[1]
        elif freq_spec[0] == "range":
            freq = rng.uniform(freq_spec[1], freq_spec[2])
        else:
            freq = rng.choice(freq_spec[1])
    elif _has(transform, "bloom"):
        # Bloom's frame 0 IS the showcase gyroid; default its density to match showcase
        # (freq 40 at radius 0.32) for whatever container radius is in use.
        freq = SHOWCASE_RF / max(1e-6, args.radius if args.radius is not None else _POV_DEFAULT_RADIUS)
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

    # `tumble` transform: build the ordered word of Givens planes whose product is the per-frame
    # N-D rotation of the whole slice basis.  Two sources:
    #   * an explicit --tumble-sequence (P3.5) — an ORDERED, possibly-overlapping word (list order
    #     = composition order); non-commuting planes reach reorientation paths the disjoint set
    #     can't.  Fully user-specified, so it overrides --tumble-lock.
    #   * otherwise the automatic DISJOINT default — couple each visible axis (0,1,2) with a
    #     distinct hidden dim so the slice tips *out of* the rendered 3-space and back, then pair
    #     leftover hidden dims among themselves.  Disjoint => each direction row mixes with at most
    #     one other, so |rotated dir| <= sqrt(2) (the marcher bound holds).
    # Drawn last (like hidden_offset) and only when needed, so the other transforms' RNG streams —
    # and thus their reproducibility — are untouched.
    tumble_planes: List[Tuple[int, int, int]] = []
    _tl = getattr(args, "tumble_lock", ())
    tumble_locked: Tuple[int, ...] = (_parse_tumble_lock(_tl) if isinstance(_tl, str)
                                      else tuple(sorted(_tl or ())))
    if _has(transform, "tumble"):
        seq = getattr(args, "tumble_sequence", None)
        if seq:
            tumble_planes = _parse_tumble_sequence(seq, D)   # explicit word overrides the default
        else:
            max_w = eff_max_w
            locked = set(a for a in tumble_locked if 0 <= a < D)
            rest = [d for d in range(3, D) if d not in locked]
            rng.shuffle(rest)                               # random hidden partners per seed
            w = 1
            for i in range(min(3, D)):                      # visible axes -> hidden partners
                if i in locked:                             # this axis is pinned; skip it
                    continue
                if not rest:
                    break
                j = rest.pop()
                tumble_planes.append((i, j, w))
                w = w % max_w + 1
            while len(rest) >= 2:                           # pair up leftover hidden dims
                i = rest.pop(); j = rest.pop()
                tumble_planes.append((i, j, w))
                w = w % max_w + 1
            if not tumble_planes and D >= 2:                # nothing paired -> spin two free axes
                free = [d for d in range(D) if d not in locked]
                if len(free) >= 2:
                    tumble_planes.append((free[0], free[-1], 1))

    surface = resolve_surface(getattr(args, "surface", "gyroid"))
    pov_values = pov_default_values(surface) if _is_pov_surface(surface) else ()
    # S4: apply any `--lock NAME=VALUE` shape-param pins, overriding the authored default in the
    # right call-order slot (names/ranges were validated once in main()).
    locks = getattr(args, "pov_param_locks", None)
    if pov_values and locks:
        amap = _pov_param_axis_map(surface)
        pv = list(pov_values)
        for nm, val in locks.items():
            if nm in amap:
                pv[amap[nm][0]] = float(val)
        pov_values = tuple(pv)
    # S7: with --param-default random, draw each UNSPECIFIED param (not pinned, not swung) uniformly
    # in its authored [lo,hi] — this is what makes a POV batch vary (POV surfaces ignore the
    # randomized dims/freq, so without it every variant is the same default shape).  Drawn last in
    # the RNG stream (after hidden_offset/tumble) so it never perturbs the field's reproducibility.
    if pov_values and getattr(args, "param_default", "default") == "random":
        pinned = set(locks or {})
        swung = set(getattr(args, "pov_swing", {}) or {})
        pv = list(pov_values)
        for i, (axis, _desc, _default, (lo, hi)) in enumerate(pov_params(surface)):
            if axis in pinned or axis in swung:
                continue                                # an explicit choice opts out of the draw
            pv[i] = rng.uniform(lo, hi)
        pov_values = tuple(pv)
    return Variant(seed=seed, dims=D, freq=freq, threshold=args.threshold,
                   thickness=args.thickness, pinned=getattr(args, "pin_axes", True),
                   bloom_params=bloom_params, bloom_amp=getattr(args, "bloom_amp", 1.0),
                   bloom_amps=dict(getattr(args, "bloom_amps", {}) or {}),
                   bloom_rates=dict(getattr(args, "bloom_rates", {}) or {}),
                   bloom_phases=dict(getattr(args, "bloom_phases", {}) or {}),
                   tumble_planes=tumble_planes,
                   tumble_mode=getattr(args, "tumble_mode", "rotate"),
                   tumble_amp=getattr(args, "tumble_amp", 0.25),
                   tumble_locked=tumble_locked, osc_phase=osc_phase,
                   surface=surface, pov_values=pov_values,
                   pov_swing=dict(getattr(args, "pov_swing", {}) or {}),
                   pov_motion=bool(getattr(args, "pov_motion", False)),
                   coupling=getattr(args, "coupling", "cyclic"),
                   pair_on=pair_on, pair_off=pair_off, dim_list=dims,
                   couple_clusters=couple_clusters)


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


def _tumbled_directions(v: "Variant", t: float) -> Dict[int, Tuple[float, float, float]]:
    """Every dim's direction row after the ``tumble`` transform's N-D slice rotation at ``t``.

    The rotation is the product of the variant's disjoint Givens planes (each an integer
    number of whole turns over the loop), applied to the *stacked* N x 3 direction matrix —
    i.e. it mixes the direction rows, which is exactly a rigid rotation of the 3-D slice
    within the N-D space.  Inert dims carry a direction too, so a plane may swing an
    oscillating axis toward a hidden (inert) one and back: that axis's visible frequency
    fades and re-forms as the slice turns.

    Two motions, per ``v.tumble_mode``:

    * ``rotate`` — each plane's angle is ``2*pi*winding*t``: the slice spins through whole
      turns, so at t=0 and t=1 every angle is a multiple of 2*pi (identity) and the field
      returns exactly to the base gyroid.
    * ``slide`` — each plane's angle is ``(2*pi*tumble_amp)*sin(2*pi*winding*t)``: the slice
      rocks back and forth between two extremes (+/- tumble_amp turns).  Because sin is 0 at
      t=0 and t=1 the loop is still seamless, but instead of a full spin the projected
      lattice frequency swells and shrinks, so the gyroid appears to breathe smaller/larger.

    Either way the disjoint planes keep each rotated direction row at |dir| <= sqrt(2).
    """
    dirs: Dict[int, List[float]] = {d.index: list(d.direction) for d in v.dim_list}
    two_pi = 2.0 * math.pi
    slide = (v.tumble_mode == "slide")
    oph = getattr(v, "osc_phase", 0.0)      # --oscillate winder phase (0 on the legacy path)
    for (i, j, wind) in v.tumble_planes:
        di = dirs.get(i)
        dj = dirs.get(j)
        if di is None or dj is None:
            continue
        if slide:
            a = (two_pi * v.tumble_amp) * math.sin(two_pi * wind * t + oph)
        else:
            a = two_pi * wind * t + oph
        ca, sa = math.cos(a), math.sin(a)
        dirs[i] = [ca * di[k] - sa * dj[k] for k in range(3)]
        dirs[j] = [sa * di[k] + ca * dj[k] for k in range(3)]
    return {idx: (d[0], d[1], d[2]) for idx, d in dirs.items()}


def _classic_gyroid_expr(freq: float) -> str:
    """The plain 3-D Schoen gyroid on the world X/Y/Z axes at ``freq`` — exactly the
    field in ``scenes/showcase.ftsl`` (``sin(f x)cos(f y) + sin(f y)cos(f z) +
    sin(f z)cos(f x)``).  This is the fixed frame-0 subject for the ``bloom`` transform."""
    f = fmt(freq)
    ux, uy, uz = f"({f}*x)", f"({f}*y)", f"({f}*z)"
    return f"sin({ux})*cos({uy})+sin({uy})*cos({uz})+sin({uz})*cos({ux})"


def _classic_primitive_expr(freq: float) -> str:
    """The plain 3-D Schwarz P (primitive) surface on X/Y/Z at ``freq``:
    ``cos(f x) + cos(f y) + cos(f z)``.  The per-node analogue of the classic gyroid —
    the fixed frame-0 subject for the ``bloom`` transform when ``--surface primitive``."""
    f = fmt(freq)
    return f"cos({f}*x)+cos({f}*y)+cos({f}*z)"


def _classic_expr(surface: str, freq: float) -> str:
    """The classic 3-D field on X/Y/Z for the chosen surface (bloom's frame-0 subject)."""
    if surface == "primitive":
        return _classic_primitive_expr(freq)
    return _classic_gyroid_expr(freq)


# The showcase gyroid's density as a radius*frequency product (radius 0.32, freq 40):
# matching it at any container radius needs freq = SHOWCASE_RF / radius, so the bloom
# base reads as *the showcase gyroid* regardless of the ball size (freq 40 at r=0.32).
SHOWCASE_RF = 0.32 * 40.0

# Supported ways the higher dimensions animate the slice over one loop.  They can be
# *layered*: --transform takes one name or a comma-separated set (e.g. 'drift,tumble' or
# 'drift,rotate,tumble,bloom'), and the field composes them — the three *motions* stack on
# each dim's argument (tumble rotates the slice basis, rotate turns each wavevector out of
# the slice, drift advances the phase) while 'bloom' wraps the whole thing in its classic->
# full cross-fade envelope (the layered motions then animate the full field it reveals).
# Every layer is the identity at t=0 and t=1, so any combination still loops seamlessly.
TRANSFORMS = ("drift", "rotate", "tumble", "bloom")

# Supported triply-periodic minimal-surface families.  Both share the whole N-D slice
# machinery (each oscillating dim d contributes an argument u_d = coeff*(dir_d . xyz) +
# phase, animated by drift/rotate/tumble/bloom); they differ ONLY in how those arguments
# are combined into the scalar field:
#   'gyroid'    — pairwise: sum over coupling edges (a,b) of sin(u_a)*cos(u_b) (the
#                 Schoen gyroid; uses the --coupling/--pair edge graph).
#   'primitive' — per-node: sum over each oscillating dim d of cos(u_d) (the Schwarz P
#                 surface; no edges, so --coupling/--pair have nothing to act on).
SURFACES = ("gyroid", "primitive")

# ---------------------------------------------------------------------------
# Surface catalog for the --list-surfaces / --surface-help discovery commands
# (OSCILLATE_GRAMMAR.md §7, P3.2).  Three honesty groups (DESIGN.md §11.7):
#   'periodic'  — triply-periodic minimal surfaces: they loop seamlessly under a phase
#                 drift and generalize to any --dims.  No shape params (driven only by the
#                 shared freq/threshold/thickness axes).
#   'nd_pov'    — POV builtins in POV_ND_GENERALIZABLE: genuinely fold into an N-D field.
#   'affine_pov'— every other POV builtin: an N-D slice is only an affine remap of x/y/z,
#                 and (being non-periodic) they loop only via a coordinate transform that
#                 returns to itself, never a linear drift.
# `--surface` selection is widened to the whole catalog in P3.3; today it accepts the two
# N-D minimal-surface families below ("gyroid"/"primitive").  The two commands here are
# pure reference output and list the whole library regardless.

# Periodic TPMS: (name, one-line description, aliases).  These are the loom-native N-D
# families plus the plain-3-D fields in loom.iso.FIELDS; none carry shape params.
_TPMS_CATALOG = [
    ("gyroid", "Schoen gyroid - pairwise sin(u_a)*cos(u_b) minimal surface", ()),
    ("primitive", "Schwarz P - per-node sum cos(u_d)", ("schwarz_p",)),
    ("schwarz_d", "Schwarz D (diamond) minimal surface", ()),
    ("neovius", "Neovius surface - 3*sum cos + 4*prod cos", ()),
]
_TPMS_ALIASES = {alias: name for name, _desc, aliases in _TPMS_CATALOG for alias in aliases}
_TPMS_SELECTABLE = frozenset(SURFACES)   # which TPMS --surface accepts today (pre-P3.3)


def surface_group(name: str) -> str:
    """The honesty group of surface ``name`` — 'periodic', 'nd_pov', or 'affine_pov'."""
    canon = _TPMS_ALIASES.get(name, name)
    if canon in {n for n, _d, _a in _TPMS_CATALOG}:
        return "periodic"
    if name in POV_ND_GENERALIZABLE:
        return "nd_pov"
    if name in POV_FUNCS:
        return "affine_pov"
    raise ValueError(f"unknown surface {name!r}")


def surface_names() -> List[str]:
    """Every surface the catalog knows (TPMS + all POV builtins), for --list-surfaces."""
    return [n for n, _d, _a in _TPMS_CATALOG] + list(POV_FUNCS)


# ---------------------------------------------------------------------------
# POV surface emission (P3.3): slice one of the POV isosurface builtins as a solid.
# ---------------------------------------------------------------------------
# TPMS families that are catalog-only reference entries — they appear in --list-surfaces
# but have no renderable N-D field yet, so --surface rejects them with a clear message
# (rather than a bare "unknown surface").
_TPMS_CATALOG_ONLY = frozenset({"schwarz_d", "neovius"})

# A conservative fallback Lipschitz bound for a POV field whose true |grad| ceiling can't be
# derived (noise / atan2 / rotation builtins).  For the algebraic builtins S2's active-band
# bounder (loom.pov_grad) computes a tight, rigorous ceiling instead; this default is only used
# when neither the bounder nor the table below applies.  NOTE: for a high-degree algebraic field
# this default is a *massive under-estimate* (f_hunt_surface's true ceiling is ~11000), so it must
# never be relied on for those — the bounder exists precisely to avoid holes there.
_POV_GRAD_DEFAULT = 8.0

# Cheap closed-form |grad f| ceilings, tried before the default when the S2 bounder declines a
# function (or as a fast path for the SDF-like primitives).  Entries here override
# _POV_GRAD_DEFAULT; the bounder (when it returns a value) overrides these.
_POV_GRAD_BOUND = {
    "f_sphere": 1.0,        # -P0 + sqrt(x^2+y^2+z^2): exact SDF, |grad| == 1
    "f_torus": 1.0,         # -P1 + sqrt((sqrt(x^2+z^2)-P0)^2 + y^2): |grad| == 1
}

# Per-function solid orientation (sign) and natural isolevel (level) for the POV builtins.
# ftrace renders the region {field < 0} as the solid.  A POV function's own convention
# decides which side of its zero set is "inside" and at what value the intended surface
# lives, and the two conventions split:
#
#   * SDF-like helpers (f_sphere, f_torus) are NEGATIVE inside and cross zero *on* the
#     surface -> (sign=+1, level=0): {f < 0} is already the solid, emit f unchanged.
#   * Most clamped algebraic builtins (f_heart, f_hunt_surface, ...) are built as
#     r = -(polynomial) then clamped, so they are POSITIVE inside and rail to -10 far
#     outside -> the bare {f < 0} renders the *exterior* (a shape-shaped crater).  We flip
#     with sign=-1 so the solid is {-f < 0} == {f > 0}, the true interior.
#   * A few (f_ellipsoid) are >= 0 everywhere with the surface at a non-zero level -> the
#     natural isolevel is stored in `level` and subtracted before the sign test.
#
# Values were read straight from src/pov_functions.h (the exact C ports).  Un-tabulated
# functions fall back to the honest (sign=+1, level=0) default: their raw {f < 0} is
# emitted as-is, which is correct for the SDF-like ones and a known-imperfect placeholder
# for the clamped ones until each is validated and added here.
_POV_SOLID_META = {
    "f_sphere": (1.0, 0.0),          # -P0 + sqrt(r^2): negative inside, surface at 0
    "f_torus": (1.0, 0.0),           # -P1 + sqrt(...): negative inside, surface at 0
    "f_ellipsoid": (1.0, 1.0),       # sqrt(x^2 P0^2+...): >= 0, surface at level 1
    "f_heart": (-1.0, 0.0),          # r = -((...)^3 - ...), clamped: positive inside
    "f_hunt_surface": (-1.0, 0.0),   # r = -(...), clamped: positive inside
    "f_kummer_surface_v1": (1.0, 0.0),  # negative inside, surface at 0
}


def _pov_solid_meta(name: str) -> Tuple[float, float]:
    """The (inside_sign, natural_level) for POV builtin ``name``.

    ``inside_sign`` is +1 when ``{f < level}`` is already the solid interior, and -1 when
    the function is positive-inside (so the interior is ``{f > level}`` and the emitted
    field must be negated).  ``natural_level`` is the isolevel the intended surface sits on
    (0 for the SDF-like builtins, non-zero for the few that never reach zero).  Falls back
    to the honest ``(+1, 0)`` for functions not yet individually validated.
    """
    return _POV_SOLID_META.get(name, (1.0, 0.0))


def _pov_grad_bound(name: str, values: Tuple[float, ...], box_half: float) -> float:
    """The ``max_gradient`` to emit for POV builtin ``name`` sliced over the cube of half-width
    ``box_half`` at shape params ``values``.

    Prefers S2's tight, rigorous *active-band* bound (:func:`loom.pov_grad.active_band_grad_bound`),
    which is essential — not merely nice — for the high-degree algebraic builtins whose true
    ``|grad f|`` ceiling dwarfs any hand-picked default (understate it and the sphere-marcher
    oversteps and punches holes).  Falls back to the cheap :data:`_POV_GRAD_BOUND` table, then
    the conservative :data:`_POV_GRAD_DEFAULT`, when the bounder can't analyze the field (a
    noise / atan2 / rotation builtin) or its optional deps (numpy/sympy) are missing.
    """
    try:
        from loom.pov_grad import active_band_grad_bound
        b = active_band_grad_bound(name, values, box_half)
        if b is not None and b > 0.0:
            return float(b)
    except Exception:                            # optional deps absent, or an unforeseen field
        pass                                     # -> honest conservative fallback below
    return _POV_GRAD_BOUND.get(name, _POV_GRAD_DEFAULT)


# Default container half-width when a shape can't be auto-sized (unbounded, or no transcribed
# field) and the user gave no explicit --radius.  The small pad multiplies a *derived* extent so
# the surface isn't flush against the clip sphere (a little air around it, and the sphere-marcher
# never grazes the exact tangent).
_POV_DEFAULT_RADIUS = 1.3
_POV_CONTAINER_PAD = 1.08

# Surfaces that are genuinely *thin* (a sheet / ribbon / self-intersecting membrane, not a solid
# body) and so render hollow by default — a solid fill would just look like a lumpy ball.  Any
# ``*_2d`` planar curve is thin too (matched by suffix).  `--shell` forces *any* POV shape thin.
_POV_THIN_SURFACES = frozenset({
    "f_klein_bottle", "f_boy_surface", "f_enneper", "f_steiner_roman",
    "f_cross_cap", "f_roman", "f_witch_hat",
})


def _pov_renders_thin(name: str) -> bool:
    """True if POV builtin ``name`` renders as a thin shell by default (a genuinely 2-manifold /
    self-intersecting surface, or a ``*_2d`` planar curve) rather than a solid body."""
    return name in _POV_THIN_SURFACES or name.endswith("_2d")


def _pov_container_radius(name: str, values: Tuple[float, ...], level: float,
                          radius_arg: Optional[float]) -> float:
    """The container half-width to fit POV builtin ``name``'s surface.

    An explicit ``radius_arg`` (user ``--radius``) always wins — it also *clips* an unbounded
    shape (paraboloid/cylinder/helix) to a finite view.  Otherwise the surface is auto-sized from
    its natural bounding box (:func:`loom.pov_grad.surface_bbox`, padded); a shape with no
    transcribed field, or one whose surface runs off to infinity (``bounded == False``), falls back
    to :data:`_POV_DEFAULT_RADIUS`.
    """
    if radius_arg is not None:
        return float(radius_arg)
    try:
        from loom.pov_grad import surface_bbox
        bb = surface_bbox(name, values, level=level)
        if bb is not None:
            extent, bounded = bb
            if bounded and extent > 1e-6:
                return float(extent) * _POV_CONTAINER_PAD
    except Exception:                                # optional deps absent, or an unforeseen field
        pass                                         # -> conservative default below
    return _POV_DEFAULT_RADIUS


def _is_pov_surface(name: str) -> bool:
    """True if ``name`` selects a POV isosurface builtin (vs a periodic TPMS family)."""
    return name in POV_FUNCS


def resolve_surface(name: str) -> str:
    """Canonicalize a ``--surface`` value to a renderable surface name.

    Resolves TPMS aliases (``schwarz_p`` -> ``primitive``), passes through the two native
    families (``gyroid``/``primitive``) and any POV builtin unchanged, and raises
    :class:`SystemExit` for a catalog-only TPMS (``schwarz_d``/``neovius`` — listed but not
    yet renderable) or an unknown name.
    """
    canon = _TPMS_ALIASES.get(name, name)
    if canon in _TPMS_SELECTABLE:
        return canon
    if canon in _TPMS_CATALOG_ONLY:
        raise SystemExit(
            f"error: surface {name!r} is a catalog-only reference entry (no renderable "
            f"field yet); pick one of {', '.join(_TPMS_SELECTABLE)} or a POV builtin "
            f"(see --list-surfaces).")
    if canon in POV_FUNCS:
        return canon
    avail = ", ".join(surface_names())
    raise SystemExit(f"error: unknown surface {name!r}; see --list-surfaces "
                     f"({len(surface_names())} available: {avail})")


def pov_default_values(name: str) -> Tuple[float, ...]:
    """The authored default shape-param values for POV builtin ``name`` (empty for a
    0-parameter helper), in call order — i.e. what an un-driven slice uses."""
    return tuple(default for _axis, _desc, default, _rng in pov_params(name))


def _pov_param_axis_map(name: str) -> Dict[str, Tuple[int, float, float]]:
    """axis-name -> (call-order index, lo, hi) for POV builtin ``name``'s named shape params.
    Used to resolve a ``--lock NAME=VALUE`` pin (S4) to the right slot in ``pov_values``."""
    return {axis: (i, lo, hi)
            for i, (axis, _desc, _default, (lo, hi)) in enumerate(pov_params(name))}


def _is_pov_param_lock_token(tok: str) -> bool:
    """True if a ``--lock`` token is a POV shape-param pin ``NAME=VALUE`` (the motion/axis
    grammar never uses ``=``, so this is unambiguous)."""
    return "=" in tok


def resolve_pov_param_locks(args: argparse.Namespace) -> None:
    """Extract ``--lock NAME=VALUE`` POV shape-param pins from ``args.lock`` and validate them
    against the resolved ``args.surface`` (S4).

    Mutates ``args`` in place: sets ``args.pov_param_locks`` (``{axis_name: value}``) and strips
    the pin tokens out of ``args.lock`` so the motion/axis grammar (which never uses ``=``) never
    sees them; a pins-only ``--lock`` collapses to ``None`` so it doesn't engage that grammar or
    trip the ``--transform``/``--oscillate`` mutual-exclusion.  Raises :class:`SystemExit` for a
    pin on a non-POV surface or for an unknown param name; warns (once) on an out-of-range value.
    Idempotent-safe: only processes tokens still carrying ``=``.
    """
    args.pov_param_locks = getattr(args, "pov_param_locks", None) or {}
    raw_lock = getattr(args, "lock", None)
    if raw_lock:
        kept = []
        for tok in raw_lock:
            if _is_pov_param_lock_token(tok):
                nm, val = _parse_pov_param_lock(tok)
                args.pov_param_locks[nm] = val
            else:
                kept.append(tok)
        args.lock = kept or None
    if not args.pov_param_locks:
        return
    surface = resolve_surface(getattr(args, "surface", "gyroid"))
    if not _is_pov_surface(surface):
        raise SystemExit(
            f"error: --lock NAME=VALUE pins a POV surface shape param, but --surface "
            f"{surface!r} is not a POV builtin (it has no named shape params; use "
            f"--freq/--threshold/--thickness for a TPMS).")
    amap = _pov_param_axis_map(surface)
    for nm, val in args.pov_param_locks.items():
        if nm not in amap:
            valid = ", ".join(amap.keys()) or "(none)"
            raise SystemExit(
                f"error: --lock {nm}=...: --surface {surface} has no shape param {nm!r} "
                f"(its params: {valid}; see --surface-help {surface}).")
        _idx, lo, hi = amap[nm]
        if not (lo <= val <= hi):
            print(f"[gyroid_nd] warning: --lock {nm}={fmt(val)} is outside {surface}'s "
                  f"authored range [{fmt(lo)}, {fmt(hi)}] (honored anyway).")


def _parse_pov_param_lock(tok: str) -> Tuple[str, float]:
    """Parse a ``NAME=VALUE`` ``--lock`` token into ``(axis_name, value)``.  ``VALUE`` may be an
    arithmetic expression (numbers, ``pi``/``tau``/``e``, ``+ - * / ** %``), like an
    ``--oscillate`` amplitude.  Raises :class:`SystemExit` on a malformed token."""
    name, _eq, val = tok.partition("=")
    name = name.strip().lower()
    if not name:
        raise SystemExit(f"error: --lock {tok!r}: missing parameter name before '='")
    if not val.strip():
        raise SystemExit(f"error: --lock {tok!r}: missing value after '='")
    try:
        value = _osc_eval_num(val, "value")
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: --lock {tok!r}: {exc}")
    return name, value


def _pov_call_expr(name: str, values: Tuple[float, ...],
                   coords: Tuple[str, str, str] = ("x", "y", "z")) -> str:
    """Emit the FTSL call string for POV builtin ``name`` on ``coords`` with shape params
    ``values`` — e.g. ``f_torus(x,y,z,0.8,0.25)``.  ``values`` must match the function's
    param count (arity - 3); a mismatch is a programming error and raises."""
    want = POV_FUNCS[name] - 3
    if len(values) != want:
        raise ValueError(f"{name} takes {want} shape param(s), got {len(values)}")
    args = list(coords) + [fmt(v) for v in values]
    return f"{name}({','.join(args)})"


def _list_surfaces_text() -> str:
    """The --list-surfaces report: every surface grouped by N-D honesty class, one line
    each with its shape-param count and whether it loops seamlessly / generalizes N-D."""
    lines = ["loom surface library - surfaces you can slice into an N-D isosurface.",
             "columns: NAME  params=<shape-param count>  [nd]=generalizes to --dims>3  "
             "[loop]=seamless under a phase drift",
             ""]
    # periodic TPMS
    lines.append("periodic minimal surfaces (nd, loop; no shape params - use freq/threshold/thickness):")
    for name, desc, aliases in _TPMS_CATALOG:
        tag = "" if name in _TPMS_SELECTABLE else "  (catalog-only: not yet renderable)"
        alias = f"  (aka {', '.join(aliases)})" if aliases else ""
        lines.append(f"  {name:<22} params=0  [nd] [loop]  {desc}{alias}{tag}")
    # N-D-generalizable POV
    lines.append("")
    lines.append("N-D-generalizable POV builtins (nd; non-periodic - loop via a returning transform):")
    for name in sorted(POV_ND_GENERALIZABLE):
        n = POV_FUNCS[name] - 3
        lines.append(f"  {name:<22} params={n}  [nd]")
    # affine-only POV
    lines.append("")
    lines.append("affine-only POV builtins (an N-D slice is an affine remap of x/y/z; non-periodic):")
    affine = sorted(n for n in POV_FUNCS if n not in POV_ND_GENERALIZABLE)
    for name in affine:
        n = POV_FUNCS[name] - 3
        lines.append(f"  {name:<22} params={n}")
    lines.append("")
    lines.append(f"{len(surface_names())} surfaces total "
                 f"({len(_TPMS_CATALOG)} periodic, {len(POV_ND_GENERALIZABLE)} N-D POV, "
                 f"{len(affine)} affine-only POV).")
    lines.append("Run --surface-help NAME for one surface's shape parameters.")
    lines.append("Note: --oscillate animates a POV builtin's shape params, and (with --dims>3) "
                 "tumble/rotate/drift move it through the slice: an nd_pov surface genuinely "
                 "folds into a true N-D field, an affine-only surface is reoriented by an affine "
                 "remap of x/y/z.")
    return "\n".join(lines)


def _surface_help_text(name: str) -> str:
    """The --surface-help NAME report: one surface's honesty group and its shape-param
    axes (name, meaning, default, range), or the shared-axis note for a param-free TPMS."""
    canon = _TPMS_ALIASES.get(name, name)
    try:
        group = surface_group(name)
    except ValueError:
        avail = ", ".join(surface_names())
        raise SystemExit(f"error: unknown surface {name!r}; see --list-surfaces "
                         f"({len(surface_names())} available: {avail})")
    lines = [f"surface: {canon}" + (f"  (alias: {name})" if canon != name else "")]
    if group == "periodic":
        desc = next(d for n, d, _a in _TPMS_CATALOG if n == canon)
        lines.append("  group   : periodic minimal surface (generalizes N-D; loops seamlessly)")
        lines.append(f"  field   : {desc}")
        lines.append("  params  : none - shaped by the shared axes freq / threshold / thickness")
        return "\n".join(lines)
    nd = " (generalizes to --dims>3)" if group == "nd_pov" else \
        " (N-D slice is only an affine remap of x/y/z)"
    lines.append(f"  group   : {'N-D-generalizable' if group == 'nd_pov' else 'affine-only'} "
                 f"POV builtin{nd}")
    lines.append("  loop    : non-periodic - seamless motion needs a returning coordinate "
                 "transform, not a linear drift")
    params = pov_params(name)
    if not params:
        lines.append("  params  : none (a 0-parameter helper; just the 3 coordinates)")
    else:
        lines.append(f"  params  : {len(params)} shape parameter(s) - pin one with "
                     f"--lock NAME=VALUE (e.g. --lock {params[0][0]}={fmt(params[0][2])}), "
                     f"or animate it with --oscillate NAME (e.g. --oscillate {params[0][0]}; "
                     f"amp=1 sweeps to the range edge at mid-loop):")
        for axis, pdesc, default, (lo, hi) in params:
            lines.append(f"    {axis:<10} {pdesc}  (default {fmt(default)}, "
                         f"range [{fmt(lo)}, {fmt(hi)}])")
    return "\n".join(lines)


def _parse_transforms(spec: str) -> str:
    """Normalize a ``--transform`` value — one name or a comma/plus-separated set — into a
    canonical ``'+'``-joined string in :data:`TRANSFORMS` order, deduped.  e.g.
    ``'tumble,drift'`` -> ``'drift+tumble'``.  Raises on an unknown name or an empty spec."""
    names = [s.strip().lower() for s in str(spec).replace("+", ",").split(",") if s.strip()]
    if not names:
        raise SystemExit(f"error: --transform needs at least one of {', '.join(TRANSFORMS)}")
    bad = [n for n in names if n not in TRANSFORMS]
    if bad:
        raise SystemExit(f"error: --transform '{', '.join(bad)}' not in "
                         f"{{{', '.join(TRANSFORMS)}}}  (layer with commas, e.g. drift,tumble)")
    seen = set(names)
    return "+".join(t for t in TRANSFORMS if t in seen)


def _has(transform: str, name: str) -> bool:
    """True if motion/envelope layer ``name`` is active in a (possibly layered) transform."""
    return name in transform.split("+")


def _motions(transform: str) -> List[str]:
    """The active per-dim *motion* layers (drift/rotate/tumble) of a transform, in canonical
    order — i.e. everything except the 'bloom' envelope."""
    return [t for t in ("drift", "rotate", "tumble") if _has(transform, t)]

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


def _parse_tumble_lock(spec: Optional[str]) -> Tuple[int, ...]:
    """Parse a ``--tumble-lock`` spec (comma-separated axis indices) into a sorted tuple.
    Empty -> ()."""
    out: List[int] = []
    for x in (spec or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            a = int(x)
        except ValueError:
            raise SystemExit(f"error: --tumble-lock '{x}' is not an integer axis index")
        if a < 0:
            raise SystemExit(f"error: --tumble-lock axis index '{a}' must be >= 0")
        if a not in out:
            out.append(a)
    return tuple(sorted(out))


def _parse_tumble_sequence(spec: Optional[str], dims: int) -> List[Tuple[int, int, int]]:
    """Parse a ``--tumble-sequence`` word into an ordered list of ``(i, j, winding)`` Givens
    planes (P3.5).  Grammar: a comma-separated list of ``i-j`` axis pairs, each optionally
    suffixed ``xN`` for a whole-turn count (default 1), e.g. ``0-3,3-4x2,0-4``.

    List order is the composition order and pairs may overlap (share an axis) — that is exactly
    what unlocks order-dependent, non-commuting reorientation the disjoint default cannot reach.
    Validates every axis into ``[0, dims)`` and rejects a self-pair / non-positive turn count.
    """
    planes: List[Tuple[int, int, int]] = []
    for raw in (spec or "").split(","):
        tok = raw.strip()
        if not tok:
            continue
        pair, sep, tc = tok.partition("x")
        winding = 1
        if sep:
            try:
                winding = int(tc)
            except ValueError:
                raise SystemExit(f"error: --tumble-sequence '{tok}': turn count after 'x' "
                                 f"must be an integer (e.g. 0-3x2)")
            if winding < 1:
                raise SystemExit(f"error: --tumble-sequence '{tok}': turn count must be >= 1")
        a, dash, b = pair.partition("-")
        if not dash:
            raise SystemExit(f"error: --tumble-sequence '{tok}': expected an axis pair 'i-j' "
                             f"(optionally 'i-jxN')")
        try:
            i, j = int(a), int(b)
        except ValueError:
            raise SystemExit(f"error: --tumble-sequence '{tok}': axis indices must be integers")
        for ax in (i, j):
            if not (0 <= ax < dims):
                raise SystemExit(f"error: --tumble-sequence '{tok}': axis {ax} is out of range "
                                 f"for --dims {dims} (valid 0..{dims - 1})")
        if i == j:
            raise SystemExit(f"error: --tumble-sequence '{tok}': a plane needs two distinct axes")
        planes.append((i, j, winding))
    return planes


def _tumble_rownorm_factor(v: "Variant") -> float:
    """The rigorous worst-case direction-row norm the tumble word can grow a unit row to (P3.5).

    The per-frame slice rotation ``R(t)`` (product of the variant's Givens planes) is applied to
    the stacked unit direction rows; row ``i`` of ``R(t)·D`` can only draw amplitude from the dims
    in ``i``'s connected component of the plane graph, so ``|row_i| <= sqrt(|component_i|)`` (
    Cauchy–Schwarz on an orthonormal row of ``R``).  The bound is therefore ``sqrt(max component
    size)`` — **t-independent, rigorous, and auto-``sqrt(2)`` for a disjoint word** (every plane its
    own size-2 component), growing only when planes overlap.  This subsumes the old ``sqrt(2)``
    shortcut with no special-case code and no change to the disjoint default's bound."""
    planes = getattr(v, "tumble_planes", ()) or ()
    if not planes:
        return 1.0
    parent: Dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j, _w) in planes:
        parent.setdefault(i, i)
        parent.setdefault(j, j)
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    sizes: Dict[int, int] = {}
    for a in list(parent):
        r = find(a)
        sizes[r] = sizes.get(r, 0) + 1
    max_size = max(sizes.values()) if sizes else 1
    return math.sqrt(max_size)


def _bloom_env(t: float) -> float:
    """The base bloom envelope w(t) = sin^2(pi t): 0 at t=0 and t=1 (so both loop ends are
    *exactly* the base gyroid and the loop is seamless), 1 at the mid-loop peak."""
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t))


def _bloom_env_p(v: "Variant", key: str, t: float) -> float:
    """The per-swinger bloom envelope for parameter ``key`` at loop phase ``t``.

    Generalizes :func:`_bloom_env` with the swinger's own clock (P3.2b):

        w(t) = 0.5 * (1 - cos(2*pi*rate*t + phase))

    where ``rate`` (``v.bloom_rates[key]``, default 1) is how many full bumps the swinger
    makes over the loop and ``phase`` (``v.bloom_phases[key]`` radians, default 0) offsets
    where it starts.  ``rate=1, phase=0`` is byte-identical to :func:`_bloom_env`.  An
    *integer* rate returns to its start at t=1 for any phase (still a seamless loop); a
    non-integer rate pulses faster but no longer closes the loop.  The range stays [0, 1]
    (peak 1) regardless, so every swing amplitude / gradient bound is unaffected."""
    rate = v.bloom_rates.get(key, 1.0)
    phase = v.bloom_phases.get(key, 0.0)
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * rate * t + phase))


def _swing_amp(v: "Variant", param: str) -> float:
    """The effective swing amplitude of a scalar swinger for this variant: its per-axis
    override in :attr:`Variant.bloom_amps` if present (the unified --oscillate grammar's
    per-item ``amp``), else the shared :attr:`Variant.bloom_amp` scalar (the legacy
    ``--bloom-amp`` path). Keeps the two input paths numerically identical when no
    per-axis override is set."""
    return v.bloom_amps.get(param, v.bloom_amp)


def bloom_freq(v: "Variant", t: float) -> float:
    """The (possibly time-varying) base frequency at loop phase ``t``.  Equals ``v.freq``
    unless 'freq' is a bloom target, in which case it swells to its peak at mid-loop."""
    if "freq" in v.bloom_params:
        return v.freq * (1.0 + _swing_amp(v, "freq") * _BLOOM_SWING["freq"] * _bloom_env_p(v, "freq", t))
    return v.freq


def bloom_threshold(v: "Variant", t: float) -> float:
    """The isosurface level set at ``t`` (shifted from ``v.threshold`` when 'threshold' blooms)."""
    if "threshold" in v.bloom_params:
        return v.threshold + _swing_amp(v, "threshold") * _BLOOM_SWING["threshold"] * _bloom_env_p(v, "threshold", t)
    return v.threshold


def bloom_thickness_scale(v: "Variant", t: float) -> float:
    """Multiplier on the sheet half-width at ``t`` (1 unless 'thickness' blooms)."""
    if "thickness" in v.bloom_params:
        return 1.0 + _swing_amp(v, "thickness") * _BLOOM_SWING["thickness"] * _bloom_env_p(v, "thickness", t)
    return 1.0


def _pov_values_at(v: "Variant", t: float) -> Tuple[float, ...]:
    """POV shape-param values at loop phase ``t`` (S5).

    ``v.pov_values`` holds each param's *base* value (default + any --lock pin); a param named
    as an ``--oscillate`` swinger (recorded in ``v.pov_swing`` as ``{name: amp}``) sweeps around
    it, range-aware so ``amp = 1`` reaches the param's authored extreme exactly at the loop peak:

        p(t) = clamp(base + amp * span * env(t),  lo, hi),   span = (hi - base) if amp >= 0
                                                                    else (base - lo)

    where ``env(t)`` is the shared sin^2 bump (its own rate/phase via ``bloom_rates``/
    ``bloom_phases``): 0 at the loop ends, rising to 1 at mid-loop, so the animation loops
    seamlessly and touches the extreme for a single instant (no plateau).  ``amp > 0`` sweeps up
    toward ``hi`` (``amp = 1`` -> exactly ``hi`` at the peak), ``amp < 0`` down toward ``lo``;
    ``|amp| > 1`` over-drives and is clamped to the authored range.  With no swingers this returns
    the base tuple unchanged.
    """
    base = tuple(getattr(v, "pov_values", ()))
    swing = getattr(v, "pov_swing", None) or {}
    if not swing or not base:
        return base
    amap = _pov_param_axis_map(getattr(v, "surface", ""))
    vals = list(base)
    for nm, amp in swing.items():
        slot = amap.get(nm)
        if slot is None:
            continue
        idx, lo, hi = slot
        span = (hi - base[idx]) if amp >= 0 else (base[idx] - lo)
        val = base[idx] + amp * span * _bloom_env_p(v, nm, t)
        vals[idx] = min(hi, max(lo, val))
    return tuple(vals)


# Rows of the 3x3 identity — the un-moved directions of the three visible slice axes.
_EYE_ROWS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _pov_affine(v: "Variant", t: float, transform: str
                ) -> Tuple[List[List[float]], List[float]]:
    """S6: the per-frame affine remap ``(M, b)`` a POV surface's ``(x, y, z)`` pass through
    before the ``f_*`` call — the honest realization of "an N-D slice of a 3-D POV field is an
    affine remap of (x,y,z)".  The emitted call becomes ``f(M0.p + b0, M1.p + b1, M2.p + b2, …)``.

    Rows 0/1/2 are the three *visible* slice axes' world directions (dims 0,1,2), composed from
    the same motion layers as the periodic field, but read as coordinate axes rather than
    wavevectors:

    * ``tumble`` — the whole slice basis rotates in N-D (``_tumbled_directions``), so a visible
      axis mixes with a hidden dim: its row tilts/foreshortens out of the rendered 3-space and
      back.  This is the marquee dims>3 effect — the shape is genuinely viewed from a turning
      N-D frame (an ellipsoid rotates; an asymmetric shape shears).  Identity at t=0 and t=1.
    * ``rotate`` — each visible axis turns edge-on independently: its row scales by ``cos(alpha)``
      (the shape stretches along that world axis as the coordinate foreshortens) and picks up a
      ``hidden_offset * sin(alpha)`` translation.  Identity at t=0 and t=1.
    * ``drift`` — each axis translates by ``winding * t`` world units: a linear pan.  Unlike the
      periodic field (where a whole-cycle phase advance loops seamlessly), a non-periodic POV
      shape does **not** return at t=1, so drift on a POV surface is deliberately *non-seamless*
      (the user opts into it; tumble/rotate are the seamless motions).

    Returns the identity remap (``M = I``, ``b = 0``) when the variant carries no explicit motion
    (``v.pov_motion`` False) — the default, keeping a plain POV render a static ``f(x, y, z)``.
    """
    if not getattr(v, "pov_motion", False):
        return [list(r) for r in _EYE_ROWS], [0.0, 0.0, 0.0]
    do_drift = _has(transform, "drift")
    do_rotate = _has(transform, "rotate")
    do_tumble = _has(transform, "tumble")
    tdirs = _tumbled_directions(v, t) if do_tumble else None
    by_index = {d.index: d for d in v.dim_list}
    two_pi = 2.0 * math.pi
    oph = getattr(v, "osc_phase", 0.0)
    M: List[List[float]] = []
    b: List[float] = []
    for i in range(3):
        dim = by_index.get(i)
        direction = tdirs[i] if (do_tumble and tdirs and i in tdirs) \
            else (dim.direction if dim is not None else _EYE_ROWS[i])
        coeff = 1.0
        offset = 0.0
        if dim is not None:
            if do_rotate:
                alpha = two_pi * dim.winding * t + oph
                coeff *= math.cos(alpha)
                offset += dim.hidden_offset * math.sin(alpha)
            if do_drift:
                offset += dim.winding * t        # linear world-unit pan (non-seamless for POV)
        M.append([coeff * c for c in direction])
        b.append(offset)
    return M, b


def _sym3_eig_extremes(a00: float, a11: float, a22: float,
                       a01: float, a02: float, a12: float) -> Tuple[float, float]:
    """Smallest and largest eigenvalues ``(eig_min, eig_max)`` of the symmetric 3x3 matrix
    ``[[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]]`` via the analytic closed form (Smith's
    trigonometric method; pure stdlib).  Shared by the 3x3 and Nx3 singular-value routines —
    both feed it the Gram matrix ``A^T A`` and take ``sigma = sqrt(eig)``."""
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    if p1 < 1e-18:                                    # already diagonal
        eigs = sorted((a00, a11, a22))
        return eigs[0], eigs[2]
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    # B = (A - qI) / p ; r = det(B) / 2
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = max(-1.0, min(1.0, detB / 2.0))
    phi = math.acos(r) / 3.0
    eig_max = q + 2.0 * p * math.cos(phi)
    eig_min = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    return eig_min, eig_max


def _mat3_singular_extremes(M: List[List[float]]) -> Tuple[float, float]:
    """The smallest and largest singular values ``(sigma_min, sigma_max)`` of a 3x3 matrix ``M``,
    computed from the eigenvalues of the symmetric PSD matrix ``A = M^T M`` via the analytic
    3x3-symmetric-eigenvalue closed form (pure stdlib; ``sigma = sqrt(eig)``).

    Used to make the S6 affine remap rigorous: the emitted field is ``f(M.p + b)`` whose gradient
    is ``M^T grad f``, so ``|grad| <= sigma_max * |grad f|`` (inflate the sphere-marcher bound) and
    the surface's world extent grows by ``1 / sigma_min`` (enlarge the container as an axis
    foreshortens)."""
    # A = M^T M (symmetric 3x3).
    a00 = M[0][0] ** 2 + M[1][0] ** 2 + M[2][0] ** 2
    a11 = M[0][1] ** 2 + M[1][1] ** 2 + M[2][1] ** 2
    a22 = M[0][2] ** 2 + M[1][2] ** 2 + M[2][2] ** 2
    a01 = M[0][0] * M[0][1] + M[1][0] * M[1][1] + M[2][0] * M[2][1]
    a02 = M[0][0] * M[0][2] + M[1][0] * M[1][2] + M[2][0] * M[2][2]
    a12 = M[0][1] * M[0][2] + M[1][1] * M[1][2] + M[2][1] * M[2][2]
    eig_min, eig_max = _sym3_eig_extremes(a00, a11, a22, a01, a02, a12)
    return math.sqrt(max(0.0, eig_min)), math.sqrt(max(0.0, eig_max))


def _matn3_singular_extremes(A: List[List[float]]) -> Tuple[float, float]:
    """The smallest and largest singular values ``(sigma_min, sigma_max)`` of a ``D x 3`` matrix
    ``A`` (``D`` rows, each a 3-vector), from the eigenvalues of the ``3x3`` Gram ``A^T A``.

    Used to make the P3.4 true-N-D remap rigorous: the emitted field is ``F(A.p + c)`` (``A`` the
    ``D x 3`` slice Jacobian), whose gradient is ``A^T grad_xi F``, so ``|grad_p F| <= sigma_max(A)
    * |grad_xi F|`` (inflate the marcher bound) and the surface's world extent grows by
    ``1 / sigma_min(A)`` (enlarge the container as the slice foreshortens)."""
    a00 = sum(r[0] * r[0] for r in A)
    a11 = sum(r[1] * r[1] for r in A)
    a22 = sum(r[2] * r[2] for r in A)
    a01 = sum(r[0] * r[1] for r in A)
    a02 = sum(r[0] * r[2] for r in A)
    a12 = sum(r[1] * r[2] for r in A)
    eig_min, eig_max = _sym3_eig_extremes(a00, a11, a22, a01, a02, a12)
    return math.sqrt(max(0.0, eig_min)), math.sqrt(max(0.0, eig_max))


# Rows of the D x 3 identity embedding: e_0,e_1,e_2 for the three visible slice axes, 0 for
# every hidden dim (the rest slice passes through the N-D field's center, so at t=0 with no
# --oscillate phase the embedded field reduces *exactly* to the base f_*(x,y,z) call).
def _pov_nd_embedding(v: "Variant", t: float, transform: str
                      ) -> Tuple[List[List[float]], List[float]]:
    """P3.4: the per-frame ``D x 3`` slice Jacobian ``A`` and offset ``c`` for a true-N-D POV
    field.  The emitted field is ``F(A_0.p + c_0, …, A_{D-1}.p + c_{D-1})`` — an honest
    ``D``-coordinate generalization of the ``nd_pov`` builtin (see :mod:`loom.pov_nd`), not a mere
    affine remap of ``(x, y, z)``.

    Rest embedding (``t=0``, no motion): ``A_i = e_i`` for the three visible dims ``i < 3`` and
    ``A_d = 0`` for every hidden dim ``d >= 3``, with ``c = 0`` — so ``xi_i = p_i`` and every hidden
    ``xi_d = 0``, collapsing the field back to ``f_*(x, y, z)`` bit-for-bit.

    Motion layers (same as :func:`_pov_affine`, composed in the same order):

    * ``tumble`` — the whole ``D``-frame rotates: the product of the variant's Givens planes mixes
      the ``A`` rows (and ``c``), folding hidden axes *into* the rendered slice.  This is the marquee
      dims>3 effect and the reason the N-D path exists — an ellipsoid genuinely rotates through the
      extra dimension rather than only shearing its 3-D shadow.  Identity at t=0 and t=1.
    * ``rotate`` — each dim's coordinate foreshortens: its row scales by ``cos(alpha)`` and picks up
      a ``hidden_offset * sin(alpha)`` shift in ``c``.  Identity at t=0 and t=1.
    * ``drift`` — each dim's coordinate pans by ``winding * t`` (non-seamless for a non-periodic POV
      field, exactly as in the affine case; the user opts in).
    """
    D = int(getattr(v, "dims", 3))
    A: List[List[float]] = [
        [1.0 if k == d else 0.0 for k in range(3)] if d < 3 else [0.0, 0.0, 0.0]
        for d in range(D)
    ]
    c: List[float] = [0.0] * D
    do_drift = _has(transform, "drift")
    do_rotate = _has(transform, "rotate")
    do_tumble = _has(transform, "tumble")
    two_pi = 2.0 * math.pi
    oph = getattr(v, "osc_phase", 0.0)
    # tumble first: mix the embedding rows (and their offsets), exactly like _tumbled_directions.
    if do_tumble:
        slide = (v.tumble_mode == "slide")
        for (i, j, wind) in v.tumble_planes:
            if i >= D or j >= D:
                continue
            if slide:
                a = (two_pi * v.tumble_amp) * math.sin(two_pi * wind * t + oph)
            else:
                a = two_pi * wind * t + oph
            ca, sa = math.cos(a), math.sin(a)
            Ai, Aj = A[i], A[j]
            A[i] = [ca * Ai[k] - sa * Aj[k] for k in range(3)]
            A[j] = [sa * Ai[k] + ca * Aj[k] for k in range(3)]
            ci, cj = c[i], c[j]
            c[i] = ca * ci - sa * cj
            c[j] = sa * ci + ca * cj
    # then per-dim rotate (foreshorten) + drift (pan) on the resulting rows.
    by_index = {d.index: d for d in v.dim_list}
    for d in range(D):
        dim = by_index.get(d)
        if dim is None:
            continue
        if do_rotate:
            alpha = two_pi * dim.winding * t + oph
            ca = math.cos(alpha)
            A[d] = [ca * x for x in A[d]]
            c[d] += dim.hidden_offset * math.sin(alpha)
        if do_drift:
            c[d] += dim.winding * t
    return A, c


def _pov_nd_coords(A: List[List[float]], c: List[float]) -> List[str]:
    """The ``D`` coordinate expressions ``A_d . (x,y,z) + c_d`` fed to the N-D field builder."""
    return [_arg_expr((row[0], row[1], row[2]), 1.0, off)
            for row, off in zip(A, c)]


def _pov_use_nd(v: "Variant", transform: str) -> bool:
    """True when a POV surface should render as an honest N-D field (P3.4) rather than the S6
    affine remap: the surface has a true-N-D form, motion is on, the ``tumble`` layer is selected
    (the only motion that folds a hidden dim into the slice) and there *is* a hidden dim (``D > 3``)
    to fold.  Every other case — no motion, drift/rotate-only, ``D <= 3``, or an ``affine_pov``
    surface — keeps the exact pre-P3.4 S6 path (byte-identical)."""
    return (bool(getattr(v, "pov_motion", False))
            and _has(transform, "tumble")
            and int(getattr(v, "dims", 3)) > 3
            and getattr(v, "surface", "gyroid") in POV_ND_GENERALIZABLE)


def _pov_affine_coords(M: List[List[float]], b: List[float]) -> Tuple[str, str, str]:
    """The three remapped coordinate expressions ``M_i . (x,y,z) + b_i`` for the ``f_*`` call."""
    return tuple(_arg_expr((row[0], row[1], row[2]), 1.0, off)  # type: ignore[return-value]
                 for row, off in zip(M, b))


def _scheme_edges(dims: List[int], scheme: str) -> List[Tuple[int, int]]:
    """The ordered ``(a, b)`` coupling edges among a run of dims under a base ``scheme``.

    ``dims`` must already be sorted.  ``'all'``/``'full'`` = every unordered pair
    ``i<j`` (the complete graph / clique).  ``'none'`` = no edges.  Anything else
    (``'cyclic'``, the default) = the consecutive ring ``(d_i, d_{i+1})`` mod m; for
    m=2 this is the two mirrored terms ``(d0,d1)`` and ``(d1,d0)`` (deliberately not
    deduplicated — the classic gyroid), and for m<=1 it is empty (an edge needs two
    dims).  Shared by the legacy ``--coupling`` base graph and each ``--couple`` cluster."""
    m = len(dims)
    if scheme in ("all", "full"):
        return [(dims[i], dims[j]) for i in range(m) for j in range(i + 1, m)]
    if scheme == "none":
        return []
    if m < 2:
        return []                       # a ring needs at least two dims
    return [(dims[i], dims[(i + 1) % m]) for i in range(m)]


def coupling_pairs(v: "Variant") -> List[Tuple[int, int]]:
    """The ordered ``(a, b)`` coupling edges of the field — one ``sin(u_a)*cos(u_b)``
    term each.

    Two source models, mutually exclusive:

    * **explicit clusters** (:attr:`Variant.couple_clusters`, from ``--couple``) — each
      cluster contributes :func:`_scheme_edges` among its own oscillating members
      (``cyclic`` ring or ``full`` clique), clusters concatenated in CLI order.  A
      cluster's edges fully define its coupling; ``--pair`` edits do not apply here.
    * **base graph + per-edge edits** (legacy ``--coupling``/``--pair``) — the base is
      the ``cyclic`` ring / ``all`` complete graph / ``none`` empty graph over all
      oscillating dims, then any edge in ``pair_off`` is deleted and any edge in
      ``pair_on`` not already present is added as an extra chord (sin of the lower
      index, cos of the higher).

    Only edges whose both endpoints oscillate survive.  The returned order is
    base-graph (or cluster) order first, then added chords sorted by endpoints (stable,
    for reproducible expressions)."""
    osc = sorted(v.oscillating)
    osc_set = set(osc)

    clusters = getattr(v, "couple_clusters", ())
    if clusters:
        out: List[Tuple[int, int]] = []
        for members, scheme in clusters:
            mem = [d for d in members if d in osc_set]      # only waving dims couple
            out.extend(_scheme_edges(mem, scheme))
        return out

    scheme = getattr(v, "coupling", "cyclic")
    base = _scheme_edges(osc, scheme)
    pair_off = getattr(v, "pair_off", frozenset())
    pair_on = getattr(v, "pair_on", frozenset())
    out: List[Tuple[int, int]] = []
    present = set()
    for (a, b) in base:
        if frozenset((a, b)) in pair_off:
            continue                    # deleted edge
        out.append((a, b))
        present.add(frozenset((a, b)))
    for edge in sorted(pair_on, key=lambda s: sorted(s)):
        a, b = sorted(edge)
        if a not in osc_set or b not in osc_set:
            continue                    # endpoint isn't oscillating -> no term to add
        if frozenset((a, b)) in pair_off or frozenset((a, b)) in present:
            continue                    # conflicting/off, or already an edge
        out.append((a, b))              # extra chord: sin(lower) * cos(higher)
        present.add(frozenset((a, b)))
    return out


def field_expr(v: Variant, t: float = 0.0, transform: str = "drift",
               freq: Optional[float] = None) -> str:
    """Emit the gyroid field at loop phase ``t`` in [0,1) under the chosen ``transform``.

    Every transform reproduces the exact static field at ``t=0`` (and loops seamlessly, so
    ``t=1`` matches ``t=0``), then moves the higher dimensions in between:

    * ``drift`` — translate the slice *through* each dimension: each dim's phase advances
      by ``2*pi*winding*t`` (an integer number of whole cycles over the loop).  The pattern
      slides.
    * ``rotate`` — rotate each dim's wavevector *out of* the 3-D slice into its own hidden
      axis by angle ``2*pi*winding*t``.  The in-slice frequency scales by ``cos`` (the
      stripes widen, vanish, and re-form as the wave turns edge-on and back) while the
      out-of-slice tilt adds a ``sin``-weighted phase from the slice's ``hidden_offset``.
      The lattice genuinely reshapes — the higher-D analogue of turning the object — rather
      than merely sliding.  The main dim (winding 0) stays put and anchors the pattern so it
      never fully dissolves.  Each dim turns *independently* in its own plane.
    * ``tumble`` — rotate the whole **3-D slice** rigidly through the N-D space: a product of
      disjoint Givens rotations (``v.tumble_planes``) is applied to the entire direction-row
      matrix once per frame, coherently reorienting the slice (the visible axes swing out and
      hidden axes swing in) instead of tilting each wavevector on its own.  With >= 4
      oscillating dims real new structure appears; with 3 it is a rigid spin.
    * ``bloom`` — frame 0 (and frame 1) is the *exact classic 3-D gyroid* (the
      ``scenes/showcase.ftsl`` field on X/Y/Z); over the loop the full N-D gyroid is
      cross-blended in and back out by an envelope ``w(t) = sin^2(pi t)`` (0 at the ends,
      1 at the midpoint).  The video therefore begins as the recognizable showcase gyroid
      and *unfolds* into its higher-dimensional structure at mid-loop, then folds back —
      a seamless "bloom".  The higher dimensions still drift while blended in.
    """
    surf = getattr(v, "surface", "gyroid")
    if _is_pov_surface(surf):
        # A POV builtin is a solid field: the surface itself, called on (x,y,z) with its shape
        # params.  It carries none of the periodic-lattice machinery (no freq / harmonics /
        # coupling), so the winder/bloom layers don't build a lattice here.  Two animations it
        # *does* honor: an --oscillate shape-param swing (S5) evaluates the params at this frame's
        # ``t``; and an explicit slice motion (S6) remaps (x,y,z) by the per-frame affine
        # _pov_affine(v,t,transform) — tumble/rotate/drift the shape through the N-D slice.  With
        # no motion the remap is the identity, so the call stays the static f(x,y,z).
        values = _pov_values_at(v, t)
        if not getattr(v, "pov_motion", False):
            return _pov_call_expr(surf, values)          # static: clean f(x,y,z)
        if _pov_use_nd(v, transform):
            # P3.4: a true-N-D field.  The slice tumbles through the extra dimensions and folds
            # them into an honest D-coordinate generalization F(A.p + c) — not just an affine
            # reorientation of f(x,y,z).  At t=0 (and t=1) the embedding is the identity, so this
            # is byte-identical to the static f(x,y,z); in between the hidden axes genuinely bend in.
            A, c = _pov_nd_embedding(v, t, transform)
            return nd_field_expr(surf, _pov_nd_coords(A, c), values)
        M, b = _pov_affine(v, t, transform)
        return _pov_call_expr(surf, values, _pov_affine_coords(M, b))
    if _has(transform, "bloom"):
        # ``bloom`` pins frame 0 (and frame 1) to the base gyroid, then oscillates the
        # selected parameters over the loop with the envelope w = sin^2(pi t).  The
        # frequency swing (if 'freq' blooms) applies to *both* the classic base and the
        # full field, so the whole pattern pulses in intricacy together.
        w = _bloom_env_p(v, "dims", t)
        fr = bloom_freq(v, t)
        g_classic = _classic_expr(getattr(v, "surface", "gyroid"), fr)
        if "dims" not in v.bloom_params:
            # No dimensional bloom: the loop is the recognizable classic gyroid the whole
            # time (frame 0 = showcase); only the scalar parameters pulse around it.
            return g_classic
        # Dimensional bloom: cross-fade the classic gyroid with the full N-D field.  The
        # envelope is 0 at t=0,1 (both ends exactly the classic gyroid, seamless loop) and
        # 1 at t=0.5 (the full higher-D gyroid at its peak).  The full field it reveals is
        # animated by whatever motion layers are *also* selected (drift by default).
        if w <= 1e-9:
            return g_classic
        inner = "+".join(_motions(transform)) or "drift"
        g_full = field_expr(v, t, inner, freq=fr)
        if w >= 1.0 - 1e-9:
            return g_full
        return f"({fmt(1.0 - w)})*({g_classic})+({fmt(w)})*({g_full})"
    fr = v.freq if freq is None else freq
    osc = sorted(v.oscillating)
    by_index = {d.index: d for d in v.dim_list}
    # The three motion layers *compose* on each dim's argument u_d = coeff*(dir . xyz) + phase:
    #   tumble — remap the whole slice basis in N-D (dir_d -> rotated row; the phases stay put
    #            since the slice turns about its anchor);
    #   rotate — turn each wavevector out of the 3-D slice: coeff *= cos(alpha), phase gets a
    #            k*sin(alpha)*hidden_offset term;
    #   drift  — advance the phase by a whole number of cycles over the loop.
    # Each is the identity at t=0,1, so any layered combination still loops seamlessly.
    do_drift, do_rotate, do_tumble = (_has(transform, n) for n in ("drift", "rotate", "tumble"))
    tdirs = _tumbled_directions(v, t) if do_tumble else None
    u = {}
    two_pi = 2.0 * math.pi
    # --oscillate winder phase: a constant radians offset on the shared clock (0 on the
    # legacy path).  It shifts where the loop starts but keeps t=0==t=1 seamless.
    oph = getattr(v, "osc_phase", 0.0)
    for d in osc:
        dim = by_index[d]
        k = dim.harmonic * fr                       # base in-slice frequency for this dim
        direction = tdirs[d] if do_tumble else dim.direction
        coeff = k
        phase = dim.phase
        if do_rotate:
            alpha = two_pi * dim.winding * t + oph
            coeff *= math.cos(alpha)
            phase += k * dim.hidden_offset * math.sin(alpha)
        if do_drift:
            phase += two_pi * dim.winding * t + oph
        # Reduce the phase modulo 2*pi so t=0 and t=1 emit the *same* constant (whole-cycle
        # advances) -> a perfectly seamless loop despite float rounding.
        u[d] = _arg_expr(direction, coeff, phase % two_pi)
    # Combine the per-dim arguments into the scalar field.  This is the ONLY surface-specific
    # step — the u_d above are shared by every surface.
    #   'primitive' (Schwarz P): one cos(u_d) per oscillating node d (no edges).
    #   'gyroid' (default): one sin(u_a)*cos(u_b) per coupling edge (a,b) — the edge set comes
    #     from coupling_pairs(): the 'cyclic' ring / 'all' complete graph / 'none' empty base,
    #     then the per-edge --pair edits (drop 'off' edges, add 'on' chords).
    if getattr(v, "surface", "gyroid") == "primitive":
        terms = [f"cos({u[d]})" for d in osc]
    else:
        terms = [f"sin({u[a]})*cos({u[b]})" for (a, b) in coupling_pairs(v)]
    return "+".join(terms) if terms else "(0.0)"


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


def build_scene(v: Variant, *, t: float = 0.0, res=(480, 480), radius=None,
                env_file: Optional[str] = None, transform: str = "drift",
                material: str = "gold", shell: bool = False,
                shape: str = "sphere", box_size: Optional[float] = None,
                stretch: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Scene:
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
    surface = getattr(v, "surface", "gyroid")
    if _is_pov_surface(surface):
        # A POV builtin renders as a *solid*: the interior bounded by the field's own
        # level set.  No abs()-shell (that would carve a thin sheet out of the solid) and
        # no frequency-scaled Lipschitz bound (the field has no lattice frequency).  Two
        # per-function conventions are honored via _POV_SOLID_META: `level` is the isolevel
        # the intended surface lives on (0 for SDF-like builtins, non-zero for a few), and
        # `sign` orients the solid — ftrace fills {field < 0}, so a positive-inside function
        # (f_heart, ...) must be negated or the render inverts (a shape-shaped crater).  The
        # user threshold shifts the level.  Gradient bound from the per-function table
        # (conservative default until S2 tabulates it); a sign flip leaves |grad| unchanged.
        sign, level = _pov_solid_meta(surface)
        # S5: shape params evaluated at this frame's t (an --oscillate param swing sweeps them);
        # static v.pov_values when nothing swings.  The container + gradient bound below are then
        # recomputed per frame from these values, so an animated param keeps a hole-free march and
        # a correctly-sized container as it grows/shrinks.
        values = _pov_values_at(v, t)
        lvl = level + v.threshold
        if abs(lvl) > 1e-9:
            inner = f"({expr})-({fmt(lvl)})"
            sheet = inner if sign > 0 else f"-({inner})"
        else:
            sheet = f"({expr})" if sign > 0 else f"-({expr})"
        # S3: a genuinely-thin surface (or an explicit --shell) renders hollow — carve a shell of
        # half-thickness v.thickness around the level set (abs(sheet)-t).  The sign flip is moot
        # under abs(); the level/threshold shift is already baked into `sheet`.  Solid shapes skip
        # this and fill the whole {sheet < 0} interior.
        if shell or _pov_renders_thin(surface):
            sheet = f"abs({sheet})-({fmt(v.thickness)})"
        # S3: auto-size the container to the surface's natural bounding box (an explicit --radius
        # overrides / clips unbounded shapes).  This is the *native* extent, before any S6 remap.
        nat_rad = _pov_container_radius(surface, values, level, radius)
        nat_box = nat_rad * 1.05                          # native contained_by half-extent
        if _pov_use_nd(v, transform):
            # P3.4: the emitted field is the honest N-D form F(A.p + c) (D x 3 slice Jacobian A,
            # offset c), whose p-gradient is A^T grad_xi F, so |grad_p F| <= sigma_max(A) *
            # |grad_xi F| and the world extent grows by 1/sigma_min(A) plus the |c| shift — exactly
            # the affine rigor with the D x 3 Jacobian in place of the 3x3 M.
            A, c = _pov_nd_embedding(v, t, transform)
            D = int(getattr(v, "dims", 3))
            sig_min, sig_max = _matn3_singular_extremes(A)
            c_norm = math.sqrt(sum(cc * cc for cc in c))
            c_max = max((abs(cc) for cc in c), default=0.0)
            if radius is None:
                rad = (nat_rad + c_norm) / max(0.15, sig_min)
            else:
                rad = nat_rad
            box = rad * 1.05
            # xi box: |xi_d| = |A_d.p + c_d| <= sigma_max * sqrt(3)*box + max|c_d| over the cube.
            xi_max = sig_max * math.sqrt(3.0) * box + c_max
            grad_xi = nd_grad_bound_xi(surface, values, xi_max, D)
            if grad_xi is None:                           # e.g. f_superellipsoid: no Lipschitz form
                grad_xi = _pov_grad_bound(surface, values, nat_box)
            grad_bound = grad_xi * max(1e-6, sig_max)
            return _assemble_iso_scene(sheet, grad_bound, rad, box, rad, res,
                                       env_file, mat_def, shape, box_size, stretch)
        # S6: the emitted field is f(M.p + b) under the per-frame affine remap.  Its gradient is
        # M^T grad f, so |grad| <= sigma_max(M) * |grad f| (inflate the marcher bound), and the
        # surface's *world* extent grows by 1/sigma_min(M) plus the |b| translation (enlarge the
        # container as an axis foreshortens / the shape pans).  With no motion M = I, b = 0 -> the
        # native values pass through unchanged (exact pre-S6 behavior).
        M, b = _pov_affine(v, t, transform)
        sig_min, sig_max = _mat3_singular_extremes(M)
        b_norm = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
        if radius is None:
            # Auto-sizing: expand the world container to still contain the remapped surface.  Floor
            # sigma_min so a near-edge-on axis (cos alpha -> 0) can't blow the container up without
            # bound; the marcher / view stay sane and an explicit --radius is the way to clip harder.
            rad = (nat_rad + b_norm) / max(0.15, sig_min)
        else:
            rad = nat_rad                                 # explicit --radius clips; don't expand
        box = rad * 1.05                                  # contained_by half-extent
        # S2 bound over the *native* box (where the surface + its un-railed active band live),
        # scaled by sigma_max for the affine.  Sign flip, level/threshold shift and the abs()-shell
        # are pure offsets/reflection — none change |grad f| — so the bound is computed on the raw
        # field and reused verbatim for the emitted `sheet`.
        grad_bound = _pov_grad_bound(surface, values, nat_box) * max(1e-6, sig_max)
        return _assemble_iso_scene(sheet, grad_bound, rad, box, rad, res,
                                   env_file, mat_def, shape, box_size, stretch)
    # Thicken the surface into a solid sheet (showcase's abs(g) - 0.5).  Scale the
    # half-width by sqrt(M/3) so walls stay visible as extra oscillating dims add
    # amplitude (M=3 reproduces the classic 0.5).
    m = max(1, len(v.oscillating))
    if _has(transform, "bloom"):
        w = _bloom_env_p(v, "dims", t)
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
    # Lipschitz bound for the sphere-marcher.  Each surface's field is a sum of terms whose
    # gradient magnitude is bounded by summing each dim's in-slice frequency k_d = freq*h_d
    # over the terms that touch it: |grad f| <= freq * weighted, where
    #   'gyroid'    — a sum of sin(u_a)*cos(u_b) edges; each edge contributes (k_a + k_b), so
    #                 weighted = Σ_d degree_d*h_d (degree_d = how many edges touch dim d).
    #                 Deriving degrees from the *actual* emitted edge list makes the bound exact
    #                 under any --coupling/--pair edit (cyclic -> every dim degree 2 = the classic
    #                 2.2*sum_h with a 10% margin; 'all' -> degree m-1; edits move it in step).
    #   'primitive' — a sum of cos(u_d), one per oscillating node; each contributes k_d, so
    #                 weighted = Σ_d h_d over the oscillating dims (every node has degree 1).
    # Over-estimating only shrinks the safe march step, so the bound never punches holes.
    by_index = {d.index: d for d in v.dim_list}
    if surface == "primitive":
        weighted = sum(by_index[d].harmonic for d in v.oscillating)
        classic_weighted = 3            # classic Schwarz P: 3 nodes, harmonic 1
    else:
        degree: Dict[int, int] = {}
        for (a, b) in coupling_pairs(v):
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        weighted = sum(deg * by_index[idx].harmonic for idx, deg in degree.items())
        classic_weighted = 6            # classic gyroid: 3 dims, degree 2, harmonic 1
    fr = v.freq
    if _has(transform, "bloom"):
        # Bloom cross-fades the full field with the classic 3-D surface, so the effective bound
        # is the larger of the two ends; and the peak (possibly 'freq'-bloomed) frequency
        # applies at this frame.
        weighted = max(weighted, classic_weighted)
        fr = bloom_freq(v, t)
    coef = 1.1
    if _has(transform, "tumble"):
        # The N-D slice rotation grows a direction row's norm, scaling that term's gradient up by
        # the same factor — inflate the bound so the sphere-marcher never oversteps the surface (no
        # holes).  _tumble_rownorm_factor is the rigorous worst-case row norm of the composed
        # rotation: sqrt(2) for the disjoint default (each plane its own size-2 component, so this
        # is byte-identical to the old shortcut) and sqrt(component size) for an overlapping
        # --tumble-sequence word.  (rotate only *shrinks* a term by cos(alpha); no inflation.)
        coef *= _tumble_rownorm_factor(v)
    grad_bound = coef * fr * max(1, weighted)
    rad = radius if radius is not None else _POV_DEFAULT_RADIUS   # TPMS keep the classic default
    box = rad * 1.05                                     # contained_by half-extent
    return _assemble_iso_scene(sheet, grad_bound, rad, box, rad, res,
                               env_file, mat_def, shape, box_size, stretch)


def _assemble_iso_scene(sheet: str, grad_bound: float, radius: float, box: float,
                        r: float, res, env_file: Optional[str], mat_def: str,
                        shape: str = "sphere", box_size: Optional[float] = None,
                        stretch: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Scene:
    """Build the shared studio scene: the isosurface ``function { expr sheet }`` clipped to a
    ball (``shape='sphere'``, default) or an axis-aligned cube (``shape='box'``) inside a
    ``box`` container with the given ``max_gradient``, plus the material and studio (or
    flat-fallback) env light.  The gyroid/primitive and POV paths of :func:`build_scene`
    differ only in how they derive ``sheet``/``grad_bound``, then hand off here.

    The base clip size is ``radius`` for the sphere (its radius).  For the box, the base
    half-extent is ``box_size/2`` when ``box_size`` is given, else the same ``radius`` (a cube
    inscribing the ball's region).  ``stretch`` = ``(sx, sy, sz)`` then scales the clip and the
    ``contained_by`` box per axis: a stretched sphere becomes an ``ellipsoid`` and a stretched
    cube a rectangular box, with the container ``min``/``max`` and camera pull-back rescaled to
    match.  Both non-uniform primitives stay Lipschitz-valid (the field builder folds the
    per-axis radii into the leaf transform), so marching is hole-free."""
    sx, sy, sz = stretch
    if shape == "box":
        he = (box_size * 0.5) if box_size is not None else radius   # base cube half-extent
        hx, hy, hz = he * sx, he * sy, he * sz                      # per-axis half-extents
        # FTSL box `size` is the full edge length per axis.
        clip = f"        box {{ size {fmt(2.0 * hx)} {fmt(2.0 * hy)} {fmt(2.0 * hz)} }}\n"
    else:
        hx, hy, hz = radius * sx, radius * sy, radius * sz          # per-axis semi-axes
        if sx == 1.0 and sy == 1.0 and sz == 1.0:
            clip = f"        sphere {{ center 0 0 0  radius {fmt(radius)} }}\n"
        else:                                                       # stretched sphere -> ellipsoid
            clip = (f"        ellipsoid {{ center 0 0 0"
                    f"  radius {fmt(hx)} {fmt(hy)} {fmt(hz)} }}\n")
    # `contained_by` half-extents (per axis, 5% air) and camera pull-back on the largest axis.
    bx, by, bz = hx * 1.05, hy * 1.05, hz * 1.05
    r = max(hx, hy, hz)

    scene = Scene(Camera(eye=(0.0, 0.7 * r, 4.0 * r), look_at=(0, 0, 0),
                         up=(0, 1, 0), fov_y=36, mode="R", res=res))

    iso = Raw(
        "isosurface {\n"
        '    material "surf"\n'
        "    intersect {\n"
        f'        function {{ expr "{sheet}" }}\n'
        + clip +
        "    }\n"
        f"    contained_by {{ min {fmt(-bx)} {fmt(-by)} {fmt(-bz)}"
        f"  max {fmt(bx)} {fmt(by)} {fmt(bz)} }}\n"
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


def surface_desc(v: Variant, full: bool = False) -> str:
    """Human-readable name of the variant's surface family.  ``full`` adds the field form."""
    surface = getattr(v, "surface", "gyroid")
    if surface == "primitive":
        return ("Schwarz P (primitive) — per-node field, sum_d cos(u_d)" if full
                else "Schwarz P")
    return ("Schoen gyroid — pairwise field, sum_(a,b) sin(u_a)*cos(u_b)" if full
            else "gyroid")


def coupling_desc(v: Variant) -> str:
    """Human-readable summary of the field's term structure.  For the pairwise gyroid this is
    its pairing scheme and sin*cos term count, e.g. 'cyclic (6 consecutive pairs)' or 'all
    pairs (15 = C(6,2))' (with any --pair edits noted).  For the per-node Schwarz P surface it
    is the cos-per-node count instead — there is no coupling graph."""
    m = len(v.oscillating)
    if getattr(v, "surface", "gyroid") == "primitive":
        return f"per-node ({m} cos term{'s' if m != 1 else ''}, one per oscillating dim)"
    actual = len(coupling_pairs(v))
    clusters = getattr(v, "couple_clusters", ())
    if clusters:                                # --couple: explicit disjoint cluster graph
        parts = []
        for members, cscheme in clusters:
            dims = ",".join(str(d) for d in members)
            kind = "clique" if cscheme in ("all", "full") else "ring"
            parts.append(f"{{{dims}}}:{kind}")
        return (f"couple {' '.join(parts)} "
                f"({actual} sin*cos term{'s' if actual != 1 else ''})")
    scheme = getattr(v, "coupling", "cyclic")
    if scheme == "all":
        base = m * (m - 1) // 2
        desc = f"all pairs ({base} = C({m},2) sin*cos terms)"
    elif scheme == "none":
        base = 0
        desc = "none (empty base graph — coupling built from --pair …:on chords)"
    else:
        base = m
        desc = f"cyclic ({m} consecutive-pair sin*cos term{'s' if m != 1 else ''})"
    if actual != base:
        desc += f"  [--pair edits -> {actual} term{'s' if actual != 1 else ''}]"
    return desc


def bloom_params_desc(v: Variant) -> str:
    """Human-readable list of what the bloom oscillates, e.g. 'higher-D structure,
    frequency (complexity)'."""
    names = {"dims": "higher-D structure", "freq": "frequency (complexity)",
             "threshold": "level set", "thickness": "sheet thickness"}
    parts = [names.get(p, p) for p in v.bloom_params]
    return ", ".join(parts) if parts else "higher-D structure"


def tumble_planes_desc(v: Variant) -> str:
    """Human-readable list of the tumble transform's rotation planes, e.g.
    'X<->d4 (1 turn), Y<->d5 (2 turns)' — each an (axis_i, axis_j, winding) Givens plane."""
    parts = []
    for (i, j, wind) in v.tumble_planes:
        turns = "turn" if wind == 1 else "turns"
        parts.append(f"{axis_name(i)}<->{axis_name(j)} ({wind} {turns})")
    return ", ".join(parts) if parts else "(none)"


def header(v: Variant, index: int, count: int, *,
           frames: Optional[int] = None, fps: Optional[float] = None,
           transform: str = "drift", material: str = "gold") -> str:
    osc = v.oscillating
    moving = [d.index for d in v.dim_list if d.oscillate and d.winding > 0]
    L = ["#" + "=" * 74,
         f"# Higher-dimensional {surface_desc(v)} slice — variant {index + 1}/{count}",
         f"# generated by gyroid_nd.py",
         "#",
         f"# variant seed          : {v.seed}   (regenerate: --variant-seed {v.seed} + the same locks)",
         f"# surface family        : {surface_desc(v, full=True)}",
         f"# dimensions (D)        : {v.dims}   (higher/extra dims beyond x,y,z: {max(0, v.dims - 3)})",
         f"# oscillating dims      : {len(osc)}  -> {axis_list(osc)}  (indices {osc})",
         f"# oscillates in         : {osc_harm_list(v)}   (dim(harmonic), the axes that wave)",
         f"# coupling / terms      : {coupling_desc(v)}",
         f"# main dimension        : {axis_name(v.main) if v.main is not None else '-'}   (fundamental, harmonic 1)",
         f"# harmonics of the main : {len(v.harmonic_dims)}  -> {axis_list(v.harmonic_dims)}",
         f"# slice orientation     : {orientation_desc(v)}",
         f"# base spatial frequency: {fmt(v.freq)}",
         f"# level set (threshold) : {fmt(v.threshold)}",
         f"# surface material      : {material}"
         + ("   (conductor / mirror — reflects the studio lights)" if material == "gold"
            else "   (clear BK7 dielectric — lattice reads through refraction)" if material == "glass"
            else "")]
    if frames is not None:
        secs = frames / fps if fps else 0.0
        # Layered transforms compose, so describe every active layer (in canonical order).
        parts = []
        if _has(transform, "drift"):
            parts.append(f"drifting dims -> {moving}")
        if _has(transform, "rotate"):
            parts.append(f"rotating each wavevector out of the slice -> {moving}")
        if _has(transform, "tumble"):
            how = ("rocking +/-{amp} turns".format(amp=fmt(v.tumble_amp))
                   if v.tumble_mode == "slide" else "spinning through full turns")
            parts.append(f"tumbling the whole slice through N-D ({how}): {tumble_planes_desc(v)}")
        if _has(transform, "bloom"):
            parts.append(f"frame 0 = classic showcase gyroid; blooms {bloom_params_desc(v)} at mid-loop")
        motion = "; ".join(parts) if parts else "static"
        L.append(f"# animation             : {frames} frames @ {fmt(fps or 30.0)} fps "
                 f"(~{fmt(secs)}s seamless loop); transform '{transform}'; {motion}")
    # The matrix + offsets view (directions = rows of A, phases = offsets, harmonics).
    L.append("#")
    L += matrix_lines(v)
    # Animation-only detail: which oscillating dims move and how fast (winding), plus role.
    # (tumble moves the whole slice, not per-dim, so it also reports its rotation planes; with
    # layered transforms both the tumble planes and the per-dim drift/rotate table can appear.)
    if frames is not None and _has(transform, "tumble"):
        if v.tumble_mode == "slide":
            L += ["#",
                  f"# tumble mode           : slide  (rocks +/-{fmt(v.tumble_amp)} turns each way,",
                  "#   theta(t) = 2*pi*tumble_amp*sin(2*pi*winding*t); the projected frequency",
                  "#   swells and shrinks, so the lattice appears to breathe smaller/larger)"]
        else:
            L += ["#",
                  "# tumble mode           : rotate  (spins through full turns; winding*t)"]
        if v.tumble_locked:
            L.append(f"# tumble locked axes    : {axis_list(list(v.tumble_locked))}  "
                     f"(indices {list(v.tumble_locked)}; excluded from the slice rotation)")
        L += ["#",
              "# tumble rotation planes — each (axis_i <-> axis_j) turns 'turns' whole",
              "#   times over the loop; together they rigidly rotate the 3-D slice in N-D:",
              "#   plane            turns",
              "#   ---------------  -----"]
        for (i, j, wind) in v.tumble_planes:
            L.append(f"#   {axis_name(i)} <-> {axis_name(j):<8}  {wind:>5}")
    if frames is not None and (_has(transform, "drift") or _has(transform, "rotate")):
        rate_col = "turns" if (_has(transform, "rotate") and not _has(transform, "drift")) else "drift"
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
    if _has(transform, "bloom"):
        gf_motion = ", ".join(_motions(transform)) or "drifting"
        L += ["#",
              f"# bloom parameters      : {', '.join(v.bloom_params)}  (amp {fmt(v.bloom_amp)})",
              f"#   oscillated over the loop by w(t) = sin^2(pi t)  (0 at t=0,1; 1 at t=0.5),",
              f"#   so frame 0 (and frame 1) is exactly the base showcase {surface_desc(v)} — seamless."]
        is_prim = getattr(v, "surface", "gyroid") == "primitive"
        classic_form = ("cos(f x) + cos(f y) + cos(f z)" if is_prim
                        else "sin(f x)cos(f y) + sin(f y)cos(f z) + sin(f z)cos(f x)")
        classic_name = "Schwarz P" if is_prim else "gyroid"
        full_name = "Schwarz P" if is_prim else "gyroid"
        if "dims" in v.bloom_params:
            L += ["#   dims:      F(t) = (1-w)*G_classic + w*G_full",
                  f"#     G_classic = {classic_form}",
                  f"#                 (the showcase {classic_name}; f = base frequency, shown at frame 0)",
                  f"#     G_full    = the full N-D {full_name} below ({gf_motion}), blended in 0->1->0"]
        else:
            L += [f"#   (no 'dims' bloom: the field stays the classic showcase {classic_name} all loop;",
                  "#    only the scalar parameters below pulse around it)"]
        if "freq" in v.bloom_params:
            L.append(f"#   freq:      f(t) = {fmt(v.freq)} * (1 + {fmt(v.bloom_amp * _BLOOM_SWING['freq'])}*w)   (intricacy/complexity pulse)")
        if "threshold" in v.bloom_params:
            L.append(f"#   threshold: thr(t) = {fmt(v.threshold)} + {fmt(v.bloom_amp * _BLOOM_SWING['threshold'])}*w   (channels open/close)")
        if "thickness" in v.bloom_params:
            L.append(f"#   thickness: half(t) *= (1 + {fmt(v.bloom_amp * _BLOOM_SWING['thickness'])}*w)   (sheet swells/thins)")
        L.append("#")
    if getattr(v, "surface", "gyroid") == "primitive":
        field_line = "# field:  sum over oscillating dims d of  cos(u_d)"
    else:
        field_line = "# field:  sum over coupling edges (a, b) of  sin(u_a) * cos(u_b)"
    L += ["#", f"{field_line}   [{coupling_desc(v)}]"]
    motions = _motions(transform)
    if motions:
        # The active motion layers compose on each dim's argument (see field_expr): tumble
        # rotates the direction row, rotate scales the in-slice frequency by cos + adds a sin
        # phase, drift advances the phase.  Build the composed u_d formula from what's on.
        freq_factor = "harmonic_d * freq" + (" * cos(a_d)" if "rotate" in motions else "")
        dir_term = "dir_d(t)" if "tumble" in motions else "dir_d"
        phase_bits = ["phase_d"]
        if "rotate" in motions:
            phase_bits.append("harmonic_d * freq * hidden_offset_d * sin(a_d)")
        if "drift" in motions:
            phase_bits.append("2*pi * winding_d * t")
        L.append(f"#   u_d = {freq_factor} * ({dir_term} . (x, y, z)) + " + " + ".join(phase_bits))
        if "rotate" in motions:
            L += ["#   a_d = 2*pi * winding_d * t   (the dim's wavevector rotates out of the 3-D",
                  "#         slice into its hidden axis; in-slice freq fades as cos, gains a sin phase)"]
        if "tumble" in motions:
            tnote = ("R(t) rocks +/-tumble_amp turns, angle 2*pi*tumble_amp*sin(2*pi*winding*t)"
                     if v.tumble_mode == "slide"
                     else "R(t) spins through whole turns, angle 2*pi*winding*t")
            L += ["#   dir_d(t) = row d of  R(t) @ A   (A = the static direction matrix above;",
                  f"#         R(t) = product of the tumble planes' Givens rotations; {tnote})"]
        L.append("#   (t runs 0->1; every layer is the identity at t=0,1 -> seamless loop)")
    else:
        # bloom-only (no explicit motion layer): the revealed full field drifts by default.
        L += ["#   u_d = harmonic_d * freq * (dir_d . (x, y, z)) + phase_d + 2*pi*winding_d*t",
              "#   (t runs 0->1 over the loop; the bloom envelope above cross-fades it in)"]
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
                  size: Tuple[int, int], raster: bool,
                  noise: Optional[float] = None, time_budget: Optional[float] = None,
                  spp: Optional[int] = None,
                  see_through: bool = False, clarity: Optional[float] = None,
                  raster_iso: Optional[int] = None, raster_gpu: bool = False,
                  pump: Optional["Callable[[], None]"] = None) -> None:
    """Render one frame ``.ftsl`` -> ``.png``, headless and non-blocking.

    ``raster`` uses ftrace ``-raster`` (fast solid-shaded z-buffer preview); otherwise
    a path-traced render whose per-frame budget is set by ``noise`` / ``time_budget`` /
    ``spp`` (path-trace only).  ``size`` is the ``(W, H)`` film resolution, passed as
    ``-r W H``.  No ``-window`` is passed so the process writes the PNG and exits,
    letting the whole frame range run unattended.  The renderer's own console chatter is
    captured (kept off the status line) and only surfaced if the frame fails.

    Path-trace budget (all optional; combine freely — ftrace stops at whichever fires
    first): ``noise`` = stop when the estimated noise falls to this percent (``-noise``);
    ``time_budget`` = wall-clock seconds per frame (``-time``); ``spp`` = fixed sample
    count per pixel (``-spp``).  If none are given, defaults to ``-noise 4``.

    ``see_through`` (raster only) passes ftrace ``-see-through`` so clear dielectrics
    (glass) render as dimmed + milky-hazed rather than a solid pale ghost (no refraction —
    that needs path tracing); ``clarity`` (0..1) sets the per-surface transmittance via
    ``-glass-clarity`` (higher = clearer; ftrace's default is 0.85).

    ``raster_iso`` (raster only) sets ftrace ``-raster-iso <n>`` — the marching-cubes grid
    resolution used to tessellate isosurfaces for the fast preview (ftrace's default 96).
    A lower value cuts the per-frame CPU tessellation cost (the raster-path bottleneck) at
    the price of a coarser surface; ``None`` leaves ftrace's default.

    ``raster_gpu`` (raster only) swaps ftrace ``-raster`` for ``-raster-gpu`` — the GPU
    primary-ray isosurface preview, which casts one ray per pixel straight at the implicit
    surface (NO marching-cubes tessellation, so ``raster_iso`` no longer applies and the
    per-frame CPU tessellation bottleneck disappears entirely).  ftrace falls back to the
    CPU rasterizer automatically when the GPU path can't handle the config (no CUDA device,
    see-through/clarity requested, or a physical mesh-lens camera), so this is always safe
    to pass.

    ``pump`` is an optional callback invoked repeatedly *while* ftrace runs (used to
    keep the preview window's event loop serviced — otherwise it would freeze for the
    whole render, since a path-traced frame can take seconds).  With it the child is
    launched via ``Popen`` and polled instead of blocking in ``subprocess.run``.
    """
    import subprocess
    w, h = size
    if raster:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png),
               "-raster-gpu" if raster_gpu else "-raster",
               "-r", str(w), str(h)]
        if raster_iso is not None and not raster_gpu:   # GPU path does no tessellation
            cmd += ["-raster-iso", str(raster_iso)]
        if clarity is not None:
            cmd += ["-glass-clarity", f"{clarity:g}"]   # implies -see-through
        elif see_through:
            cmd.append("-see-through")
    else:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-r", str(w), str(h),
               "-interval", "8", "-checkpoint"]
        if noise is None and time_budget is None and spp is None:
            noise = 4.0                                  # back-compat default budget
        if noise is not None:
            cmd += ["-noise", f"{noise:g}"]
        if time_budget is not None:
            cmd += ["-time", f"{time_budget:g}"]
        if spp is not None:
            cmd += ["-spp", str(spp)]
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
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]
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
               raster: bool, fmt: str, noise: Optional[float] = None,
               time_budget: Optional[float] = None, spp: Optional[int] = None,
               env_file: Optional[str] = None,
               transform: str = "drift", material: str = "gold",
               clarity: Optional[float] = None, raster_iso: Optional[int] = None,
               raster_gpu: bool = False,
               preview: Optional["_PreviewWindow"] = None,
               video_dir: Optional[Path] = None, shell: bool = False,
               shape: str = "sphere", box_size: Optional[float] = None,
               stretch: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    """Emit ``frames`` morphing scene files, render them, and assemble one video.

    The per-frame ``.ftsl``/``.png`` files land in ``frames_dir`` (its own subdirectory),
    while the assembled video is written to ``video_dir`` if given, else ``out_dir`` (the
    shared collection directory).
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
                            transform=transform, material=material, shell=shell,
                            shape=shape, box_size=box_size, stretch=stretch)
        body = scene.emit(Clock(t=t, frame=k, frames=frames, fps=fps),
                          Cache(), assets_dir=frames_dir, tag=f"{k:0{fw}d}")
        fp = frames_dir / f"{base}_{k:0{fw}d}.ftsl"
        fp.write_text(body, encoding="utf-8")
        png = fp.with_suffix(".png")
        _status(f"{label} | {verb} frame {k + 1}/{frames}")
        _render_frame(ftrace, root, fp, png, size=size, raster=raster, noise=noise,
                      time_budget=time_budget, spp=spp,
                      see_through=clear, clarity=frame_clarity, raster_iso=raster_iso,
                      raster_gpu=raster_gpu,
                      pump=(preview.pump if preview is not None else None))
        pngs.append(png)
        if preview is not None:
            preview.show(png, f"{title_base}  |  frame {k + 1}/{frames}")
    vdir = video_dir if video_dir is not None else out_dir
    vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"{base}.{_video_ext(fmt)}"
    # ffmpeg reads the frames from frames_dir (its cwd) and writes the video to `out`
    # (an absolute path), so the video's directory is independent of where the frames live.
    pattern = f"{base}_%0{fw}d.png"
    _status(f"{label} | assembling {out.suffix.lstrip('.')}")
    _assemble_video(pngs, out, fps=fps, pattern=pattern, cwd=frames_dir)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stretch_factor(spec: str) -> float:
    """Parse a ``--stretch-{x,y,z}`` factor: a strictly-positive float."""
    try:
        v = float(spec)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--stretch: '{spec}' is not a number")
    if not (v > 0.0):
        raise argparse.ArgumentTypeError(f"--stretch: factor {v} must be > 0")
    return v


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
        epilog=("axis-lock format for --axis (repeatable) — operates on nodes (dimensions):\n"
                "  INDEX is a single axis (4) or an inclusive range (3-6 = axes 3,4,5,6)\n"
                "  INDEX:on          force this axis to oscillate (random harmonic role)\n"
                "  INDEX:off         force this axis inert (no term; surface invariant along it)\n"
                "  INDEX:on:H        force it to oscillate at integer harmonic H "
                "(H>=2 = overtone of the main)\n\n"
                "edge-edit format for --pair (repeatable) — operates on coupling edges:\n"
                "  I,J:off           delete the single sin(u_I)*cos(u_J) coupling term\n"
                "  I,J:on            add that term as an extra chord (forces I and J to oscillate)\n"
                "  (comma = one edge's two endpoints; contrast --axis's hyphen node range LO-HI)\n\n"
                "value-lock spec for --dims / --oscillating / --harmonics / --freq:\n"
                "  V                 fixed (never randomized; draws nothing from the RNG)\n"
                "  LO-HI             inclusive range — a uniform random pick in [LO, HI]\n"
                "  A,B,C             set — a uniform random pick among the listed values\n\n"
                "base polarity + override (like --coupling's base edge set):\n"
                "  --axis-default on|off   default oscillation for un-named axes; flip with --axis\n"
                "  --coupling none         empty base graph; build coupling from --pair I,J:on\n\n"
                "surface family (--surface):\n"
                "  gyroid            pairwise Schoen gyroid, sum sin(u_a)*cos(u_b) (default; uses --coupling/--pair)\n"
                "  primitive         Schwarz P, per-node sum cos(u_d) (no edges; --coupling/--pair N/A)\n"
                "  --list-surfaces   print the whole surface library (periodic TPMS / N-D POV / affine POV)\n"
                "  --surface-help N  print one surface's shape parameters (name, default, range)\n\n"
                "examples:\n"
                "  python examples/gyroid_nd.py --count 10\n"
                "  python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 "
                "--oscillating 4 --harmonics 2\n"
                "  python examples/gyroid_nd.py --count 5 --dims 5-8 --oscillating 4,6,8 "
                "--freq 3-5   # range/set locks\n"
                "  python examples/gyroid_nd.py --dims 6 --axis-default on --axis 2:off   "
                "# all axes on but 2\n"
                "  python examples/gyroid_nd.py --dims 6 --coupling none --pair 0,1:on "
                "--pair 1,2:on   # hand-built coupling\n"
                "  python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 "
                "--axis 2:on:1   # classic gyroid\n"
                "  python examples/gyroid_nd.py --dims 7 --axis 3-6:off   "
                "# range: axes 3,4,5,6 inert in one flag\n"
                "  python examples/gyroid_nd.py --dims 6 --oscillating 6 --coupling all   "
                "# all 15 pairs, not 6\n"
                "  python examples/gyroid_nd.py --dims 6 --oscillating 6 --coupling all --pair 0,3:off   "
                "# 15 pairs minus edge 0-3\n"
                "  python examples/gyroid_nd.py --dims 6 --couple 0,1,2 3,4,5   "
                "# two independent 3-dim coupling rings\n"
                "  python examples/gyroid_nd.py --dims 5 --couple 0,1,2:full 3,4   "
                "# one clique cluster + one ring\n"
                "  python examples/gyroid_nd.py --dims 6 --axis 4:on:3 --axis 1:off\n"
                "  python examples/gyroid_nd.py --dims 3 --no-pin-axes   # freely-tilted gyroid slice\n"
                "  python examples/gyroid_nd.py --dims 6 --oscillate bloom   # opens on the showcase gyroid\n"
                "  python examples/gyroid_nd.py --surface primitive --dims 5   # Schwarz P (per-node cos)\n"
                "  python examples/gyroid_nd.py --list-surfaces                # browse the surface library\n"
                "  python examples/gyroid_nd.py --surface-help f_torus         # one surface's params"))

    g = p.add_argument_group("output")
    g.add_argument("-n", "--count", type=int, default=1,
                   help="number of variant .ftsl files to generate (default 1)")
    g.add_argument("--out", type=str, default=None,
                   help="output directory. Accepts an absolute path or one relative to the "
                        "current directory, and writes straight there (no runNNN subdir unless "
                        "you also pass --run-subdir). If omitted, defaults to <repo>/png/gyroid_nd "
                        "with a fresh runNNN subdir per run.")
    g.add_argument("--name", type=str, default="gyroid_nd",
                   help="base filename for the outputs (default gyroid_nd)")
    g.add_argument("--video-out", type=str, default=None,
                   help="directory for the assembled video file(s) ONLY (the per-frame "
                        ".ftsl/.png and sidecars still go in the output dir). Accepts an "
                        "absolute path or one relative to the current directory. If omitted, "
                        "the video is written alongside the frames in the output dir.")
    g.add_argument("--run-subdir", action=argparse.BooleanOptionalAction, default=None,
                   help="put each run in a fresh numbered subdirectory (run001, run002, ...) "
                        "under the output dir so runs never overwrite each other. Default: ON "
                        "when --out is omitted (the auto png/<name> location), OFF when --out is "
                        "given (write straight into the directory you named). --run-subdir / "
                        "--no-run-subdir force it either way.")
    g.add_argument("--seed", type=int, default=None,
                   help="master RNG seed for a reproducible batch (default: random; the "
                        "chosen value is printed so you can reproduce the run)")
    g.add_argument("--variant-seed", type=int, action="append", default=[], metavar="S",
                   help="generate exactly the variant(s) with this seed, bypassing the master "
                        "(repeatable). Reproduces a single variant when combined with the same "
                        "locks it was made with.")

    g = p.add_argument_group("locks (fix a value instead of randomizing it)")
    g.add_argument("--dims", type=_dims_spec, default=None, metavar="V|LO-HI|A,B,C",
                   help="lock the total number of dimensions D (>=3). Accepts a fixed value "
                        "(5), an inclusive range (4-8, random), or a set (4,6,8, random pick); "
                        "a range/set here supersedes --dims-range. (see epilog)")
    g.add_argument("--dims-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                   default=(3, 8), help="range for a random D when --dims is unset (default 3 8)")
    g.add_argument("--oscillating", type=_oscillating_spec, default=None, metavar="V|LO-HI|A,B,C",
                   help="lock how many dimensions oscillate (>=2). Fixed value, range, or set "
                        "(see --dims); a range/set is intersected with the feasible span.")
    g.add_argument("--harmonics", type=_harmonics_spec, default=None, metavar="V|LO-HI|A,B,C",
                   help="lock how many oscillating dims are harmonics (overtones) of the main "
                        "dim. Fixed value, range, or set (see --dims).")
    g.add_argument("--max-harmonic", type=int, default=5,
                   help="largest integer harmonic drawn for an overtone dim (default 5)")
    g.add_argument("--max-winding", type=int, default=2,
                   help="fastest per-dim drift rate: integer cycles a dimension advances "
                        "over one video loop as the slice moves through it (default 2)")
    g.add_argument("--axis-default", choices=("random", "on", "off"), default="random",
                   help="base oscillation polarity for axes NOT named by an explicit --axis: "
                        "'random' (default) lets the picker choose; 'on' makes every un-named "
                        "axis oscillate (then turn individuals off with --axis d:off); 'off' "
                        "makes them all inert (then turn individuals on with --axis d:on). "
                        "Explicit --axis locks and --pair …:on endpoints always win.")
    g.add_argument("--axis", action="append", default=[], metavar="SPEC",
                   help="force an axis (or a LO-HI range of axes) on/off and optionally its "
                        "harmonic; repeatable. INDEX may be a single number (4:off) or an "
                        "inclusive range (3-6:off turns axes 3,4,5,6 off in one flag). "
                        "(see epilog)")
    g.add_argument("--freq", type=_freq_spec, default=None, metavar="V|LO-HI|A,B,C",
                   help="lock the base spatial frequency (cells packed into the ball). Fixed "
                        "value, range, or set (see --dims); a range/set supersedes --freq-range.")
    g.add_argument("--freq-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=(3.0, 7.0), help="range for a random freq when --freq is unset "
                                            "(default 3 7)")
    g.add_argument("--phase0", action="store_true",
                   help="set every phase to 0 (deterministic pattern position) instead of random")
    g.add_argument("--param-default", choices=("default", "random"), default="default",
                   help="how a POV surface's UNSPECIFIED shape params (not pinned by --lock nor "
                        "animated by --oscillate) get their value: 'default' (each param's authored "
                        "default; the current behavior) or 'random' (draw each uniformly within its "
                        "authored [lo,hi] range, per variant seed). 'random' is what gives a POV "
                        "batch (-n N) actual variety — otherwise every variant shares the one default "
                        "shape (POV surfaces ignore the randomized dims/freq/harmonics). No effect on "
                        "a TPMS (its shape comes from freq/threshold, already randomized).")
    g.add_argument("--pin-axes", action=argparse.BooleanOptionalAction, default=True,
                   help="pin the first three dimensions to the world X/Y/Z axes so the slice "
                        "always contains the ordinary xyz volume (default; this is what lets "
                        "D=3 all-on reproduce the exact classic gyroid). --no-pin-axes instead "
                        "gives every dimension a random direction — a freely-oriented N-D slice, "
                        "so even the base gyroid comes out tilted.")
    g.add_argument("--surface", default="gyroid", metavar="NAME",
                   help="which surface to slice (validated at runtime; --list-surfaces shows all). "
                        "'gyroid' (default): the pairwise Schoen gyroid, sum over coupling edges "
                        "(a,b) of sin(u_a)*cos(u_b) — uses the --coupling/--pair edge graph. "
                        "'primitive' (aka schwarz_p): the Schwarz P surface, sum over each "
                        "oscillating dim d of cos(u_d) — a per-node field with no edges, so "
                        "--coupling/--pair do not apply to it (a warning is printed if they are "
                        "given). Both TPMS families share the whole N-D slice machinery "
                        "(--dims/--oscillating/--harmonics/--transform/bloom all work identically). "
                        "Or any POV isosurface builtin (f_sphere, f_torus, f_heart, ...): a solid "
                        "primitive sliced from src/pov_functions.h with its default shape params "
                        "(P3.3; param animation/N-D remap arrive in later slices). Run "
                        "--list-surfaces to see the full library and --surface-help NAME for one "
                        "surface's shape parameters.")
    g.add_argument("--list-surfaces", action="store_true",
                   help="print the whole surface library (periodic TPMS / N-D POV / affine-only "
                        "POV), grouped with each surface's shape-param count and N-D status, then "
                        "exit.")
    g.add_argument("--surface-help", metavar="NAME", default=None,
                   help="print one surface's shape parameters (axis name, meaning, default, range) "
                        "then exit; NAME is any surface from --list-surfaces.")
    g.add_argument("--coupling", choices=("cyclic", "all", "none"), default="cyclic",
                   help="base graph of sin*cos pairs the field sums over (the pair polarity). "
                        "'cyclic' (default): the m consecutive oscillating pairs (o_i, o_{i+1}) "
                        "wrapping around — the standard Schoen-gyroid generalization, m terms. "
                        "'all': every unordered pair i<j — C(m,2) terms (e.g. 15 for 6 "
                        "oscillating dims), a denser lattice (the two agree in count only for "
                        "m<=3). 'none': an empty base graph — build the coupling up entirely "
                        "from --pair I,J:on chords. Edit any base with --pair (see epilog).")
    g.add_argument("--pair", action="append", default=[], metavar="I,J:on|off",
                   help="edit an individual coupling *edge* on top of the --coupling base "
                        "graph; repeatable. 'I,J:off' deletes the sin(u_I)*cos(u_J) term; "
                        "'I,J:on' adds it as an extra chord (both endpoints are forced to "
                        "oscillate). The comma names an edge's two endpoints — distinct from "
                        "--axis's hyphen range (LO-HI) that names a run of nodes. (see epilog)")
    g.add_argument("--couple", nargs="+", default=None, metavar="CLUSTER",
                   help="spatial-coupling clusters — the field counterpart of --oscillate. "
                        "Each CLUSTER is comma-joined dims that share sin*cos coupling terms "
                        "among themselves; spaces separate disjoint clusters (a dim lives in at "
                        "most one). Cluster members are forced to oscillate (like --pair …:on). "
                        "e.g. '--couple 0,1,2 3,4' makes two independent rings. Per-cluster "
                        "scheme tag ':full' (clique) or ':cyclic' (ring, default) overrides "
                        "--couple-scheme. Mutually exclusive with --coupling/--pair (put the two "
                        "dims in one cluster instead of a --pair chord).")
    g.add_argument("--couple-scheme", choices=("cyclic", "full"), default="cyclic",
                   help="default within-cluster wiring for --couple: 'cyclic' (ring of "
                        "consecutive pairs, the Schoen-gyroid generalization) or 'full' (clique, "
                        "every unordered pair). Override per cluster with a ':full'/':cyclic' tag.")

    g = p.add_argument_group("scene")
    g.add_argument("--threshold", type=float, default=0.0,
                   help="isosurface level set f = threshold (default 0; ~+/-0.7 thins the walls)")
    g.add_argument("--shape", choices=("sphere", "box"), default="sphere",
                   help="shape the surface is CSG-clipped to: 'sphere' (default; a ball) or "
                        "'box' (an axis-aligned cube). Sized by --radius (sphere) / --box-size "
                        "(box); the marching container and camera framing follow the shape.")
    g.add_argument("--radius", type=float, default=None,
                   help="radius of the spherical clip/container the surface fills (default: 1.3 "
                        "for TPMS; a POV builtin auto-sizes to its own bounding box, so pass this "
                        "only to override — e.g. to clip an unbounded shape like a paraboloid). "
                        "With --shape box and no --box-size, the cube's half-extent is this radius.")
    g.add_argument("--box-size", type=float, default=None, metavar="S",
                   help="full edge length of the --shape box cube (spans -S/2..S/2). Default: "
                        "twice the sphere --radius (a cube inscribing the same region). Ignored "
                        "unless --shape box.")
    g.add_argument("--stretch-x", type=_stretch_factor, default=1.0, metavar="FX",
                   help="factor to stretch the bounding box/sphere along X (default 1.0): scales "
                        "the clip shape and the contained_by min/max on X. >1 elongates, <1 "
                        "squashes; a stretched --shape sphere becomes an ellipsoid.")
    g.add_argument("--stretch-y", type=_stretch_factor, default=1.0, metavar="FY",
                   help="factor to stretch the bounding box/sphere along Y (default 1.0).")
    g.add_argument("--stretch-z", type=_stretch_factor, default=1.0, metavar="FZ",
                   help="factor to stretch the bounding box/sphere along Z (default 1.0).")
    g.add_argument("--shell", action="store_true",
                   help="render a POV surface hollow (a thin shell of --thickness half-width) "
                        "instead of a solid body; a few genuinely-thin surfaces (klein_bottle, "
                        "boy_surface, enneper, *_2d curves) are thin by default. No effect on TPMS.")
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
    g.add_argument("--oscillate", nargs="*", default=None, metavar="GROUP",
                   help="unified change-axis grammar (OSCILLATE_GRAMMAR.md): space-separated "
                        "GROUPs, each a comma-joined list of [amp*]axis items sharing one "
                        "oscillator. Axes: motions drift/rotate/tumble, the bloom crossfade, "
                        "scalar swingers freq/threshold/thickness (each takes its own amp, e.g. "
                        "'2*freq'), bare spatial-dim indices, and — with a POV --surface — that "
                        "surface's named shape params (e.g. '--surface f_torus --oscillate minor'; "
                        "amp=1 sweeps the param to its authored range edge at mid-loop, amp<0 the "
                        "other way; see --surface-help NAME). A group takes 'rate <expr>' and "
                        "'phase <expr>' (the shared clock): on a motion group 'rate' caps the "
                        "per-dim winding cycle (like --max-winding); a bare dim index winds exactly "
                        "round(amp*rate) turns (e.g. '3 rate 2' winds dim 3 twice) and forces that "
                        "dim on; 'phase' (radians, 2*pi=one turn) offsets the winder clock. Replaces "
                        "--transform + --bloom/--bloom-amp/--tumble-* (mutually exclusive with "
                        "--transform). (swinger 'rate'/'phase' is not wired yet — the bloom "
                        "envelope's clock is fixed.)")
    g.add_argument("--lock", nargs="*", default=None, metavar="GROUP",
                   help="same grammar as --oscillate, naming axes to HOLD FIXED. Currently dim "
                        "indices map to the tumble lock (axes excluded from the slice rotation). "
                        "A 'NAME=VALUE' token instead PINS a POV surface shape param to a fixed "
                        "value (e.g. --lock minor=0.5; see --surface-help NAME for a surface's "
                        "param names/ranges); space-separated pins set several (--lock rx=2 rz=0.5).")
    # Deprecated legacy motion flags (superseded by --oscillate; see OSCILLATE_GRAMMAR.md).
    # Still fully supported — using --transform prints a one-line deprecation note (main) —
    # but hidden from --help so the grammar in --oscillate is the single documented surface.
    g.add_argument("--transform", type=str, default=None, metavar="T[,T...]",
                   help=argparse.SUPPRESS)
    g.add_argument("--bloom", type=str, default=None, metavar="P[,P...]",
                   help=argparse.SUPPRESS)
    g.add_argument("--bloom-amp", type=float, default=1.0, help=argparse.SUPPRESS)
    g.add_argument("--tumble-mode", choices=("rotate", "slide"), default="rotate",
                   help=argparse.SUPPRESS)
    g.add_argument("--tumble-amp", type=float, default=0.25, help=argparse.SUPPRESS)
    g.add_argument("--tumble-lock", type=str, default=None, metavar="A[,A...]",
                   help=argparse.SUPPRESS)
    g.add_argument("--tumble-sequence", type=str, default=None, metavar="i-j[xN],...",
                   help="tumble only: an explicit ORDERED word of Givens planes replacing the "
                        "automatic disjoint pairing — a comma-separated list of axis pairs like "
                        "'0-3,3-4,0-4' (each optionally carrying a whole-turn count, e.g. "
                        "'0-3x2'). List order is the composition order and pairs MAY overlap "
                        "(share an axis), which yields order-dependent reorientation the disjoint "
                        "default cannot reach. Overrides --tumble-lock. Plain --oscillate tumble "
                        "with no --tumble-sequence keeps the tidy automatic default.")
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
    g.add_argument("--raster-iso", type=int, default=None, metavar="N",
                   help="rasterizer only: marching-cubes grid resolution ftrace uses to "
                        "tessellate isosurfaces each frame (ftrace default 96). Lower = "
                        "faster per frame (less CPU tessellation, the raster-path bottleneck) "
                        "but a coarser surface. Ignored under --no-raster.")
    g.add_argument("--raster-gpu", action="store_true",
                   help="rasterizer only: render each frame with ftrace -raster-gpu, the GPU "
                        "primary-ray isosurface preview (one ray per pixel straight at the "
                        "implicit surface, NO marching-cubes tessellation — so --raster-iso is "
                        "moot and the CPU tessellation bottleneck disappears). Falls back to the "
                        "CPU rasterizer automatically when the GPU can't handle the config (no "
                        "CUDA device, see-through/clear glass, or a physical lens camera). "
                        "Ignored under --no-raster.")
    g.add_argument("--preview", action="store_true",
                   help="show each frame as it is rendered in one reusable preview window "
                        "whose title tracks the current gyroid / values / frame")
    g.add_argument("--render-noise", type=float, default=None, metavar="PCT",
                   help="path-traced frame budget (--no-raster): stop each frame when the "
                        "estimated noise falls to PCT percent (lower = cleaner, slower). If "
                        "none of --render-noise/--render-time/--render-spp is given, defaults "
                        "to 4%%. Combine budgets: ftrace stops at whichever fires first.")
    g.add_argument("--render-time", type=float, default=None, metavar="SEC",
                   help="path-traced frame budget (--no-raster): wall-clock seconds to spend "
                        "per frame before moving on (regardless of noise reached).")
    g.add_argument("--render-spp", type=int, default=None, metavar="N",
                   help="path-traced frame budget (--no-raster): render exactly N samples per "
                        "pixel per frame (a fixed sample count; the mode-R analogue of forward "
                        "photons). Combine with --render-time/--render-noise as an upper bound.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Discovery commands (P3.2): print the surface library / one surface's params, then exit
    # before any generation work.  --surface-help raises SystemExit on an unknown name.
    if args.list_surfaces:
        print(_list_surfaces_text())
        return 0
    if args.surface_help is not None:
        print(_surface_help_text(args.surface_help))
        return 0

    # (--dims >= 3, --oscillating >= 2, --freq > 0 are validated by their spec parsers.)
    if args.count < 1:
        raise SystemExit("error: --count must be >= 1")
    # --transform (and its --bloom/--tumble-* satellites) is deprecated in favour of the
    # unified --oscillate grammar; still fully supported, but nudge the user once. Checked
    # before resolve_oscillate() runs, since that may synthesize args.transform itself.
    if args.transform is not None:
        print("[gyroid_nd] note: --transform (and --bloom/--bloom-amp/--tumble-*) is "
              "deprecated; use --oscillate instead (see OSCILLATE_GRAMMAR.md). It still works.")
    # Validate + canonicalize --surface up front (resolves schwarz_p->primitive, rejects a
    # catalog-only TPMS or an unknown name) so a bad value fails cleanly before any rendering.
    # Resolved BEFORE the --lock/--oscillate grammar so those can classify a token against the
    # surface's named shape params (S4 pins, S5 param swingers).
    args.surface = resolve_surface(args.surface)

    # S4: pull POV shape-param value pins (`--lock NAME=VALUE`) out of --lock and validate them
    # against the surface.  MUST run before resolve_oscillate(), which parses the remaining
    # --lock tokens through the motion/axis grammar (which never uses '=' and would reject a
    # pin token); a pins-only --lock collapses to None so it doesn't engage that grammar or
    # trip the --transform mutual-exclusion.
    resolve_pov_param_locks(args)

    # Resolve the unified --oscillate/--lock grammar (if used) onto the canonical
    # transform/bloom/tumble fields, or default --transform to 'drift'. Then normalize
    # --transform (one name or a comma/plus-separated layered set) to canonical form.
    resolve_oscillate(args)
    args.transform = _parse_transforms(args.transform)
    if args.bloom is not None and not _has(args.transform, "bloom"):
        raise SystemExit("error: --bloom only applies when 'bloom' is in --transform")
    _parse_bloom_params(args.bloom)     # validate early (raises on a bad parameter name)
    # A swinger with a non-integer envelope rate pulses faster but no longer returns to its
    # start at t=1, so the loop is no longer seamless.  Allowed (looping is a nicety, not a
    # requirement), but note it once so a silently non-looping video isn't a surprise.
    _nonlooping = sorted(k for k, r in getattr(args, "bloom_rates", {}).items()
                         if float(r) != round(float(r)))
    if _nonlooping:
        print(f"[gyroid_nd] note: non-integer --oscillate rate on {', '.join(_nonlooping)} "
              f"pulses faster but won't loop seamlessly (last frame != first).")

    # tumble options only apply when 'tumble' is in --transform.
    if not _has(args.transform, "tumble"):
        if args.tumble_mode != "rotate":
            raise SystemExit("error: --tumble-mode only applies to --transform tumble")
        if args.tumble_lock is not None:
            raise SystemExit("error: --tumble-lock only applies to --transform tumble")
    args.tumble_lock = _parse_tumble_lock(args.tumble_lock)   # normalize to a tuple

    # Video/frame pixel size: explicit --size (N or WxH) wins, else square --res.
    size = args.size if args.size is not None else (args.res, args.res)

    axis_locks: Dict[int, AxisLock] = {}
    for spec in args.axis:
        try:
            parse_axis_lock(spec, axis_locks)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    # --pair edits to the coupling *edges* (on top of the --coupling base graph).
    pair_on: set = set()
    pair_off: set = set()
    for spec in args.pair:
        try:
            parse_pair_lock(spec, pair_on, pair_off)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
    args.pair_on = frozenset(pair_on)
    args.pair_off = frozenset(pair_off)

    # --couple explicit clusters (the field counterpart of --oscillate); mutually exclusive
    # with a non-default --coupling / any --pair (resolve_couple raises on conflict).
    resolve_couple(args)

    # The coupling graph (--coupling/--pair/--couple) is a pairwise-gyroid concept; a per-node
    # surface (Schwarz P) has no edges to wire, so those flags are inert there.  Warn rather than
    # error so a batch script that sets a house-style coupling can still switch --surface freely.
    if args.surface != "gyroid":
        ignored = []
        if getattr(args, "coupling", "cyclic") != "cyclic":
            ignored.append("--coupling")
        if args.pair_on or args.pair_off:
            ignored.append("--pair")
        if getattr(args, "couple_clusters", ()):
            ignored.append("--couple")
        if ignored:
            print(f"[gyroid_nd] note: {' and '.join(ignored)} only affect --surface gyroid "
                  f"(the pairwise coupling graph); ignored for --surface {args.surface} "
                  f"(no coupling edges to wire).")

    # Resolve --out to an absolute path (relative to the invoking cwd): the frames are
    # rendered by ftrace with cwd = repo_root, so a relative outdir would be written under
    # the user's cwd but looked for under repo_root and fail to open.  (The default outdir
    # is already absolute, repo_root/png/<name>.)
    base_outdir = Path(args.out).resolve() if args.out else _default_outdir(args.name)
    # Run-subdir default: fresh runNNN when --out is omitted (the auto png/<name> location, so
    # runs never collide), but write straight into an explicit --out dir.  Either is forceable
    # with --run-subdir / --no-run-subdir.
    use_run_subdir = args.run_subdir if args.run_subdir is not None else (args.out is None)
    if use_run_subdir:
        outdir = _next_run_dir(base_outdir)
        print(f"[gyroid_nd] run dir: {outdir}  (--no-run-subdir to write into {base_outdir})")
    else:
        outdir = base_outdir
        print(f"[gyroid_nd] output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional separate destination for the assembled video(s); frames/sidecars stay in
    # outdir.  Resolved (absolute or relative to the invoking cwd) just like --out; when
    # omitted the video lands alongside the frames.
    video_outdir = Path(args.video_out).resolve() if args.video_out else None
    if video_outdir is not None:
        video_outdir.mkdir(parents=True, exist_ok=True)
        print(f"[gyroid_nd] video dir: {video_outdir}")

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
                               noise=args.render_noise, time_budget=args.render_time,
                               spp=args.render_spp, fmt=args.format,
                               env_file=env_file, transform=args.transform,
                               material=args.material, clarity=args.glass_clarity,
                               raster_iso=args.raster_iso, raster_gpu=args.raster_gpu,
                               preview=preview, video_dir=video_outdir, shell=args.shell,
                               shape=args.shape, box_size=args.box_size,
                               stretch=(args.stretch_x, args.stretch_y, args.stretch_z))
            made.append(video)
            _status_commit(f"{label} | done -> {video.name} ({args.frames} frames)")
        else:
            # No video: one static scene file (t=0) with the full comment header.
            scene = build_scene(v, t=0.0, res=size, radius=args.radius, env_file=env_file,
                                transform=args.transform, material=args.material,
                                shell=args.shell, shape=args.shape, box_size=args.box_size,
                                stretch=(args.stretch_x, args.stretch_y, args.stretch_z))
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
    dest = video_outdir if (args.video and video_outdir is not None) else outdir
    print(f"[gyroid_nd] wrote {len(made)} {kind} to {dest}")
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
