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
  * a **direction** in (x,y,z):  dims 0/1/2 are the x/y/z axes; dims >= 3 get a
    generic random unit direction (a higher-D axis seen edge-on in the slice);
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

The rendered subject is just the gyroid from ``scenes/showcase.ftsl`` — a **gold thickened
gyroid sheet** (``abs(g) - t``), CSG-clipped to a ball — on its own, with no Cornell box or
glass sphere.  Gold is a mirror, so it shows whatever surrounds it; a flat uniform light
would wash the lattice out.  Instead it is lit like a product shot by a procedural **studio
environment** — a dark neutral base plus a few bright soft "softbox" lights, written once as
an equirectangular ``studio_env.pfm`` beside the outputs and fed to ftrace's image-based
``light env`` — so the gold picks up crisp highlights that trace every facet while deep
shadows give depth, over a clean neutral background.  (The gold look only develops under path
tracing, ``--no-raster``; the fast rasterizer previews the same geometry flat-shaded.)

This script **randomly picks** the field parameters.  For each of ``--count N`` variants it
**renders a seamless morphing video**: every non-main oscillating dimension is given an
integer *drift* rate (a "winding") so its phase advances a whole number of cycles over
the loop, translating the visible 3-D slice *through* that dimension — literally the
transformation through the higher dimensions.  The assembled animated ``.gif`` (or ``.mp4``
via ``--format mp4``) and a ``.txt`` listing every chosen value collect together in the
output directory; each variant's per-frame ``.ftsl``/``.png`` files live in their own
subdir ``<outdir>/<variant>/``.  Frames render with ftrace's fast headless rasterizer by
default.  Use ``--no-video`` to instead emit a single static ``.ftsl`` per variant (with a
full comment header).  Any choice can be **locked** from the CLI (see ``--help``): the
dimension count, how many dims oscillate, how many are harmonics of the main, the base
frequency, and — per axis — whether it oscillates and at what harmonic.

Examples::

    # 10 fully random variants, each a morphing video, into png/gyroid_nd/<variant>/
    python examples/gyroid_nd.py --count 10

    # reproducible; lock 6 dims, 4 oscillating, 2 of them harmonics of the main
    python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 --oscillating 4 --harmonics 2

    # the classic gyroid, animated: x,y,z on at harmonic 1, 90 frames as an mp4
    python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 --axis 2:on:1 \
        --frames 90 --format mp4

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
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    winding: int = 0                    # animation drift: integer cycles this dim's
    #                                     phase advances over one video loop (0 = fixed;
    #                                     the main dim anchors at 0, others drift so the
    #                                     slice translates through their dimension)


@dataclass
class Variant:
    seed: int
    dims: int
    freq: float
    threshold: float
    thickness: float = 0.5              # half-width of the thickened gold sheet
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
    else:
        H = rng.randint(lo_h, hi_h) if hi_h >= lo_h else lo_h
    rng.shuffle(free)
    chosen_harm = set(forced_gt1) | set(free[:max(0, H - len(forced_gt1))])

    # 5) assemble every dimension ---------------------------------------------
    dims: List[Dim] = []
    for d in range(D):
        direction = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))[d] \
            if d < 3 else _rand_unit(rng)
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

    freq = args.freq if args.freq is not None else rng.uniform(*args.freq_range)
    return Variant(seed=seed, dims=D, freq=freq, threshold=args.threshold,
                   thickness=args.thickness, dim_list=dims)


# ---------------------------------------------------------------------------
# field expression
# ---------------------------------------------------------------------------

def _u_expr(dim: Dim, freq: float, phase: float) -> str:
    """The per-dimension argument u_d = harmonic*freq*(dir . (x,y,z)) + phase."""
    coeff = dim.harmonic * freq
    parts = []
    for c, var in zip(dim.direction, ("x", "y", "z")):
        if abs(c) < 1e-9:
            continue
        parts.append(f"({fmt(c)})*{var}")
    lin = "+".join(parts) if parts else "0"
    if abs(phase) < 1e-9:
        return f"({fmt(coeff)}*({lin}))"
    return f"({fmt(coeff)}*({lin})+({fmt(phase)}))"


def field_expr(v: Variant, t: float = 0.0) -> str:
    """Emit the gyroid field at loop phase ``t`` in [0,1).

    Each dim's phase is advanced by ``2*pi*winding*t`` — an integer number of full
    cycles over the loop — so ``t=0`` and ``t=1`` are identical (seamless) and the
    pattern drifts through its dimensions in between.  ``t=0`` reproduces the static
    field exactly.
    """
    osc = sorted(v.oscillating)
    m = len(osc)
    by_index = {d.index: d for d in v.dim_list}
    u = {}
    two_pi = 2.0 * math.pi
    for d in osc:
        dim = by_index[d]
        # Reduce the drifted phase modulo 2*pi so t=0 and t=1 emit the *same* constant
        # (a whole-cycle advance) -> a perfectly seamless loop despite float rounding.
        phase = (dim.phase + two_pi * dim.winding * t) % two_pi
        u[d] = _u_expr(dim, v.freq, phase)
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
                env_file: Optional[str] = None) -> Scene:
    """The lone gold thickened gyroid ball whose lattice is the morphing higher-D field.

    ``env_file`` is a path to an equirectangular environment map (see ``studio_env_pfm``)
    used for image-based lighting; with it the gold picks up the studio highlights that
    reveal the lattice.  Without it the scene falls back to a plain uniform env (flatter).
    The gold look only develops under path tracing (``mode R``, i.e. ``--no-raster``); the
    fast rasterizer previews the same geometry flat-shaded.
    """
    expr = field_expr(v, t)
    # Thicken the surface into a solid sheet (showcase's abs(g) - 0.5).  Scale the
    # half-width by sqrt(M/3) so walls stay visible as extra oscillating dims add
    # amplitude (M=3 reproduces the classic 0.5).
    m = max(1, len(v.oscillating))
    half = v.thickness * math.sqrt(m / 3.0)
    thr = v.threshold
    inner = f"({expr})-({fmt(thr)})" if abs(thr) > 1e-9 else f"({expr})"
    sheet = f"abs({inner})-({fmt(half)})"
    # Lipschitz bound for the sphere-marcher: |grad f| <= 2*freq*sum(harmonic_d).
    sum_h = sum(d.harmonic for d in v.dim_list if d.oscillate)
    grad_bound = 2.2 * v.freq * max(1, sum_h)
    box = radius * 1.05                                  # contained_by half-extent
    r = radius

    scene = Scene(Camera(eye=(0.0, 0.7 * r, 4.0 * r), look_at=(0, 0, 0),
                         up=(0, 1, 0), fov_y=36, mode="R", res=res))

    iso = Raw(
        "isosurface {\n"
        '    material "gold"\n'
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
        Raw('material "gold" { preset gold }'),
        iso,
        light,
    )
    return scene


def header(v: Variant, index: int, count: int, *,
           frames: Optional[int] = None, fps: Optional[float] = None) -> str:
    osc = v.oscillating
    drifting = [d.index for d in v.dim_list if d.oscillate and d.winding > 0]
    L = ["#" + "=" * 74,
         f"# Higher-dimensional gyroid slice — variant {index + 1}/{count}",
         f"# generated by gyroid_nd.py",
         "#",
         f"# variant seed          : {v.seed}   (regenerate: --variant-seed {v.seed} + the same locks)",
         f"# dimensions (D)        : {v.dims}   (higher/extra dims beyond x,y,z: {max(0, v.dims - 3)})",
         f"# oscillating dims      : {len(osc)}  -> {osc}",
         f"# main dimension        : {v.main}   (fundamental, harmonic 1)",
         f"# harmonics of the main : {len(v.harmonic_dims)}  -> {v.harmonic_dims}",
         f"# base spatial frequency: {fmt(v.freq)}",
         f"# level set (threshold) : {fmt(v.threshold)}"]
    if frames is not None:
        secs = frames / fps if fps else 0.0
        L.append(f"# animation             : {frames} frames @ {fmt(fps or 30.0)} fps "
                 f"(~{fmt(secs)}s seamless loop); drifting dims -> {drifting}")
    L += ["#",
          "# axis  osc  harmonic  drift  direction (x y z)                 phase     role",
          "# ----  ---  --------  -----  --------------------------------  --------  -----------"]
    for d in v.dim_list:
        dirs = "(" + " ".join(fmt(c) for c in d.direction) + ")"
        if d.oscillate:
            drift = f"{d.winding}" if d.winding > 0 else "-"
            L.append(f"#  {d.index:>3}  yes  {d.harmonic:>6}  {drift:>5}  {dirs:<32}  "
                     f"{d.phase:>7.4f}   {d.role}")
        else:
            L.append(f"#  {d.index:>3}   no       -      -  {dirs:<32}  {'-':>7}   inert")
    L += ["#",
          "# field:  sum over cyclic oscillating pairs (i, i+1) of  sin(u_i) * cos(u_j)",
          "#   u_d = harmonic_d * freq * (dir_d . (x, y, z)) + phase_d + 2*pi*winding_d*t",
          "#   (t runs 0->1 over the loop; winding is the integer 'drift' column above)",
          "#" + "=" * 74, ""]
    return "\n".join(L)


def sidecar_text(v: Variant, index: int, count: int, *,
                 ftsl_name: str = "", video_name: str = "",
                 frames: Optional[int] = None, fps: Optional[float] = None) -> str:
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
    for line in header(v, index, count, frames=frames, fps=fps).splitlines():
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
                  size: Tuple[int, int], raster: bool, noise: float) -> None:
    """Render one frame ``.ftsl`` -> ``.png``, headless and non-blocking.

    ``raster`` uses ftrace ``-raster`` (fast solid-shaded z-buffer preview); otherwise
    a path-traced render to a per-frame noise budget.  ``size`` is the ``(W, H)`` film
    resolution, passed as ``-r W H``.  No ``-window`` is passed so the process writes
    the PNG and exits, letting the whole frame range run unattended.  The renderer's
    own console chatter is captured (kept off the status line) and only surfaced if the
    frame fails.
    """
    import subprocess
    w, h = size
    if raster:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-raster",
               "-r", str(w), str(h)]
    else:
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-r", str(w), str(h),
               "-interval", "8", "-checkpoint", "-noise", f"{noise:g}"]
    r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write("\n")
        sys.stdout.write((r.stdout or "") + (r.stderr or ""))
        raise SystemExit(f"ftrace failed on {fp.name} (exit {r.returncode})")


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
    pngs: List[Path] = []
    for k in range(frames):
        t = k / frames                                  # seamless loop: t in [0,1)
        _status(f"{label} | emit ftsl  frame {k + 1}/{frames}")
        scene = build_scene(v, t=t, res=size, radius=radius, env_file=env_file)
        body = scene.emit(Clock(t=t, frame=k, frames=frames, fps=fps),
                          Cache(), assets_dir=frames_dir, tag=f"{k:0{fw}d}")
        fp = frames_dir / f"{base}_{k:0{fw}d}.ftsl"
        fp.write_text(body, encoding="utf-8")
        png = fp.with_suffix(".png")
        _status(f"{label} | {verb} frame {k + 1}/{frames}")
        _render_frame(ftrace, root, fp, png, size=size, raster=raster, noise=noise)
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
                "  python examples/gyroid_nd.py --dims 6 --axis 4:on:3 --axis 1:off"))

    g = p.add_argument_group("output")
    g.add_argument("-n", "--count", type=int, default=1,
                   help="number of variant .ftsl files to generate (default 1)")
    g.add_argument("--out", type=str, default=None,
                   help="output directory (default: <repo>/png/gyroid_nd)")
    g.add_argument("--name", type=str, default="gyroid_nd",
                   help="base filename for the outputs (default gyroid_nd)")
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

    g = p.add_argument_group("scene")
    g.add_argument("--threshold", type=float, default=0.0,
                   help="isosurface level set f = threshold (default 0; ~+/-0.7 thins the walls)")
    g.add_argument("--radius", type=float, default=1.3,
                   help="radius of the spherical container the lattice fills (default 1.3)")
    g.add_argument("--thickness", type=float, default=0.5,
                   help="half-width of the thickened gold sheet (showcase abs(g)-t style; "
                        "default 0.5; auto-scaled up with the oscillating-dim count)")
    g.add_argument("--size", type=_parse_size, default=None, metavar="N|WxH",
                   help="video/frame size in pixels: a single number for square (e.g. 720) "
                        "or WIDTHxHEIGHT (e.g. 1280x720); the preview window matches it "
                        "(overrides --res)")
    g.add_argument("--res", type=int, default=480,
                   help="square render size when --size is unset (default 480)")

    g = p.add_argument_group("video (per variant)")
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

    # Video/frame pixel size: explicit --size (N or WxH) wins, else square --res.
    size = args.size if args.size is not None else (args.res, args.res)

    axis_locks: Dict[int, AxisLock] = {}
    for spec in args.axis:
        try:
            parse_axis_lock(spec, axis_locks)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    outdir = Path(args.out) if args.out else _default_outdir(args.name)
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
        # Brief, in-place status: which gyroid, its values, and the live frame/phase.
        label = (f"[gyroid_nd] gyroid {k + 1}/{count}  D={v.dims} osc={v.oscillating} "
                 f"harm={v.harmonic_dims} freq={fmt(v.freq)} seed={v.seed}")
        if args.video:
            # Videos and their .txt sidecars collect in the shared outdir; each video's
            # per-frame .ftsl/.png files live in their own subdir <outdir>/<base>/.
            frames_dir = outdir / base
            frames_dir.mkdir(parents=True, exist_ok=True)
            ext = _video_ext(args.format)
            (outdir / f"{base}.txt").write_text(
                sidecar_text(v, k, count, ftsl_name=f"{base}/{base}_NNN.ftsl",
                             video_name=f"{base}.{ext}",
                             frames=args.frames, fps=args.fps),
                encoding="utf-8")
            video = make_video(frames_dir, outdir, base, v, label=label,
                               frames=args.frames, fps=args.fps, size=size,
                               radius=args.radius, raster=args.raster,
                               noise=args.render_noise, fmt=args.format,
                               env_file=env_file, preview=preview)
            made.append(video)
            _status_commit(f"{label} | done -> {video.name} ({args.frames} frames)")
        else:
            # No video: one static scene file (t=0) with the full comment header.
            scene = build_scene(v, t=0.0, res=size, radius=args.radius, env_file=env_file)
            body = scene.emit(Clock(t=0.0), Cache(), assets_dir=outdir,
                              tag=f"{k:0{width}d}")
            fp = outdir / f"{base}.ftsl"
            fp.write_text(header(v, k, count) + body, encoding="utf-8")
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


if __name__ == "__main__":
    sys.exit(main())
