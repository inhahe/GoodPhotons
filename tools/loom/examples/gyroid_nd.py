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

This script **randomly picks** all of the above and writes ``--count N`` complete,
renderable ``.ftsl`` scene files, each with a full comment header recording exactly
what was chosen (so a variant can be reproduced or hand-edited).  By default it also
**rasterizes** each variant to a ``.png`` (ftrace ``-raster``, a fast headless z-buffer
preview) and drops a ``.txt`` beside it listing every chosen value; use ``--no-images``
to emit only the scene files.  Any choice can be **locked** from the CLI (see
``--help``): the dimension count, how many dims oscillate, how many are harmonics of
the main, the base frequency, and — per axis — whether it oscillates and at what
harmonic.

Examples::

    # 10 fully random variants (each -> .ftsl + .png + .txt) into png/gyroid_nd/
    python examples/gyroid_nd.py --count 10

    # reproducible; lock 6 dims, 4 oscillating, 2 of them harmonics of the main
    python examples/gyroid_nd.py --count 5 --seed 42 --dims 6 --oscillating 4 --harmonics 2

    # force the classic gyroid: x,y,z on at harmonic 1, nothing else
    python examples/gyroid_nd.py --dims 3 --axis 0:on:1 --axis 1:on:1 --axis 2:on:1

    # just the .ftsl scene files, no rasterized images
    python examples/gyroid_nd.py --count 3 --no-images

    # generate and also full path-trace each (windowed, crash-safe checkpointing)
    python examples/gyroid_nd.py --count 3 --render
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

from loom import Scene, Camera, Material, Light, Raw, Isosurface  # noqa: E402
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


@dataclass
class Variant:
    seed: int
    dims: int
    freq: float
    threshold: float
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

    freq = args.freq if args.freq is not None else rng.uniform(*args.freq_range)
    return Variant(seed=seed, dims=D, freq=freq, threshold=args.threshold, dim_list=dims)


# ---------------------------------------------------------------------------
# field expression
# ---------------------------------------------------------------------------

def _u_expr(dim: Dim, freq: float) -> str:
    """The per-dimension argument u_d = harmonic*freq*(dir . (x,y,z)) + phase."""
    coeff = dim.harmonic * freq
    parts = []
    for c, var in zip(dim.direction, ("x", "y", "z")):
        if abs(c) < 1e-9:
            continue
        parts.append(f"({fmt(c)})*{var}")
    lin = "+".join(parts) if parts else "0"
    if abs(dim.phase) < 1e-9:
        return f"({fmt(coeff)}*({lin}))"
    return f"({fmt(coeff)}*({lin})+({fmt(dim.phase)}))"


def field_expr(v: Variant) -> str:
    osc = sorted(v.oscillating)
    m = len(osc)
    by_index = {d.index: d for d in v.dim_list}
    u = {d: _u_expr(by_index[d], v.freq) for d in osc}
    terms = []
    for i in range(m):
        a = osc[i]
        b = osc[(i + 1) % m]
        terms.append(f"sin({u[a]})*cos({u[b]})")
    return "+".join(terms)


# ---------------------------------------------------------------------------
# scene + header
# ---------------------------------------------------------------------------

def build_scene(v: Variant, *, res=(480, 480), radius=1.3, material="shell") -> Scene:
    expr = field_expr(v)
    fn = (lambda cx, cy, cz, _e=expr: _e)   # ignore transformed coords; freq baked in
    iso = Isosurface(fn, freq=1.0, threshold=v.threshold, container="sphere",
                     center=(0, 0, 0), radius=radius, material=material,
                     name="gyroid_nd")
    scene = Scene(Camera(eye=(0.0, 0.7, 5.2), look_at=(0, 0, 0), up=(0, 1, 0),
                         fov_y=34, mode="R", res=res))
    scene.add(
        Material("shell", "diffuse", reflect=0.85),
        Material("wall", "diffuse", reflect=0.78),
        iso,
        Raw('quad { origin -2 -1.7 -2  u 4 0 0  v 0 0 4  material "wall" }'),
        Raw('quad { origin -2  1.7 -2  u 4 0 0  v 0 0 4  material "wall" }'),
        Raw('quad { origin -2 -1.7 -2  u 4 0 0  v 0 3.4 0  material "wall" }'),
        Raw('quad { origin -2 -1.7 -2  u 0 0 4  v 0 3.4 0  material "wall" }'),
        Raw('quad { origin  2 -1.7 -2  u 0 0 4  v 0 3.4 0  material "wall" }'),
        Light("area", origin="-0.9 1.68 -0.9", u="1.8 0 0", v="0 0 1.8",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


def header(v: Variant, index: int, count: int) -> str:
    osc = v.oscillating
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
         f"# level set (threshold) : {fmt(v.threshold)}",
         "#",
         "# axis  osc  harmonic  direction (x y z)                 phase     role",
         "# ----  ---  --------  --------------------------------  --------  -----------"]
    for d in v.dim_list:
        dirs = "(" + " ".join(fmt(c) for c in d.direction) + ")"
        if d.oscillate:
            L.append(f"#  {d.index:>3}  yes  {d.harmonic:>6}    {dirs:<32}  "
                     f"{d.phase:>7.4f}   {d.role}")
        else:
            L.append(f"#  {d.index:>3}   no       -    {dirs:<32}  {'-':>7}   inert")
    L += ["#",
          "# field:  sum over cyclic oscillating pairs (i, i+1) of  sin(u_i) * cos(u_j)",
          "#   u_d = harmonic_d * freq * (dir_d . (x, y, z)) + phase_d",
          "#" + "=" * 74, ""]
    return "\n".join(L)


def sidecar_text(v: Variant, index: int, count: int, *,
                 ftsl_name: str = "", png_name: str = "") -> str:
    """Plain-text (non-comment) dump of every chosen value, saved beside each image.

    Reuses :func:`header` verbatim (stripped of its ``#`` comment prefixes) so the
    ``.txt`` sidecar and the ``.ftsl`` header can never drift apart.
    """
    lines: List[str] = []
    if ftsl_name or png_name:
        if ftsl_name:
            lines.append(f"scene file : {ftsl_name}")
        if png_name:
            lines.append(f"image file : {png_name}")
        lines.append("")
    for line in header(v, index, count).splitlines():
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
# image / render generation
# ---------------------------------------------------------------------------

def rasterize_files(paths: List[Path], *, res: int) -> List[Path]:
    """Rasterize each ``.ftsl`` to a PNG with ftrace ``-raster`` (fast z-buffer preview).

    Headless and non-blocking: no ``-window`` is passed, so ftrace writes the PNG to
    ``-o`` and exits, letting a whole batch of N run unattended (with ``-window`` a
    single ``-raster`` still becomes an interactive, blocking fly camera).
    """
    import subprocess
    from loom.drive import find_ftrace, repo_root
    ftrace = find_ftrace()
    pngs: List[Path] = []
    for i, fp in enumerate(paths):
        png = fp.with_suffix(".png")
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-raster", "-r", str(res)]
        print(f"[gyroid_nd] rasterize {i + 1}/{len(paths)}: {png.name}", flush=True)
        r = subprocess.run(cmd, cwd=str(repo_root()))
        if r.returncode != 0:
            raise SystemExit(f"ftrace -raster failed on {fp} (exit {r.returncode})")
        pngs.append(png)
    return pngs


def render_files(paths: List[Path], *, noise: float, res: int) -> None:
    """Full path-traced render of each ``.ftsl`` (windowed, checkpointed) — opt-in."""
    import subprocess
    from loom.drive import find_ftrace, repo_root
    ftrace = find_ftrace()
    for i, fp in enumerate(paths):
        png = fp.with_suffix(".png")
        last = (i == len(paths) - 1)
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png), "-r", str(res),
               "-interval", "8", "-checkpoint", "-noise", f"{noise:g}",
               "-keepwindow" if last else "-window"]
        print(f"[gyroid_nd] render {i + 1}/{len(paths)}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(repo_root()))
        if r.returncode != 0:
            raise SystemExit(f"ftrace failed on {fp} (exit {r.returncode})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    g.add_argument("--res", type=int, default=480,
                   help="render resolution written into each scene's film (default 480)")

    g = p.add_argument_group("images")
    g.add_argument("--images", action=argparse.BooleanOptionalAction, default=True,
                   help="rasterize each variant to a PNG and write a .txt of its values "
                        "(default on; --no-images to only emit the .ftsl files)")

    g = p.add_argument_group("render (optional)")
    g.add_argument("--render", action="store_true",
                   help="also path-trace each generated file with ftrace (windowed, checkpointed)")
    g.add_argument("--render-noise", type=float, default=4.0,
                   help="per-frame noise-floor budget for --render (default 4%%)")
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

    axis_locks: Dict[int, AxisLock] = {}
    for spec in args.axis:
        try:
            parse_axis_lock(spec, axis_locks)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    outdir = Path(args.out) if args.out else _default_outdir(args.name)
    outdir.mkdir(parents=True, exist_ok=True)

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

    count = len(seeds)
    width = max(3, len(str(count - 1)))
    written: List[Path] = []
    for k, vseed in enumerate(seeds):
        v = pick_variant(vseed, args, axis_locks)
        scene = build_scene(v, res=(args.res, args.res), radius=args.radius)
        from loom import Clock, Cache
        body = scene.emit(Clock(t=0.0), Cache(), assets_dir=outdir, tag=f"{k:0{width}d}")
        text = header(v, k, count) + body
        fp = outdir / f"{args.name}{k:0{width}d}.ftsl"
        fp.write_text(text, encoding="utf-8")
        written.append(fp)
        if args.images:
            # A plain-text record of every chosen value, saved beside each image.
            txt = fp.with_suffix(".txt")
            txt.write_text(
                sidecar_text(v, k, count, ftsl_name=fp.name,
                             png_name=fp.with_suffix(".png").name),
                encoding="utf-8")
        print(f"[gyroid_nd] {fp.name}: D={v.dims} osc={v.oscillating} "
              f"harmonics={v.harmonic_dims} freq={fmt(v.freq)} seed={v.seed}")

    print(f"[gyroid_nd] wrote {len(written)} scene file(s) to {outdir}")
    if args.images:
        pngs = rasterize_files(written, res=args.res)
        print(f"[gyroid_nd] rasterized {len(pngs)} image(s) (+ .txt values) to {outdir}")
    if args.render:
        render_files(written, noise=args.render_noise, res=args.res)
    return 0


def _default_outdir(name: str) -> Path:
    try:
        from loom.drive import repo_root
        return repo_root() / "png" / name
    except Exception:
        return Path.cwd() / name


if __name__ == "__main__":
    sys.exit(main())
