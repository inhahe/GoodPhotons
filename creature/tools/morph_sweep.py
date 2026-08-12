"""What fraction of randomised bodies actually stand up?

P4 wants to train one policy across a whole distribution of bodies, which only works if
sampling that distribution reliably produces bodies worth training on. That is a *number*,
not an opinion, and it is not obtainable by looking at rigs -- an unstandable morph loads
and renders exactly like a good one. This measures it.

The number is also the honest way to choose the randomisation width. `--scale` is the
curriculum knob in `sample_morph`: 0 is the declared defaults, 1 is the full declared
range on every parameter independently. Independence is the catch. Nothing couples
`stance_femur` to `femur_len`, so at scale 1 the sampler will happily draw a long thigh
with a shallow crouch and a short shin, which is not a dog anyone has met. Sweeping scale
shows where that starts to bite:

    python tools/morph_sweep.py rigs/canis.ftcl -n 24
    python tools/morph_sweep.py rigs/canis.ftcl -n 24 --scale 0.25 0.5 0.75 1.0
    python tools/morph_sweep.py rigs/canis.ftcl -n 8 --verbose

`--morph` moves the centre of the distribution off the rig's defaults and onto a *fitted*
animal, which is the question P4 actually asks. Stage C hands you `out/theta_rex.json`;
`train.py --morph out/theta_rex.json --morph-scale 0.5` then trains over a neighbourhood of
that dog. Whether that neighbourhood is habitable is not a property of the rig — a θ sitting
near the edge of a declared range has most of its neighbourhood clipped against that edge,
and a real animal is far likelier to be near an edge than the defaults are. So sweep the
same centre you intend to train on, and read the yield before spending the training run:

    python tools/morph_sweep.py rigs/canis.ftcl --morph out/theta_rex.json -n 24 \\
                                --scale 0.25 0.5 1.0
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creaturelab.console import use_utf8  # noqa: E402

use_utf8()   # before argparse: --help is the thing a cp1252 console cannot print

from ftcl.errors import FtclError                      # noqa: E402
from creaturelab.build import load, sample_morph       # noqa: E402
from creaturelab.emit_mjcf import to_mjcf              # noqa: E402
from creaturelab.morph_io import morph_from_args, parse_sets   # noqa: E402
from creaturelab.tune import UnstablePose, build_tuned  # noqa: E402
from creaturelab.validate import stand_test            # noqa: E402


def one(rig: str, params, seed: int, scale: float, center=None):
    """Build one randomised body and report (outcome, detail). Never raises."""
    import mujoco
    morph = sample_morph(params, random.Random(seed), scale, center)
    try:
        creature, loads = build_tuned(rig, morph)
    except UnstablePose as e:
        # Not a bug -- a fact about this draw. The pose cannot be held by any ground
        # reaction, so there is nothing to tune and the sample should simply be rejected.
        return "unstable", f"CoM {-e.margin*1000:.0f} mm outside the support polygon"
    except (FtclError, RuntimeError) as e:
        return "error", str(e).split("\n")[0][:90]
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    r = stand_test(model, mujoco.MjData(model))
    # `peak` is the worst tilt reached while recovering from the shove, and is what the verdict
    # is decided on; `pitch` alone reads tidy on a body that folded sideways and came back.
    detail = (f"sank {r.drop*1000:+5.0f} mm ({r.rel_drop*100:4.1f}%), pitch "
              f"{r.tilt:+5.1f} deg, peak {r.tilt_peak:5.1f} deg, "
              f"margin {r.margin*1000:+5.0f} mm, "
              f"peak k {max(L.stiffness for L in loads):7.0f}")
    return ("stands" if r.ok else "collapses"), detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig")
    ap.add_argument("-n", "--count", type=int, default=16, help="samples per scale")
    ap.add_argument("--scale", type=float, nargs="*", default=[1.0],
                    help="randomisation widths to test")
    ap.add_argument("--seed", type=int, default=0, help="first seed; samples are seed..seed+n")
    ap.add_argument("--morph", metavar="FILE",
                    help="centre the distribution on a fitted morph (out/theta_*.json) "
                         "instead of the rig defaults -- the same body train.py trains around")
    ap.add_argument("--set", nargs="*", metavar="NAME=VAL", default=[],
                    help="override individual centre params; wins over --morph")
    ap.add_argument("--verbose", action="store_true", help="one line per sample")
    args = ap.parse_args()

    try:
        params = load(args.rig).params
        center = morph_from_args(args.rig, args.morph, parse_sets(args.set), params) or None
    except (FtclError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    where = f"centred on {args.morph}" if args.morph else "centred on the rig defaults"
    if center and not args.morph:
        where = f"centred on the rig defaults with {len(center)} override(s)"
    print(f"{args.rig}: {len(params)} morph parameters, {args.count} samples per scale, "
          f"{where}")
    worst = 1.0
    for scale in args.scale:
        tally: dict[str, int] = {}
        t0 = time.time()
        for i in range(args.count):
            seed = args.seed + i
            outcome, detail = one(args.rig, params, seed, scale, center)
            tally[outcome] = tally.get(outcome, 0) + 1
            if args.verbose or outcome in ("error", "collapses"):
                print(f"  scale {scale:.2f} seed {seed:3d}  {outcome:9s}  {detail}")
        n = float(args.count)
        rate = tally.get("stands", 0) / n
        # An unstandable draw is a legitimate rejection, not a failure of the tuner, so
        # score the two separately: the first number is yield, the second is correctness.
        viable = n - tally.get("unstable", 0)
        held = tally.get("stands", 0) / viable if viable else float("nan")
        worst = min(worst, held if viable else 1.0)
        parts = "  ".join(f"{k} {v}" for k, v in sorted(tally.items()))
        print(f"scale {scale:.2f}: {rate*100:5.1f}% stand of all draws, "
              f"{held*100:5.1f}% of the {int(viable)} statically viable ones   "
              f"[{parts}]   {time.time()-t0:.1f}s")
    # Exit non-zero if the tuner failed a body that was standable in principle -- that is
    # the part this project is responsible for. A low yield at scale 1 is a statement
    # about the rig's parameter ranges, and is reported rather than treated as an error.
    return 0 if worst > 0.999 else 3


if __name__ == "__main__":
    raise SystemExit(main())
