"""Compile a .ftcl rig to MJCF, and optionally load/settle/inspect it in MuJoCo.

    python tools/ftcl_build.py rigs/canis.ftcl -o out/canis.xml --check
    python tools/ftcl_build.py rigs/canis.ftcl --set body_scale=1.4 limb_gracility=0.7
    python tools/ftcl_build.py rigs/canis.ftcl --view
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftcl.errors import FtclError                     # noqa: E402
from creaturelab.build import load, sample_morph      # noqa: E402
from creaturelab.emit_mjcf import place_on_ground, to_mjcf   # noqa: E402
from creaturelab.tune import build_tuned              # noqa: E402
from creaturelab.validate import (NUDGE_FRACTION, stand_test,   # noqa: E402
                                  withers_height)


def parse_sets(pairs) -> dict[str, float]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects name=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig")
    ap.add_argument("-o", "--out", help="write MJCF here (default out/<name>.xml)")
    ap.add_argument("--set", nargs="*", metavar="NAME=VAL", help="override morph params")
    ap.add_argument("--random", action="store_true", help="sample a random morph vector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="randomisation spread about the defaults (curriculum knob)")
    ap.add_argument("--check", action="store_true",
                    help="load in MuJoCo, settle under gravity, report")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        morph = parse_sets(args.set)
        if args.random:
            import random
            # One cheap load just to learn the parameter list, then sample within it.
            params = load(args.rig, morph).params
            sampled = sample_morph(params, random.Random(args.seed), args.scale)
            sampled.update(morph)                    # explicit --set still wins
            morph = sampled
        # `build_tuned`, not `load`: an untuned rig has no passive tone and folds up the
        # instant gravity touches it, so --check on a bare `load` would measure the
        # collapse of a body nobody intends to simulate. The tone pass is part of what
        # "compile this rig" means.
        creature, loads = build_tuned(args.rig, morph)
    except FtclError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    xml = to_mjcf(creature)
    out = args.out or os.path.join("out", f"{creature.name}.xml")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(xml)

    if not args.quiet:
        nj = len(creature.actuated_joints)
        print(f"{creature.name}: {len(creature.bones)} bones, {nj} actuated joints "
              f"-> {out}")
        if loads:
            print(f"  passive tone: {len(loads)} joints sprung, peak k "
                  f"{max(L.stiffness for L in loads):.0f} N*m/rad, peak |tau_hold| "
                  f"{max(abs(L.tau_static) for L in loads):.2f} N*m "
                  f"({len(creature.contact_excludes)} contact pairs excluded)")
        else:
            print("  no `posture` block: bare skeleton, no passive tone")

    if args.check or args.view:
        import mujoco
        model = mujoco.MjModel.from_xml_path(out)
        data = mujoco.MjData(model)
        z = place_on_ground(model, data)
        total = float(sum(model.body_mass))
        if not args.quiet:
            print(f"  loaded: nq={model.nq} nv={model.nv} nu={model.nu} "
                  f"ngeom={model.ngeom}")
            print(f"  mass={total:.2f} kg   spawn z={z:.3f} m")
            # Withers height is the number a dog person would sanity-check.
            print(f"  withers {withers_height(model, data):.3f} m")

        if args.check:
            r = stand_test(model, data)
            print(f"  motors off, 3 s: sank {r.drop*1000:+.0f} mm "
                  f"({r.rel_drop*100:.1f}% of withers), pitch {r.tilt:+.1f} deg, "
                  f"support margin {r.margin*1000:+.0f} mm  {r.verdict}")
            # The verdict is decided on the shove, so the shove has to be reported -- otherwise
            # a body that folds sideways prints a tidy pitch next to *** COLLAPSES ***.
            print(f"  shoved at {r.nudge:.3f} m/s ({NUDGE_FRACTION*100:.0f}% of the "
                  f"{r.v_tip:.3f} m/s that would tip it): peak tilt {r.tilt_peak:.1f} deg")
            if not r.ok:
                return 2 if not r.finite else 3

        if args.view:
            import mujoco.viewer
            mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
