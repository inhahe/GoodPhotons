"""Measure a rig instead of guessing at it.

design.md's directability section says every layer has to be independently inspectable.
This is that, for the skeleton layer. It answers the questions you cannot get from looking
at the XML:

* **What torque does each joint actually have to hold?** Computed by inverse dynamics at
  the standing pose *with the feet in contact*, so it is the real static load, not a
  free-body approximation. This is the number that should set motor gears and passive
  stiffness -- picking those by feel is how you end up with a model that can't hold itself
  up, or one so stiff the policy learns nothing.
* **Is the rig left/right symmetric?** An asymmetric rig teaches an asymmetric gait, and
  it is nearly invisible by eye in a 25-body tree.
* **Where does the standing pose actually put the feet?** A limb that doesn't reach the
  ground in the reference pose makes every later result meaningless.

* **Where are the landmarks?** The sites are the rig's half of the keypoint<->rig
  interface, so a fit that reprojects badly needs a way to ask whether the rig is putting
  the withers somewhere sane before blaming the detector.

    python tools/rig_report.py rigs/canis.ftcl
    python tools/rig_report.py rigs/canis.ftcl --set body_scale=1.6
    python tools/rig_report.py rigs/canis.ftcl --morph out/theta_rex.json --sites
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creaturelab.console import use_utf8  # noqa: E402

use_utf8()   # before argparse: --help is the thing a cp1252 console cannot print

import numpy as np                                     # noqa: E402

from ftcl.errors import FtclError                      # noqa: E402
from creaturelab.build import load                     # noqa: E402
from creaturelab.emit_mjcf import (geom_z_extent, natural_mass,  # noqa: E402
                                   place_on_ground, to_mjcf)
from creaturelab.morph_io import morph_from_args, parse_sets   # noqa: E402
from creaturelab.tune import (SEAT, build_tuned,        # noqa: E402
                              steps_per_cycle_floor)
from creaturelab.validate import (NUDGE_FRACTION, stand_test,   # noqa: E402
                                  withers_height)


def build(rig: str, sets: dict[str, float], clearance: float = 0.01):
    """Compile, tune the passive tone, and stand the result on the ground."""
    import mujoco
    creature, loads = build_tuned(rig, sets)
    nat = natural_mass(creature)
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)
    place_on_ground(model, data, clearance)
    return creature, model, data, nat, loads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig")
    ap.add_argument("--set", nargs="*", metavar="NAME=VAL", default=[])
    ap.add_argument("--morph", metavar="FILE",
                    help="load a fitted morph vector (out/theta_*.json); --set still wins")
    ap.add_argument("--sites", action="store_true",
                    help="list the landmarks and where the standing pose puts them")
    ap.add_argument("--top", type=int, default=12, help="how many joints to list")
    args = ap.parse_args()

    import mujoco
    try:
        sets = morph_from_args(args.rig, args.morph, parse_sets(args.set),
                               load(args.rig).params)
        creature, model, data, nat, loads = build(args.rig, sets, -SEAT)
    except (FtclError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    total = float(sum(model.body_mass))
    print(f"=== {creature.name} " + "=" * (60 - len(creature.name)))
    print(f"bones {len(creature.bones)}   actuated joints {len(creature.actuated_joints)}"
          f"   nq {model.nq}  nv {model.nv}  nu {model.nu}")
    if creature.target_mass is not None:
        print(f"mass  {total:.2f} kg  (target {creature.target_mass:.2f}; geometry alone "
              f"would give {nat:.2f}, density scaled by {creature.target_mass/nat:.3f})")
    else:
        print(f"mass  {total:.2f} kg  (from densities)")

    # --- pose geometry ------------------------------------------------------------------
    def gz(name):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return None if gid < 0 else data.geom_xpos[gid]

    lows = {}
    for paw in ("hpaw_l", "hpaw_r", "fpaw_l", "fpaw_r"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, paw)
        if gid >= 0:
            lows[paw] = geom_z_extent(model, data, gid)[0]
    if lows:
        print("foot height relative to ground (m): " +
              "  ".join(f"{k} {v:+.4f}" for k, v in lows.items()))
        spread = max(lows.values()) - min(lows.values())
        verdict = "level" if spread < 0.01 else "*** UNEVEN ***"
        print(f"  spread {spread:.4f}  {verdict}")

    th, pv = gz("thorax"), gz("pelvis")
    if th is not None and pv is not None:
        print(f"withers {withers_height(model, data):.3f} m   croup {pv[2]:.3f} m   "
              f"back rise {th[2]-pv[2]:+.3f} m")

    # --- landmarks -----------------------------------------------------------------------
    # Always summarised, listed in full only on request: 21 rows is too much for the
    # default report, but "how many landmarks does this rig even have" is exactly the
    # question you ask when a fit reprojects badly, and the answer used to be unavailable.
    if creature.sites:
        n_soft = sum(1 for s in creature.sites if s.kind == "soft")
        print(f"landmarks {len(creature.sites)}  ({len(creature.sites) - n_soft} rigid, "
              f"{n_soft} soft)" + ("" if args.sites else "   -- use --sites to list"))
    elif args.sites:
        print("landmarks: none -- this rig has no `site` blocks, so it cannot be fit "
              "to video (notes/pipeline.md stage B)")

    if args.sites and creature.sites:
        print(f"\n  {'site':14s} {'bone':10s} {'kind':5s} "
              f"{'world x':>8s} {'y':>8s} {'z':>8s}   {'bone-local x':>12s} "
              f"{'y':>8s} {'z':>8s}")
        for s in creature.sites:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s.name)
            p = data.site_xpos[sid]
            print(f"  {s.name:14s} {s.bone:10s} {s.kind:5s} "
                  f"{p[0]:+8.4f} {p[1]:+8.4f} {p[2]:+8.4f}   "
                  f"{s.at[0]:+12.4f} {s.at[1]:+8.4f} {s.at[2]:+8.4f}")

        # Landmark symmetry, checked separately from the torque symmetry below. A site
        # mirrored wrongly is invisible in the joint torques -- it has no mass and exerts
        # no force -- but it would put a constant left/right bias into E_kp, which the fit
        # can only absorb by yawing the whole skeleton. Nothing else in this report can
        # see that.
        by = {s.name: data.site_xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s.name)] for s in creature.sites}
        worst, pair = 0.0, None
        for name, p in by.items():
            mate = (name.replace("_l_", "_r_") if "_l_" in name
                    else name[:-2] + "_r" if name.endswith("_l") else None)
            if mate and mate in by:
                q = by[mate]
                # mirror in the y=0 plane: x and z must match, y must negate
                d = max(abs(p[0] - q[0]), abs(p[2] - q[2]), abs(p[1] + q[1]))
                # `pair is None or ...`, not a bare `d > worst`. Mirrored site positions
                # come out BIT-IDENTICAL, so every d is exactly 0.0, so a bare `>` never
                # fires and the check silently prints nothing on precisely the rigs that
                # pass it. (The torque check below had the same shape and survived only
                # because inverse dynamics never returns two exactly equal floats.)
                if pair is None or d > worst:
                    worst, pair = d, (name, mate)
        if pair:
            print(f"  left/right landmark mirror error: {worst:.3e} m "
                  f"({pair[0]} vs {pair[1]})  "
                  + ("symmetric" if worst < 1e-9 else "*** ASYMMETRIC ***"))

    # --- static load and the passive tone sized from it ----------------------------------
    ncon = int(data.ncon)
    print(f"\n{ncon} active contact(s) at the standing pose"
          + ("  *** NONE -- the load below would be a free fall ***" if ncon == 0 else ""))
    if not loads:
        print("no `posture` block: this is a bare skeleton with no passive tone")
        return 0

    p = creature.posture
    print(f"passive tone: sag {np.degrees(p.sag):.1f}deg  tone_floor {p.tone_floor:.2f}  "
          f"buckle_margin {p.buckle_margin:.2f}  damping {p.damping_ratio:.2f} of critical")
    loads.sort(key=lambda L: -L.stiffness)
    print(f"\n  {'joint':20s} {'tau_hold':>9s} {'tau_ref':>8s} {'k_buckle':>9s} "
          f"{'k':>9s} {'c':>7s} {'f_n':>7s}  set by")
    for L in loads[:args.top]:
        print(f"  {L.name:20s} {L.tau_static:+9.2f} {L.tau_ref:8.2f} {L.k_buckle:9.1f} "
              f"{L.stiffness:9.1f} {L.damping:7.2f} {L.hz:6.1f}Hz  {L.limited_by}")
    if len(loads) > args.top:
        print(f"  ... {len(loads)-args.top} more")
    print(f"units: tau N*m, k N*m/rad, c N*m*s/rad, f_n the passive joint's natural "
          f"frequency")
    peak = max(abs(L.tau_static) for L in loads)
    print(f"peak |tau_hold| {peak:.2f} N*m   sum {sum(abs(L.tau_static) for L in loads):.1f}"
          f"   peak k {max(L.stiffness for L in loads):.0f} N*m/rad")

    # A spring the integrator cannot follow is the one failure mode this pass can create,
    # and it shows up as an exploding model rather than as a wrong number -- so check it
    # here, against this creature's own declared timestep AND integrator. `implicit` and
    # `implicitfast` integrate joint stiffness and damping implicitly, which is exactly
    # what they exist for, so they tolerate far stiffer springs than explicit Euler; using
    # one threshold for both would either cry wolf here or miss a real problem there.
    # `need` comes from tune, never from a literal repeated here: `stiffness_ceiling` sizes
    # springs to land exactly on this bound, so a copy that drifted -- or a `>` where the
    # bound is inclusive -- makes the tuner's own output fail the tuner's own check. It did:
    # with armature no longer inflating the distal joints, the clamped joints sit at exactly
    # 12 steps/cycle and `>` reported the correctly-tuned rig as too stiff.
    fmax = max(L.hz for L in loads)
    steps = 1.0 / (fmax * model.opt.timestep) if fmax > 0 else float("inf")
    need = steps_per_cycle_floor(creature)
    ok = steps >= need * (1.0 - 1e-9)
    print(f"stiffest joint oscillates at {fmax:.1f} Hz = {steps:.0f} steps/cycle "
          f"({creature.world.integrator}, wants >= {need}) "
          + ("fine" if ok else "*** TOO STIFF FOR THIS TIMESTEP ***"))

    # --- symmetry -----------------------------------------------------------------------
    by_name = {L.name: L.tau_static for L in loads}
    worst, worst_pair = 0.0, None
    for name, v in by_name.items():
        if name.endswith("_l") or "_l_" in name:
            mate = name.replace("_l_", "_r_") if "_l_" in name else name[:-2] + "_r"
            if mate in by_name:
                d = abs(abs(v) - abs(by_name[mate]))
                if worst_pair is None or d > worst:   # see the landmark check above
                    worst, worst_pair = d, (name, mate)
    if worst_pair:
        ok = "symmetric" if worst < 1e-6 else "*** ASYMMETRIC ***"
        print(f"\nleft/right torque mismatch: {worst:.3e} N*m "
              f"({worst_pair[0]} vs {worst_pair[1]})  {ok}")

    # --- does it actually stand? ----------------------------------------------------------
    # The whole point of the tone pass. Everything above is a prediction; this is the test.
    r = stand_test(model, data)
    print(f"\nunactuated stand test (3 s, motors off): sank {r.drop*1000:+.0f} mm "
          f"({r.rel_drop*100:.1f}% of withers), pitch {r.tilt:+.1f} deg, "
          f"support margin {r.margin*1000:+.0f} mm  {r.verdict}")
    # The verdict is decided on the shove, so the shove has to be reported -- otherwise a body
    # that folds sideways prints a tidy pitch next to *** COLLAPSES ***. A symmetric settle
    # cannot excite an antisymmetric mode at all, which is the reason this phase exists.
    print(f"  then shoved at {r.nudge:.3f} m/s ({NUDGE_FRACTION*100:.0f}% of the "
          f"{r.v_tip:.3f} m/s that would tip this stance over): peak tilt {r.tilt_peak:.1f} deg")
    return 0 if r.ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
