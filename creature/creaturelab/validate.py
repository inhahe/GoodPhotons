"""Does the built body actually work?

Every question here is one that reading the rig cannot answer, which is the reason the
module exists. A `.ftcl` file that is completely wrong -- a limb that does not reach the
ground, a spring that folds under its own animal, a stance whose centre of mass sits
behind the hind feet -- parses cleanly, compiles cleanly, loads in MuJoCo cleanly, and
renders as a perfectly plausible dog. The failure only appears once gravity is applied,
and by then it is a policy that mysteriously will not learn to walk.

`stand_test` is the acceptance bar P0 is held to: motors off, gravity, a shove, does it
still look like an animal standing up. Motors *off* is the point -- a body that needs its
controller to avoid collapsing has pushed the job of not falling over into the policy,
where it costs training budget forever, instead of into the ligaments, where real animals
put it.
"""
from __future__ import annotations

from dataclasses import dataclass


# How long "settled" means, shared by the acceptance test and by the tuner that has to
# pass it. These must be the same number. When they were not -- the tuner settling for
# 1.5 s, the test for 3 s -- the tuner declared victory on a body that was still sinking,
# and the extra 1.5 s took it from 6% to 9.2% and a failure it had no way to see.
SETTLE_SECONDS = 3.0

#: How hard `stand_test` shoves the settled body, as a fraction of the toppling speed its
#: own stance geometry provides (`tune.tipping_velocity`).
#:
#: A symmetric settle cannot see an antisymmetric mode. Not "is unlikely to" -- cannot, at
#: any duration: the default rig is passively unstable in roll with a 0.67 s e-folding time,
#: and it sat at *exactly* 0.00 deg of roll for 19 of 20 simulated seconds, because a
#: perfectly left-right symmetric body released from a perfectly symmetric pose has nothing
#: to roll towards. The instability was real, the release was clean, and the test therefore
#: reported a body that would fall over the instant a policy breathed on it as STANDS. So
#: the shove is not robustness testing bolted on top of the stand test; it is what makes the
#: stand test able to answer its own question.
#:
#: Sized against `v_c` rather than the Froude speed `sqrt(gL)` because past `v_c` *no*
#: stiffness helps -- see `tune.tipping_velocity` for the measurement that settled this. As
#: a fraction of `v_c` the shove means the same thing on every morph: 0.10 carries 1% of the
#: energy needed to tip the body over as one rigid piece, so anything that falls, fell
#: because it *bent* -- the only failure passive tone is responsible for, and the one this
#: bar exists to measure.
#:
#: The value is the middle of a plateau, not a round number. Swept against both things that
#: matter -- does it still catch the known-unstable body, and how many randomised morphs
#: survive it -- over 24 morphs at each of three morph-range scales:
#:
#:     fraction   catches the unbraced rig?   yield @0.5 / @0.75 / @1.0
#:       0.00              no (!)                 100% / 100% /  92%
#:       0.05              yes                     96% /  71% /  67%
#:       0.08              yes                     92% /  71% /  67%
#:       0.10              yes                     96% /  71% /  67%
#:       0.12              yes                     92% /  71% /  67%
#:       0.15              yes                     88% /  71% /  58%
#:       0.20              yes                     83% /  62% /  58%
#:
#: 0.05 through 0.12 give the same verdict on all 48 of the two harder scales' bodies, and the
#: 92-vs-96% wobble at scale 0.5 is one morph either way. So the bar does not hinge on the exact
#: figure -- which is the property a constant like this needs, and the same kind of plateau
#: argument as `tune.brace`'s mode-selection threshold. 0.10 is the middle of that plateau.
#: Picking 0.15 for roundness would cost real morphs at both ends for nothing; going softer than
#: 0.05 runs at the edge of catching the thing the shove is for, and buys no yield at all.
#:
#: The 0.00 row is why the whole thing exists: with no shove, the bar passes a body that
#: falls over on its own.
NUDGE_FRACTION = 0.10


@dataclass
class StandResult:
    drop: float             # m the floating base sank (positive = downward)
    rel_drop: float         # ... as a fraction of withers height
    tilt: float             # deg between the trunk's own up-axis and world up
    withers: float          # m, the length scale everything above is judged against
    margin: float           # m from the CoM's ground projection to the support edge
    finite: bool            # False = the integrator blew up
    ok: bool
    # The shove phase. `v_tip` is the stance's own toppling speed, `nudge` the fraction of
    # it actually applied, `tilt_peak` the worst tilt reached while recovering -- which is
    # what the verdict is judged on, because a body that swings out to 40 deg and happens to
    # come back has not passed anything a policy could rely on.
    v_tip: float = 0.0
    nudge: float = 0.0
    tilt_peak: float = 0.0

    @property
    def verdict(self) -> str:
        if not self.finite:
            return "*** NON-FINITE ***"
        return "STANDS" if self.ok else "*** COLLAPSES ***"


def withers_height(model, data) -> float:
    """Shoulder height, or the tallest point of anything that has no shoulder.

    Used as the length scale for judging sag, so it has to exist for *any* rig -- a
    creature with no bone called `thorax` is not a reason to crash.
    """
    import mujoco

    from .emit_mjcf import geom_z_extent
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "thorax")
    if gid >= 0:
        return geom_z_extent(model, data, gid)[1]
    return max(geom_z_extent(model, data, g)[1] for g in range(model.ngeom))


def trunk_tilt_of(data) -> float:
    """Degrees between the trunk's own up-axis and world up, from an ALREADY-forward `data`.

    Extracting a pitch from the quaternion as `asin(2*(qy*qw - qx*qz))` looks equivalent
    and is not: `asin` saturates, so it folds large rotations back into small ones. A draw
    that ended up completely inverted reported a tidy "+10 deg pitch" and was only caught
    because it had also sunk most of a metre. Taking the angle between two axis vectors has
    no such branch -- upside down is 180, and cannot be anything else.

    Split out from `trunk_tilt` because P1's episode loop asks this question once per
    control step, right after `mj_step` has already updated the kinematics -- and the
    `mj_forward` inside `trunk_tilt` would then be a redundant full dynamics evaluation in
    the hot loop. Splitting rather than reimplementing keeps the training termination
    condition and the P0 acceptance bar *literally the same measure*, which matters: a
    policy that can satisfy one and not the other is a policy that learned to game the
    difference.
    """
    import numpy as np

    # Body 1 is the first body after the world, i.e. the floating root the rig hangs from.
    return float(np.degrees(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0))))


def trunk_tilt(model, data) -> float:
    """`trunk_tilt_of`, forcing the kinematics up to date first."""
    import mujoco

    mujoco.mj_forward(model, data)
    return trunk_tilt_of(data)


def stand_test(model, data, seconds: float = SETTLE_SECONDS, sag_frac: float = 0.09,
               tilt_deg: float = 8.0,
               nudge_frac: float = NUDGE_FRACTION) -> StandResult:
    """Drop the creature on the ground with its motors off, shove it, see whether it stays up.

    The sag budget is a *fraction of the animal's own height*, never a distance. An
    absolute limit picked on one dog fails a Great Dane for sagging less, proportionally,
    than the dog the limit came from -- the identical units mistake that a stiffness
    literal makes, one level up, and the reason `posture.sag` is declared as an angle.

    Two phases, and the second is not optional decoration. Settling from the reference pose
    is left-right symmetric, so it excites only symmetric modes; an antisymmetric one is
    invisible to it *forever*, not merely for the first three seconds (see `NUDGE_FRACTION`).
    Phase two therefore shoves the settled body out over the weakest edge of its own support
    polygon at `nudge_frac` of the speed that would tip it, and settles again. Everything
    reported -- `drop`, `tilt`, `margin` -- is measured after both phases, so it describes
    the state the body actually ends up in; `tilt_peak` covers the excursion in between.

    `nudge_frac=0.0` skips the shove, for callers that want to attribute a failure to one
    phase or the other. It is not the default because a body that only passes without the
    shove has not passed.
    """
    import mujoco
    import numpy as np

    from .emit_mjcf import place_on_ground
    from .tune import support_polygon, tipping_velocity

    mujoco.mj_resetData(model, data)
    place_on_ground(model, data)
    withers = withers_height(model, data)
    z0 = float(data.qpos[2])
    steps = int(seconds / model.opt.timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    finite = bool(np.all(np.isfinite(data.qpos)))
    v_tip = nudge = 0.0
    peak = trunk_tilt(model, data) if finite else float("nan")
    if finite and nudge_frac > 0.0:
        import copy

        try:
            v_tip, direction = tipping_velocity(model, data)
        except Exception:
            v_tip, direction = 0.0, np.zeros(2)   # no contacts: nothing to shove it off
        if not np.isfinite(v_tip) or v_tip <= 0.0:
            direction = np.zeros(2)
        else:
            nudge = nudge_frac * v_tip
        # Both signs of the weakest axis, from the same settled state, and the report is the
        # worse of the two -- a body that holds together shoved left and folds shoved right
        # has not passed anything. Testing both is also what keeps a *symmetric* rig's tuned
        # gains symmetric (`tune.brace` reads the mode from this same pair): the roll is not
        # antisymmetric in its joint response, because the side that unloads goes slack
        # instead of deflecting, so one direction alone reports one shoulder giving twice as
        # much as its mirror and would spring them differently. Nothing here needs to know
        # which joints are mirrors of which, or that the body has a left and a right at all.
        settled, worst = copy.deepcopy(data), None
        for sign in (+1.0, -1.0):
            trial = copy.deepcopy(settled)
            trial.qvel[0:2] += sign * nudge * direction
            hi = trunk_tilt_of(trial)
            for _ in range(steps):
                mujoco.mj_step(model, trial)
                if not np.all(np.isfinite(trial.qpos)):
                    hi = float("inf")
                    break
                hi = max(hi, trunk_tilt_of(trial))
            if worst is None or hi > worst[0]:
                worst = (hi, trial)
        peak = max(peak, worst[0])
        finite = finite and np.isfinite(worst[0])
        # Everything below is reported from whichever shove went worst, so the numbers
        # describe the state the body actually ends up in rather than its better side. Copied
        # back into the caller's `data` rather than rebound locally, because the caller's
        # object is the contract: `ftcl_build --view` hands exactly this `data` to the MuJoCo
        # viewer, so a caller that asks to look at the body after the test must be shown the
        # body the test judged.
        data.qpos[:] = worst[1].qpos
        data.qvel[:] = worst[1].qvel
        mujoco.mj_forward(model, data)

    drop = z0 - float(data.qpos[2]) if finite else float("nan")
    tilt = trunk_tilt(model, data) if finite else float("nan")
    if not finite:
        peak = float("nan")

    margin = float("nan")
    if finite:
        try:
            margin = support_polygon(model, data)[3]
        except Exception:
            pass                                # no contacts at all: leave it as nan

    rel = abs(drop) / max(withers, 1e-6) if finite else float("inf")
    ok = finite and rel < sag_frac and abs(peak) < tilt_deg
    return StandResult(drop, rel, tilt, withers, margin, finite, ok,
                       v_tip=v_tip, nudge=nudge, tilt_peak=peak)
