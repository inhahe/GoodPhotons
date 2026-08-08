"""Does the built body actually work?

Every question here is one that reading the rig cannot answer, which is the reason the
module exists. A `.ftcl` file that is completely wrong -- a limb that does not reach the
ground, a spring that folds under its own animal, a stance whose centre of mass sits
behind the hind feet -- parses cleanly, compiles cleanly, loads in MuJoCo cleanly, and
renders as a perfectly plausible dog. The failure only appears once gravity is applied,
and by then it is a policy that mysteriously will not learn to walk.

`stand_test` is the acceptance bar P0 is held to: motors off, three seconds of gravity,
does it still look like an animal standing up. Motors *off* is the point -- a body that
needs its controller to avoid collapsing has pushed the job of not falling over into the
policy, where it costs training budget forever, instead of into the ligaments, where real
animals put it.
"""
from __future__ import annotations

from dataclasses import dataclass


# How long "settled" means, shared by the acceptance test and by the tuner that has to
# pass it. These must be the same number. When they were not -- the tuner settling for
# 1.5 s, the test for 3 s -- the tuner declared victory on a body that was still sinking,
# and the extra 1.5 s took it from 6% to 9.2% and a failure it had no way to see.
SETTLE_SECONDS = 3.0


@dataclass
class StandResult:
    drop: float             # m the floating base sank (positive = downward)
    rel_drop: float         # ... as a fraction of withers height
    tilt: float             # deg between the trunk's own up-axis and world up
    withers: float          # m, the length scale everything above is judged against
    margin: float           # m from the CoM's ground projection to the support edge
    finite: bool            # False = the integrator blew up
    ok: bool

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
               tilt_deg: float = 8.0) -> StandResult:
    """Drop the creature on the ground with its motors off and see whether it stays up.

    The sag budget is a *fraction of the animal's own height*, never a distance. An
    absolute limit picked on one dog fails a Great Dane for sagging less, proportionally,
    than the dog the limit came from -- the identical units mistake that a stiffness
    literal makes, one level up, and the reason `posture.sag` is declared as an angle.
    """
    import mujoco
    import numpy as np

    from .emit_mjcf import place_on_ground
    from .tune import support_polygon

    mujoco.mj_resetData(model, data)
    place_on_ground(model, data)
    withers = withers_height(model, data)
    z0 = float(data.qpos[2])

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

    finite = bool(np.all(np.isfinite(data.qpos)))
    drop = z0 - float(data.qpos[2]) if finite else float("nan")
    tilt = trunk_tilt(model, data) if finite else float("nan")

    margin = float("nan")
    if finite:
        try:
            margin = support_polygon(model, data)[3]
        except Exception:
            pass                                # no contacts at all: leave it as nan

    rel = abs(drop) / max(withers, 1e-6) if finite else float("inf")
    ok = finite and rel < sag_frac and abs(tilt) < tilt_deg
    return StandResult(drop, rel, tilt, withers, margin, finite, ok)
