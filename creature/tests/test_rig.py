"""Regression tests for the rig-generation and passive-tone pipeline.

Biased deliberately towards the failures that *do not raise*. Every serious bug found
while building this layer produced a plausible number rather than an error -- feet seated
3.6 cm above the floor reported every joint torque as 0.00, a dog standing on its own
self-collision reported clean statics, a body flipped 170 degrees reported "+10 deg pitch".
None of that is caught by "does it load", so none of these tests check that.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mujoco = pytest.importorskip("mujoco")

from creaturelab.build import load, sample_morph                    # noqa: E402
from creaturelab.emit_mjcf import (geom_z_extent, place_on_ground,  # noqa: E402
                                   to_mjcf)
from creaturelab.tune import (SEAT, UnstablePose, build_tuned,      # noqa: E402
                              ground_reaction, stiffness_ceiling,
                              support_polygon, tipping_velocity)
from creaturelab.validate import (SETTLE_SECONDS, stand_test,       # noqa: E402
                                  trunk_tilt, withers_height)

RIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rigs", "canis.ftcl")
SQRT_HALF = math.sqrt(0.5)


@pytest.fixture(scope="module")
def tuned():
    creature, loads = build_tuned(RIG)
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    return creature, loads, model, mujoco.MjData(model)


# --- geometry ----------------------------------------------------------------------------

def _one_geom(body_attrs: str, geom: str):
    xml = f"""<mujoco><worldbody><body {body_attrs}>{geom}</body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_geom_z_extent_sphere():
    m, d = _one_geom('pos="0 0 1"', '<geom type="sphere" size="0.25"/>')
    assert geom_z_extent(m, d, 0) == pytest.approx((0.75, 1.25))


def test_geom_z_extent_capsule_is_not_the_bounding_sphere():
    """A horizontal capsule reaches `radius` above centre, not `half_length + radius`.

    This exact confusion -- `geom_rbound` is a bounding *sphere* -- seated the rig's feet
    3.6 cm off the floor, which made every inverse-dynamics torque a free-fall measurement
    and therefore identically zero.
    """
    m, d = _one_geom('pos="0 0 1"',
                     '<geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.05"/>')
    assert geom_z_extent(m, d, 0) == pytest.approx((0.95, 1.05))
    assert m.geom_rbound[0] > 0.5            # the bound really is much larger


def test_geom_z_extent_rotated_box():
    m, d = _one_geom('pos="0 0 1" euler="0 45 0"', '<geom type="box" size="0.5 0.2 0.1"/>')
    half = (0.5 + 0.1) * math.sqrt(0.5)      # |R_z . size| for a 45-degree tilt
    lo, hi = geom_z_extent(m, d, 0)
    assert hi - 1.0 == pytest.approx(half, rel=1e-9)
    assert 1.0 - lo == pytest.approx(half, rel=1e-9)


def test_place_on_ground_puts_the_lowest_point_at_the_clearance(tuned):
    _, _, model, data = tuned
    place_on_ground(model, data, 0.02)
    mujoco.mj_forward(model, data)
    lowest = min(geom_z_extent(model, data, g)[0] for g in range(model.ngeom)
                 if model.geom_bodyid[g] != 0)
    assert lowest == pytest.approx(0.02, abs=1e-6)


# --- the reference pose ------------------------------------------------------------------

def test_all_four_feet_reach_the_ground(tuned):
    """A limb that does not touch down makes every later measurement meaningless."""
    _, _, model, data = tuned
    place_on_ground(model, data, 0.0)
    mujoco.mj_forward(model, data)
    lows = {}
    for paw in ("hpaw_l", "hpaw_r", "fpaw_l", "fpaw_r"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, paw)
        assert gid >= 0, f"{paw} missing from the rig"
        lows[paw] = geom_z_extent(model, data, gid)[0]
    assert max(lows.values()) - min(lows.values()) < 1e-4, f"feet are uneven: {lows}"


def test_reference_pose_is_statically_supported(tuned):
    _, _, model, data = tuned
    place_on_ground(model, data, -SEAT)
    mujoco.mj_forward(model, data)
    pts, bids, com, margin = support_polygon(model, data)
    assert len(pts) == 4, "expected exactly the four paws in contact"
    assert margin > 0.02, f"centre of mass is only {margin*1000:.0f} mm inside the feet"


def test_ground_reaction_balances_weight_and_never_pulls(tuned):
    _, _, model, data = tuned
    place_on_ground(model, data, -SEAT)
    mujoco.mj_forward(model, data)
    frozen = ground_reaction(model, data)
    total = sum(F for _, F, _ in frozen)
    weight = float(np.sum(model.body_mass)) * 9.81
    assert total[2] == pytest.approx(weight, rel=1e-6)
    assert abs(total[0]) < 1e-6 * weight and abs(total[1]) < 1e-6 * weight
    for _, F, _ in frozen:
        assert F[2] >= -1e-9 * weight, "a foot is being asked to pull down on the ground"


def test_support_polygon_rejects_a_com_outside_the_feet():
    """Underdetermined wrench balance always finds a solution; the geometry must be checked.

    With four feet this is 6 equations in 12 unknowns, so `lstsq` returns an exact fit even
    when it requires a foot to pull downward -- a residual check never fires. Shifting the
    mass well behind the feet has to be caught geometrically or not at all.
    """
    xml = """<mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="trunk" pos="0 0 0.5">
          <freejoint/>
          <geom type="box" pos="-1.5 0 0" size="0.1 0.1 0.1" density="4000"/>
          <geom name="f" type="sphere" pos="0.2 0 -0.5" size="0.05"/>
          <geom name="b" type="sphere" pos="-0.2 0 -0.5" size="0.05"/>
        </body>
      </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    place_on_ground(m, d, -SEAT)
    mujoco.mj_forward(m, d)
    with pytest.raises(UnstablePose):
        ground_reaction(m, d)


# --- the passive tone --------------------------------------------------------------------

def test_self_collisions_are_excluded(tuned):
    """The reference pose overlaps by design -- a humerus sits inside the body outline.

    Left alone, MuJoCo resolves that overlap as contact and the dog stands on its own
    interpenetration at several times body weight, with the feet carrying almost nothing.
    """
    creature, _, model, data = tuned
    assert creature.contact_excludes, "no pairs excluded; the load path is probably wrong"
    place_on_ground(model, data, -SEAT)
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        assert (b1 == 0) != (b2 == 0), "a creature-creature contact survived exclusion"


def test_every_joint_gets_finite_positive_gains(tuned):
    creature, loads, _, _ = tuned
    assert loads
    for L in loads:
        assert math.isfinite(L.stiffness) and L.stiffness > 0, L.name
        assert math.isfinite(L.damping) and L.damping >= 0, L.name
    for bone in creature.bones:
        for j in bone.joints:
            if j.actuated:
                assert j.stiffness is not None and j.damping is not None, j.name


def test_gains_respect_the_integrator_timestep(tuned):
    """A spring the integrator cannot follow explodes rather than reporting a wrong number."""
    creature, loads, model, _ = tuned
    for L in loads:
        assert L.stiffness <= stiffness_ceiling(creature, L.inertia) * 1.001, L.name
        assert L.hz * model.opt.timestep < 1.0 / 12.0 + 1e-9, L.name


def test_armature_scales_with_each_joint_s_own_inertia(tuned):
    """An absolute armature literal is invisible: the joint is just heavier than its bone.

    The rig's old `0.008 kg*m^2` was 0.7% of the spine's inertia and 3790% of the paw's --
    unmorphed, no morph vector involved -- so the paw joints were 97.4% fictitious rotor.
    Nothing raised, and the inflated inertia went on to license 38x more stiffness through
    `stiffness_ceiling` and to overdamp the same joints, both worst at the ground contact.
    So assert the property that a literal cannot have: a *constant fraction*.
    """
    creature, loads, _, _ = tuned
    ratio = creature.defaults.joint_armature_ratio
    armature = {j.name: j.armature
                for bone in creature.bones for j in bone.joints if j.armature is not None}
    assert len(armature) >= 20, "armature was not sized for most joints"

    inertias = [L.inertia for L in loads]
    assert max(inertias) / min(inertias) > 100, \
        "this rig no longer spans enough inertia for the test to be meaningful"

    for L in loads:
        a = armature.get(L.name)
        if a is None:
            continue
        # `L.inertia` already includes the armature, so the limb's own inertia is I - a.
        limb = L.inertia - a
        assert limb > 0, f"{L.name}: armature {a:.3g} exceeds total inertia {L.inertia:.3g}"
        assert a == pytest.approx(ratio * limb, rel=1e-6), L.name


def test_left_right_torques_match(tuned):
    """An asymmetric rig teaches an asymmetric gait, and is invisible in a 25-body tree."""
    _, loads, _, _ = tuned
    by_name = {L.name: L.tau_static for L in loads}
    pairs = 0
    for name, v in by_name.items():
        mate = (name.replace("_l_", "_r_") if "_l_" in name
                else name[:-2] + "_r" if name.endswith("_l") else None)
        if mate in by_name:
            pairs += 1
            assert abs(abs(v) - abs(by_name[mate])) < 1e-9, f"{name} vs {mate}"
    assert pairs >= 8, "expected several mirrored joint pairs"


def test_buckling_joints_are_sprung_far_above_their_static_load(tuned):
    """The stifle holds almost no torque standing square and folds instantly without this.

    Sizing it from `tau_static` alone gave it k=1.4 N*m/rad and it went straight to its
    -85 degree limit. The compressive load is what has to set it.
    """
    _, loads, _, _ = tuned
    stifle = [L for L in loads if L.name.startswith("stifle")]
    assert stifle
    for L in stifle:
        assert L.k_buckle > 0.0
        assert L.stiffness > L.k_buckle


# --- the acceptance bar ------------------------------------------------------------------

def test_the_default_rig_stands(tuned):
    _, _, model, data = tuned
    r = stand_test(model, data)
    assert r.finite and r.ok, (f"sank {r.drop*1000:.0f} mm ({r.rel_drop*100:.1f}%), "
                               f"tilt {r.tilt:.1f} deg")
    assert r.margin > 0.0


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.7])
def test_it_stands_across_the_body_scale_range(scale):
    creature, _ = build_tuned(RIG, {"body_scale": scale})
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    r = stand_test(model, mujoco.MjData(model))
    assert r.ok, f"body_scale={scale}: sank {r.rel_drop*100:.1f}%, tilt {r.tilt:.1f} deg"


@pytest.mark.parametrize("mass", [10.0, 90.0, 320.0])
def test_sag_is_invariant_to_mass(mass):
    """Gains are measured from the load, so a 32x heavier animal must sag the same *fraction*.

    This is the property a stiffness literal cannot have, and the reason the `posture`
    block declares an angle instead of a spring constant.
    """
    creature, _ = build_tuned(RIG, {"body_mass": mass})
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    r = stand_test(model, mujoco.MjData(model))
    assert r.ok
    assert r.rel_drop == pytest.approx(0.049, abs=0.01), f"{mass} kg sagged {r.rel_drop:.3f}"


@pytest.mark.parametrize("quat, expect", [
    ((1.0, 0.0, 0.0, 0.0), 0.0),                 # upright
    ((0.0, 1.0, 0.0, 0.0), 180.0),               # 180 deg about x: fully inverted
    ((SQRT_HALF, SQRT_HALF, 0.0, 0.0), 90.0),    # 90 deg about x: on its side
    ((0.0, 0.0, 1.0, 0.0), 180.0),               # 180 deg about y: inverted the other way
])
def test_tilt_does_not_fold_a_large_rotation_into_a_small_one(quat, expect):
    """`asin` saturates; an inverted body once reported "+10 deg pitch".

    Measured on the trunk directly rather than through `stand_test`, which resets the pose
    before it settles -- there is no way to hand it an orientation, and it would be wrong
    if there were.
    """
    creature, _ = build_tuned(RIG)
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)
    place_on_ground(model, data)
    data.qpos[3:7] = quat
    assert trunk_tilt(model, data) == pytest.approx(expect, abs=1e-6)


def test_tipping_velocity_is_the_stance_geometry_and_nothing_else(tuned):
    """`v_c = w*sqrt(g/h)`, from the support polygon -- not a fraction of the Froude speed.

    Pinned because the temptation to quote the shove in `sqrt(gL)` is strong -- it is the unit
    everything else in this project uses -- and it is wrong: past `v_c` the body rotates about
    a foot as one rigid piece, so no stiffness helps, and 0.10*sqrt(gL) is not survived at a
    hundredfold abduction stiffness. The direction has to come out of the hull too, so that
    nothing in the tuner needs to know the body has a left and a right.
    """
    _, _, model, _ = tuned
    data = mujoco.MjData(model)
    place_on_ground(model, data)
    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)

    pts, _, com, margin = support_polygon(model, data)
    h = com[2] - np.mean([p[2] for p in pts])
    v_c, direction = tipping_velocity(model, data)
    assert v_c == pytest.approx(margin * math.sqrt(9.81 / h), rel=1e-6)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    # A quadruped's weakest support edge is a long one running fore-aft, so the way it topples
    # is sideways. This asserts the geometry came out right, not that the code knew in advance.
    assert abs(direction[1]) > abs(direction[0])
    # And the reason the scale had to be geometric at all: it is nowhere near sqrt(gL).
    assert v_c < 0.25 * math.sqrt(9.81 * withers_height(model, data))


def test_the_shove_is_what_makes_the_stand_test_able_to_answer(tuned):
    """The regression that matters: without the shove the bar cannot see a real collapse.

    canis is passively unstable in roll with a 0.67 s e-folding time. It sat at *exactly*
    0.00 deg of roll for 19 of 20 simulated seconds, because a symmetric body released from a
    symmetric pose has nothing to roll towards -- so a symmetric settle called it STANDS, and
    P1 would have trained a policy on a dog that falls over on its own. Both halves are
    pinned here: the shove catches it, and the tuner's `brace` pass fixes it.
    """
    from creaturelab import tune

    # The tuned rig -- `brace` included -- must survive its own shove with room to spare.
    _, _, model, _ = tuned
    r = stand_test(model, mujoco.MjData(model))
    assert r.ok, f"tuned rig fell: sag {r.rel_drop*100:.1f}%, peak tilt {r.tilt_peak:.1f} deg"
    assert r.nudge > 0.0 and r.tilt_peak < 4.0

    # The same rig with `brace` skipped: what the tuner produced before this existed.
    creature = load(RIG)
    real, tune.brace = tune.brace, lambda *a, **k: 0.0
    try:
        tune.apply_posture(creature)
    finally:
        tune.brace = real
    unbraced = mujoco.MjModel.from_xml_string(to_mjcf(creature))

    assert stand_test(unbraced, mujoco.MjData(unbraced), nudge_frac=0.0).ok, \
        "a symmetric settle is supposed to be fooled by this body -- that is the whole point"
    fell = stand_test(unbraced, mujoco.MjData(unbraced))
    assert not fell.ok and fell.tilt_peak > 45.0, \
        f"the shove failed to expose the roll mode: peak tilt only {fell.tilt_peak:.1f} deg"


def test_brace_stiffens_the_mode_and_not_the_skeleton(tuned):
    """`brace` must stay surgical, because stiffening broadly makes the body *worse*.

    Measured, and the reason the mode-selection threshold is where it is: stiffening all 31
    joints by 2x or 3x collapses the shove test outright, while stiffening exactly the four
    abduction joints by 3x passes it. Stiffness is not monotonically stabilising. So a pass
    that quietly widened its selection would look like a fix and be a regression, and it
    would not fail any other test in this file.
    """
    creature, _, _, _ = tuned
    plain = load(RIG)
    from creaturelab.tune import apply_posture

    real = None
    try:
        from creaturelab import tune
        real, tune.brace = tune.brace, lambda *a, **k: 0.0
        apply_posture(plain)
    finally:
        from creaturelab import tune
        tune.brace = real

    before = {j.name: j.stiffness for b in plain.bones for j in b.joints}
    raised = {j.name for b in creature.bones for j in b.joints
              if j.stiffness > before.get(j.name, 0.0) * 1.001}
    assert raised, "brace changed nothing at all"
    assert len(raised) <= 12, f"brace stiffened {len(raised)} joints: {sorted(raised)}"
    # It must be symmetric on a symmetric rig. `brace` has no notion of mirrored joints; it
    # gets this by shoving both signs of the weakest axis, because the roll's joint response
    # is *not* antisymmetric -- the side that unloads goes slack instead of deflecting, and
    # one direction alone reports one shoulder giving 1.9x its mirror.
    for name in raised:
        if "_l" in name:
            assert name.replace("_l", "_r") in raised, f"{name} raised but not its mirror"


def test_randomised_morphs_mostly_stand():
    """P4's premise: sampling the morph distribution has to yield trainable bodies.

    All eight, and not marginally -- the worst peak tilt under the shove is 2.6 deg against
    an 8 deg bar -- so this asserts the full count rather than some fraction of it.

    It briefly did not, and that is the part worth recording. When `stand_test` began
    shoving the body, seed 3 failed, and the tempting reading was that a harder bar is
    simply harder and the expected count should follow it down. That reading was wrong.
    `brace` had identified seed 3's mode correctly and then abandoned it one secant step
    short of a body that stands, for two bugs that only a failing morph was ever going to
    surface: it judged progress on peak tilt, which saturates near 95 deg once the body is
    past its balance point and so carries no gradient in exactly the regime the decision is
    made in, and it re-measured the mode's joints every pass, which is self-erasing --
    stiffening a joint is what stops it deflecting, so the fix drops out of the next read --
    and which silently invalidated the secant's own model. Both are fixed, and documented at
    length in `tune.brace`. Relaxing this assertion would have concealed both of them.
    """
    import random
    params = load(RIG).params
    ok = 0
    for seed in range(8):
        creature, _ = build_tuned(RIG, sample_morph(params, random.Random(seed), 0.5))
        model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
        ok += stand_test(model, mujoco.MjData(model)).ok
    assert ok == 8, f"only {ok}/8 randomised bodies stood"


def test_withers_height_survives_a_rig_with_no_thorax():
    m, d = _one_geom('pos="0 0 1"', '<geom type="sphere" size="0.25"/>')
    assert withers_height(m, d) == pytest.approx(1.25)


# --- authored values win -----------------------------------------------------------------

def test_an_authored_stiffness_is_not_overridden():
    """design.md, directability: the sim proposes, the author disposes."""
    creature = load(RIG)
    target = creature.bone("femur_l").joints[0]
    target.stiffness = 12345.0
    from creaturelab.tune import apply_posture
    apply_posture(creature)
    assert creature.bone("femur_l").joints[0].stiffness == 12345.0
