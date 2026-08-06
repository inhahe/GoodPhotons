"""Measure the body, then size its passive tone to fit -- instead of guessing gains.

A bare skeleton folds up. Real ones do not, and not because the animal is thinking about
it: ligaments, joint capsules and resting muscle tone carry a standing quadruped with the
nervous system barely involved. So the rig needs passive stiffness before P1 can hope to
work -- a fully floppy 31-DOF body makes PPO spend its whole budget discovering "don't
collapse" instead of "walk".

The obvious way to add it is to write `stiffness 150` on each joint. That is exactly the
mistake this whole project exists to avoid. A stiffness literal is only correct for one
body: it has units of N*m/rad, so it scales as mass x length^2 x time^-2, and every one of
those changes when `body_scale`, `body_mass`, `femur_len` or any stance angle moves. Under
the domain randomisation of P4 the rig sweeps a Chihuahua-to-Great-Dane range -- a spring
tuned for the middle of it leaves one end of the distribution collapsing and the other end
rigid, and *nothing reports this*, because both still load and both still simulate. You
would find out as a policy that mysteriously fails on large morphs.

So the rig declares the *goal* -- "no joint may sag more than 5 degrees holding the
standing pose" -- and this module measures what that costs on the body that the current
morph vector actually produced:

* **Static holding torque** comes from inverse dynamics at the standing pose with the feet
  seated on the ground, so it is the true load path through the skeleton (`tau_static`).
* **Load scale** covers the joints that carry nothing when standing square. A dog standing
  still needs zero torque at every yaw and abduction joint, by symmetry -- but they still
  need tone, or the body flops laterally the instant it is disturbed. Their floor comes
  from the subtree they carry: mass x g x the moment arm to that subtree's centre of mass
  (`tau_ref`), which is a real, geometry-derived quantity that scales correctly.
* **Damping** is set as a fraction of critical for each joint's own effective inertia,
  read off the mass matrix. Damping literals suffer from the identical scaling problem --
  worse, actually, since critical damping goes as sqrt(k * I) and so depends on the
  stiffness that was itself just measured.

Everything here is a default. A joint that names its own `stiffness` or `damping` in the
rig keeps it (design.md, directability: the sim proposes, the author disposes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .emit_mjcf import place_on_ground, to_mjcf
from .model import Creature

# The feet are seated this far into the floor for the inverse-dynamics pass. MuJoCo only
# builds a contact constraint once the gap goes negative, and inverse dynamics with no
# active contact silently measures a *free fall* -- in which every internal torque is
# exactly zero, because free fall is a rigid-body motion and needs no internal force. That
# reads as a plausible "all joints unloaded" report rather than as an error, which is
# precisely why it is worth a named constant and this comment. 1 mm is far inside MuJoCo's
# default contact softness, so the measured pose is not visibly disturbed.
SEAT = 0.001


class UnstablePose(RuntimeError):
    """The reference pose cannot be held by *any* physical ground reaction.

    Raised rather than worked around, because there is nothing to measure: sizing springs
    for a pose the creature cannot stand in produces gains for a body that would have
    fallen over before the springs mattered. P4's randomiser is expected to catch this and
    reject the sample -- an unstandable morph is a fact about the morph, not a failure of
    the tuner. `margin` is the signed distance in metres (negative here).
    """

    def __init__(self, msg: str, margin: float):
        super().__init__(msg)
        self.margin = margin


def _dist_to_segment(c, a, b) -> float:
    import numpy as np
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 <= 0.0 else min(1.0, max(0.0, float((c - a) @ ab) / L2))
    return float(np.linalg.norm(c - (a + t * ab)))


def support_polygon(model, data):
    """Foot contacts, the centre of mass, and the static stability margin.

    The margin is the signed distance from the centre of mass's *ground projection* to the
    edge of the convex hull of the contact points -- positive inside, negative outside.
    This is the textbook criterion for whether a pose can be held at all, and it is the one
    thing the wrench balance in `ground_reaction` cannot tell you on its own: with two or
    more feet that balance is six equations in 3N unknowns, so it is underdetermined and
    `lstsq` returns an exact solution *always*, including solutions where a foot has to
    pull down on the ground to stop the creature toppling. A residual check therefore never
    fires, and a morph that falls flat on its face reports a clean bill of health.

    That is not hypothetical: randomised morphs sample stance angles independently of limb
    lengths, so they routinely produce a body whose feet are nowhere near under its mass.
    """
    import numpy as np

    pts, bids = [], []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 == 0) == (b2 == 0):          # want exactly one side to be the floor
            continue
        pts.append(np.array(c.pos))
        bids.append(b2 if b1 == 0 else b1)
    if not pts:
        raise RuntimeError("no foot contacts at the reference pose -- the creature is not "
                           "standing on anything, so there is no static load to measure")

    com = np.array(data.subtree_com[1])     # body 1 is the root bone: the whole creature
    P = np.array([p[:2] for p in pts])
    c2 = com[:2]

    margin = None
    if len(P) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(P)
            # `equations` rows are [nx, ny, offset], outward-normalised, so a*x + c <= 0
            # inside. The largest value is the (signed) distance to the nearest edge.
            margin = -float(np.max(hull.equations[:, :2] @ c2 + hull.equations[:, 2]))
        except Exception:
            margin = None                   # collinear feet: Qhull refuses, and rightly
    if margin is None:
        # One foot, two feet, or all feet in a line. The support polygon has no area, so no
        # pose is statically stable on it; report how far off the line the mass is.
        if len(P) == 1:
            d = float(np.linalg.norm(c2 - P[0]))
        else:
            k = np.argmax(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            i, j = divmod(int(k), len(P))
            d = _dist_to_segment(c2, P[i], P[j])
        margin = -d
    return pts, bids, com, float(margin)


@dataclass
class JointLoad:
    """What one joint has to hold, and what it would take to hold it."""
    name: str
    dof: int
    kind: str
    tau_static: float          # N*m needed to hold the standing pose (signed)
    tau_ref: float             # N*m scale of the subtree this joint carries
    k_buckle: float = 0.0      # N*m/rad the joint must beat to not buckle
    inertia: float = 0.0       # kg*m^2 seen at this joint, including armature
    stiffness: float = 0.0     # N*m/rad, filled by size_tone
    damping: float = 0.0       # N*m*s/rad
    limited_by: str = ""       # 'static' | 'floor' | 'buckle' | 'cap'
    sag_final: float = 0.0     # rad actually settled to, measured by `relax`
    relaxed: float = 1.0       # how much `relax` had to raise the predicted stiffness

    @property
    def hz(self) -> float:
        """Natural frequency of the resulting passive joint, for a timestep sanity check."""
        if self.inertia <= 0 or self.stiffness <= 0:
            return 0.0
        return math.sqrt(self.stiffness / self.inertia) / (2.0 * math.pi)


def _full_mass_matrix(model, data):
    """Dense joint-space inertia, across MuJoCo's two `mj_fullM` signatures.

    3.11 renamed `mjData.qM` to `mjData.M`, changed its sparse layout, and changed
    `mj_fullM` from (model, dst, qM) to (model, data, dst). Indexing the sparse array by
    `dof_Madr` -- the old way to read the diagonal -- now silently returns *off-diagonal*
    entries under the new layout, which can be negative; that is how this was caught, and
    it is exactly the kind of version drift that produces a plausible wrong number rather
    than an error. So: ask MuJoCo, and probe which signature it has.
    """
    import mujoco
    import numpy as np
    dst = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, dst)          # >= 3.11
    except TypeError:
        mujoco.mj_fullM(model, dst, data.qM)       # older
    return dst


def size_armature(creature: Creature) -> dict[str, float]:
    """Give every joint an armature proportional to the inertia it actually carries.

    Armature is reflected rotor inertia: `n^2 * I_rotor`. The body-independent quantity is
    therefore the *ratio* to the load, because a drive matched to a heavier limb has a
    bigger motor behind a similar gear ratio. An absolute kg*m^2 is the same units mistake
    a stiffness literal makes, and it hides better -- nothing about the model looks wrong,
    the joint is simply heavier than the bone attached to it.

    How bad it was, on this rig, unmorphed, with no morph vector involved at all: the
    single default `0.01 kg*m^2` was **0.7% of the spine's own inertia and 3790% of the
    paw's**. The paw joints were 97.4% fictitious rotor -- 0.00821 against a true 0.00021
    -- so their dynamics were essentially invented. It then propagated: `stiffness_ceiling`
    is `I*(2*pi*f_max)^2`, so an inflated `I` licensed 38x more stiffness than the real
    limb could follow, and damping is `2*zeta*sqrt(k*I)`, so those joints came out heavily
    overdamped as well. Both errors are largest exactly where the contact happens.

    Measured with armature at zero, which is what the model already has if nothing set it,
    so the inertia read here is the limb's own. Runs before `measure`, since every torque,
    ceiling and damping value downstream is computed through the mass matrix this changes.
    An authored per-joint `armature`, or an absolute `defaults.joint_armature`, still wins.
    """
    import mujoco
    import numpy as np

    ratio = creature.defaults.joint_armature_ratio
    override = creature.defaults.joint_armature
    if override is None and not ratio:
        return {}

    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)
    place_on_ground(model, data, -SEAT)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    Mdiag = np.diag(_full_mass_matrix(model, data))

    inertia: dict[str, float] = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        inertia[name] = float(Mdiag[int(model.jnt_dofadr[jid])])

    out: dict[str, float] = {}
    for bone in creature.bones:
        for j in bone.joints:
            if j.armature is not None or j.name not in inertia:
                continue                    # authored: not argued with
            j.armature = override if override is not None else ratio * inertia[j.name]
            out[j.name] = j.armature
    return out


def auto_exclude(creature: Creature, margin: float = 0.006) -> list[tuple[str, str]]:
    """Find the body pairs whose geometry overlaps in the reference pose, and exclude them.

    A skeleton built from primitives has proximal limb segments buried inside the trunk,
    because that is where they are in the animal: a humerus really does sit inside the body
    outline, and a scapula really does lie against the neck. To the collision solver those
    look like deep interpenetrations, and it does what it is supposed to do -- pushes them
    apart, hard. This rig was generating 1253 N between thorax and humerus, six times the
    dog's own weight, while the four feet carried 44 N of a 206 N animal. It was standing on
    its own self-collisions.

    Nothing about that is visible. The model loads, simulates, stays finite, and reports a
    full set of plausible joint torques -- every one of them measured through the wrong load
    path. It surfaced only because the frozen-reaction self-check in `measure_buckling`
    stopped agreeing with the contact solve.

    The fix is not to move the bones apart; they are where they belong. It is to say that
    volumes which share space *in the reference pose* share it by design. Detecting that
    rather than hand-listing the pairs is what keeps it correct under randomisation -- a
    longer scapula or a wider chest changes which pairs overlap, and a hand-written
    exclusion list would quietly stop covering them.

    `margin` also excludes pairs that merely come close, so a body that is barely clear at
    the default morph does not start fighting itself at the edge of the range.
    """
    import mujoco
    import numpy as np

    creature.contact_excludes = []
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)
    model.geom_margin[:] = np.maximum(model.geom_margin, margin)   # catch near-misses too
    mujoco.mj_forward(model, data)

    pairs: set[tuple[str, str]] = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 == 0 or b2 == 0:                    # touching the floor is the point
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        pairs.add((n1, n2) if n1 < n2 else (n2, n1))
    creature.contact_excludes = sorted(pairs)
    return creature.contact_excludes


def ground_reaction(model, data):
    """The ground reaction that *exactly* balances the creature, solved from statics.

    Not read out of the contact solver, and that is the whole point. MuJoCo's contacts are
    soft: at the 1 mm seating depth used here its solver supplied only 92 N under a 206 N
    dog, even through inverse dynamics, and quietly charged the missing 114 N to the
    floating base -- where it looks like nothing at all, because nobody reads the root's
    row of `qfrc_inverse`. Every joint torque measured that way is the torque for a body
    that is *partly held up by a crane*. Seating the feet deeper would trade one arbitrary
    number for another.

    Statics has no such freedom. With `N` foot contacts at world points `p_i`, the wrench
    balance about the creature's centre of mass is six equations in `3N` unknowns:

        sum(F_i) = M*g_up            sum((p_i - com) x F_i) = 0

    which fixes everything that is actually determined -- notably the fore/aft split, which
    follows from where the centre of mass sits, and is why a dog carries ~60% on its
    forelegs. What is left over is the genuinely indeterminate part (which of four legs
    takes the redundant share), and the minimum-norm solution spreads that evenly, which is
    both the conventional choice and the symmetric one.

    Returns `(body_id, world_force, point_in_body_frame)` per contact. The point is stored
    in the paw's own frame because a contact point is a *material point of the foot*: the
    buckling measurement needs the force to stay fixed in the world while its application
    point rides with the limb.
    """
    import numpy as np

    pts, bids, com, margin = support_polygon(model, data)
    if margin <= 0.0:
        raise UnstablePose(
            f"the reference pose is not statically stable: the centre of mass projects "
            f"{-margin*1000:.0f} mm outside the support polygon, so no ground reaction "
            f"with non-negative normals can hold it -- this creature topples before any "
            f"passive tone is relevant", margin)

    m_total = float(np.sum(model.body_mass))
    n = len(pts)
    A = np.zeros((6, 3 * n))
    for i, p in enumerate(pts):
        r = p - com
        A[0:3, 3 * i:3 * i + 3] = np.eye(3)
        A[3:6, 3 * i:3 * i + 3] = np.array([[0, -r[2], r[1]],
                                            [r[2], 0, -r[0]],
                                            [-r[1], r[0], 0]])
    b = np.concatenate([-m_total * np.array(model.opt.gravity), np.zeros(3)])
    F, *_ = np.linalg.lstsq(A, b, rcond=None)

    # The minimum-norm solution ignores the fact that a foot can only push. Inside the
    # support polygon a non-negative solution is guaranteed to exist, but min-norm does not
    # have to *find* it -- a splayed stance can put a small negative normal on one foot. Ask
    # again with the normals bounded when that happens, and only then, because the bounded
    # solve needs a regulariser to pick among exact solutions and that costs the exact
    # left/right symmetry that makes min-norm such a good asymmetry detector.
    W = m_total * 9.81
    if float(np.min(F[2::3])) < -1e-9 * W:
        from scipy.optimize import lsq_linear
        lam = 1e-8 * W
        lo = np.tile([-np.inf, -np.inf, 0.0], n)
        F = lsq_linear(np.vstack([A, lam * np.eye(3 * n)]),
                       np.concatenate([b, np.zeros(3 * n)]),
                       bounds=(lo, np.full(3 * n, np.inf))).x

    residual = float(np.max(np.abs(A @ F - b)))
    if residual > 1e-6 * max(1.0, W):
        raise UnstablePose(f"no static ground reaction balances this pose (wrench residual "
                           f"{residual:.4g} N)", margin)

    out = []
    for i, bid in enumerate(bids):
        R = data.xmat[bid].reshape(3, 3)
        out.append((bid, F[3 * i:3 * i + 3], R.T @ (pts[i] - data.xpos[bid])))
    return out


def _hold_torque(model, data, frozen):
    """Joint torque needed to hold the current pose, given the frozen ground reaction.

    Two MuJoCo details, both of which produce a wrong answer rather than an error:

    * Contacts must be DISABLED around this. Otherwise the constraint solver simply reduces
      the real contact force by whatever external force is pushing the foot up, so putting
      the reaction back as an applied force nets to nothing -- and, because that cancellation
      is exact, even a self-check against the contact-solved torque passes. The symptom is
      that every perturbation derivative comes out as zero.
    * `qfrc_inverse` is the *total* generalised force the pose requires. It does not have
      applied forces subtracted from it; the documented identity is the opposite, that
      qfrc_inverse equals the sum of all applied forces when forward and inverse agree. So
      setting `xfrc_applied` has no effect on it at all. The reaction has to be projected
      into joint space here and subtracted by hand, which `mj_applyFT` does exactly -- and
      it takes a world point, so the wrench never needs shifting to a COM either.
    """
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)              # kinematics for the application points
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0                          # ... after forward, which writes qacc
    mujoco.mj_inverse(model, data)

    ext = np.zeros(model.nv)
    zero = np.zeros(3)
    for bid, force, r_local in frozen:
        p = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ r_local
        mujoco.mj_applyFT(model, data, force, zero, p, bid, ext)
    return np.array(data.qfrc_inverse) - ext


def measure_buckling(model, data, frozen, eps: float = 0.035):
    """How fast each joint's required holding torque *falls away* as the joint deflects.

    This is the term that the static holding torque cannot see, and it is the one that was
    actually collapsing the rig. A standing quadruped stacks its joints close to the line
    of the ground reaction -- that is good design, it is why an animal can sleep standing
    up -- and a joint on that line needs ~zero torque to hold the pose. Sizing its spring
    from that torque gives it ~zero spring. But the load through it is *compressive*, so
    the moment it deflects, the load line moves off the joint and generates a moment that
    grows with the deflection. That is Euler buckling, and the stifle and shoulder folded
    straight to their limits from it while every well-loaded joint held to within 5 deg.

    Linearising, with `f(q)` the torque needed to hold the joint at `q`:

        I*q_ddot = -k*q - f(q)     =>     stable iff  k > -f'(q*)

    so `max(0, -df/dq)` is the stiffness a joint must beat to stand at all -- independent
    of, and often far larger than, the stiffness it needs to carry its share of the weight.
    """
    import mujoco
    import numpy as np

    q0 = np.array(data.qpos)
    saved_flags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        base = _hold_torque(model, data, frozen)
        out = {}
        for jid in range(model.njnt):
            jt = model.jnt_type[jid]
            if jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                continue
            adr = int(model.jnt_qposadr[jid])
            dof = int(model.jnt_dofadr[jid])
            taus = []
            for s in (+1.0, -1.0):
                data.qpos[:] = q0
                data.qpos[adr] += s * eps
                taus.append(_hold_torque(model, data, frozen)[dof])
            dfdq = (taus[0] - taus[1]) / (2.0 * eps)
            out[jid] = max(0.0, -dfdq)
    finally:
        model.opt.disableflags = saved_flags
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
    return out, base


def measure(creature: Creature) -> list[JointLoad]:
    """Load a throwaway MuJoCo model of `creature` and measure every scalar joint."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)
    place_on_ground(model, data, -SEAT)      # seat the feet so the contact set is found

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    Mdiag = np.diag(_full_mass_matrix(model, data))
    frozen = ground_reaction(model, data)    # solved from statics, not from the solver

    # The holding torque and the buckling gradient now come from the same load, with
    # MuJoCo's contacts switched off in both -- so they are consistent by construction
    # rather than by a check that turned out to be able to pass vacuously.
    buckle, tau = measure_buckling(model, data, frozen)

    # Self-check: under a ground reaction that balances the creature, the *floating base*
    # needs no force. That is what "standing" means. This is the check that would have
    # caught the soft-contact shortfall immediately: the root's row of qfrc_inverse was
    # carrying 114 N of a 206 N dog, in silence, while every joint torque below it looked
    # entirely reasonable.
    root_residual = float(np.max(np.abs(tau[:6])))
    if root_residual > 1e-4 * float(np.sum(model.body_mass)) * 9.81:
        raise RuntimeError(f"the standing pose is not in static equilibrium: the floating "
                           f"base still needs {root_residual:.3g} N to hold it up")

    g = float(np.linalg.norm(model.opt.gravity))
    out: list[JointLoad] = []
    for jid in range(model.njnt):
        jt = model.jnt_type[jid]
        if jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue                        # the floating base has no passive spring
        dof = int(model.jnt_dofadr[jid])
        bid = int(model.jnt_bodyid[jid])
        axis = np.array(data.xaxis[jid])
        anchor = np.array(data.xanchor[jid])

        m_sub = float(model.body_subtreemass[bid])
        com = np.array(data.subtree_com[bid])
        if jt == mujoco.mjtJoint.mjJNT_HINGE:
            r = com - anchor
            # Only the component of the moment arm perpendicular to the axis can generate
            # torque about it; the parallel component is along the hinge and does nothing.
            r_perp = r - float(r @ axis) * axis
            tau_ref = m_sub * g * float(np.linalg.norm(r_perp))
        else:                               # slide: a force, not a torque
            tau_ref = m_sub * g

        out.append(JointLoad(
            name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            dof=dof,
            kind="hinge" if jt == mujoco.mjtJoint.mjJNT_HINGE else "slide",
            tau_static=float(tau[dof]),
            tau_ref=tau_ref,
            k_buckle=float(buckle.get(jid, 0.0)),
            inertia=float(Mdiag[dof]),
        ))
    return out


def steps_per_cycle_floor(creature: Creature) -> int:
    """How many integration steps one oscillation of a passive spring has to span.

    `implicit` and `implicitfast` integrate joint stiffness and damping implicitly -- that
    is what they exist for -- so they stay stable on springs that would detonate under
    explicit Euler. Using one threshold for both would either cry wolf on the implicit
    integrators or miss a real problem on the explicit one.
    """
    return 12 if creature.world.integrator in ("implicit", "implicitfast") else 30


def stiffness_ceiling(creature: Creature, inertia: float) -> float:
    """The stiffest spring this creature's own timestep and integrator can actually follow.

    A stiffness cap is genuinely necessary -- but `max_stiffness 4000` in the rig is the
    same units mistake as every other literal this module deletes, and a worse one for
    being invisible: it binds on the large end of the morph range and nowhere else, so the
    rig quietly under-springs exactly the bodies that needed the most support, and reports
    a tidy 4000 while doing it.

    The real constraint is not a torque-per-radian at all, it is numerical: a spring whose
    natural frequency approaches the sampling rate cannot be integrated. That constraint is
    per-joint, because it runs through the joint's own inertia -- `k = I*(2*pi*f)^2` -- and
    it is derivable from quantities the rig already declares. So derive it. A joint that
    then needs more stiffness than this is telling you the timestep is too large, which is
    a real and reportable fact about the model, not something to paper over.
    """
    f_max = 1.0 / (steps_per_cycle_floor(creature) * creature.world.timestep)
    return max(inertia, 0.0) * (2.0 * math.pi * f_max) ** 2


def size_tone(loads: list[JointLoad], creature: Creature) -> list[JointLoad]:
    """Turn measured loads into stiffness and damping, per the creature's `posture` goals."""
    p = creature.posture
    for L in loads:
        # Three requirements; the joint has to satisfy all of them, so take the largest.
        # They are genuinely different questions -- "hold the weight", "have any tone at
        # all", "do not buckle" -- and for the stifle the third is ~50x the first.
        #
        # The `k_buckle` term in the first two is the part that is easy to get wrong, and it
        # cost a 5x sag overshoot on randomised morphs before it was there. Buckling does
        # not merely threaten collapse, it *consumes stiffness*: the torque the joint has to
        # hold is not the constant `tau_static`, it grows as the joint deflects, at exactly
        # the rate `k_buckle` that was measured for the stability criterion. Equilibrium is
        # therefore where `k*dq = tau_static + k_buckle*dq`, so the stiffness actually
        # available to resist sag is `k - k_buckle`, and the sag goal has to be met by that
        # difference rather than by `k` itself. Sizing from `tau_static/sag` alone left the
        # stifle and hip at 20-30 deg of sag against a 4 deg goal, on a rig that reported
        # every joint as comfortably within budget. When `k_buckle` is zero -- every joint
        # that is not on the load line -- this reduces exactly to the naive form.
        candidates = [
            (L.k_buckle + abs(L.tau_static) / p.sag, "static"),
            (L.k_buckle + p.tone_floor * L.tau_ref / p.sag, "floor"),
            (p.buckle_margin * L.k_buckle, "buckle"),
        ]
        L.stiffness, L.limited_by = max(candidates)
        for cap, why in ((p.max_stiffness, "cap"), (stiffness_ceiling(creature, L.inertia),
                                                    "timestep")):
            if cap is not None and L.stiffness > cap:
                L.stiffness, L.limited_by = cap, why
        # Critical damping for this joint's own effective inertia. Writing a damping
        # literal instead would be doubly wrong under randomisation: it would miss both the
        # inertia change and the stiffness change that the same morph edit caused.
        L.damping = p.damping_ratio * 2.0 * math.sqrt(max(L.stiffness, 0.0) * L.inertia)
    return loads


def _gains(creature: Creature) -> dict[str, tuple]:
    return {j.name: (j.stiffness, j.damping)
            for b in creature.bones for j in b.joints}


def _restore_gains(creature: Creature, gains: dict[str, tuple], by_name: dict) -> None:
    for b in creature.bones:
        for j in b.joints:
            if j.name not in gains:
                continue
            j.stiffness, j.damping = gains[j.name]
            L = by_name.get(j.name)
            if L is not None and j.stiffness is not None:
                L.stiffness, L.damping = j.stiffness, j.damping or 0.0


def peak_stiffness(creature: Creature) -> float:
    return max((j.stiffness or 0.0 for b in creature.bones for j in b.joints), default=0.0)


def relax(creature: Creature, loads: list[JointLoad], owned_k: set[str],
          owned_c: set[str], iters: int = 8, seconds: float | None = None,
          trace_out: list | None = None) -> float:
    """Settle the body under gravity and stiffen whatever actually sagged past the goal.

    `size_tone` predicts the sag from a linearisation about the reference pose, and that
    prediction is systematically optimistic: it said 4 degrees for randomised morphs that
    then settled to 18, and the difference is the whole margin between standing and lying
    down. The reason is structural, not a tuning constant. The linear model asks each joint
    "what if *you* deflect?", holding every other joint fixed, and a limb does not fold that
    way -- hip, stifle and hock give together, the trunk descends, the load redistributes
    between fore and hind feet, and the centre of mass migrates until it leaves the support
    polygon and the animal tips. Every one of those effects is a coupling term.

    The obvious repair -- keep the full `d(tau)/dq` matrix instead of its diagonal, which
    the perturbation loop already computes and throws away -- does not work either, and it
    is worth writing down why so nobody re-derives it. With the ground reaction frozen, a
    rigid translation of the whole creature changes no relative geometry and therefore no
    joint torque, so the six floating-base columns of that matrix are identically zero. The
    matrix is singular by construction. What restores the base is *contact* -- dropping the
    body presses the feet harder -- and freezing the reaction is precisely what removed it.
    Modelling that properly means carrying the closed kinematic loop through the feet.

    So measure it instead. Simulate the settle, read the deflection each joint actually
    reached, and scale its spring by the ratio it missed the goal by. For a linear
    uncoupled joint that ratio is the exact correction, which is why this converges in a
    handful of passes rather than crawling; for the real coupled body it is a very good
    search direction. This costs a few seconds of simulated time per build and buys
    agreement with the only number anyone cares about -- where the body ends up -- rather
    than with a model of it. It is the same argument as the rest of the module, applied one
    level further out: the goal was always the settled pose, and the linearisation was only
    ever a way to guess at it.

    Returns the worst final sag in radians.
    """
    import mujoco
    import numpy as np

    from .validate import SETTLE_SECONDS, withers_height

    seconds = SETTLE_SECONDS if seconds is None else seconds
    p = creature.posture
    goal = max(p.sag, 1e-6)
    by_name = {L.name: L for L in loads}
    worst = 0.0
    trace: list[tuple[float, float]] = trace_out if trace_out is not None else []
    best: tuple[float, dict] = (float("inf"), _gains(creature))

    for _ in range(iters):
        model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
        data = mujoco.MjData(model)
        place_on_ground(model, data)
        q0 = np.array(data.qpos)
        withers = withers_height(model, data)
        z0 = float(data.qpos[2])
        for _ in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError("the model went non-finite while settling -- the passive "
                               "springs are almost certainly too stiff for the timestep")

        # A per-joint budget does not bound the whole-body drop, and the difference is not
        # academic: a limb is a chain, so thirty-one joints each sagging a legal 3.9 deg
        # put the chest on the floor while every single joint reports itself within budget.
        # There is nothing left for the per-joint rule to stiffen, because nothing is over.
        # So the settled *height* is a goal in its own right, and it needs no new constant:
        # the body may sink by the same fraction of its own height that its joints are
        # allowed to rotate in radians. At `sag` = 4 deg that is 7% of withers, which sits
        # just under the 9% the stand test fails at -- the tuner aims inside its own bar.
        sank = max(0.0, (z0 - float(data.qpos[2])) / max(withers, 1e-6))
        whole_body = min(sank / goal, 2.5) if sank > goal else 1.0

        # Keep the best gains seen and stop when stiffening stops paying, because not every
        # collapse is a stiffness problem and the loop cannot tell the difference from the
        # inside. A body whose centre of mass leaves its support polygon topples about its
        # feet no matter how rigid its joints are; chasing that with the multiplier ran one
        # 314 kg draw up to 2.3e6 N*m/rad -- a number with no physical meaning, arrived at
        # by eight rounds of a loop that was never going to converge. Diverging quietly is
        # worse than failing, so record the best and let the stand test report the failure.
        improved = sank < best[0] * 0.98
        if sank < best[0]:
            best = (sank, _gains(creature))
        trace.append((sank, peak_stiffness(creature)))
        if trace and not improved:
            break

        worst, changed = 0.0, False
        for bone in creature.bones:
            for j in bone.joints:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
                if jid < 0 or model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                                          mujoco.mjtJoint.mjJNT_SLIDE):
                    continue
                adr = int(model.jnt_qposadr[jid])
                dq = abs(float(data.qpos[adr] - q0[adr]))
                L = by_name.get(j.name)
                if L is not None:
                    L.sag_final = dq
                worst = max(worst, dq)
                # Only touch joints whose gains this module owns. An authored stiffness is
                # a deliberate statement about the animal and is not up for negotiation
                # just because the body sagged somewhere else.
                if j.name not in owned_k:
                    continue
                # Cap the per-pass correction. A creature that toppled reports enormous
                # deflections, and following them literally would slam every joint to the
                # ceiling in one step and lose the information in the next pass.
                factor = max(whole_body, min(dq / goal, 2.5) if dq > goal else 1.0)
                if factor <= 1.0:
                    continue
                new = j.stiffness * factor
                if p.max_stiffness is not None:
                    new = min(new, p.max_stiffness)
                if L is not None:
                    new = min(new, stiffness_ceiling(creature, L.inertia))
                if new > j.stiffness * 1.001:
                    if L is not None:
                        L.relaxed *= new / j.stiffness
                        L.stiffness = new
                    j.stiffness = new
                    if j.name in owned_c and L is not None:
                        j.damping = L.damping = (p.damping_ratio * 2.0 *
                                                 math.sqrt(max(new, 0.0) * L.inertia))
                    changed = True
        if not changed:
            break
    _restore_gains(creature, best[1], by_name)
    return worst


def apply_posture(creature: Creature) -> list[JointLoad]:
    """Measure `creature` and write the resulting passive gains onto its joints.

    Returns the per-joint measurements so a caller can report them. A no-op (empty list)
    when the rig declares no `posture` block -- a bare skeleton is a legitimate thing to
    ask for, and P0 built one before this existed.
    """
    if creature.posture is None:
        return []
    # Must come first: every torque below is measured through the load path, and a rig that
    # is propping itself up on a self-collision has the wrong load path entirely.
    auto_exclude(creature)
    # Then armature, because it is *part of* the mass matrix that every measurement below
    # reads -- the stiffness ceiling, the damping, and the buckling gradient all run through
    # it. Sizing it afterwards would tune the body to a mass matrix it does not have.
    size_armature(creature)
    loads = size_tone(measure(creature), creature)
    by_name = {L.name: L for L in loads}
    owned_k: set[str] = set()
    owned_c: set[str] = set()
    for bone in creature.bones:
        for j in bone.joints:
            L = by_name.get(j.name)
            if L is None:
                continue
            # An authored value always wins. This is the escape hatch that keeps the
            # measurement a *default*: a rig that knows something the measurement cannot
            # (a joint braced by a ligament with no gravitational load, say) says so, and
            # is not argued with.
            if j.stiffness is None:
                j.stiffness, _ = L.stiffness, owned_k.add(j.name)
            if j.damping is None:
                j.damping, _ = L.damping, owned_c.add(j.name)
    # The prediction above is an initial guess; this makes the settled body agree with it.
    relax(creature, loads, owned_k, owned_c)
    return loads


def build_tuned(path: str, morph: dict[str, float] | None = None,
                which: str | None = None) -> tuple[Creature, list[JointLoad]]:
    """`load()` plus the passive-tone pass -- the normal way to get a simulable creature."""
    from .build import load
    creature = load(path, morph, which)
    return creature, apply_posture(creature)
