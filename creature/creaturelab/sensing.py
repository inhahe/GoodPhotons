"""The proprioceptive interface: what the policy sees, in what frame, how late, how noisy.

This module is the P1 item that is really a P4 prerequisite. The observation vector is not
a training detail you tune later -- it is the contract that decides whether a policy
survives being morphed at all, and getting it wrong costs a full retrain to discover. Two
decisions are enforced here rather than left to the trainer:

**1. Every channel is body-local and normalised by the body's own scale.**
Global state -- world position, absolute orientation, velocity in m/s -- trains fastest and
is the wrong choice. A number like "1.8 m/s forward" means *sprinting* to a terrier and
*ambling* to a wolfhound, so a policy trained on it has memorised one body. The fix is
dynamic similarity: divide every length by the animal's own withers height `L` and every
time by its own pendulum period `T = sqrt(L/g)`. Speed then arrives as a Froude number,
joint angles as a fraction of their own range, contact as a fraction of body weight -- and
the same number means the same thing on a Chihuahua and a Great Dane. That property is
exactly what P4's morph generalisation rests on, which is why it is imposed at the
observation layer instead of hoped for at the policy layer.

Everything here is also *biologically available*: joint angle (spindles), joint rate
(spindle Ia), contact force (cutaneous mechanoreceptors), gravity direction and body rates
(vestibular), efference copy of the last action (corollary discharge). Nothing reads a
world coordinate. Root height above the floor is deliberately ABSENT even though it would
help early training, because no animal has that sensor and the policy would come to depend
on it; the same information is recoverable from limb extension plus contact.

**2. Signals arrive late, by an amount derived from the body.**
Conduction is finite, so real proprioception is 10-40 ms old -- more in a large animal,
and more from a hind limb than a fore. Train on instantaneous truth and you get reflexes no
animal has; the tell is a twitchy, over-corrected gait, which is the motor-control analogue
of a fixed-pivot knee. So each channel is delayed by `central_delay + path_length /
conduction`, where `path_length` is measured *through the built kinematic tree* to the
declared sensing hub. Because it is measured, scaling the creature up makes it move heavier
for free instead of needing a second hand-tuned constant, and the fore/hind asymmetry falls
out of the anatomy rather than being authored.

The motor side is delayed the same way: an action is applied `central_delay + path /
conduction` after it is chosen. Modelling only the afferent half would understate the loop
delay -- which is the thing that actually destabilises fast feedback -- by about half.

**Noise is not scaled by body size, on purpose.** It is declared as a fraction of each
channel's own scale (`angle_noise` of the joint's range, `force_noise` of body weight), and
a receptor's fractional error is already size-independent. Scaling a fraction by size would
be scaling it twice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Creature


# Channel groups, in the order they are laid out in the observation vector. Named because
# the trainer, the tests and any future asymmetric-noise work all need to talk about
# "the vestibular block" without hardcoding a slice.
GROUPS = ("joint_angle", "joint_rate", "efference", "vestibular", "contact",
          "command", "morph")


@dataclass
class ObsSpec:
    """Layout, normalisation and per-channel sensing lag for one built body.

    Built once per model. Everything in it is a plain array, so a vectorised env can hold
    one spec and apply it to N bodies without re-deriving anything per step.
    """
    dim: int
    slices: dict[str, slice]
    names: list[str]

    # --- normalisation constants, all derived from the body itself -----------------
    length: float                  # withers height L, m -- the body's own length unit
    time: float                    # sqrt(L/g), s -- its own pendulum period
    speed: float                   # sqrt(g*L), m/s -- its own Froude-1 speed
    weight: float                  # m*g, N -- its own weight, the contact force unit

    # --- the actuated-joint tables (nu-long, in actuator order) --------------------
    jnt_qposadr: np.ndarray
    jnt_dofadr: np.ndarray
    jnt_lo: np.ndarray
    jnt_hi: np.ndarray
    jnt_limited: np.ndarray

    # --- feet ----------------------------------------------------------------------
    foot_bodies: np.ndarray        # body ids, stable order
    foot_names: list[str]
    geom_foot: np.ndarray          # (ngeom,) -> foot index, or -1. See `gather_contact`.

    # --- fast selectors -------------------------------------------------------------
    # `jnt_qposadr` / `jnt_dofadr` as something numpy can slice cheaply. Reading
    # `data.qvel[array_of_indices]` costs ~1.6 us; reading `data.qvel[a:b]` costs ~0.7 us,
    # and these are read several times per physics chunk per env inside the thread pool,
    # where every microsecond is GIL-held and therefore serialises the whole vec env. The
    # addresses are contiguous whenever the actuated joints are a contiguous run of the
    # dof array, which is the normal case for a rig with one free root; when they are not,
    # these fall back to the index arrays and nothing else changes.
    qpos_sel: object               # slice | np.ndarray
    dof_sel: object                # slice | np.ndarray

    # Precomputed joint-angle normalisation, so the per-step form is one fused
    # multiply-add rather than a subtract, a divide and a subtract.
    ang_scale: np.ndarray          # 2 / (hi - lo)
    ang_bias: np.ndarray           # -2*lo/(hi - lo) - 1

    # The morph vector this body was built from, in [-1, 1]. A constant per body, but it
    # rides in the observation from day one: a conditioning channel added after training
    # is a channel the policy has already learned to ignore, and the fix is a retrain.
    morph_norm: np.ndarray

    # --- sensing lag / noise --------------------------------------------------------
    # Lags are kept as FRACTIONAL control steps and interpolated at read time. Rounding
    # them to whole steps looks harmless and quietly destroys the thing this module is
    # for: at a 50 Hz control rate the dog's fore-limb lag (19 ms) and hind-limb lag
    # (25 ms) both round to 1 step, so every derived-from-anatomy delay collapses to one
    # global constant and the fore/hind asymmetry disappears. A linear interpolation
    # between the two neighbouring samples is a first-order hold on a smooth 50 Hz signal
    # -- cheap, and it keeps the gradient the morph vector actually produces.
    obs_delay: np.ndarray          # per observation element, in control steps (float)
    act_delay: np.ndarray          # per actuator, in control steps (float)
    obs_delay_s: np.ndarray        # ... the same, in seconds, for reporting
    obs_noise: np.ndarray          # per observation element, sigma in NORMALISED units
    max_delay: int                 # ring length driver: ceil of the largest lag

    def group(self, name: str) -> slice:
        return self.slices[name]


def _selector(adr: np.ndarray):
    """A contiguous run of addresses as a `slice`, otherwise the index array unchanged."""
    if adr.size and bool(np.all(np.diff(adr) == 1)):
        return slice(int(adr[0]), int(adr[-1]) + 1)
    return adr


def _path_lengths_to(model, hub_body: int) -> np.ndarray:
    """Distance from every body to `hub_body`, measured along the kinematic tree.

    Straight-line distance would be wrong and would also be *unstable*: a hind paw folded
    up under the belly is close to the head in space and far from it along the animal.
    Nerves run along the limb, so the tree path is the physical one -- and it changes with
    the morph vector automatically, since every segment offset came from a morph parameter.
    """
    n = model.nbody
    seg = np.linalg.norm(np.asarray(model.body_pos), axis=1)      # to each body's parent
    parent = np.asarray(model.body_parentid)

    # Depth-to-root distance for every body, parents-first (MuJoCo guarantees that order).
    to_root = np.zeros(n)
    for b in range(1, n):
        to_root[b] = to_root[parent[b]] + seg[b]

    # Path length between two bodies is the sum of their depths minus twice their lowest
    # common ancestor's -- the standard tree metric.
    def ancestors(b: int) -> list[int]:
        out = []
        while b != 0:
            out.append(b)
            b = int(parent[b])
        out.append(0)
        return out

    hub_chain = ancestors(hub_body)
    hub_set = {b: i for i, b in enumerate(hub_chain)}
    out = np.zeros(n)
    for b in range(n):
        c = b
        while c not in hub_set:
            c = int(parent[c])
        out[b] = to_root[b] + to_root[hub_body] - 2.0 * to_root[c]
    return out


def find_feet(model, data, creature: Creature) -> list[str]:
    """Which bodies are feet, MEASURED from the settled standing pose.

    Not authored, for the same reason armature and passive tone are not authored: a rig
    that declares its feet is a rig that lies the moment a morph makes a different part of
    it take the load. Standing the creature up and reading which bodies the floor pushes
    back on is the definition rather than a proxy for it.

    The set is resolved ONCE, on the reference body, and then held fixed by name across
    every randomised morph -- otherwise the observation vector would change length between
    episodes, which no policy can consume. A morph whose paw genuinely stops touching the
    ground therefore reports zero contact on that channel, which is the truth.
    """
    import mujoco

    from .emit_mjcf import place_on_ground
    from .validate import SETTLE_SECONDS

    mujoco.mj_resetData(model, data)
    place_on_ground(model, data)
    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)

    hits: set[int] = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 == 0) == (b2 == 0):              # exactly one side must be the floor
            continue
        hits.add(b2 if b1 == 0 else b1)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in sorted(hits)]
    if not names:
        raise RuntimeError(
            "no ground contacts in the settled reference pose, so there is nothing to "
            "call a foot -- the rig is not standing on anything (run `ftcl_build --check`)")
    return names


def build_spec(model, data, creature: Creature, *, control_dt: float,
               foot_names: list[str] | None = None) -> ObsSpec:
    """Derive the whole observation contract for one built body."""
    import mujoco

    from .validate import withers_height

    s = creature.sensing
    g = float(abs(creature.world.gravity[2])) or 9.81

    mujoco.mj_resetData(model, data)
    from .emit_mjcf import place_on_ground
    place_on_ground(model, data)
    L = float(withers_height(model, data))
    T = float(np.sqrt(L / g))
    V = float(np.sqrt(g * L))
    mass = float(np.sum(model.body_mass))
    W = mass * g

    if foot_names is None:
        foot_names = find_feet(model, data, creature)
    foot_bodies = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names],
        dtype=np.int32)
    if np.any(foot_bodies < 0):
        missing = [n for n, b in zip(foot_names, foot_bodies) if b < 0]
        raise KeyError(f"foot bodies not in this model: {missing}")

    # ---- actuated joints, in ACTUATOR order (so obs and action index the same joint) ----
    nu = model.nu
    jids = np.array([int(model.actuator_trnid[a, 0]) for a in range(nu)], dtype=np.int32)
    qposadr = np.array([int(model.jnt_qposadr[j]) for j in jids], dtype=np.int32)
    dofadr = np.array([int(model.jnt_dofadr[j]) for j in jids], dtype=np.int32)
    limited = np.array([bool(model.jnt_limited[j]) for j in jids])
    rng = np.array([model.jnt_range[j] for j in jids], dtype=float)
    lo, hi = rng[:, 0].copy(), rng[:, 1].copy()
    # An unlimited joint has no range to normalise against; give it a nominal +-pi so the
    # channel still lands in [-1, 1] instead of silently dominating the input scale.
    lo[~limited], hi[~limited] = -np.pi, np.pi

    # Which geoms belong to a foot, as an O(1) lookup keyed by geom id. The per-step
    # alternative -- searching `foot_bodies` for each contact -- is a numpy call per
    # contact inside the physics threads, i.e. exactly where it costs the most.
    geom_foot = np.full(model.ngeom, -1, dtype=np.int32)
    for k, b in enumerate(foot_bodies):
        geom_foot[model.geom_bodyid == b] = k

    # ---- layout -----------------------------------------------------------------------
    nfoot = len(foot_bodies)
    nmorph = len(creature.params)
    sizes = {"joint_angle": nu, "joint_rate": nu, "efference": nu,
             "vestibular": 9, "contact": nfoot, "command": 3, "morph": nmorph}
    slices, names, o = {}, [], 0
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in jids]
    label = {
        "joint_angle": [f"q/{n}" for n in jnt_names],
        "joint_rate": [f"qd/{n}" for n in jnt_names],
        "efference": [f"a-/{n}" for n in jnt_names],
        "vestibular": ["grav_x", "grav_y", "grav_z", "omega_x", "omega_y", "omega_z",
                       "vel_x", "vel_y", "vel_z"],
        "contact": [f"contact/{n}" for n in foot_names],
        "command": ["cmd_vx", "cmd_vy", "cmd_yaw"],
        "morph": [f"morph/{p.name}" for p in creature.params],
    }
    for k in GROUPS:
        slices[k] = slice(o, o + sizes[k])
        names.extend(label[k])
        o += sizes[k]
    dim = o

    # ---- conduction lag ---------------------------------------------------------------
    hub_name = s.hub or creature.root
    hub = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hub_name)
    if hub < 0:
        raise KeyError(f"sensing hub body '{hub_name}' is not in the model")
    path = _path_lengths_to(model, hub)

    jnt_body = np.array([int(model.jnt_bodyid[j]) for j in jids], dtype=np.int32)
    jnt_lag = s.central_delay + path[jnt_body] / s.conduction
    foot_lag = s.central_delay + path[foot_bodies] / s.conduction
    # The vestibular organ sits in the head, i.e. at the hub, so it carries the central
    # latency alone -- which is why a falling animal rights itself faster than it can
    # correct a foot.
    vest_lag = np.full(9, s.central_delay)

    obs_lag_s = np.zeros(dim)
    obs_lag_s[slices["joint_angle"]] = jnt_lag
    obs_lag_s[slices["joint_rate"]] = jnt_lag
    obs_lag_s[slices["contact"]] = foot_lag
    obs_lag_s[slices["vestibular"]] = vest_lag
    # efference / command / morph are internal signals the animal issues or *is*; they are
    # not measured through a nerve, so they are not delayed.
    obs_delay = obs_lag_s / control_dt
    act_delay = (s.central_delay + path[jnt_body] / s.conduction) / control_dt

    noise = np.zeros(dim)
    noise[slices["joint_angle"]] = s.angle_noise * 2.0    # channel spans [-1, 1]
    noise[slices["joint_rate"]] = s.rate_noise
    noise[slices["vestibular"]] = s.vestibular_noise
    noise[slices["contact"]] = s.force_noise

    return ObsSpec(
        dim=dim, slices=slices, names=names,
        length=L, time=T, speed=V, weight=W,
        jnt_qposadr=qposadr, jnt_dofadr=dofadr, jnt_lo=lo, jnt_hi=hi, jnt_limited=limited,
        foot_bodies=foot_bodies, foot_names=list(foot_names), geom_foot=geom_foot,
        qpos_sel=_selector(qposadr), dof_sel=_selector(dofadr),
        ang_scale=2.0 / (hi - lo), ang_bias=-2.0 * lo / (hi - lo) - 1.0,
        morph_norm=np.array(creature.morph_vector_normalized(), dtype=float),
        obs_delay=obs_delay, act_delay=act_delay, obs_delay_s=obs_lag_s, obs_noise=noise,
        max_delay=int(np.ceil(max(obs_delay.max(initial=0.0),
                                  act_delay.max(initial=0.0)))),
    )


# ============================================================ the per-step reading, batched
#
# Everything below works on N bodies at once. That is a performance decision with a hard
# measurement behind it, and it is worth stating because the per-env form was much easier
# to read.
#
# MuJoCo's `mj_step` releases the GIL, so N creatures really do step in parallel on a
# thread pool -- but only for as long as the threads stay inside C. Measured on this rig
# (31 actuators, 135 channels, 12 cores): the physics is ~343 us per control step and scales
# to 3.9x across 16 threads, while the surrounding numpy was ~500 us per env and did not
# scale at all, because it holds the GIL. Interleaving even 58 us of it between the physics
# steps dropped a physics-only loop from 7900 to 1800 env-steps/s -- a 4.4x loss to a 15 %
# serial fraction, far worse than Amdahl predicts, because what costs is the GIL hand-off,
# not the work. Lowering `sys.setswitchinterval` does not help, and worker processes reach
# only ~2900 env-steps/s because the Windows pipe barrier eats what the GIL gave back.
#
# So the split is: the threads touch `MjData` and nothing else, copying raw state into
# per-env rows of shared (N, ...) buffers; every normalisation, delay, noise, reward and
# termination computation then happens once on the whole batch, outside the pool. The numpy
# call count per control step becomes O(1) in N instead of O(N). Net effect on the vec env:
# 762 -> 3139 env-steps/s, i.e. a 2e7-step PPO run goes from ~7 hours to ~1.8. What is left
# is genuinely close to the floor -- 318 us per env-step against 343 us of single-core
# physics -- so the next real lever is the physics itself, not this layer.

@dataclass
class RawState:
    """Un-normalised physical state for N bodies, as gathered straight out of `MjData`.

    The only thing the physics threads write, and the only thing the batched pass reads.
    Keeping it a plain struct of (N, ...) arrays is what makes the boundary between "in the
    thread, GIL-held, expensive" and "outside, batched, cheap" a boundary you can see.
    """
    qpos_j: np.ndarray             # (N, nu)   actuated joint angles, rad
    qvel_j: np.ndarray             # (N, nu)   actuated joint rates, rad/s
    xmat: np.ndarray               # (N, 3, 3) root body orientation
    qvel_free: np.ndarray          # (N, 6)    free-joint velocity: 0:3 world lin, 3:6 body ang
    root_z: np.ndarray             # (N,)      root height, m -- termination only, never sensed
    contact: np.ndarray            # (N, nfoot) normal force on each foot, N
    work_tau: np.ndarray           # (N, C+1, nu) actuator generalised force, per energy sample
    work_vel: np.ndarray           # (N, C+1, nu) matching joint rate
    sane: np.ndarray               # (N,) bool -- did the integrator survive. See `gather_state`
    warn: np.ndarray               # (N,) int  -- cumulative MuJoCo bad-value warning count


def make_raw(spec: ObsSpec, n: int, energy_samples: int) -> RawState:
    nu = len(spec.jnt_lo)
    c = energy_samples + 1
    return RawState(
        qpos_j=np.zeros((n, nu)), qvel_j=np.zeros((n, nu)),
        xmat=np.zeros((n, 3, 3)), qvel_free=np.zeros((n, 6)), root_z=np.zeros(n),
        contact=np.zeros((n, len(spec.foot_bodies))),
        work_tau=np.zeros((n, c, nu)), work_vel=np.zeros((n, c, nu)),
        sane=np.ones(n, dtype=bool), warn=np.zeros(n, dtype=np.int64))


def gather_contact(model, data, spec: ObsSpec, out: np.ndarray) -> None:
    """Sum the normal force on each foot into `out`. Called inside the physics thread.

    Any contact involving a foot geom counts, whatever the foot is touching. The narrower
    "only contacts against the world body" rule reads as more careful and is not: a paw
    resting on another paw is a real load on a real mechanoreceptor, and P3's terrain may
    not be a world geom at all, so the narrow rule would silently blank the channel on the
    first rig it was not written for.
    """
    import mujoco

    out[:] = 0.0
    ncon = data.ncon
    if not ncon:
        return
    g = data.contact.geom[:ncon]               # (ncon, 2), one attribute access
    f0 = spec.geom_foot[g[:, 0]]
    f1 = spec.geom_foot[g[:, 1]]
    hits = np.nonzero((f0 >= 0) | (f1 >= 0))[0]
    if not hits.size:
        return
    buf = np.empty(6)
    for i in hits:
        mujoco.mj_contactForce(model, data, int(i), buf)
        fn = abs(buf[0])                       # normal component, contact frame
        if f0[i] >= 0:
            out[f0[i]] += fn
        if f1[i] >= 0:
            out[f1[i]] += fn


#: `mjWARN_BADQPOS`, `mjWARN_BADQVEL`, `mjWARN_BADQACC` -- contiguous in the enum, so this
#: is a slice rather than a fancy index, which matters because it is read inside the threads.
_BAD_WARN = slice(3, 6)


def gather_state(model, data, spec: ObsSpec, raw: RawState, i: int) -> None:
    """Copy env `i`'s physical state out of `MjData` into row `i` of the batch buffers.

    Also decides whether the step is usable at all, and that check needs both halves.

    Testing `isfinite` alone does not work, and finding out why is the reason this comment
    exists: when MuJoCo detects a bad qpos/qvel/qacc it warns and then calls `mj_resetData`
    itself, so by the time the step returns the state is finite again -- it is just no longer
    the state of the animal that was walking. It is the default pose, at the origin, at rest.
    A NaN reaching the rollout buffer at least crashes the optimiser and points a traceback
    somewhere; a silent teleport mid-episode does not, and instead teaches the value function
    that a particular state transitions to a fresh standing pose for free. So the actual
    detector is MuJoCo's own warning counter, and the finiteness test only remains as the
    belt-and-braces case where a NaN appears without tripping a warning at all.
    """
    raw.qpos_j[i] = data.qpos[spec.qpos_sel]
    raw.qvel_j[i] = data.qvel[spec.dof_sel]
    raw.xmat[i] = data.xmat[1].reshape(3, 3)   # body 1 = the floating root
    raw.qvel_free[i] = data.qvel[0:6]
    raw.root_z[i] = data.qpos[2]
    gather_contact(model, data, spec, raw.contact[i])
    warn = int(data.warning.number[_BAD_WARN].sum())
    # Checked on the full state, not just the copied subset: an unactuated joint can be the
    # one that blew up, and a NaN anywhere propagates to everything next step.
    raw.sane[i] = (warn == raw.warn[i]
                   and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()))
    raw.warn[i] = warn


#: Saturation limit for every rate-like channel, in the body's own Froude units.
#:
#: Real afferents saturate, and this is not a safety net bolted on after the fact -- it is
#: part of what the sensor *is*. A muscle spindle's Ia firing rate flattens out long before
#: the joint does, and a semicircular canal stops reporting past roughly 6 rad/s. Peak joint
#: rates in dog locomotion reach ~15-20 rad/s, which for canis (T = 0.26 s) is 4-5 in these
#: units, so 8 clears the physiological range with headroom and still bounds the input.
#:
#: Without it the channel is unbounded in practice, not merely in theory: 200 control steps
#: of +-0.3 random actuation drive `qd/tail2_pitch` to 49.6, because the last tail segment is
#: light, long and barely damped. That is honest physics, not an exploded integrator -- which
#: is exactly what makes it dangerous. One channel at eight times the scale of every other is
#: the channel the first layer's weights get organised around, and the failure is silent: the
#: policy trains fine, and quietly treats a whipping tail as its dominant input.
RATE_CLIP = 8.0


def assemble(spec: ObsSpec, raw: RawState, prev_action: np.ndarray,
             command: np.ndarray, morph: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Normalise a whole batch of raw state into observations. No delay, no noise yet."""
    s = spec.slices
    out[:, s["joint_angle"]] = np.clip(raw.qpos_j * spec.ang_scale + spec.ang_bias,
                                       -2.0, 2.0)
    np.clip(raw.qvel_j * spec.time, -RATE_CLIP, RATE_CLIP, out=out[:, s["joint_rate"]])
    out[:, s["efference"]] = prev_action

    v = s["vestibular"].start
    R = raw.xmat
    # `R[:, 2, :]` is world +z expressed in each root's own frame; gravity is its negation
    # and the sign is a wash for the policy, but writing +up keeps it readable next to
    # `trunk_tilt`, which reads the same row.
    out[:, v + 0:v + 3] = R[:, 2, :]
    # MuJoCo free-joint velocity: qvel[0:3] is linear in the WORLD frame, qvel[3:6] angular
    # in the BODY frame. So the angular part is already local; the linear part needs R.T.
    np.clip(raw.qvel_free[:, 3:6] * spec.time, -RATE_CLIP, RATE_CLIP,
            out=out[:, v + 3:v + 6])
    np.einsum("nji,nj->ni", R, raw.qvel_free[:, 0:3], out=out[:, v + 6:v + 9])
    out[:, v + 6:v + 9] /= spec.speed
    np.clip(out[:, v + 6:v + 9], -RATE_CLIP, RATE_CLIP, out=out[:, v + 6:v + 9])

    np.minimum(raw.contact / spec.weight, 2.0, out=out[:, s["contact"]])
    out[:, s["command"]] = command
    out[:, s["morph"]] = morph
    return out


def actuator_work(spec: ObsSpec, raw: RawState, chunk_dt: float) -> np.ndarray:
    """Mechanical work each body's actuators delivered over the last control step, J.

    Trapezoid-integrated over `C + 1` samples of |tau . qdot| taken inside the frame skip.
    Sampling every *physics* step is exact, and is what this did first; each sample also
    costs a GIL hand-off in the middle of the physics, so the count is a throughput knob.
    Both sides of that trade were measured rather than assumed -- accuracy on 2400 control
    steps of smoothed random actuation against the every-physics-step sum, throughput at 128
    envs on 12 threads:

        samples/step   mean |err|   p95 |err|   bias on episode total   env-steps/s
                   1       27 %        50 %              -22 %              4315
                   2       12 %        22 %               -9 %              3862
                   5        3 %         8 %               -1.2 %            3139
                  10        2 %         6 %                0 % (exact)      2433

    One sample is not acceptable at any price: the error is a systematic *under*-report,
    because |power| is spiky within a control step, so the energy penalty would be quietly
    disabled exactly when the animal is flailing and the penalty is most needed. Five buys
    an 18x accuracy improvement over one for 27 % of the throughput, and a 1.2 % bias on a
    term weighted 0.02 is not a term anyone can distinguish from exact -- so five is the
    default. Note also that the naive "sample once at the end and multiply by the step"
    estimator is no better than the 1-sample trapezoid (28 % vs 27 %), so the trapezoid is
    free accuracy and there is never a reason to use the rectangle.
    """
    p = np.abs(raw.work_tau * raw.work_vel).sum(axis=2)         # (N, C+1)
    return 0.5 * (p[:, :-1] + p[:, 1:]).sum(axis=1) * chunk_dt


class BatchProprioception:
    """Delay lines for N bodies at once: one ring buffer, indexed per channel AND per env.

    The observation buffer is written every control step and read per-element at that
    element's own lag, so a hind-paw spindle and a vestibular canal are genuinely different
    ages within the same input vector -- which is the point, and is not expressible as a
    single "observation delay" hyperparameter.

    All N envs share one `head`, because a vec env steps them in lockstep. That is what lets
    the whole read be two fancy-index gathers on a (N, hist, dim) array instead of N
    separate ones.
    """

    def __init__(self, spec: ObsSpec, n_env: int, rng: np.random.Generator):
        self.spec = spec
        self.n_env = n_env
        self.rng = rng
        self._obs_lo = np.floor(spec.obs_delay).astype(np.int32)
        self._act_lo = np.floor(spec.act_delay).astype(np.int32)
        self._obs_w = spec.obs_delay - self._obs_lo
        self._act_w = spec.act_delay - self._act_lo
        # The lerp reads `floor(lag)` and one step further back, so the history must hold
        # max(floor(lag)) + 2 entries. Sizing it from ceil(lag) instead is off by one
        # whenever a lag lands exactly on a step boundary; the stale sample it then reads is
        # multiplied by a zero weight and does no harm, which is precisely why that bug
        # would survive every test.
        self.hist = int(max(self._obs_lo.max(initial=0),
                            self._act_lo.max(initial=0))) + 2
        self.obs_buf = np.zeros((n_env, self.hist, spec.dim))
        self.act_buf = np.zeros((n_env, self.hist, len(spec.act_delay)))
        self.head = 0
        self._obs_chan = np.arange(spec.dim)
        self._act_chan = np.arange(len(spec.act_delay))
        self._noise_idx = np.nonzero(spec.obs_noise > 0)[0]
        self._noise_sig = spec.obs_noise[self._noise_idx]

    def reset(self, idx, obs0: np.ndarray, action0: np.ndarray) -> None:
        """Fill the whole history of envs `idx` with a freshly-reset state.

        Zero-filling instead would hand the policy a first observation claiming the animal
        was inverted and airborne a moment ago, and the resulting flail is easy to mistake
        for a physics problem.
        """
        self.obs_buf[idx] = obs0[:, None, :]
        self.act_buf[idx] = action0[:, None, :]

    def sense(self, raw_obs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Record fresh readings and return what each policy actually gets to see."""
        self.obs_buf[:, self.head] = raw_obs
        lo = (self.head - self._obs_lo) % self.hist
        hi = (lo - 1) % self.hist                     # one step further into the past
        a = self.obs_buf[:, lo, self._obs_chan]
        b = self.obs_buf[:, hi, self._obs_chan]
        o = np.add(a, (b - a) * self._obs_w, out=out)
        if self._noise_idx.size:
            o[:, self._noise_idx] += (
                self.rng.standard_normal((self.n_env, self._noise_idx.size))
                * self._noise_sig)
        return o

    def actuate(self, action: np.ndarray) -> np.ndarray:
        """Record fresh commands and return what actually reaches the muscles now."""
        self.act_buf[:, self.head] = action
        lo = (self.head - self._act_lo) % self.hist
        hi = (lo - 1) % self.hist
        a = self.act_buf[:, lo, self._act_chan]
        b = self.act_buf[:, hi, self._act_chan]
        return a + (b - a) * self._act_w

    def advance(self) -> None:
        self.head = (self.head + 1) % self.hist


def describe(spec: ObsSpec) -> str:
    """One-screen summary, for the training log and for `tools/obs_report.py`."""
    lines = [f"observation: {spec.dim} channels"]
    for k in GROUPS:
        sl = spec.slices[k]
        d = spec.obs_delay_s[sl] * 1000.0
        lag = ""
        if sl.stop > sl.start and d.max() > 0:
            lag = (f"   lag {d.min():.1f}-{d.max():.1f} ms"
                   if d.max() - d.min() > 0.05 else f"   lag {d.max():.1f} ms")
        lines.append(f"  {k:12s} [{sl.start:3d}:{sl.stop:3d}]  "
                     f"{sl.stop - sl.start:3d}{lag}")
    lines.append(f"  scales: L={spec.length:.3f} m  T={spec.time:.3f} s  "
                 f"V={spec.speed:.3f} m/s  W={spec.weight:.1f} N")
    lines.append(f"  feet: {', '.join(spec.foot_names)}")
    return "\n".join(lines)
