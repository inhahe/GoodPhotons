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
        foot_bodies=foot_bodies, foot_names=list(foot_names),
        obs_delay=obs_delay, act_delay=act_delay, obs_delay_s=obs_lag_s, obs_noise=noise,
        max_delay=int(np.ceil(max(obs_delay.max(initial=0.0),
                                  act_delay.max(initial=0.0)))),
    )


# ------------------------------------------------------------------ the per-step reading
def raw_observation(model, data, spec: ObsSpec, prev_action: np.ndarray,
                    command: np.ndarray, morph: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
    """Sample every channel from the current state, already normalised. No delay, no noise.

    Called once per control step per env, so it avoids allocation and keeps everything in
    numpy; the delay/noise pass is applied afterwards by `Proprioception`.
    """
    import mujoco

    o = np.empty(spec.dim) if out is None else out

    q = data.qpos[spec.jnt_qposadr]
    o[spec.slices["joint_angle"]] = np.clip(
        2.0 * (q - spec.jnt_lo) / (spec.jnt_hi - spec.jnt_lo) - 1.0, -2.0, 2.0)
    o[spec.slices["joint_rate"]] = data.qvel[spec.jnt_dofadr] * spec.time
    o[spec.slices["efference"]] = prev_action

    R = data.xmat[1].reshape(3, 3)                      # root body orientation
    v = spec.slices["vestibular"]
    o[v.start + 0:v.start + 3] = R[2, :]                # world -z ... see note below
    # `R[2, :]` is world +z expressed in the root's own frame; gravity is its negation, and
    # the sign is a wash for the policy, but writing +up keeps it readable next to
    # `trunk_tilt`, which reads the same row.
    # MuJoCo free-joint velocity: qvel[0:3] linear in WORLD, qvel[3:6] angular in the
    # BODY frame. So the angular part is already local; the linear part must be rotated.
    o[v.start + 3:v.start + 6] = data.qvel[3:6] * spec.time
    o[v.start + 6:v.start + 9] = (R.T @ data.qvel[0:3]) / spec.speed

    c = np.zeros(len(spec.foot_bodies))
    if data.ncon:
        buf = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if (b1 == 0) == (b2 == 0):
                continue
            b = b2 if b1 == 0 else b1
            hit = np.nonzero(spec.foot_bodies == b)[0]
            if hit.size:
                mujoco.mj_contactForce(model, data, i, buf)
                c[hit[0]] += abs(buf[0])           # normal component, contact frame
    o[spec.slices["contact"]] = np.minimum(c / spec.weight, 2.0)

    o[spec.slices["command"]] = command
    o[spec.slices["morph"]] = morph
    return o


class Proprioception:
    """Per-env delay lines for the sensed observation and the issued action.

    One ring buffer each. The observation buffer is written every control step and read
    per-element at that element's own lag, so a hind-paw spindle and a vestibular canal are
    genuinely different ages within the same input vector -- which is the point, and is not
    expressible as a single "observation delay" hyperparameter.
    """

    def __init__(self, spec: ObsSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        n = spec.max_delay + 1
        self.obs_buf = np.zeros((n, spec.dim))
        self.act_buf = np.zeros((n, len(spec.act_delay)))
        self.head = 0
        self.n = n
        self._obs_take = np.arange(spec.dim)
        self._act_take = np.arange(len(spec.act_delay))
        # Precomputed halves of the fractional-delay lerp; the per-step work is then two
        # gathers and a fused multiply-add rather than any index arithmetic.
        self._obs_lo = np.floor(spec.obs_delay).astype(np.int32)
        self._obs_w = (spec.obs_delay - self._obs_lo)
        self._act_lo = np.floor(spec.act_delay).astype(np.int32)
        self._act_w = (spec.act_delay - self._act_lo)
        self._noise_idx = np.nonzero(spec.obs_noise > 0)[0]
        self._noise_sig = spec.obs_noise[self._noise_idx]

    def reset(self, obs0: np.ndarray, action0: np.ndarray) -> None:
        """Fill the whole history with the reset state.

        Zero-filling instead would hand the policy a first observation claiming the animal
        was inverted and airborne a moment ago, and the resulting flail is easy to mistake
        for a physics problem.
        """
        self.obs_buf[:] = obs0
        self.act_buf[:] = action0
        self.head = 0

    def sense(self, raw: np.ndarray) -> np.ndarray:
        """Record a fresh reading and return what the policy actually gets to see."""
        self.obs_buf[self.head] = raw
        lo = (self.head - self._obs_lo) % self.n
        hi = (lo - 1) % self.n                       # one step further into the past
        a = self.obs_buf[lo, self._obs_take]
        b = self.obs_buf[hi, self._obs_take]
        out = a + (b - a) * self._obs_w
        if self._noise_idx.size:
            out[self._noise_idx] += (
                self.rng.standard_normal(self._noise_idx.size) * self._noise_sig)
        return out

    def actuate(self, action: np.ndarray) -> np.ndarray:
        """Record a fresh command and return what actually reaches the muscles now."""
        self.act_buf[self.head] = action
        lo = (self.head - self._act_lo) % self.n
        hi = (lo - 1) % self.n
        a = self.act_buf[lo, self._act_take]
        b = self.act_buf[hi, self._act_take]
        return a + (b - a) * self._act_w

    def advance(self) -> None:
        self.head = (self.head + 1) % self.n


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
