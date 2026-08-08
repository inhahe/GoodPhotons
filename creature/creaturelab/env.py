"""The locomotion task: a Gym-style environment over a generated creature.

P1's job is to smoke-test the whole loop -- rig -> MJCF -> physics -> observation ->
policy -> torque -> reward -> PPO -- on a body whose statics are already trusted by P0's
acceptance bar. The gait it produces will look bad, and that is the correct outcome; what
is being tested is the plumbing.

Two things here are decided rather than tuned, because they outlive P1:

**The reward is dimensionless.** Every term is normalised by the body's own scales
(`sensing.ObsSpec`), so the same weights mean the same trade-off on any body in the morph
range. A reward of "0.5 * forward velocity in m/s minus 0.001 * torque in N*m" is silently
a *different objective* for every morph -- a big animal gets more reward for the same
effort and a small one is crushed by the torque penalty -- and re-tuning weights per body
is exactly the thing the whole parameterisation exists to avoid. So speed error is in
Froude numbers, energy is mechanical power as a fraction of `weight * Froude speed` (a cost
of transport), and the action penalties are on a control signal that is already in [-1, 1].

**Gait is not commanded.** The command channel carries a desired velocity, not a gait, an
phase, or a footfall pattern -- design.md's standing constraint. Walk/trot/gallop are
expected to fall out of the speed command plus the energy penalty, and if they do not, the
fix is the energy model, not a new command channel.

`step()` returns the 5-tuple `(obs, reward, terminated, truncated, info)`. `terminated` and
`truncated` are kept apart because PPO must bootstrap the value function through a
time-limit truncation and must not through a fall; collapsing them into one `done` flag
teaches the policy that the world ends after 20 seconds, which shortens every gait it
learns.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import sensing
from .model import Creature


@dataclass
class EnvConfig:
    rig: str = "rigs/canis.ftcl"
    control_hz: float = 50.0
    episode_seconds: float = 20.0

    # --- command distribution, in the body's OWN units --------------------------------
    # Froude number v / sqrt(g*L): ~0.3 is a walk, ~0.7 a trot, >1 a gallop for a
    # quadruped, on any body. Expressing the curriculum this way is what lets the same
    # numbers stay meaningful when P4 randomises the animal underneath it.
    speed_range: tuple[float, float] = (0.0, 0.8)
    lateral_range: tuple[float, float] = (-0.15, 0.15)
    yaw_range: tuple[float, float] = (-0.4, 0.4)      # rad per pendulum period
    stand_fraction: float = 0.1                        # episodes commanded to hold still

    # --- reward weights (all terms dimensionless) --------------------------------------
    w_speed: float = 1.0
    w_yaw: float = 0.3
    w_energy: float = 0.02
    w_action_rate: float = 0.05
    w_torque: float = 0.005
    speed_tol: float = 0.25            # Froude units, the tracking kernel's width
    yaw_tol: float = 0.4

    # --- termination --------------------------------------------------------------------
    fall_tilt: float = 50.0            # deg between trunk up-axis and world up
    min_height: float = 0.45           # root z as a fraction of reference withers height

    # --- reset randomisation --------------------------------------------------------------
    init_joint_noise: float = 0.05     # fraction of each joint's range
    init_rate_noise: float = 0.05      # fraction of the body's own rate scale

    # --- integration accounting ------------------------------------------------------------
    # How many times per control step the actuator power is sampled for the energy penalty.
    # Rounded down to a divisor of the frame skip. This is a throughput/accuracy trade with
    # a measured table behind it -- see `sensing.actuator_work`. 5 costs ~1% bias on the
    # episode energy total and keeps the vec env parallel; 1 costs 22% and would quietly
    # switch the energy penalty off exactly when the animal flails.
    energy_samples: int = 5

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz


@dataclass
class Body:
    """One built creature plus everything derived from it that never changes again."""
    creature: Creature
    model: object                      # mujoco.MjModel
    spec: sensing.ObsSpec
    frame_skip: int
    withers: float


def build_body(cfg: EnvConfig, morph: dict[str, float] | None = None,
               foot_names: list[str] | None = None) -> Body:
    """Compile a rig to a simulatable body, tone included.

    `build_tuned`, not `load`: an untuned skeleton has no passive tone, folds up the
    instant gravity touches it, and would hand the policy the job of not collapsing --
    which costs training budget forever. The tone pass is part of what "this body" means.
    """
    import mujoco

    from .emit_mjcf import to_mjcf
    from .tune import build_tuned

    creature, _ = build_tuned(cfg.rig, morph)
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)

    skip = int(round(cfg.control_dt / model.opt.timestep))
    if skip < 1:
        raise ValueError(f"control_hz {cfg.control_hz} is faster than the physics "
                         f"timestep {model.opt.timestep}")
    actual = skip * model.opt.timestep
    spec = sensing.build_spec(model, data, creature, control_dt=actual,
                              foot_names=foot_names)
    return Body(creature=creature, model=model, spec=spec, frame_skip=skip,
                withers=spec.length)


def _energy_chunks(frame_skip: int, requested: int) -> int:
    """Largest divisor of `frame_skip` that is at most `requested`.

    The frame skip has to divide evenly into chunks or the last chunk integrates the wrong
    interval, so the requested sample count is a ceiling rather than a promise.
    """
    return max(d for d in range(1, frame_skip + 1)
               if frame_skip % d == 0 and d <= max(1, requested))


class VecCreatureEnv:
    """N creatures stepped in lockstep, with auto-reset. The single-env class wraps this.

    Threaded rather than multiprocess, and batched rather than per-env -- see the long note
    above `RawState` in `sensing.py` for the measurements that forced this shape. In one
    sentence: the thread pool may only touch `MjData`, because everything else holds the GIL
    and the GIL is what caps this loop.

    Auto-reset returns the FIRST observation of the next episode in `obs`, and stashes the
    terminal observation in `info["final_obs"]`. PPO needs the terminal one to bootstrap a
    truncated episode's value, and silently substituting the reset observation there is a
    classic quiet bug: the value target for the last step of every episode becomes the value
    of a completely different state.
    """

    def __init__(self, bodies: list[Body], cfg: EnvConfig, seed: int = 0,
                 workers: int | None = None, auto_reset: bool = True):
        import mujoco

        self.cfg = cfg
        self.bodies = bodies
        self.n = len(bodies)
        self.auto_reset = auto_reset
        self.spec = bodies[0].spec
        self.obs_dim = self.spec.dim
        self.act_dim = len(self.spec.act_delay)
        for b in bodies:
            if b.spec.dim != self.obs_dim or len(b.spec.act_delay) != self.act_dim:
                raise ValueError("all bodies in a vector env must share an observation and "
                                 "action layout (same rig, same foot set)")
            if b.frame_skip != bodies[0].frame_skip:
                raise ValueError("all bodies in a vector env must share a frame skip")

        self.frame_skip = bodies[0].frame_skip
        self.chunks = _energy_chunks(self.frame_skip, cfg.energy_samples)
        self.chunk_steps = self.frame_skip // self.chunks
        self.chunk_dt = self.chunk_steps * bodies[0].model.opt.timestep
        self.max_steps = int(round(cfg.episode_seconds * cfg.control_hz))

        self.datas = [mujoco.MjData(b.model) for b in bodies]
        self.rng = np.random.default_rng(seed)
        self.prop = sensing.BatchProprioception(self.spec, self.n, self.rng)
        self.raw = sensing.make_raw(self.spec, self.n, self.chunks)

        w = max(1, workers if workers is not None else min(self.n, (os.cpu_count() or 4)))
        self.pool = ThreadPoolExecutor(max_workers=w) if self.n > 1 else None
        # One task per WORKER, each stepping a contiguous span of envs -- not one task per
        # env. `ThreadPoolExecutor.map` builds a Future, a condition variable and a queue
        # entry per task, all in GIL-held Python; at 64 envs that dispatch alone measured
        # ~10 us per env against ~45 us of GIL-held work inside the task, so it was a fifth
        # of the serial budget. Spans make the dispatch cost O(workers) instead of O(n).
        edges = np.linspace(0, self.n, w + 1).astype(int)
        self._spans = [(int(edges[k]), int(edges[k + 1])) for k in range(w)
                       if edges[k] < edges[k + 1]]

        # Per-env episode state and the batch buffers the step writes through.
        self.obs = np.zeros((self.n, self.obs_dim))
        self.rew = np.zeros(self.n)
        self.term = np.zeros(self.n, dtype=bool)
        self.trunc = np.zeros(self.n, dtype=bool)
        self.final_obs = np.zeros((self.n, self.obs_dim))
        self.command = np.zeros((self.n, 3))
        self.morph = np.array([b.spec.morph_norm for b in bodies])
        self.withers = np.array([b.withers for b in bodies])
        self._raw_obs = np.zeros((self.n, self.obs_dim))
        self._scratch = np.zeros((self.n, self.obs_dim))
        self._prev_action = np.zeros((self.n, self.act_dim))
        self._ctrl = np.zeros((self.n, self.act_dim))
        self._steps = np.zeros(self.n, dtype=np.int64)
        self._return = np.zeros(self.n)
        self._tilt = np.zeros(self.n)
        self._speed = np.zeros(self.n)
        self._cot = np.zeros(self.n)
        self._r_speed = np.zeros(self.n)
        # `info` hands out these buffers by reference to avoid a per-step allocation, so every
        # array it exposes must be one that the auto-reset at the END of `step` cannot touch.
        # `command` and `raw.sane` are both live state the reset overwrites -- the reset draws
        # a new command and re-gathers the fresh body -- so they get snapshotted here instead.
        # Reading them directly was a real bug and an instructive one: the corrupted rows were
        # exactly the finished episodes, i.e. the only rows a logger looks at, and the values
        # stayed plausible (a valid command, `sane` True) so nothing ever raised.
        self._info_cmd = np.zeros((self.n, 3))
        self._sane = np.zeros(self.n, dtype=bool)

    # ------------------------------------------------------------------------- commands
    def sample_commands(self, k: int) -> np.ndarray:
        """Draw `k` desired body-frame velocities, in the bodies' own units."""
        c = self.cfg
        cmd = np.stack([self.rng.uniform(*c.speed_range, k),
                        self.rng.uniform(*c.lateral_range, k),
                        self.rng.uniform(*c.yaw_range, k)], axis=1)
        cmd[self.rng.random(k) < c.stand_fraction] = 0.0
        return cmd

    # ---------------------------------------------------------------------------- reset
    def _reset_one(self, i: int) -> None:
        import mujoco

        from .emit_mjcf import place_on_ground

        m, d, s, c = self.bodies[i].model, self.datas[i], self.spec, self.cfg
        mujoco.mj_resetData(m, d)
        place_on_ground(m, d)

        # Perturb the start pose. Every episode starting from the identical settled stance
        # lets a policy memorise one opening rather than learn to move from wherever it
        # finds itself, and the failure only shows when something disturbs it.
        span = s.jnt_hi - s.jnt_lo
        d.qpos[s.jnt_qposadr] += (self.rng.uniform(-0.5, 0.5, len(span)) * span
                                  * c.init_joint_noise)
        d.qvel[s.jnt_dofadr] += (self.rng.uniform(-0.5, 0.5, len(span))
                                 * (c.init_rate_noise / s.time))
        mujoco.mj_forward(m, d)
        place_on_ground(m, d)

        # `mj_resetData` zeroed MuJoCo's warning counters too, so the baseline `gather_state`
        # compares against has to be zeroed with them or the fresh env reads as insane.
        self.raw.warn[i] = 0
        sensing.gather_state(m, d, s, self.raw, i)
        self.raw.work_tau[i] = 0.0
        self.raw.work_vel[i] = 0.0
        self._steps[i] = 0
        self._return[i] = 0.0
        self._prev_action[i] = 0.0

    def _reset_idx(self, idx: np.ndarray) -> None:
        """Reset the given envs and refill their whole delay history with the new state."""
        for i in idx:
            self._reset_one(int(i))
        self.command[idx] = self.sample_commands(len(idx))
        # Assembled on the full batch (one numpy call beats len(idx) of them) but only the
        # reset rows are published, so a mid-batch auto-reset cannot disturb the envs that
        # are still running.
        fresh = sensing.assemble(self.spec, self.raw, self._prev_action, self.command,
                                 self.morph, self._scratch)
        self._raw_obs[idx] = fresh[idx]
        self.prop.reset(idx, fresh[idx], self._prev_action[idx])
        self.obs[idx] = self.prop.sense(self._raw_obs)[idx]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.prop.rng = self.rng
        self._reset_idx(np.arange(self.n))
        return self.obs, {"command": self.command}

    # ----------------------------------------------------------------------------- step
    def _physics(self, i: int) -> None:
        """The only code that runs in the thread pool. Touches `MjData` and nothing else."""
        import mujoco

        m, d, s = self.bodies[i].model, self.datas[i], self.spec
        d.ctrl[:] = self._ctrl[i]
        tau, vel, sel = self.raw.work_tau[i], self.raw.work_vel[i], s.dof_sel
        tau[0] = d.qfrc_actuator[sel]
        vel[0] = d.qvel[sel]
        for c in range(1, self.chunks + 1):
            mujoco.mj_step(m, d, nstep=self.chunk_steps)
            tau[c] = d.qfrc_actuator[sel]
            vel[c] = d.qvel[sel]
        sensing.gather_state(m, d, s, self.raw, i)

    def _physics_span(self, span: tuple[int, int]) -> None:
        for i in range(span[0], span[1]):
            self._physics(i)

    def step(self, actions: np.ndarray):
        cfg, s = self.cfg, self.spec
        action = np.clip(np.asarray(actions, dtype=float).reshape(self.n, self.act_dim),
                         -1.0, 1.0)

        # What the muscles receive now is what the brain sent a conduction delay ago.
        self._ctrl[:] = self.prop.actuate(action)

        if self.pool is None:
            self._physics(0)
        else:
            list(self.pool.map(self._physics_span, self._spans))
        self._steps += 1

        # ---- observation -----------------------------------------------------------------
        sane = self.raw.sane
        fresh = sensing.assemble(s, self.raw, action, self.command, self.morph,
                                 self._scratch)
        # An exploded integrator has no meaningful observation; keep the last good one rather
        # than letting NaNs -- or MuJoCo's own silent reset-to-default-pose, see
        # `sensing.gather_state` -- into the delay line, where either would contaminate every
        # later read.
        np.copyto(self._raw_obs, fresh, where=sane[:, None])
        self.prop.advance()
        self.prop.sense(self._raw_obs, out=self.obs)

        # ---- reward, every term dimensionless --------------------------------------------
        R = self.raw.xmat
        v_local = np.einsum("nji,nj->ni", R, self.raw.qvel_free[:, 0:3]) / s.speed
        yaw_rate = self.raw.qvel_free[:, 5] * s.time
        cmd = self.command

        e_v = ((v_local[:, 0] - cmd[:, 0]) ** 2 + (v_local[:, 1] - cmd[:, 1]) ** 2)
        r_speed = np.exp(-e_v / (cfg.speed_tol ** 2))
        r_yaw = np.exp(-((yaw_rate - cmd[:, 2]) ** 2) / (cfg.yaw_tol ** 2))

        power = sensing.actuator_work(s, self.raw, self.chunk_dt) / cfg.control_dt
        c_energy = power / (s.weight * s.speed)              # ~ cost of transport
        c_rate = np.mean((action - self._prev_action) ** 2, axis=1)
        c_torque = np.mean(action ** 2, axis=1)

        self.rew[:] = (cfg.w_speed * r_speed + cfg.w_yaw * r_yaw
                       - cfg.w_energy * c_energy - cfg.w_action_rate * c_rate
                       - cfg.w_torque * c_torque)
        self.rew[~sane] = -1.0
        self._prev_action[:] = action
        self._return += self.rew

        # ---- termination -------------------------------------------------------------------
        # Identical measure to `validate.trunk_tilt_of`, deliberately: a policy that can
        # satisfy the training termination and not P0's acceptance bar is a policy that
        # learned to game the difference between them.
        self._tilt[:] = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
        self.term[:] = ((self._tilt > cfg.fall_tilt)
                        | (self.raw.root_z < cfg.min_height * self.withers)
                        | ~sane)
        self.trunc[:] = (~self.term) & (self._steps >= self.max_steps)
        self._speed[:], self._cot[:], self._r_speed[:] = v_local[:, 0], c_energy, r_speed

        self._info_cmd[:] = cmd                      # snapshots -- see __init__
        self._sane[:] = sane
        info = {"command": self._info_cmd, "tilt": self._tilt, "speed": self._speed,
                "cot": self._cot, "r_speed": self._r_speed, "sane": self._sane}
        done = np.nonzero(self.term | self.trunc)[0]
        if done.size:
            info["episode_return"] = self._return[done].copy()
            info["episode_length"] = self._steps[done].copy()
            info["episode_idx"] = done
            if self.auto_reset:
                self.final_obs[done] = self.obs[done]
                info["final_obs"] = self.final_obs
                self._reset_idx(done)
        return self.obs, self.rew, self.term, self.trunc, info

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


class CreatureEnv:
    """One creature, one episode at a time. Gym-style, without the gymnasium dependency.

    A thin view over a one-element `VecCreatureEnv` rather than a separate implementation:
    the reward and termination rules *are* the definition of the task, and two copies of them
    would drift the moment either is tuned. Auto-reset is off here, so `step` returns the
    terminal observation and the caller resets, which is the single-env Gym convention.

    The gymnasium dependency is skipped on purpose: the only things it would provide are the
    `spaces` declaration and the 5-tuple convention, both of which are four lines, and the
    parts of it that cost something -- its vector-env wrappers, its auto-reset semantics --
    are exactly the parts P2's AMP discriminator and P4's per-env morphs would have to fight.
    The 5-tuple signature is kept identical so a gymnasium wrapper stays trivial.
    """

    def __init__(self, body: Body, cfg: EnvConfig, seed: int = 0):
        self.vec = VecCreatureEnv([body], cfg, seed=seed, workers=1, auto_reset=False)
        self.body = body
        self.cfg = cfg
        self.spec = body.spec
        self.model = body.model
        self.data = self.vec.datas[0]
        self.obs_dim = self.vec.obs_dim
        self.act_dim = self.vec.act_dim
        self.max_steps = self.vec.max_steps

    @staticmethod
    def _row(info: dict) -> dict:
        out = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim else v)
               for k, v in info.items() if not k.startswith("episode_")}
        if "episode_return" in info:
            out["episode"] = {"r": float(info["episode_return"][0]),
                              "l": int(info["episode_length"][0])}
        return out

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self.vec.reset(seed=seed)
        return obs[0], self._row(info)

    def step(self, action: np.ndarray):
        obs, rew, term, trunc, info = self.vec.step(np.asarray(action)[None, :])
        return obs[0], float(rew[0]), bool(term[0]), bool(trunc[0]), self._row(info)

    def close(self) -> None:
        self.vec.close()
