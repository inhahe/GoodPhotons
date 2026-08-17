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

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import sensing, terrain
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

    # --- command curriculum ---------------------------------------------------------------
    # The commanded *speed* is widened only as fast as the policy earns it. This is not a
    # convenience: with the full range on from the start, over half of every batch is drawn
    # from a region where the reward is flat, and the run reliably converges to standing still.
    # See `_advance_curriculum` for the measurement and the escape argument.
    curriculum: bool = True
    curriculum_start: float = 0.3      # initial upper bound on |commanded speed|
    curriculum_step: float = 0.05      # how much to widen by, per promotion
    curriculum_window: int = 256       # episodes that must finish before a promotion is judged
    # The bar is expressed as a fraction of the gap between "parked" and "perfect", not as an
    # absolute tracking reward, because an absolute number asks a different question at every
    # cap width. A motionless animal scores 0.64 of the tracking reward when commands only run
    # to 0.3 Froude and 0.31 when they run to 0.8 -- so a fixed 0.75 is "track a bit better than
    # standing" at the start and "track almost perfectly" later, which is backwards: wider
    # commands are the harder task. `_standstill_score` computes the parked baseline in closed
    # form and the bar sits this far above it. 0.25 reproduces ~0.73 at the initial width.
    curriculum_margin: float = 0.25

    # --- terrain (P1b) ----------------------------------------------------------------------
    # Off by default: a flat plane is what P1's numbers were measured on, and the terrain bar
    # is explicitly "flat-ground performance does not regress", which needs the flat task to
    # stay reachable unchanged.
    #
    # Everything here is in the body's OWN units, for the same reason the command range is:
    # `terrain_amp` and `terrain_cell` are withers heights, so the same config means the same
    # terrain on a terrier and a wolfhound rather than the same absolute stones.
    terrain: bool = False
    terrain_kinds: tuple[str, ...] = ("flat", "rolling", "rubble", "steps", "slope", "stairs")
    terrain_amp: float = 0.35          # peak-to-peak of the rough classes at difficulty 1
    terrain_cell: float = 0.25         # target sample spacing
    terrain_margin: float = 1.25       # patch covers this multiple of an episode's max travel
    terrain_max_slope: float = 12.0    # deg; the MEASURED learnable ceiling, see TerrainSpec
    terrain_max_cells: int = 513       # ceiling on the grid; cell size gives before this does

    # The terrain curriculum shares the command curriculum's window, bar and measurement --
    # see `_advance_curriculum`. It has its own start/step because difficulty is a different
    # axis with a different natural granularity.
    terrain_curriculum: bool = True
    terrain_start: float = 0.0         # difficulty in [0, 1]; 0 is flat
    terrain_step: float = 0.1

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
    #: The MJCF this model was compiled from, kept so a vector env can clone the model per
    #: env. That is only needed for terrain -- `hfield_data` lives in `MjModel`, so envs
    #: sharing a `Body` would share one terrain and a reset in env 3 would move the ground
    #: under envs 0-2 mid-episode. See `VecCreatureEnv.__init__`.
    mjcf: str = ""
    #: The hfield's shape, when this body was compiled with terrain. `None` means a plane.
    terrain: object | None = None


def build_body(cfg: EnvConfig, morph: dict[str, float] | None = None,
               foot_names: list[str] | None = None) -> Body:
    """Compile a rig to a simulatable body, tone included.

    `build_tuned`, not `load`: an untuned skeleton has no passive tone, folds up the
    instant gravity touches it, and would hand the policy the job of not collapsing --
    which costs training budget forever. The tone pass is part of what "this body" means.

    With `cfg.terrain` the model is compiled twice, and that is not waste. The terrain patch
    is sized in the body's own units -- amplitude and cell in withers heights, extent from
    how far this animal can travel in an episode -- and the withers height is a *measurement*
    off the compiled model (`spec.length`), so it does not exist until the first compile. The
    tuning pass, which is what actually costs (~1.1 s), runs once either way; the second pass
    is XML parsing.
    """
    import mujoco

    from .emit_mjcf import to_mjcf
    from .tune import build_tuned

    creature, _ = build_tuned(cfg.rig, morph)

    def compile_(tspec):
        xml = to_mjcf(creature, terrain=tspec)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        skip = int(round(cfg.control_dt / m.opt.timestep))
        if skip < 1:
            raise ValueError(f"control_hz {cfg.control_hz} is faster than the physics "
                             f"timestep {m.opt.timestep}")
        s = sensing.build_spec(m, d, creature, control_dt=skip * m.opt.timestep,
                               foot_names=foot_names)
        return xml, m, skip, s

    xml, model, skip, spec = compile_(None)
    tspec = None
    if cfg.terrain:
        tspec = terrain.TerrainSpec.for_body(
            spec.length, amp=cfg.terrain_amp, cell=cfg.terrain_cell,
            margin=cfg.terrain_margin, max_speed=cfg.speed_range[1],
            seconds=cfg.episode_seconds, max_slope_deg=cfg.terrain_max_slope,
            max_cells=cfg.terrain_max_cells)
        xml, model, skip, spec = compile_(tspec)
    return Body(creature=creature, model=model, spec=spec, frame_skip=skip,
                withers=spec.length, mjcf=xml, terrain=tspec)


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
        # `spec` is the LAYOUT reference only -- dims, slices, the index tables that address
        # `MjData`. Every quantity that depends on the body (its length and pendulum period,
        # its joint ranges, its conduction lags) is read from `bspec`, which is per-env; and
        # the per-env physics functions take `self.bodies[i].spec` directly. Reaching for
        # `self.spec` to normalise something is the bug `BatchSpec` exists to prevent.
        self.spec = bodies[0].spec
        self.bspec = sensing.batch_spec([b.spec for b in bodies])
        self.obs_dim = self.bspec.dim
        self.act_dim = self.bspec.act_dim
        for b in bodies:
            if b.frame_skip != bodies[0].frame_skip:
                raise ValueError("all bodies in a vector env must share a frame skip")

        self.frame_skip = bodies[0].frame_skip
        self.chunks = _energy_chunks(self.frame_skip, cfg.energy_samples)
        self.chunk_steps = self.frame_skip // self.chunks
        self.chunk_dt = self.chunk_steps * bodies[0].model.opt.timestep
        self.max_steps = int(round(cfg.episode_seconds * cfg.control_hz))

        # Per-env models, but only when terrain is on. `MjModel` is read-only during a step,
        # so on flat ground every env can share one -- which is the whole reason `build_envs`
        # compiles a single body for 64 envs. Terrain breaks that: elevation data lives in
        # `MjModel.hfield_data`, so envs sharing a model share a terrain, and since episodes
        # end at different steps a reset in one env would move the ground under the others
        # mid-stride. Cloning is per env rather than per distinct body for the same reason.
        self.models = [b.model for b in bodies]
        self.patches: list[terrain.Patch] | None = None
        if any(b.terrain is not None for b in bodies):
            if not all(b.terrain is not None for b in bodies):
                raise ValueError("some bodies were compiled with terrain and some without")
            self.models = [self._clone_model(b) for b in bodies]
            self.patches = [terrain.Patch(b.terrain, m)
                            for b, m in zip(bodies, self.models)]

        self.datas = [mujoco.MjData(m) for m in self.models]
        self.rng = np.random.default_rng(seed)
        self.prop = sensing.BatchProprioception(self.bspec, self.n, self.rng)
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
        self.withers = self.bspec.length
        self._raw_obs = np.zeros((self.n, self.obs_dim))
        self._scratch = np.zeros((self.n, self.obs_dim))
        self._prev_action = np.zeros((self.n, self.act_dim))
        self._ctrl = np.zeros((self.n, self.act_dim))
        self._steps = np.zeros(self.n, dtype=np.int64)
        self._return = np.zeros(self.n)
        # Episode-total tracking reward, which is what the command curriculum is judged on --
        # deliberately not `_return`, see `_advance_curriculum`.
        self._ep_track = np.zeros(self.n)
        self._speed_cap = (float(np.clip(cfg.curriculum_start, *cfg.speed_range))
                           if cfg.curriculum else cfg.speed_range[1])
        # Terrain difficulty in [0, 1]. Zero without terrain, so `info["terrain_level"]` is a
        # number in every run and a log parser never has to special-case the flat case.
        self._terrain_level = 0.0 if self.patches is None else (
            float(np.clip(cfg.terrain_start, 0.0, 1.0)) if cfg.terrain_curriculum else 1.0)
        self._cur_track = self._cur_steps = 0.0     # the promotion window, see `_advance_...`
        self._cur_eps = 0
        # The last completed window's score, and the standstill score it has to beat. Both are
        # exposed in `info` because a curriculum whose decision rule is invisible in the log is
        # a curriculum you can only debug by rerunning it: the first thing worth knowing about a
        # cap that has not moved in 900k steps is whether the score is at 0.74 or at 0.40, and
        # nothing else in the rollout answers that. `r_speed` in the rollout stats is close but
        # not the same number -- it averages every step in the buffer, including the currently
        # unfinished episodes, which are exactly the ones that have not fallen over yet.
        self.cur_score = float("nan")
        self.cur_bar = self._promotion_bar()
        self._tilt = np.zeros(self.n)
        self._clear = np.zeros(self.n)      # root height above the ground under it
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

    @staticmethod
    def _clone_model(body: Body):
        """An independent `MjModel` for one env. Deep copy where the bindings allow it.

        `copy.deepcopy` is ~two orders of magnitude cheaper than re-parsing the XML, which
        matters at 64 envs, but it is a binding feature rather than a guarantee -- so the
        MJCF is kept on the `Body` and re-parsed if the copy is refused.
        """
        import copy

        try:
            return copy.deepcopy(body.model)
        except Exception:                       # noqa: BLE001 -- any failure means re-parse
            import mujoco
            return mujoco.MjModel.from_xml_string(body.mjcf)

    # ------------------------------------------------------------------------- commands
    def sample_commands(self, k: int) -> np.ndarray:
        """Draw `k` desired body-frame velocities, in the bodies' own units."""
        c = self.cfg
        hi = self.speed_cap if c.curriculum else c.speed_range[1]
        cmd = np.stack([self.rng.uniform(c.speed_range[0], hi, k),
                        self.rng.uniform(*c.lateral_range, k),
                        self.rng.uniform(*c.yaw_range, k)], axis=1)
        cmd[self.rng.random(k) < c.stand_fraction] = 0.0
        return cmd

    @property
    def speed_cap(self) -> float:
        """Current upper bound on the commanded forward speed; see `_advance_curriculum`."""
        return self._speed_cap

    @speed_cap.setter
    def speed_cap(self, v: float) -> None:
        # A property rather than a plain attribute solely so `cur_bar` cannot go stale. The
        # bar is a function of the cap, and the cap is written from outside this class exactly
        # once -- `tools/train.py` restores it from a checkpoint on `--resume` -- which is the
        # one place a recompute is easiest to forget and hardest to notice, since the run would
        # simply promote against the wrong bar for the rest of its life.
        self._speed_cap = float(v)
        self.cur_bar = self._promotion_bar()

    @property
    def terrain_level(self) -> float:
        """Current terrain difficulty in [0, 1]; see `_advance_curriculum`."""
        return self._terrain_level

    @terrain_level.setter
    def terrain_level(self, v: float) -> None:
        # Written from outside exactly where `speed_cap` is: a checkpoint restore. Resuming a
        # terrain run at difficulty 0 would hand a competent policy the flat task it solved
        # millions of steps ago, and the curve would step backwards for reasons entirely
        # internal to the resume.
        self._terrain_level = float(np.clip(v, 0.0, 1.0))

    def _standstill_score(self) -> float:
        """The tracking reward a *motionless* animal collects under the current cap.

        This is the number the promotion bar is measured from, and it is worth having exactly
        rather than by intuition, because it moves by a factor of two across the curriculum.
        `r_speed` at zero body velocity is `exp(-(u^2 + w^2) / speed_tol^2)` for a commanded
        forward/lateral pair `(u, w)`; the two are drawn independently and uniformly, so the
        expectation factorises into two one-dimensional Gaussian integrals, each of which is an
        `erf` difference. `stand_fraction` of commands are set to exactly zero and score exactly
        1, so they are mixed in separately.
        """
        tol = self.cfg.speed_tol

        def mean_kernel(lo: float, hi: float) -> float:
            if hi <= lo:                                    # a degenerate range is a point mass
                return math.exp(-(lo / tol) ** 2)
            k = tol * math.sqrt(math.pi) / 2
            return k * (math.erf(hi / tol) - math.erf(lo / tol)) / (hi - lo)

        lo = self.cfg.speed_range[0]
        hi = self.speed_cap if self.cfg.curriculum else self.cfg.speed_range[1]
        moving = mean_kernel(lo, hi) * mean_kernel(*self.cfg.lateral_range)
        sf = self.cfg.stand_fraction
        return sf + (1.0 - sf) * moving

    def _promotion_bar(self) -> float:
        """Tracking reward a window must reach to widen the cap: `curriculum_margin` of the way
        from a motionless animal's score to a perfect one's."""
        stand = self._standstill_score()
        return stand + self.cfg.curriculum_margin * (1.0 - stand)

    def _advance_curriculum(self, idx: np.ndarray) -> None:
        """Widen the commanded speed range when the finished episodes earned it.

        **The problem this solves, measured rather than assumed.** `speed_tol` is 0.25 Froude
        and the declared range runs to 0.8, so the tracking kernel `exp(-e^2 / speed_tol^2)`
        evaluated at a *standing* animal decays across the range like this:

            commanded  0.00  0.15  0.29  0.44  0.58  0.73  0.80
            r_speed    0.99  0.71  0.26  0.05  0.005 0.001 0.000

        Above about 0.44 there is no reward to be had and, far more importantly, no *gradient*
        pointing anywhere -- the surface is flat, so those samples tell the policy nothing at
        all. With the full range on from the start that is over half of every batch. Meanwhile
        standing still is a genuinely good policy on the rest of it: it scores ~1.0 on the
        `stand_fraction` and near-zero commands, pays a cost of transport of 0.07 against the
        2.4 that moving costs, and never terminates. A 1M-step run found exactly that and sat
        there: commanded 0.0 through 0.8, the measured speed was 0.000 in every column, and the
        evaluation return oscillated for half a million steps without climbing.

        So the range starts where the gradient is (0.3, i.e. a slow walk) and widens by
        `curriculum_step` each time completed episodes come back tracking `curriculum_margin`
        of the way from `_standstill_score` to perfect. The animal is never asked for a speed it
        has no way to discover it should want.

        Judged on the *tracking* term alone, not on the return. The return is dominated by
        episode length, so a policy that survives well and tracks badly would promote itself
        out of the range it can actually handle -- which is the failure this exists to prevent,
        arrived at from the other side.

        Only ever widens. A curriculum that also narrows on a bad batch oscillates, and the
        thing being measured is a mean over finished episodes, which is noisy.

        **Judged over a window of `curriculum_window` episodes, and that is the whole
        difference between this working and not.** The first version scored whichever handful
        of envs happened to finish on the current step -- so it ran the promotion test dozens
        of times per rollout, on samples of one to five episodes, drawn from a population
        selected precisely for having ended. It went 0.30 to the full 0.80 in 82k steps, before
        the animal could stand, and reproduced the standstill it was written to prevent. The
        window is one accumulator over ~4 episodes per env, cleared on each decision, so the
        test runs on a sample large enough to mean something and at most once per window.

        Accumulated per *step*, not per episode, so a policy that falls over after 20 well
        tracked steps does not count the same as one that held the command for 20 seconds.

        **Terrain (P1b) rides on exactly this machinery, on a second axis.** Rough ground and
        a faster command are both "the task got harder", both show up in the same tracking
        reward, and both must only be widened once the policy has earned it -- so they share
        the window, the measurement and the bar, and differ only in what a promotion spends
        itself on. `_promote` gives each earned promotion to whichever axis has progressed
        less, in fractions of its own range. Advancing both at once would double the step
        change the policy has to absorb per promotion; advancing speed to its maximum first
        would train a flat-ground gallop and then ask it to relearn on rubble.
        """
        n = self._steps[idx]                    # zeroed by `_reset_one`, so this is the episode
        if not (self._can_promote_speed() or self._can_promote_terrain()):
            return
        self._cur_track += float(np.sum(self._ep_track[idx]))
        self._cur_steps += float(np.sum(n))
        self._cur_eps += int(idx.size)
        if self._cur_eps < self.cfg.curriculum_window or self._cur_steps <= 0:
            return
        self.cur_score = score = self._cur_track / self._cur_steps
        self._cur_track = self._cur_steps = 0.0
        self._cur_eps = 0
        if score >= self.cur_bar:
            self._promote()

    def _can_promote_speed(self) -> bool:
        return bool(self.cfg.curriculum and self.speed_cap < self.cfg.speed_range[1])

    def _can_promote_terrain(self) -> bool:
        return bool(self.patches is not None and self.cfg.terrain_curriculum
                    and self._terrain_level < 1.0)

    def _promote(self) -> None:
        """Spend one earned promotion on the axis that is further behind."""
        c = self.cfg
        span = c.speed_range[1] - c.curriculum_start
        speed_at = 1.0 if span <= 0 else (self.speed_cap - c.curriculum_start) / span
        # An axis that cannot move counts as finished, so the comparison always picks the
        # other one rather than stalling the curriculum on a maxed-out axis.
        if not self._can_promote_speed():
            speed_at = float("inf")
        terrain_at = self._terrain_level if self._can_promote_terrain() else float("inf")
        if speed_at <= terrain_at:
            self.speed_cap = min(c.speed_range[1],        # the setter moves `cur_bar` too
                                 self.speed_cap + c.curriculum_step)
        else:
            self.terrain_level = self._terrain_level + c.terrain_step

    # ---------------------------------------------------------------------------- reset
    def _reset_one(self, i: int) -> None:
        import mujoco

        from .emit_mjcf import place_on_ground

        m, d, s, c = self.models[i], self.datas[i], self.bodies[i].spec, self.cfg
        mujoco.mj_resetData(m, d)

        # A fresh surface every episode, not one memorised terrain. Rolled before the body is
        # placed, because where the ground is decides where the body starts.
        patch = None if self.patches is None else self.patches[i]
        if patch is not None:
            patch.regenerate(self.rng, self._terrain_level, c.terrain_kinds)
        place_on_ground(m, d, patch=patch)

        # Perturb the start pose. Every episode starting from the identical settled stance
        # lets a policy memorise one opening rather than learn to move from wherever it
        # finds itself, and the failure only shows when something disturbs it.
        span = s.jnt_hi - s.jnt_lo
        d.qpos[s.jnt_qposadr] += (self.rng.uniform(-0.5, 0.5, len(span)) * span
                                  * c.init_joint_noise)
        d.qvel[s.jnt_dofadr] += (self.rng.uniform(-0.5, 0.5, len(span))
                                 * (c.init_rate_noise / s.time))
        mujoco.mj_forward(m, d)
        place_on_ground(m, d, patch=patch)

        # `mj_resetData` zeroed MuJoCo's warning counters too, so the baseline `gather_state`
        # compares against has to be zeroed with them or the fresh env reads as insane.
        self.raw.warn[i] = 0
        sensing.gather_state(m, d, s, self.raw, i)
        if patch is not None:
            self.raw.ground_z[i] = patch.height_at(d.qpos[0], d.qpos[1])
        self.raw.work_tau[i] = 0.0
        self.raw.work_vel[i] = 0.0
        self._steps[i] = 0
        self._return[i] = 0.0
        self._ep_track[i] = 0.0
        self._prev_action[i] = 0.0

    def _reset_idx(self, idx: np.ndarray) -> None:
        """Reset the given envs and refill their whole delay history with the new state."""
        for i in idx:
            self._reset_one(int(i))
        self.command[idx] = self.sample_commands(len(idx))
        # Assembled on the full batch (one numpy call beats len(idx) of them) but only the
        # reset rows are published, so a mid-batch auto-reset cannot disturb the envs that
        # are still running.
        fresh = sensing.assemble(self.bspec, self.raw, self._prev_action, self.command,
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

        m, d, s = self.models[i], self.datas[i], self.bodies[i].spec
        d.ctrl[:] = self._ctrl[i]
        tau, vel, sel = self.raw.work_tau[i], self.raw.work_vel[i], s.dof_sel
        tau[0] = d.qfrc_actuator[sel]
        vel[0] = d.qvel[sel]
        for c in range(1, self.chunks + 1):
            mujoco.mj_step(m, d, nstep=self.chunk_steps)
            tau[c] = d.qfrc_actuator[sel]
            vel[c] = d.qvel[sel]
        sensing.gather_state(m, d, s, self.raw, i)
        # The one terrain query in the hot loop. It lives here, in the pool, rather than in
        # the batched pass outside it, because each env has its own elevation array -- there
        # is nothing to batch, and N one-point numpy calls would cost more GIL time than N
        # scalar interpolations. See `terrain.Patch.height_at`.
        if self.patches is not None:
            self.raw.ground_z[i] = self.patches[i].height_at(d.qpos[0], d.qpos[1])

    def _physics_span(self, span: tuple[int, int]) -> None:
        for i in range(span[0], span[1]):
            self._physics(i)

    def step(self, actions: np.ndarray):
        cfg, s = self.cfg, self.bspec
        action = np.clip(np.asarray(actions, dtype=float).reshape(self.n, self.act_dim),
                         -1.0, 1.0)

        # What the muscles receive now is what the brain sent a conduction delay ago.
        self._ctrl[:] = self.prop.actuate(action)

        if self.pool is None:
            self._physics(0)
        else:
            list(self.pool.map(self._physics_span, self._spans))
        self._steps += 1

        # MuJoCo's own warning counters only fire on NaN/Inf. A body can also be *finite* and
        # still not be physics -- hundreds of Froude of root speed -- which reaches the reward
        # (`c_energy`) and the progress trace looking like a real sample. See known-issues #6.
        sensing.flag_divergence(s, self.raw)

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
        v_local = np.einsum("nji,nj->ni", R, self.raw.qvel_free[:, 0:3]) / s.speed_col
        yaw_rate = self.raw.qvel_free[:, 5] * s.time
        cmd = self.command

        e_v = ((v_local[:, 0] - cmd[:, 0]) ** 2 + (v_local[:, 1] - cmd[:, 1]) ** 2)
        r_speed = np.exp(-e_v / (cfg.speed_tol ** 2))
        r_yaw = np.exp(-((yaw_rate - cmd[:, 2]) ** 2) / (cfg.yaw_tol ** 2))

        power = sensing.actuator_work(self.raw, self.chunk_dt) / cfg.control_dt
        c_energy = power / (s.weight * s.speed)              # ~ cost of transport
        c_rate = np.mean((action - self._prev_action) ** 2, axis=1)
        c_torque = np.mean(action ** 2, axis=1)

        self.rew[:] = (cfg.w_speed * r_speed + cfg.w_yaw * r_yaw
                       - cfg.w_energy * c_energy - cfg.w_action_rate * c_rate
                       - cfg.w_torque * c_torque)
        self.rew[~sane] = -1.0
        self._prev_action[:] = action
        self._return += self.rew
        self._ep_track += r_speed

        # ---- termination -------------------------------------------------------------------
        # Identical measure to `validate.trunk_tilt_of`, deliberately: a policy that can
        # satisfy the training termination and not P0's acceptance bar is a policy that
        # learned to game the difference between them.
        #
        # The height test is on clearance above the ground *under the root*, not on world z.
        # On a plane `ground_z` is zero and this is the pre-terrain test unchanged; on terrain
        # world z is wrong in both directions at once -- cresting a rise reads as unusually
        # tall, standing healthily in a dip reads as fallen -- and the second of those ends
        # episodes the policy was doing nothing wrong in, which is the more expensive error.
        self._tilt[:] = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
        self._clear[:] = self.raw.root_z - self.raw.ground_z
        self.term[:] = ((self._tilt > cfg.fall_tilt)
                        | (self._clear < cfg.min_height * self.withers)
                        | ~sane)
        self.trunc[:] = (~self.term) & (self._steps >= self.max_steps)
        self._speed[:], self._cot[:], self._r_speed[:] = v_local[:, 0], c_energy, r_speed

        self._info_cmd[:] = cmd                      # snapshots -- see __init__
        self._sane[:] = sane
        info = {"command": self._info_cmd, "tilt": self._tilt, "speed": self._speed,
                "cot": self._cot, "r_speed": self._r_speed, "sane": self._sane,
                "speed_cap": self.speed_cap, "cur_score": self.cur_score,
                "cur_bar": self.cur_bar, "terrain_level": self.terrain_level}
        done = np.nonzero(self.term | self.trunc)[0]
        if done.size:
            info["episode_return"] = self._return[done].copy()
            info["episode_length"] = self._steps[done].copy()
            info["episode_idx"] = done
            if self.auto_reset:
                # Before the reset, which zeroes the very counters the promotion is judged on.
                # Gated on `auto_reset` so an evaluation env -- which is handed a fixed command
                # grid and never trains -- cannot advance a curriculum it is not part of.
                self._advance_curriculum(done)
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
        # The vec env's model, not `body.model`: with terrain they are different objects, and
        # the one that matters is the one `self.data` was allocated against and whose hfield
        # the resets write into.
        self.model = self.vec.models[0]
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
