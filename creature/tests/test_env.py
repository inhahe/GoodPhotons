"""Regression tests for the observation contract and the locomotion environment.

Same bias as `test_rig.py`: aimed at the failures that produce a plausible number instead
of an exception. Everything this layer can get wrong is silent. A delay line rounded to
whole control steps still trains, and only collapses the fore/hind asymmetry the module
exists to create. An auto-reset that reports the *reset* observation as the terminal one
still trains, and only corrupts the value target of the last step of every episode. A
batched rewrite that disagrees with the single-env path still trains, and only makes every
number measured at N=1 a lie about the run. None of that raises.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mujoco = pytest.importorskip("mujoco")

from creaturelab import sensing                                     # noqa: E402
from creaturelab.env import (CreatureEnv, EnvConfig, VecCreatureEnv,  # noqa: E402
                             _energy_chunks, build_body)

RIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rigs", "canis.ftcl")


def cfg(**kw) -> EnvConfig:
    kw.setdefault("rig", RIG)
    return EnvConfig(**kw)


@pytest.fixture(scope="module")
def body():
    return build_body(cfg())


# --------------------------------------------------------------------- the observation spec
def test_scales_are_the_bodys_own(body):
    """L, T, V, W must be dynamically similar, not arbitrary constants."""
    s = body.spec
    g = 9.81
    assert s.length == pytest.approx(body.withers)
    assert s.time == pytest.approx(np.sqrt(s.length / g), rel=1e-3)
    assert s.speed == pytest.approx(np.sqrt(g * s.length), rel=1e-3)
    assert s.speed * s.time == pytest.approx(s.length, rel=1e-3)   # V*T == L, by definition


def test_layout_is_contiguous_and_complete(body):
    s = body.spec
    edge = 0
    for k in sensing.GROUPS:
        assert s.slices[k].start == edge, f"{k} does not abut the previous group"
        edge = s.slices[k].stop
    assert edge == s.dim
    assert len(s.names) == s.dim


def test_no_world_frame_channel_leaks_in(body):
    """Root height and world position are deliberately absent -- no animal has that sensor.

    Guarding it by name, because the tempting fix for slow early training is to add exactly
    one of these, and it trains beautifully right up to the first morph.
    """
    banned = ("root_z", "height", "world", "pos_x", "pos_y", "pos_z", "xpos")
    for n in body.spec.names:
        assert not any(b in n for b in banned), f"world-frame channel '{n}' in observation"


def test_delays_are_fractional_and_ordered(body):
    """The fore/hind lag gradient must survive, which means lags stay non-integer.

    At 50 Hz the dog's fore-paw lag is ~27 ms and its hind-paw ~32 ms; rounding either to a
    whole control step makes them the same number and silently deletes the anatomy.
    """
    s = body.spec
    lag = dict(zip(s.names, s.obs_delay_s))
    fore = np.mean([lag[f"contact/fpaw_{v}"] for v in "lr"])
    hind = np.mean([lag[f"contact/hpaw_{v}"] for v in "lr"])
    assert hind > fore + 1e-3, "hind afferents must be later than fore ones (hub is the head)"

    # And the stored delays must not be whole numbers of control steps.
    used = s.obs_delay[s.obs_delay > 0]
    assert np.any(np.abs(used - np.round(used)) > 1e-6), "delays were quantised to steps"

    # Vestibular sits at the hub, so it carries the central latency alone and is the fastest
    # measured channel -- which is why a falling animal rights itself before it can correct
    # a foot.
    vest = s.obs_delay_s[s.slices["vestibular"]]
    assert np.all(vest <= s.obs_delay_s[s.slices["contact"]].min() + 1e-12)

    # Internal signals are not measured through a nerve and must not be delayed at all.
    for k in ("efference", "command", "morph"):
        assert np.all(s.obs_delay_s[s.slices[k]] == 0.0)


def test_delay_scales_with_the_body(body):
    """A bigger animal must move heavier for free, from measured anatomy not a constant."""
    small = build_body(cfg(), morph={"body_scale": 0.6})
    large = build_body(cfg(), morph={"body_scale": 1.7})
    assert large.spec.length > body.spec.length > small.spec.length
    for k in ("joint_angle", "contact"):
        a = small.spec.obs_delay_s[small.spec.slices[k]].max()
        b = large.spec.obs_delay_s[large.spec.slices[k]].max()
        assert b > a * 1.3, f"{k} lag barely changed between a 0.6x and a 1.7x body"


def test_feet_are_the_things_touching_the_floor(body):
    assert sorted(body.spec.foot_names) == ["fpaw_l", "fpaw_r", "hpaw_l", "hpaw_r"]
    assert np.all(body.spec.geom_foot[body.spec.geom_foot >= 0] < 4)


def test_selectors_match_the_index_arrays(body):
    """The slice fast path must select exactly what the index arrays would."""
    s = body.spec
    d = mujoco.MjData(body.model)
    mujoco.mj_forward(body.model, d)
    assert np.array_equal(d.qpos[s.qpos_sel], d.qpos[s.jnt_qposadr])
    assert np.array_equal(d.qvel[s.dof_sel], d.qvel[s.jnt_dofadr])


def test_angle_normalisation_maps_range_to_pm_one(body):
    s = body.spec
    assert np.allclose(s.jnt_lo * s.ang_scale + s.ang_bias, -1.0)
    assert np.allclose(s.jnt_hi * s.ang_scale + s.ang_bias, +1.0)


# ------------------------------------------------------------------------ the delay lines
def test_delay_line_returns_signals_late():
    """A step injected now must not appear immediately on a lagged channel."""
    spec = _fake_spec(dim=3, lags=[0.0, 1.0, 2.5])
    prop = sensing.BatchProprioception(spec, 1, np.random.default_rng(0))
    prop.reset(np.array([0]), np.zeros((1, 3)), np.zeros((1, 1)))
    seen = []
    for k in range(6):
        raw = np.full((1, 3), 1.0 if k >= 1 else 0.0)
        prop.advance()
        seen.append(prop.sense(raw)[0].copy())
    seen = np.array(seen)
    assert seen[1, 0] == pytest.approx(1.0)          # zero lag: arrives at once
    assert seen[1, 1] == pytest.approx(0.0)          # one step: not yet
    assert seen[2, 1] == pytest.approx(1.0)
    # 2.5 steps: half-way between the 2- and 3-step-old samples, i.e. a genuine fraction,
    # which is the whole reason lags are not rounded.
    assert seen[3, 2] == pytest.approx(0.5)
    assert seen[4, 2] == pytest.approx(1.0)


def test_delay_line_history_is_seeded_not_zeroed():
    """Reset must fill the past with the reset state.

    Zero-filling gives the policy a first observation claiming the animal was inverted and
    airborne a moment ago; the resulting flail looks exactly like a physics bug.
    """
    spec = _fake_spec(dim=2, lags=[3.0, 3.0])
    prop = sensing.BatchProprioception(spec, 2, np.random.default_rng(0))
    obs0 = np.array([[5.0, -5.0], [1.0, -1.0]])
    prop.reset(np.array([0, 1]), obs0, np.zeros((2, 1)))
    got = prop.sense(obs0)
    assert np.allclose(got, obs0)
    assert np.allclose(prop.obs_buf[0], 5.0 * np.array([1.0, -1.0]))


def test_delay_line_history_is_long_enough_for_the_lerp():
    """The lerp reads floor(lag) AND one step further back, so sizing on ceil is off by one.

    A lag landing exactly on a step boundary gets weight zero on that extra sample, so the
    stale value it would read does no harm and the bug survives every behavioural test.
    """
    for lag in (2.0, 2.5, 3.0):
        spec = _fake_spec(dim=1, lags=[lag])
        prop = sensing.BatchProprioception(spec, 1, np.random.default_rng(0))
        assert prop.hist >= int(np.floor(lag)) + 2


def _fake_spec(dim: int, lags: list[float]) -> sensing.ObsSpec:
    """A minimal spec, so the delay-line tests do not need a body."""
    z = np.zeros(dim)
    return sensing.ObsSpec(
        dim=dim, slices={}, names=[f"c{i}" for i in range(dim)],
        length=1.0, time=1.0, speed=1.0, weight=1.0,
        jnt_qposadr=np.zeros(1, int), jnt_dofadr=np.zeros(1, int),
        jnt_lo=-np.ones(1), jnt_hi=np.ones(1), jnt_limited=np.ones(1, bool),
        foot_bodies=np.zeros(0, int), foot_names=[], geom_foot=np.zeros(0, np.int32),
        qpos_sel=slice(0, 1), dof_sel=slice(0, 1),
        ang_scale=np.ones(1), ang_bias=np.zeros(1), morph_norm=np.zeros(0),
        obs_delay=np.array(lags, dtype=float), act_delay=np.zeros(1),
        obs_delay_s=np.array(lags, dtype=float), obs_noise=z.copy(),
        max_delay=int(np.ceil(max(lags))))


# ------------------------------------------------------------------------- actuator work
def test_energy_chunks_divide_the_frame_skip():
    for skip in (1, 2, 6, 10, 12):
        for want in (1, 2, 5, 10, 99):
            c = _energy_chunks(skip, want)
            assert skip % c == 0 and c <= max(1, want)
    assert _energy_chunks(10, 5) == 5
    assert _energy_chunks(10, 7) == 5          # 7 is not a divisor; round down to one
    assert _energy_chunks(10, 3) == 2


def test_actuator_work_integrates_a_known_power():
    """Constant power over the step must integrate to power * dt, exactly."""
    spec = _fake_spec(dim=1, lags=[0.0])
    raw = sensing.make_raw(spec, 2, energy_samples=4)
    raw.work_tau[:] = 3.0
    raw.work_vel[:] = 2.0                       # |tau.qdot| = 6 per joint, 1 joint
    w = sensing.actuator_work(spec, raw, chunk_dt=0.005)
    assert np.allclose(w, 6.0 * 4 * 0.005)


def test_actuator_work_does_not_cancel_negative_work():
    """A joint driven against its own motion still burns energy."""
    spec = _fake_spec(dim=1, lags=[0.0])
    raw = sensing.make_raw(spec, 1, energy_samples=2)
    raw.work_tau[0, :, 0] = [1.0, 1.0, 1.0]
    raw.work_vel[0, :, 0] = [1.0, -1.0, 1.0]
    w = sensing.actuator_work(spec, raw, chunk_dt=1.0)
    assert w[0] == pytest.approx(2.0), "power was signed, so braking looked free"


# ------------------------------------------------------------------------------- the env
def test_passive_body_holds_its_stance(body):
    """Zero torque from the reference stance must not fall over -- at the reset noise the
    training loop actually uses, which is the whole point.

    This is the P0 acceptance bar restated inside the env: if the body needs its controller
    to stay upright, the cost of not falling over has been moved into the policy, where it is
    paid for the whole life of the project.

    It used to pin `init_joint_noise=0.0` and say so, because at the default 0.05 the body
    fell over in 3.5-6.5 s under zero torque (known-issues #3). That made the test agree with
    the physics while disagreeing with every episode PPO would ever run: it asserted the
    property only in the one condition where an antisymmetric instability is unobservable,
    since a perfectly symmetric body released from a perfectly symmetric pose has nothing to
    roll towards. Now that `tune.brace` sizes the roll mode, the default holds under 2.6 deg
    for 10 s at *twice* the default noise, so the test asserts what it always meant to.
    """
    e = CreatureEnv(body, cfg())          # default reset noise -- deliberately not zeroed
    e.reset(seed=0)
    a = np.zeros(e.act_dim)
    worst = 0.0
    for k in range(int(20.0 * 50)):
        obs, r, term, trunc, info = e.step(a)
        worst = max(worst, info["tilt"])
        assert not term, f"collapsed at t={k / 50:.2f}s, tilt {info['tilt']:.1f} deg"
        if trunc:
            break
    assert trunc
    # Not merely "did not trip the termination threshold": a body that drifts most of the way
    # there is still one the policy would have to spend its budget propping up.
    assert worst < 8.0, f"stayed up, but wandered to {worst:.1f} deg doing it"


def test_observation_stays_bounded(body):
    """Nothing may dominate the input scale. An unnormalised channel is invisible until the
    policy quietly learns to read it and nothing else.

    This one caught a live case: the last tail segment is light, long and barely damped, so
    +-0.3 random actuation drove `qd/tail2_pitch` to 49.6 -- eight times every other channel,
    with no exception and no NaN to notice. Hence `sensing.RATE_CLIP`.
    """
    e = CreatureEnv(body, cfg())
    obs, _ = e.reset(seed=3)
    rng = np.random.default_rng(0)
    worst = np.abs(obs)
    for _ in range(200):
        obs, r, term, trunc, _ = e.step(rng.uniform(-0.3, 0.3, e.act_dim))
        worst = np.maximum(worst, np.abs(obs))
        if term or trunc:
            obs, _ = e.reset()
    assert np.all(np.isfinite(worst))
    # The bound is the saturation limit itself plus room for the noise added after it, not a
    # number chosen to fit: a channel that can only be pinned at RATE_CLIP is still bounded.
    limit = sensing.RATE_CLIP + 5.0 * float(body.spec.obs_noise.max())
    assert worst.max() <= limit, f"channel '{body.spec.names[int(worst.argmax())]}' reached " \
                                 f"{worst.max():.1f}"


def test_rate_channels_actually_saturate(body):
    """`RATE_CLIP` must be applied to every rate-like channel, not just the joint rates.

    The vestibular angular rate and body-frame velocity are normalised by the same Froude
    scales and are just as unbounded; forgetting one of them leaves exactly the failure the
    constant exists to prevent, in the channel a falling animal relies on most.
    """
    e = CreatureEnv(body, cfg())
    e.reset(seed=0)
    s, raw = body.spec, e.vec.raw
    raw.qvel_j[:] = 1e4 / s.time                       # absurd but finite
    raw.qvel_free[:, 3:6] = 1e4 / s.time
    raw.qvel_free[:, 0:3] = 1e4 * s.speed
    out = sensing.assemble(s, raw, np.zeros((1, e.act_dim)), e.vec.command, e.vec.morph,
                           np.zeros((1, e.obs_dim)))
    for group in ("joint_rate", "vestibular"):
        v = np.abs(out[0, s.slices[group]])
        assert v.max() <= sensing.RATE_CLIP + 1e-9, f"{group} reached {v.max():.1f}"
    # ...and the gravity direction is a unit vector, so it must NOT have been touched.
    g = s.slices["vestibular"].start
    assert abs(np.linalg.norm(out[0, g:g + 3]) - 1.0) < 1e-9


def test_reward_is_dimensionless_across_morphs():
    """The same weights must mean the same trade-off on a small and a large animal.

    Standing still under a zero command should score about the same on both. If it does not,
    some term still carries units and every weight is secretly per-body.
    """
    c = cfg(init_joint_noise=0.0, init_rate_noise=0.0, stand_fraction=1.0)
    scores = []
    for scale in (0.7, 1.4):
        b = build_body(c, morph={"body_scale": scale})
        e = CreatureEnv(b, c)
        e.reset(seed=0)
        tot = 0.0
        for _ in range(50):
            _, r, term, trunc, _ = e.step(np.zeros(e.act_dim))
            tot += r
        scores.append(tot / 50)
    assert abs(scores[0] - scores[1]) < 0.15 * max(abs(s) for s in scores), \
        f"standing scored {scores[0]:.3f} vs {scores[1]:.3f} on a 0.7x and a 1.4x body"


def test_batched_matches_single_env(body):
    """The batched path is the only implementation, so it must equal itself at any width.

    A drift here would make every number measured at N=1 -- in a debugger, in a render, in a
    unit test -- a claim about a different environment than the one that trains.
    """
    c = cfg(init_joint_noise=0.0, init_rate_noise=0.0)
    out = []
    for n, workers in ((1, 1), (5, 3)):
        v = VecCreatureEnv([body] * n, c, seed=11, workers=workers)
        v.reset(seed=11)
        v.command[:] = np.array([0.4, 0.05, -0.1])       # one task for every row
        rng = np.random.default_rng(0)
        rew, tilt = [], []
        for _ in range(40):
            a = np.repeat(rng.uniform(-0.3, 0.3, (1, v.act_dim)), n, axis=0)
            _, r, _, _, info = v.step(a)
            rew.append(r.copy())
            tilt.append(info["tilt"].copy())
        v.close()
        out.append((np.array(rew), np.array(tilt)))
    # Every row of the wide batch must agree with every other, and with the single env.
    for rew, tilt in out:
        assert np.abs(rew - rew[:, :1]).max() == 0.0
        assert np.abs(tilt - tilt[:, :1]).max() == 0.0
    assert np.abs(out[0][0][:, 0] - out[1][0][:, 0]).max() == 0.0
    assert np.abs(out[0][1][:, 0] - out[1][1][:, 0]).max() == 0.0


def test_autoreset_reports_the_terminal_observation(body):
    """`info["final_obs"]` must be the last state of the finished episode, not the new one.

    Substituting the reset observation makes the value target for the final step of every
    episode the value of an unrelated state -- and training still works, just worse.
    """
    c = cfg(episode_seconds=0.4)                  # 20 steps
    v = VecCreatureEnv([body] * 3, c, seed=2, workers=2)
    v.reset(seed=2)
    a = np.zeros((3, v.act_dim))
    for _ in range(20):
        obs, r, term, trunc, info = v.step(a)
    assert np.all(trunc) and not np.any(term)
    idx = info["episode_idx"]
    assert list(idx) == [0, 1, 2]
    assert np.all(info["episode_length"] == 20)
    final = info["final_obs"][idx]
    assert np.all(np.isfinite(final))
    assert not np.allclose(final, obs[idx]), "final_obs is the reset observation"
    assert np.all(v._steps == 0), "step counter survived the auto-reset"


def test_info_describes_the_step_not_the_auto_reset(body):
    """Everything in `info` must describe the step that just happened.

    `info` hands out the env's own buffers by reference, and the auto-reset runs at the end of
    the same `step` -- so any buffer the reset also writes gets read by the caller in its
    post-reset state. This bit for real, in `command` and `sane`, and it bit in the worst
    possible place: only the rows that finished were wrong, which is precisely the set of rows
    a logger or an AMP buffer looks at, and the wrong values were a valid command and
    `sane=True`, so there was nothing to notice.
    """
    c = cfg(episode_seconds=0.4)                  # 20 steps, all three finish together
    v = VecCreatureEnv([body] * 3, c, seed=4, workers=2)
    v.reset(seed=4)
    a = np.zeros((3, v.act_dim))
    for k in range(20):
        pre = v.command.copy()
        obs, r, term, trunc, info = v.step(a)
    assert np.all(trunc)
    assert np.array_equal(info["command"], pre), \
        "info['command'] is the command the auto-reset drew, not the one that was followed"
    assert not np.array_equal(v.command, pre), "the auto-reset did not resample the command"


def test_termination_uses_the_same_tilt_as_the_acceptance_bar(body):
    """`env` and `validate` must measure tilt identically, or the policy games the gap."""
    from creaturelab.validate import trunk_tilt_of

    e = CreatureEnv(body, cfg())
    e.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(30):
        _, _, term, trunc, info = e.step(rng.uniform(-0.5, 0.5, e.act_dim))
        assert info["tilt"] == pytest.approx(trunk_tilt_of(e.data), abs=1e-9)
        if term or trunc:
            break


def test_exploded_physics_terminates_without_poisoning_the_observation(body):
    """An exploded integrator must end the episode and never emit a NaN.

    One NaN reaching the rollout buffer destroys a whole PPO update, and the traceback
    points at the optimiser rather than at the physics.

    The subtler half is that MuJoCo usually does NOT hand back a NaN: on a bad qvel it warns
    and calls `mj_resetData` itself, so the step returns a perfectly finite state that happens
    to be the default pose at the origin at rest. That is worse than a NaN, because nothing
    downstream can tell -- the animal simply teleports mid-episode and the value function
    learns the transition. So this asserts on `sane`, which reads MuJoCo's warning counter,
    and would fail against the `isfinite`-only check this test was first written for.
    """
    c = cfg()
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    v.reset(seed=0)
    v.datas[1].qvel[:] = np.nan                    # blow up env 1 only
    obs, r, term, trunc, info = v.step(np.zeros((2, v.act_dim)))
    assert not info["sane"][1] and term[1]
    assert np.all(np.isfinite(obs)), "a NaN escaped into the observation"
    assert r[1] == pytest.approx(-1.0)
    assert info["sane"][0] and not term[0], "the healthy env was affected too"
    # MuJoCo swallowed the NaN, which is exactly why the counter is the detector.
    assert np.all(np.isfinite(v.datas[1].qvel))


def test_a_recovered_env_is_sane_again_after_reset(body):
    """One blown step must not brand the env insane for the rest of the run.

    The warning counters live in `MjData` and are cumulative, so the check is against a stored
    baseline. Miss the baseline reset and every env poisoned once terminates on every step
    forever after -- episodes of length 1, a policy that never learns, and no error anywhere.
    """
    c = cfg()
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    v.reset(seed=0)
    v.datas[1].qvel[:] = np.nan
    _, _, term, _, info = v.step(np.zeros((2, v.act_dim)))
    assert term[1] and not info["sane"][1]
    for _ in range(3):
        _, _, term, trunc, info = v.step(np.zeros((2, v.act_dim)))
        assert np.all(info["sane"]), "the auto-reset env is still reading as insane"
        assert not np.any(term)


# ------------------------------------------------------------------- the command curriculum
def test_curriculum_starts_narrow_and_never_exceeds_the_declared_range(body):
    """The commanded speed is what the curriculum widens; the declared range is the ceiling."""
    c = cfg(curriculum=True, curriculum_start=0.3)
    v = VecCreatureEnv([body] * 8, c, seed=0, workers=1)
    v.reset(seed=0)
    assert v.speed_cap == pytest.approx(0.3)
    assert v.command[:, 0].max() <= 0.3 + 1e-12

    v.speed_cap = c.speed_range[1]
    v.command[:] = v.sample_commands(8)
    assert v.command[:, 0].max() <= c.speed_range[1] + 1e-12


def test_curriculum_off_uses_the_full_range_immediately(body):
    c = cfg(curriculum=False)
    v = VecCreatureEnv([body] * 4, c, seed=0, workers=1)
    assert v.speed_cap == pytest.approx(c.speed_range[1])


def test_curriculum_promotes_on_tracking_and_not_on_survival(body):
    """The bar is the *tracking* term, and choosing that over the return is load-bearing.

    The episode return is dominated by episode length -- a 20 s episode of standing still
    scores several hundred -- so a curriculum judged on return promotes a policy that survives
    well and tracks badly straight out of the speed range it can actually handle. Which is the
    exact failure the curriculum exists to prevent, arrived at from the other side.
    """
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_step=0.05,
            curriculum_margin=0.25, curriculum_window=4)
    v = VecCreatureEnv([body] * 4, c, seed=0, workers=1)
    v.reset(seed=0)
    idx = np.arange(4)

    # Long, well-rewarded episodes that tracked badly: no promotion, however big the return.
    v._steps[:] = 1000
    v._ep_track[:] = 1000 * 0.2
    v._return[:] = 5000.0
    v._advance_curriculum(idx)
    assert v.speed_cap == pytest.approx(0.3)

    # Short episodes that tracked well: promoted.
    v._steps[:] = 50
    v._ep_track[:] = 50 * 0.9
    v._return[:] = -100.0
    v._advance_curriculum(idx)
    assert v.speed_cap == pytest.approx(0.35)


def test_curriculum_needs_a_full_window_before_it_promotes(body):
    """The bug that made the first version useless, and it is not a tuning matter.

    Scoring whichever handful of envs happened to finish on the current step runs the promotion
    test dozens of times per rollout, on samples of one to five episodes, drawn from a
    population selected precisely for having ended. It took the range from 0.30 to the full
    0.80 in 82k steps -- before the animal could stand -- and so reproduced the exact standstill
    the curriculum was written to prevent.
    """
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_window=64, curriculum_margin=0.25)
    v = VecCreatureEnv([body] * 4, c, seed=0, workers=1)
    v.reset(seed=0)
    v._steps[:], v._ep_track[:] = 100, 100.0        # flawless tracking, every time
    for _ in range(15):                             # 60 episodes: one window short
        v._advance_curriculum(np.arange(4))
    assert v.speed_cap == pytest.approx(0.3), "promoted on a partial window"
    v._advance_curriculum(np.arange(4))              # the 64th
    assert v.speed_cap == pytest.approx(0.35)
    # ...and the window is cleared, so the next promotion needs another full one.
    v._advance_curriculum(np.arange(4))
    assert v.speed_cap == pytest.approx(0.35)


def test_curriculum_weights_by_step_not_by_episode(body):
    """A policy that falls over after 20 well-tracked steps must not count the same as one
    that held the command for 20 seconds."""
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_window=2, curriculum_margin=0.25)
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    v.reset(seed=0)
    # One long, badly tracked episode and one short, perfect one. Per *episode* the mean is
    # 0.75; per *step* it is (1000*0.5 + 20*1.0)/1020 = 0.51. The bar sits between the two, so
    # the weighting is the only thing deciding the outcome and the test cannot pass by accident.
    assert 0.51 < v.cur_bar < 0.75
    v._steps[:] = [1000, 20]
    v._ep_track[:] = [1000 * 0.5, 20 * 1.0]
    v._advance_curriculum(np.arange(2))
    assert v.speed_cap == pytest.approx(0.3)


def test_curriculum_only_widens(body):
    """A curriculum that also narrows oscillates, and the signal is a noisy mean over however
    many episodes happened to finish in one step."""
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_margin=0.25, curriculum_window=4)
    v = VecCreatureEnv([body] * 4, c, seed=0, workers=1)
    v.reset(seed=0)
    idx = np.arange(4)
    v._steps[:], v._ep_track[:] = 100, 100 * 0.9
    v._advance_curriculum(idx)
    assert v.speed_cap == pytest.approx(0.35)
    v._ep_track[:] = 0.0                       # a terrible batch
    v._advance_curriculum(idx)
    assert v.speed_cap == pytest.approx(0.35), "the curriculum narrowed"


def test_curriculum_stops_at_the_ceiling(body):
    c = cfg(curriculum=True, curriculum_start=0.78, curriculum_step=0.05, curriculum_window=2)
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    v.reset(seed=0)
    idx = np.arange(2)
    for _ in range(10):
        v._steps[:], v._ep_track[:] = 100, 100.0
        v._advance_curriculum(idx)
    assert v.speed_cap == pytest.approx(c.speed_range[1])


def test_an_eval_env_does_not_advance_the_curriculum(body):
    """`auto_reset=False` is how an evaluation env is built, and it is handed a fixed command
    grid it never sampled. Letting it promote would make the curriculum depend on how often
    the run was scored."""
    c = cfg(curriculum=True, curriculum_start=0.3, episode_seconds=0.1)
    v = VecCreatureEnv([body] * 2, c, seed=0, auto_reset=False, workers=1)
    v.reset(seed=0)
    for _ in range(int(0.1 * c.control_hz) + 2):
        v._ep_track[:] = v._steps * 1.0        # perfect tracking, by construction
        _, _, _, trunc, _ = v.step(np.zeros((2, v.act_dim)))
        if trunc.any():
            break
    assert v.speed_cap == pytest.approx(0.3)


def test_standstill_score_matches_a_motionless_animal(body):
    """The promotion bar is measured up from this number, so it had better be the real one.

    `_standstill_score` is a closed form -- two `erf` differences and the `stand_fraction`
    mixture -- and the whole point of it is to be exact where a guess would not be. Checked
    against the empirical mean of `r_speed` over actual sampled commands at zero velocity.
    """
    c = cfg(curriculum=True, curriculum_start=0.3)
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    for cap in (0.3, 0.5, c.speed_range[1]):
        v.speed_cap = cap
        cmd = v.sample_commands(200_000)                 # velocity is zero, so error == command
        empirical = np.exp(-(cmd[:, 0] ** 2 + cmd[:, 1] ** 2) / c.speed_tol ** 2).mean()
        assert v._standstill_score() == pytest.approx(empirical, abs=3e-3)


def test_the_promotion_bar_tracks_the_cap(body):
    """A fixed absolute bar asks a different question at every width.

    Parked scores 0.64 when commands run to 0.3 Froude and 0.31 when they run to 0.8, so a
    constant 0.75 means "a little better than standing" at the start and "near perfect" later --
    backwards, since wider commands are the harder task. The bar is a fixed fraction of the gap
    from parked to perfect instead, which is the same claim about the policy at every width.
    """
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_margin=0.25)
    v = VecCreatureEnv([body] * 2, c, seed=0, workers=1)
    narrow_stand, narrow_bar = v._standstill_score(), v.cur_bar
    v.speed_cap = c.speed_range[1]                       # the setter must move the bar with it
    wide_stand, wide_bar = v._standstill_score(), v.cur_bar

    assert wide_stand < narrow_stand, "a wider command range is harder to stand through"
    assert wide_bar < narrow_bar, "the bar did not follow the cap"
    for stand, bar in ((narrow_stand, narrow_bar), (wide_stand, wide_bar)):
        assert bar == pytest.approx(stand + 0.25 * (1.0 - stand))
        assert stand < bar < 1.0


def test_curriculum_advances_from_a_real_rollout(body):
    """End to end through `step`, because the promotion has to happen *before* the auto-reset
    zeroes the very counters it is judged on."""
    c = cfg(curriculum=True, curriculum_start=0.3, curriculum_step=0.05,
            curriculum_margin=0.25, curriculum_window=4, episode_seconds=0.2)
    v = VecCreatureEnv([body] * 4, c, seed=0, workers=1)
    v.reset(seed=0)
    v.command[:] = 0.0                         # standing still tracks a zero command perfectly
    a = np.zeros((4, v.act_dim))
    for _ in range(int(0.2 * c.control_hz) + 1):
        v.step(a)
    assert v.speed_cap > 0.3, "a full episode of perfect tracking earned no promotion"
