"""Tests for the PPO implementation itself, separate from whether it learns to walk.

Writing PPO rather than importing SB3 (see `ppo.py`'s docstring for why) means owning the
handful of details that are easy to get wrong and impossible to notice: a run with any of
them broken still trains, still produces a falling learning curve, and simply arrives
somewhere worse than it should have. None of them announce themselves, so each one gets an
assertion here rather than a comment saying it was thought about.

Deliberately no MuJoCo in this file. Everything below runs against a `_FakeVec` whose
dynamics are one line, so the properties under test are visible in closed form and the suite
stays fast enough to run on every change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creaturelab import ppo                                     # noqa: E402


# --------------------------------------------------------------------------------- fixtures
class _FakeVec:
    """A vector env with the same contract as `VecCreatureEnv` and no physics.

    The observation is a scalar counter broadcast over `obs_dim`, the reward is 1 per step,
    and episodes end on a schedule the test dictates. That is enough to exercise every path
    in `collect` -- terminate, truncate, auto-reset, final-obs bootstrap -- while keeping the
    correct answer something a person can work out on paper.
    """

    def __init__(self, n=4, obs_dim=3, act_dim=2, term_at=None, trunc_at=None):
        self.n, self.obs_dim, self.act_dim = n, obs_dim, act_dim
        self.term_at = term_at or {}          # {step index: [env ids]}
        self.trunc_at = trunc_at or {}
        self.t = 0
        self._c = np.zeros(n)
        self.final_obs = np.zeros((n, obs_dim))

    def _obs(self):
        return np.repeat(self._c[:, None], self.obs_dim, axis=1).astype(np.float64)

    def reset(self, seed=None):
        self.t, self._c = 0, np.zeros(self.n)
        return self._obs(), {}

    def step(self, a):
        self._c += 1
        obs = self._obs()
        rew = np.ones(self.n)
        term = np.zeros(self.n, dtype=bool)
        trunc = np.zeros(self.n, dtype=bool)
        term[self.term_at.get(self.t, [])] = True
        trunc[self.trunc_at.get(self.t, [])] = True
        self.t += 1
        done = term | trunc
        info = {"tilt": np.zeros(self.n), "speed": np.zeros(self.n),
                "r_speed": np.zeros(self.n), "cot": np.zeros(self.n),
                "sane": np.ones(self.n, dtype=bool)}
        if done.any():
            info["episode_return"] = np.zeros(done.sum())
            info["episode_length"] = np.zeros(done.sum())
            info["final_obs"] = obs.copy()
            self.final_obs = obs.copy()
            self._c[done] = 0.0                       # auto-reset
            obs = self._obs()
        return obs, rew, term, trunc, info

    def close(self):
        pass


def _cfg(**kw):
    base = dict(num_envs=4, horizon=8, epochs=2, minibatches=2, device="cpu",
                hidden=(8, 8), total_steps=1000)
    base.update(kw)
    return ppo.PPOConfig(**base)


def _rollout(**env_kw):
    cfg = _cfg()
    env = _FakeVec(n=cfg.num_envs, **env_kw)
    ac = ppo.ActorCritic(env.obs_dim, env.act_dim, cfg)
    norm = ppo.RunningNorm(env.obs_dim)
    obs, _ = env.reset()
    ro, _ = ppo.collect(env, ac, norm, cfg, obs)
    return ro, ac, norm, cfg


# ------------------------------------------------------------------ truncation != termination
def test_termination_does_not_bootstrap_but_truncation_does():
    """The single most consequential line in the file, and the one with no symptom.

    A terminal state has, by definition, no future: the correct target is the reward alone.
    A *truncated* state has a perfectly good future that the rollout simply stopped watching,
    and dropping its value teaches the critic that surviving to the time limit is worth as
    little as falling over. On a 20 s episode at 50 Hz that is a lie told at the end of every
    successful episode and never at the end of a failed one, so it biases the policy against
    exactly the behaviour the task is trying to produce -- while the loss still falls and the
    return still climbs, just to a worse place.
    """
    ro, *_ = _rollout(term_at={3: [0]}, trunc_at={3: [1]})
    assert ro.done[3, 0] and ro.done[3, 1]
    assert ro.next_val[3, 0] == 0.0, "a terminated episode must not bootstrap"
    assert ro.next_val[3, 1] != 0.0, "a truncated episode must bootstrap off its final obs"


def test_next_val_of_a_live_step_is_the_next_rows_value():
    """No wasted forward pass, and no off-by-one in the one place it would be invisible."""
    ro, *_ = _rollout()
    assert np.allclose(ro.next_val[:-1], ro.val[1:], atol=1e-6)


def test_final_obs_is_read_only_where_the_episode_ended():
    """`final_obs` is a persistent buffer; its non-done rows are stale by contract.

    Bootstrapping from a stale row would use the terminal observation of some *earlier*
    episode of a different env, which is not a wrong number so much as a meaningless one.
    Env 2 never ends here, so its bootstrap must come from the live path, not the buffer.
    """
    ro, *_ = _rollout(trunc_at={2: [0]})
    assert not ro.done[2, 2]
    assert ro.next_val[2, 2] == pytest.approx(ro.val[3, 2], abs=1e-6)


# ----------------------------------------------------------------------------------- GAE
def test_gae_matches_the_closed_form_on_an_uninterrupted_stretch():
    cfg = _cfg(gamma=0.9, gae_lambda=0.8)
    T, N = 5, 1
    rng = np.random.default_rng(0)
    val = rng.normal(size=(T, N)).astype(np.float32)
    ro = ppo.Rollout(obs=np.zeros((T, N, 1), np.float32), act=np.zeros((T, N, 1), np.float32),
                     logp=np.zeros((T, N), np.float32), val=val,
                     rew=np.ones((T, N), np.float32), done=np.zeros((T, N), bool),
                     next_val=np.concatenate([val[1:], np.zeros((1, N), np.float32)]))
    adv, ret = ppo.advantages(ro, cfg)
    delta = ro.rew + cfg.gamma * ro.next_val - ro.val
    want = np.zeros_like(adv)
    run = 0.0
    for t in range(T - 1, -1, -1):
        run = delta[t] + cfg.gamma * cfg.gae_lambda * run
        want[t] = run
    assert np.allclose(adv, want, atol=1e-5)
    assert np.allclose(ret, adv + val, atol=1e-5)


def test_advantage_does_not_leak_backwards_across_an_episode_boundary():
    """The `(1 - done)` on the recursion, which is a *different* thing from the bootstrap.

    With auto-reset, row t+1 of an env that ended at t belongs to a brand new episode. Without
    the gate, that episode's TD errors flow backwards into the previous one, so a policy is
    credited for a reward collected after it had already fallen over -- and with `done_frac`
    around 3% per step at the start of training, that is not a rare edge case, it is most of
    the rollout.
    """
    cfg = _cfg(gamma=0.99, gae_lambda=0.95)
    T, N = 4, 1
    done = np.zeros((T, N), bool)
    done[1] = True
    ro = ppo.Rollout(obs=np.zeros((T, N, 1), np.float32), act=np.zeros((T, N, 1), np.float32),
                     logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32),
                     rew=np.array([[0.0], [0.0], [1.0], [1.0]], np.float32),
                     done=done, next_val=np.zeros((T, N), np.float32))
    adv, _ = ppo.advantages(ro, cfg)
    # Steps 2 and 3 pay 1 apiece; steps 0 and 1 are in the episode that ended at step 1 and
    # collected nothing, so their advantage is exactly zero rather than merely small.
    assert adv[1, 0] == 0.0
    assert adv[0, 0] == 0.0
    assert adv[2, 0] > 0.9


# ------------------------------------------------------------------------ observation norm
def test_running_norm_matches_a_batch_computation():
    """Chan's parallel Welford, checked against the thing it is an online version of.

    Worth a test because the failure is quiet: a naive incremental variance drifts slowly, so
    the normaliser stays plausible for thousands of updates and is wrong by then in a way no
    training curve distinguishes from a bad learning rate.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(3.0, 7.0, size=(1000, 5))
    n = ppo.RunningNorm(5)
    for chunk in np.array_split(data, 17):
        n.update(chunk)
    assert np.allclose(n.mean, data.mean(axis=0), atol=1e-8)
    assert np.allclose(n.var, data.var(axis=0), atol=1e-6)
    assert n.count == pytest.approx(1000, abs=1e-3)   # seeded with an epsilon, not with 0


def test_running_norm_clips_in_standard_deviations():
    n = ppo.RunningNorm(2)
    n.update(np.random.default_rng(2).normal(0.0, 1.0, size=(4000, 2)))
    out = n(np.array([[0.0, 1e6]]), clip=10.0)
    assert abs(out[0, 0]) < 0.1
    assert out[0, 1] == 10.0


# --------------------------------------------------------------------------------- update
def test_update_moves_the_policy_towards_the_better_of_two_actions():
    """The surrogate's sign, which no amount of staring at the loss expression confirms.

    Setting it up takes more care than it looks, and the first attempt here was wrong in a way
    worth keeping a note about: it gave *every* sample the same action and the same positive
    reward and asserted the mean must rise. It does not, and the implementation is right to
    refuse. `update` normalises the advantage within each minibatch, so a uniformly positive
    reward becomes a mean-zero advantage and half the samples end up pushing the policy *down*.
    PPO only ever learns "better than the rest of this batch"; there is no such thing as a
    batch where everything improves. So the test has to offer a choice.

    One constant observation, so the policy has a single scalar to move and cannot instead
    learn to tell the two groups of samples apart by their inputs.
    """
    import torch
    cfg = _cfg(epochs=8, minibatches=1, target_kl=1e9, lr=3e-3)
    ac = ppo.ActorCritic(2, 1, cfg)
    T, N = 4, 4
    obs = np.ones((T, N, 2), dtype=np.float32)
    act = np.where(np.arange(N) < N // 2, 1.0, -1.0)
    act = np.broadcast_to(act[None, :, None], (T, N, 1)).astype(np.float32).copy()
    rew = np.broadcast_to((np.arange(N) < N // 2).astype(np.float32)[None, :],
                          (T, N)).copy()
    with torch.no_grad():
        d = ac._dist(torch.as_tensor(obs.reshape(-1, 2)))
        logp = d.log_prob(torch.as_tensor(act.reshape(-1, 1))).sum(-1).numpy()
    before = float(ac.mean_action(obs.reshape(-1, 2)).mean())
    ro = ppo.Rollout(obs=obs, act=act, logp=logp.reshape(T, N).astype(np.float32),
                     val=np.zeros((T, N), np.float32), rew=rew,
                     # Every step its own episode, so the advantage is exactly the reward and
                     # the assertion does not depend on the GAE recursion as well.
                     done=np.ones((T, N), bool), next_val=np.zeros((T, N), np.float32))
    ppo.update(ac, ro, cfg)
    after = float(ac.mean_action(obs.reshape(-1, 2)).mean())
    assert after > before + 1e-3, "action +1 paid and -1 did not; the mean must move up"


def test_early_stop_fires_on_kl_and_is_reported():
    cfg = _cfg(epochs=8, minibatches=1, target_kl=1e-9, lr=1e-2)
    ro, ac, _, _ = _rollout()
    logs = ppo.update(ac, ro, cfg)
    assert logs["early_stop"] and logs["epochs"] == 1


def test_kl_is_measured_on_the_whole_batch_and_is_non_negative():
    """Schulman's estimator, and the reason it replaced a per-minibatch read.

    The per-minibatch version measured the policy on the very samples whose gradient it had
    just stepped along and kept only the last minibatch's number, so at 128 samples it read
    0.05-0.11 against a 0.02 target and early-stopped after one epoch on almost every update.
    The run was quietly doing a fifth of the optimisation it was configured for.
    """
    cfg = _cfg(epochs=3, minibatches=4, target_kl=1e9)
    ro, ac, _, _ = _rollout()
    logs = ppo.update(ac, ro, cfg)
    assert logs["epochs"] == 3
    assert logs["kl"] >= 0.0                 # the naive estimator can go negative; this cannot


# ----------------------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip_restores_the_normaliser_too(tmp_path):
    """A checkpoint without the observation statistics loads cleanly and behaves untrained.

    That is the whole failure: `torch.load` succeeds, the weights are bit-identical, and the
    policy is fed observations in units it has never seen. `--eval best.pt` then reports a
    number that has nothing to do with the number `best.pt` was selected on.
    """
    cfg = _cfg()
    ac = ppo.ActorCritic(6, 3, cfg)
    norm = ppo.RunningNorm(6)
    norm.update(np.random.default_rng(4).normal(5.0, 2.0, size=(500, 6)))
    probe = np.random.default_rng(5).normal(size=(10, 6))
    want = ac.mean_action(norm(probe, cfg.obs_clip))

    p = tmp_path / "ck.pt"
    ppo.save(p, ac, norm, cfg, step=1234, extra={"best": 9.5})

    ac2 = ppo.ActorCritic(6, 3, cfg)
    norm2 = ppo.RunningNorm(6)
    assert not np.allclose(ac2.mean_action(norm2(probe, cfg.obs_clip)), want)
    step, extra = ppo.load(p, ac2, norm2)
    assert step == 1234 and extra["best"] == 9.5
    assert np.allclose(norm2.mean, norm.mean) and norm2.count == norm.count
    assert np.allclose(ac2.mean_action(norm2(probe, cfg.obs_clip)), want, atol=1e-6)


def test_checkpoint_carries_optimiser_state():
    """Resuming with a fresh Adam puts a dent in the curve exactly where the run resumed,
    which is very easy to misread as instability in the algorithm rather than in the resume."""
    cfg = _cfg()
    ac = ppo.ActorCritic(4, 2, cfg)
    ro, _, _, _ = _rollout()
    ro = ppo.Rollout(obs=np.random.default_rng(6).normal(size=(4, 4, 4)).astype(np.float32),
                     act=np.zeros((4, 4, 2), np.float32), logp=np.zeros((4, 4), np.float32),
                     val=np.zeros((4, 4), np.float32), rew=np.ones((4, 4), np.float32),
                     done=np.zeros((4, 4), bool), next_val=np.zeros((4, 4), np.float32))
    ppo.update(ac, ro, cfg)
    d = ac.state_dict()
    assert d["opt"]["state"], "Adam moments are empty -- nothing would be restored"


# --------------------------------------------------------------------------------- plumbing
def test_collect_returns_the_shapes_the_update_expects():
    cfg = _cfg()
    ro, *_ = _rollout()
    assert ro.obs.shape == (cfg.horizon, cfg.num_envs, 3)
    assert ro.act.shape == (cfg.horizon, cfg.num_envs, 2)
    for f in (ro.logp, ro.val, ro.rew, ro.done, ro.next_val):
        assert f.shape == (cfg.horizon, cfg.num_envs)


def test_insane_steps_are_excluded_from_the_logged_means():
    """One diverged env read 806 Froude units of speed and made the run's own progress trace
    unreadable. The training signal for that step is well defined (-1 and terminate); its
    measured speed is not, and an unphysical number does not belong in a mean."""
    cfg = _cfg()
    env = _FakeVec(n=cfg.num_envs, obs_dim=3, act_dim=2)
    real_step = env.step

    def poisoned(a):
        obs, rew, term, trunc, info = real_step(a)
        info["speed"] = info["speed"].copy()
        info["speed"][0] = 1e6
        info["sane"] = info["sane"].copy()
        info["sane"][0] = False
        return obs, rew, term, trunc, info

    env.step = poisoned
    ac = ppo.ActorCritic(3, 2, cfg)
    norm = ppo.RunningNorm(3)
    obs, _ = env.reset()
    ro, _ = ppo.collect(env, ac, norm, cfg, obs)
    assert ro.stats["speed"] == pytest.approx(0.0)
    assert ro.stats["insane_frac"] == pytest.approx(0.25)


def test_pick_device_is_cpu_unless_asked_otherwise():
    assert ppo.pick_device("auto") == "cpu"
    assert ppo.pick_device("cuda") == "cuda"
