"""PPO, written out rather than imported.

The obvious move is Stable-Baselines3, and it is the wrong one *here* specifically. P2 wants
an AMP discriminator sharing the rollout, and P10 wants ASE latents resampled mid-episode and
carried into the observation. Both of those reach straight through the parts SB3 owns -- its
rollout buffer, its vec-env contract, its `learn()` loop -- so adopting it means adopting a
frame we would immediately be subclassing our way out of. The algorithm underneath is a few
hundred lines; the integration surface is the expensive part, so keep the cheap thing and own
the expensive one.

The three details that actually decide whether PPO works, all of which are easy to get subtly
wrong and none of which announce themselves:

* **Truncation is not termination.** An episode cut off at the time limit has a future, and
  its last value target must bootstrap `V(s_T)`; an episode that ended because the animal fell
  over does not. Conflating them teaches the critic that standing at t = 20 s is worth exactly
  as much as lying on your side, which is a bias that grows with how good the policy gets.
  `VecCreatureEnv` already separates `term` from `trunc` and stashes the pre-reset observation
  in `info["final_obs"]` precisely so this is possible -- see its docstring.
* **Observation normalisation is part of the policy.** A checkpoint that saves the network and
  not the running statistics restores a policy being fed a different input distribution than
  it was trained on. It does not crash; it just performs like an untrained net, and the
  natural conclusion is that the checkpoint format is broken rather than incomplete.
* **The GAE recursion must not cross an episode boundary**, in *either* direction. With
  auto-reset the buffer holds interleaved episodes, so the `(1 - done)` factor is what keeps a
  new episode's advantage from leaking backwards into the previous one's last step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class PPOConfig:
    # --- rollout ------------------------------------------------------------------------
    num_envs: int = 64
    horizon: int = 64                  # control steps per env per update -> 4096 sample batch
    total_steps: int = 20_000_000      # env steps, the budget the run is quoted in

    # --- objective ----------------------------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    # Value clipping is deliberately absent. It is in the original code release but not the
    # paper, and the ablation literature finds it neutral-to-harmful; leaving it out removes a
    # knob that would otherwise have to be justified.
    entropy_coef: float = 0.0025
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- optimisation ---------------------------------------------------------------------
    lr: float = 3e-4
    epochs: int = 5
    minibatches: int = 4
    target_kl: float = 0.02            # early-stop the epoch loop; see `update`

    # --- network ----------------------------------------------------------------------------
    hidden: tuple[int, ...] = (256, 256)
    # exp(-1.6) = 0.20 of the action range. The usual continuous-control default is -0.5, and
    # it is wrong here by enough to stop the run learning at all -- because an action of 1.0 on
    # this creature is the *full* motor gear, and that gear was sized from the joint's static
    # holding torque, so sigma=0.6 is a 60%-of-hold-torque white-noise shove on each of 31
    # joints, 50 times a second. Measured on the untrained policy (32 envs, mean over the
    # rollout):
    #
    #:    sigma   episode      cost of      mean
    #:            length      transport    reward
    #:    0.000    20.00 s        0.00      +0.657     <- passive stance, holds the full episode
    #:    0.100     2.90 s        0.58      +0.524
    #:    0.202     1.98 s        1.96      +0.446
    #:    0.400     0.92 s       16.65      +0.235
    #:    0.607     0.72 s      106.35      -0.035     <- the -0.5 default
    #
    # Two things go wrong at once at 0.607, and they compound. Episodes end in 0.72 s, so a
    # 64-step horizon contains almost no post-transient behaviour to learn from; and `c_energy`
    # reaches 106, which against `w_energy=0.02` is a 2.1/step penalty that swamps the 1.3 the
    # tracking terms can pay at their maximum -- the reward is net *negative* for existing. A
    # 400k-step run at -0.5 has its evaluation return fall 54 -> 19; at -1.6 it reaches 579
    # with every animal surviving the full 20 s. -1.2 also works (474); -2.3 is worse (412) for
    # the opposite reason, that KL scales as (delta mu / sigma)^2, so a narrow policy trips
    # `target_kl` after a single epoch every update and throttles its own optimisation.
    init_log_std: float = -1.6

    # --- plumbing -----------------------------------------------------------------------------
    device: str = "cpu"                # see `pick_device` for why this is not "cuda"
    seed: int = 0
    obs_clip: float = 10.0             # after normalisation, in standard deviations


def pick_device(requested: str = "auto") -> str:
    """CPU unless told otherwise, and that is a measurement rather than an oversight.

    The nets here are 135->256->256->31. Acting costs one forward pass on a (num_envs, 135)
    batch per control step, which is far too small to amortise a host-device round trip, and
    the environment -- MuJoCo, on the CPU, at ~2800 env-steps/s -- is the bottleneck by a wide
    margin either way. Moving the policy to the GPU therefore adds transfer latency to the
    critical path to speed up something that was never the constraint.
    """
    if requested != "auto":
        return requested
    return "cpu"


class RunningNorm:
    """Welford mean/variance over observations, checkpointed with the policy.

    Batched, because it is fed the whole rollout at once rather than a sample at a time; the
    parallel (Chan et al.) update is used so that is exact rather than an approximation.
    """

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, self.mean.size)
        n = x.shape[0]
        if n == 0:
            return
        bm, bv = x.mean(axis=0), x.var(axis=0)
        delta, tot = bm - self.mean, self.count + n
        self.mean += delta * (n / tot)
        m_a = self.var * self.count
        m_b = bv * n
        self.var = (m_a + m_b + delta ** 2 * (self.count * n / tot)) / tot
        self.count = tot

    def __call__(self, x: np.ndarray, clip: float) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean[:] = d["mean"]
        self.var[:] = d["var"]
        self.count = d["count"]


def _mlp(sizes, out_dim, out_gain):
    """Orthogonal-init MLP with tanh activations.

    Tanh rather than ReLU, and a small final-layer gain, because both matter more than they
    look: the policy head is initialised near zero so the first actions are ~the mean of the
    action range rather than a random lunge, and a quadruped that spends its first thousand
    episodes falling over from the initial policy learns termination and nothing else.
    """
    import torch
    import torch.nn as nn

    layers, prev = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    nn.init.zeros_(net[-1].bias)
    return net


class ActorCritic:
    """Gaussian policy and value function, as two separate trunks.

    Separate rather than shared: with a shared trunk the value loss -- whose gradients are
    much larger early, when returns are unnormalised and the critic is wrong about everything
    -- dominates the representation the policy is trying to learn from. Sharing buys parameter
    efficiency this problem does not need at 135 inputs.

    `log_std` is a state-independent learnable parameter rather than a network output. A
    state-dependent std lets the policy drive exploration to zero in exactly the states it is
    already confident about, which is the mechanism behind premature convergence to a
    stand-still on a locomotion task that pays for standing.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pi = _mlp((obs_dim, *cfg.hidden), act_dim, 0.01).to(self.device)
        self.vf = _mlp((obs_dim, *cfg.hidden), 1, 1.0).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), cfg.init_log_std, device=self.device))
        self.params = list(self.pi.parameters()) + list(self.vf.parameters()) + [self.log_std]
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)

    def _dist(self, obs):
        import torch
        mean = self.pi(obs)
        return torch.distributions.Normal(mean, torch.exp(self.log_std))

    def act(self, obs_np: np.ndarray):
        """Sample actions for a batch of observations. Returns numpy, no graph."""
        import torch
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            d = self._dist(obs)
            a = d.sample()
            return (a.cpu().numpy(),
                    d.log_prob(a).sum(-1).cpu().numpy(),
                    self.vf(obs).squeeze(-1).cpu().numpy())

    def value(self, obs_np: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            return self.vf(obs).squeeze(-1).cpu().numpy()

    def mean_action(self, obs_np: np.ndarray) -> np.ndarray:
        """The deterministic policy, for evaluation and for the viewer."""
        import torch
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            return self.pi(obs).cpu().numpy()

    def state_dict(self) -> dict:
        return {"pi": self.pi.state_dict(), "vf": self.vf.state_dict(),
                "log_std": self.log_std.detach().cpu(), "opt": self.opt.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        import torch
        self.pi.load_state_dict(d["pi"])
        self.vf.load_state_dict(d["vf"])
        with torch.no_grad():
            self.log_std.copy_(d["log_std"].to(self.device))
        self.opt.load_state_dict(d["opt"])


@dataclass
class Rollout:
    obs: np.ndarray                 # (T, N, obs_dim), normalised
    act: np.ndarray                 # (T, N, act_dim)
    logp: np.ndarray                # (T, N)
    val: np.ndarray                 # (T, N)
    rew: np.ndarray                 # (T, N)
    done: np.ndarray                # (T, N) term | trunc -- cuts the GAE recursion
    next_val: np.ndarray            # (T, N) V(s_{t+1}), already 0 wherever the episode ended
    stats: dict = field(default_factory=dict)


def collect(env, ac: ActorCritic, norm: RunningNorm, cfg: PPOConfig,
            obs: np.ndarray) -> tuple[Rollout, np.ndarray]:
    """Run `cfg.horizon` control steps. Returns the rollout and the observation to carry on from.

    `obs` in and out is the *raw* observation; normalisation happens here so that the running
    statistics are updated exactly once per sample, on the samples actually used.
    """
    T, N = cfg.horizon, env.n
    o = np.empty((T, N, env.obs_dim), dtype=np.float32)
    a = np.empty((T, N, env.act_dim), dtype=np.float32)
    lp = np.empty((T, N), dtype=np.float32)
    v = np.empty((T, N), dtype=np.float32)
    r = np.empty((T, N), dtype=np.float32)
    dn = np.empty((T, N), dtype=bool)
    nv = np.zeros((T, N), dtype=np.float32)
    ep_ret, ep_len, tilts, speeds, r_speeds, cots = [], [], [], [], [], []

    for t in range(T):
        o[t] = norm(obs, cfg.obs_clip)
        a[t], lp[t], v[t] = ac.act(o[t])
        # The env clips to [-1, 1] itself. The *unclipped* sample is what is stored, because
        # the log-probability recorded here has to be the log-probability of the thing that
        # was actually sampled or the importance ratio is wrong.
        obs, rew, term, trunc, info = env.step(a[t])
        r[t] = rew
        done = term | trunc
        dn[t] = done

        idx = np.nonzero(done)[0]
        if idx.size:
            # Bootstrap the truncated episodes off their own final observation, and only
            # those: `term` rows keep the zero they were initialised with. `info["final_obs"]`
            # is a persistent buffer whose non-`done` rows are stale by contract, so it is
            # read only at `idx`.
            cut = idx[trunc[idx]]
            if cut.size:
                nv[t, cut] = ac.value(norm(info["final_obs"][cut], cfg.obs_clip))
            ep_ret.append(info["episode_return"].copy())
            ep_len.append(info["episode_length"].copy())
        # `sane` masks the diagnostics only -- the *training* signal for an exploded env is
        # already well defined (`env.step` pays it -1 and terminates it). What is not defined
        # is its measured speed, which is read out of a diverged `qvel` and came back as 806
        # Froude units in one rollout, dragging the logged mean far enough that the run's own
        # progress trace became unreadable. An unphysical number does not belong in a mean.
        ok = info["sane"]
        tilts.append(info["tilt"][ok])
        speeds.append(info["speed"][ok])
        r_speeds.append(info["r_speed"][ok])
        cots.append(info["cot"][ok])

    # Steps that did not end carry the value of the state the env actually went on to, which
    # for t < T-1 is simply the next row's value, already computed.
    nv[:-1][~dn[:-1]] = v[1:][~dn[:-1]]
    tail = ~dn[-1]
    if tail.any():
        nv[-1, tail] = ac.value(norm(obs, cfg.obs_clip)[tail])

    norm.update(obs.reshape(-1, env.obs_dim))

    def m(rows) -> float:
        # Concatenated, not stacked: the `sane` mask above makes the rows ragged.
        flat = np.concatenate(rows) if rows else np.zeros(0)
        return float(flat.mean()) if flat.size else float("nan")

    stats = {
        "tilt": m(tilts), "speed": m(speeds), "r_speed": m(r_speeds), "cot": m(cots),
        "reward": float(r.mean()), "done_frac": float(dn.mean()),
        "insane_frac": float(1.0 - sum(x.size for x in tilts) / max(1, T * N)),
    }
    if ep_ret:
        stats["ep_return"] = float(np.mean(np.concatenate(ep_ret)))
        stats["ep_len"] = float(np.mean(np.concatenate(ep_len)))
        stats["ep_count"] = int(sum(x.size for x in ep_ret))
    return Rollout(o, a, lp, v, r, dn, nv, stats), obs


def advantages(ro: Rollout, cfg: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda). Returns (advantage, value target), both (T, N).

    `next_val` already encodes the termination rule -- zero where the episode genuinely ended,
    `V(final_obs)` where it was merely cut off -- so the delta below needs no `(1 - term)`
    factor. The `(1 - done)` on the recursion is a separate thing and is still required: it is
    what stops an advantage propagating backwards across a reset into an unrelated episode.
    """
    adv = np.zeros_like(ro.rew)
    run = np.zeros(ro.rew.shape[1], dtype=np.float32)
    for t in range(ro.rew.shape[0] - 1, -1, -1):
        delta = ro.rew[t] + cfg.gamma * ro.next_val[t] - ro.val[t]
        run = delta + cfg.gamma * cfg.gae_lambda * (~ro.done[t]) * run
        adv[t] = run
    return adv, adv + ro.val


def update(ac: ActorCritic, ro: Rollout, cfg: PPOConfig) -> dict:
    """Clipped-surrogate epochs over the rollout. Returns diagnostics.

    Stops early once the sampled KL passes `target_kl`. That guard is not optional decoration:
    the clip alone bounds the ratio per sample but not the total movement over several epochs,
    and a policy that jumps too far in one update on this task does not gently regress -- it
    falls over, which truncates every episode, which starves the next rollout of exactly the
    states it needed to recover from.
    """
    import torch

    T, N = ro.rew.shape
    adv, ret = advantages(ro, cfg)
    dev = ac.device
    b_obs = torch.as_tensor(ro.obs.reshape(T * N, -1), device=dev)
    b_act = torch.as_tensor(ro.act.reshape(T * N, -1), device=dev)
    b_logp = torch.as_tensor(ro.logp.reshape(-1), device=dev)
    b_adv = torch.as_tensor(adv.reshape(-1), device=dev)
    b_ret = torch.as_tensor(ret.reshape(-1), device=dev)

    n = T * N
    size = max(1, n // cfg.minibatches)
    idx = np.arange(n)
    rng = np.random.default_rng(cfg.seed)
    kl = 0.0
    logs = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "epochs": 0}
    stop = False

    for ep in range(cfg.epochs):
        rng.shuffle(idx)
        for s in range(0, n, size):
            mb = torch.as_tensor(idx[s:s + size], device=dev)
            d = ac._dist(b_obs[mb])
            logp = d.log_prob(b_act[mb]).sum(-1)
            ratio = torch.exp(logp - b_logp[mb])

            # Normalised per minibatch, which is what makes the clip threshold mean the same
            # thing regardless of the reward scale the run happens to be at.
            a = b_adv[mb]
            a = (a - a.mean()) / (a.std() + 1e-8)

            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * a
            pi_loss = -torch.min(unclipped, clipped).mean()
            v_loss = ((ac.vf(b_obs[mb]).squeeze(-1) - b_ret[mb]) ** 2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent

            ac.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.params, cfg.max_grad_norm)
            ac.opt.step()

            # Inside `no_grad` only so that `float()` on a graph-attached tensor does not warn;
            # the values are read after `opt.step()` and the graph is already spent.
            with torch.no_grad():
                logs["clip_frac"] += float(
                    ((ratio - 1).abs() > cfg.clip_ratio).float().mean())
                logs["pi_loss"] += float(pi_loss)
                logs["v_loss"] += float(v_loss)
                logs["entropy"] += float(ent)
        logs["epochs"] = ep + 1

        # KL over the *whole* rollout, on a fresh pass, once per epoch. Both halves of that
        # matter and both were wrong at first. Reading it inside the minibatch loop measures
        # the policy on the very samples whose gradient it just took a step along, which
        # overstates the move; and keeping only the last minibatch's value estimates the
        # epoch's KL from n/minibatches samples, which at this batch size is noise of the same
        # size as `target_kl`. Together they early-stopped after one epoch on nearly every
        # update -- sampled KL read 0.05-0.11 against a 0.02 target while the true full-batch
        # figure was well inside it, so the run was quietly doing a fifth of the optimisation
        # it was configured for. One extra forward pass per epoch buys the honest number.
        with torch.no_grad():
            # Schulman's low-variance, always-positive estimator; the naive
            # (logp_old - logp).mean() is unbiased but can go negative, which reads as
            # "the policy moved backwards" rather than as the sampling noise it is.
            lr_ = ac._dist(b_obs).log_prob(b_act).sum(-1) - b_logp
            kl = float(((torch.exp(lr_) - 1) - lr_).mean())
        if kl > cfg.target_kl:
            stop = True
            break

    steps = max(1, logs["epochs"] * int(np.ceil(n / size)))
    for k in ("pi_loss", "v_loss", "entropy", "clip_frac"):
        logs[k] /= steps
    logs["kl"] = kl
    logs["early_stop"] = stop
    logs["log_std"] = float(ac.log_std.detach().mean())
    return logs


# ------------------------------------------------------------------------------ checkpoints
def save(path, ac: ActorCritic, norm: RunningNorm, cfg: PPOConfig,
         step: int, extra: dict | None = None) -> None:
    """Everything needed to resume *or* to evaluate, in one file.

    The observation normaliser and the optimiser state are both in here on purpose. Omitting
    the normaliser produces a checkpoint that loads cleanly and behaves like an untrained
    network; omitting the optimiser produces a resume that silently restarts Adam's moment
    estimates and puts a visible dent in the learning curve at exactly the point where the run
    was interrupted, which is very easy to misread as instability in the algorithm.
    """
    import torch

    torch.save({"version": 1, "step": step, "cfg": asdict(cfg),
                "ac": ac.state_dict(), "norm": norm.state_dict(),
                "extra": extra or {}}, path)


def load(path, ac: ActorCritic, norm: RunningNorm) -> tuple[int, dict]:
    import torch

    d = torch.load(path, map_location=ac.device, weights_only=False)
    ac.load_state_dict(d["ac"])
    norm.load_state_dict(d["norm"])
    return int(d["step"]), d.get("extra", {})
