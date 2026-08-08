"""Train a locomotion policy, headless, resumably.

    python tools/train.py --steps 20e6 --out runs/canis
    python tools/train.py --resume runs/canis/latest.pt          # picks up mid-run
    python tools/train.py --eval runs/canis/best.pt --view       # watch it

Checkpoints are written on a wall-clock interval rather than an update count, because the
thing they exist to survive is a session ending, and sessions end in minutes rather than in
updates. `latest.pt` is rewritten in place for resuming; `best.pt` tracks the best evaluated
return so a late collapse cannot destroy the run's best policy.

Evaluation is deterministic (the distribution mean, not a sample) and uses a *fixed* command
set, so consecutive evaluations differ only by the policy. Scoring on the training rollout
instead would mix policy improvement with whatever commands the sampler happened to draw, and
that noise is the same order as the signal early on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                        # noqa: E402

from creaturelab import env as envmod                     # noqa: E402
from creaturelab import ppo                               # noqa: E402


def build_envs(cfg: envmod.EnvConfig, n: int, seed: int, auto_reset: bool = True):
    """One compiled body shared by every env.

    Sharing is safe and deliberate: `Body` holds the `MjModel` and the derived constants, all
    of which are read-only during a step, while the per-env mutable state is `MjData`, which
    `VecCreatureEnv` allocates one of per env. Compiling 64 identical models instead would
    cost 64x the tune pass (~1.1 s each) to produce 64 identical results. P4 is where the
    bodies start differing, and that is a list of distinct `Body` objects at this same call.
    """
    body = envmod.build_body(cfg)
    return envmod.VecCreatureEnv([body] * n, cfg, seed=seed, auto_reset=auto_reset)


def evaluate(env, ac, norm, pcfg) -> dict:
    """Deterministic rollout over a fixed command grid. Returns mean return and tracking.

    The env passed here must be built with `auto_reset=False`, and that is load-bearing rather
    than tidiness. `VecCreatureEnv._reset_idx` draws a *fresh random command* for every env it
    resets, so under auto-reset the fixed grid assigned below survives only until the first
    animal falls over -- after which the "deterministic evaluation over a fixed command set"
    is quietly scoring a random one. It looked like algorithm noise: consecutive evaluations
    of a steadily improving policy came back 20, 263, 22.

    A separate env also keeps evaluation from disturbing the training rollout, which otherwise
    has to be re-reset afterwards, throwing away `num_envs` partial episodes every time.
    """
    n = env.n
    # A fixed spread of forward speeds in Froude units so the score covers the commanded
    # range rather than whatever the sampler happened to draw.
    grid = np.linspace(env.cfg.speed_range[0], env.cfg.speed_range[1], n)
    obs, _ = env.reset(seed=12345)
    env.command[:, 0], env.command[:, 1], env.command[:, 2] = grid, 0.0, 0.0
    ret = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    acc = {"tilt": 0.0, "speed": 0.0, "r_speed": 0.0, "cot": 0.0}
    live, steps = np.zeros(n), 0
    for _ in range(env.max_steps):
        a = ac.mean_action(norm(obs, pcfg.obs_clip))
        obs, rew, term, trunc, info = env.step(a)
        ret += rew * alive
        live += alive
        for k in acc:
            acc[k] += float(np.sum(info[k] * alive))
        steps += 1
        alive &= ~term          # a fallen env stops contributing and is not restarted
        if not alive.any():
            break
    tot = max(1.0, live.sum())
    out = {f"eval_{k}": v / tot for k, v in acc.items()}
    out.update(eval_return=float(ret.mean()), eval_survived=float(alive.mean()),
               eval_len=float(live.mean()), eval_steps=steps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default=str(ROOT / "rigs" / "canis.ftcl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "canis"))
    ap.add_argument("--steps", type=float, default=2e7, help="env steps to train for")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint-minutes", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=20, help="updates between evaluations")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--eval", default=None, help="evaluate a checkpoint and exit")
    ap.add_argument("--view", action="store_true", help="with --eval, open the MuJoCo viewer")
    args = ap.parse_args()

    ecfg = envmod.EnvConfig(rig=args.rig)
    pcfg = ppo.PPOConfig(num_envs=args.envs, horizon=args.horizon, lr=args.lr,
                         seed=args.seed, total_steps=int(args.steps),
                         device=ppo.pick_device(args.device))

    if args.eval:
        return run_eval(args, ecfg, pcfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = build_envs(ecfg, pcfg.num_envs, pcfg.seed)
    # A second, separate set of envs used only for scoring. See `evaluate`: it must not
    # auto-reset (that would re-randomise the fixed command grid), and keeping it apart from
    # the training envs means a scoring pass costs the rollout nothing.
    eval_env = build_envs(ecfg, pcfg.num_envs, pcfg.seed + 1, auto_reset=False)
    ac = ppo.ActorCritic(env.obs_dim, env.act_dim, pcfg)
    norm = ppo.RunningNorm(env.obs_dim)

    step, best = 0, -np.inf
    if args.resume:
        step, extra = ppo.load(args.resume, ac, norm)
        best = extra.get("best", -np.inf)
        print(f"resumed {args.resume} at {step:,} steps (best eval {best:.2f})", flush=True)

    obs, _ = env.reset(seed=pcfg.seed)
    per_update = pcfg.horizon * pcfg.num_envs
    log_path = out / "log.jsonl"
    t_ck = t0 = time.time()
    upd = 0
    # Every progress line is flushed. A run this long is normally started detached with its
    # stdout redirected to a file, and Python block-buffers a redirected stream at 8 KB -- which
    # is about forty of these lines, or twenty minutes of silence at this rate. The whole
    # premise of the tool is that a run can be watched and left alone, and an unflushed log
    # looks exactly like a hang.
    say = lambda s: print(s, flush=True)                                     # noqa: E731
    say(f"obs {env.obs_dim}  act {env.act_dim}  envs {pcfg.num_envs}  "
        f"device {pcfg.device}  {per_update} samples/update")

    while step < pcfg.total_steps:
        ro, obs = ppo.collect(env, ac, norm, pcfg, obs)
        logs = ppo.update(ac, ro, pcfg)
        step += per_update
        upd += 1

        row = {"step": step, "update": upd, "sps": step / max(1e-9, time.time() - t0),
               **ro.stats, **logs}
        if upd % args.eval_every == 0:
            row.update(evaluate(eval_env, ac, norm, pcfg))
            if row["eval_return"] > best:
                best = row["eval_return"]
                ppo.save(out / "best.pt", ac, norm, pcfg, step, {"best": best})
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if upd % 10 == 0 or "eval_return" in row:
            ev = (f"  eval {row['eval_return']:7.1f} surv {row['eval_survived']:.2f}"
                  if "eval_return" in row else "")
            say(f"{step:>10,}  ret {row.get('ep_return', float('nan')):7.1f}  "
                f"len {row.get('ep_len', float('nan')):6.1f}  "
                f"rspd {row['r_speed']:.3f}  tilt {row['tilt']:5.1f}  "
                f"kl {row['kl']:.4f}  sps {row['sps']:5.0f}{ev}")

        if time.time() - t_ck > args.checkpoint_minutes * 60:
            ppo.save(out / "latest.pt", ac, norm, pcfg, step, {"best": best})
            t_ck = time.time()

    ppo.save(out / "latest.pt", ac, norm, pcfg, step, {"best": best})
    say(f"done: {step:,} steps in {(time.time() - t0) / 60:.1f} min, best eval {best:.2f}")
    env.close()
    eval_env.close()
    return 0


def run_eval(args, ecfg, pcfg) -> int:
    # Scoring needs `auto_reset=False` (see `evaluate`); the viewer wants the opposite, so a
    # fall puts the animal back on its feet instead of leaving it lying there.
    env = build_envs(ecfg, 1 if args.view else pcfg.num_envs, pcfg.seed,
                     auto_reset=args.view)
    ac = ppo.ActorCritic(env.obs_dim, env.act_dim, pcfg)
    norm = ppo.RunningNorm(env.obs_dim)
    step, extra = ppo.load(args.eval, ac, norm)
    print(f"{args.eval}: trained {step:,} steps, best {extra.get('best', float('nan')):.2f}")
    if not args.view:
        for k, v in evaluate(env, ac, norm, pcfg).items():
            print(f"  {k:16s} {v:.3f}")
        env.close()
        return 0

    import mujoco.viewer
    obs, _ = env.reset(seed=0)
    env.command[:] = [0.5, 0.0, 0.0]
    with mujoco.viewer.launch_passive(env.bodies[0].model, env.datas[0]) as v:
        dt = env.cfg.control_dt
        while v.is_running():
            t = time.time()
            obs, _, _, _, _ = env.step(ac.mean_action(norm(obs, pcfg.obs_clip)))
            v.sync()
            time.sleep(max(0.0, dt - (time.time() - t)))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
