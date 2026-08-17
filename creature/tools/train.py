"""Train a locomotion policy, headless, resumably.

    python tools/train.py --steps 20e6 --out runs/canis
    python tools/train.py --resume runs/canis/latest.pt          # picks up mid-run
    python tools/train.py --eval runs/canis/best.pt --view       # watch it

    # train on a FITTED animal, over a neighbourhood of it (P4)
    python tools/train.py --morph out/theta_rex.json --morph-scale 0.4 --out runs/rex
    python tools/train.py --eval runs/rex/best.pt --morph out/theta_rex.json

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
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                        # noqa: E402

from creaturelab import env as envmod                     # noqa: E402
from creaturelab import ppo                               # noqa: E402
from creaturelab import terrain                           # noqa: E402
from creaturelab.console import use_utf8                  # noqa: E402

use_utf8()   # before argparse: --help is the thing a cp1252 console cannot print


def _parse_kinds(ap, s: str | None) -> tuple[str, ...] | None:
    """`--terrain-kinds a,b,c` -> a validated tuple, or None if the flag was not given.

    Validated rather than passed through because an unknown name is not an error anywhere
    downstream: `terrain.generate` would simply never draw it, so `--terrain-kinds stiars`
    trains on the five classes that *did* parse and reports nothing. A typo in the set of
    terrain a run is trained on is precisely the kind of silent task change P1b already had
    two of.
    """
    if s is None:
        return None
    kinds = tuple(k.strip() for k in s.split(",") if k.strip())
    if not kinds:
        ap.error("--terrain-kinds needs at least one class")
    if bad := [k for k in kinds if k not in terrain.KINDS]:
        ap.error(f"unknown terrain class {', '.join(bad)} "
                 f"(choose from {', '.join(terrain.KINDS)})")
    return kinds


def build_envs(cfg: envmod.EnvConfig, n: int, seed: int, auto_reset: bool = True,
               center: dict[str, float] | None = None, morph_scale: float = 0.0,
               bodies: int | None = None, quiet: bool = False):
    """Compile the bodies the envs will run.

    `morph_scale == 0` (the default) is the single-body case: one compiled `Body` shared by
    every env. Sharing is safe rather than sloppy -- `Body` holds the `MjModel` and the
    derived constants, all read-only during a step, while the per-env mutable state is
    `MjData`, which `VecCreatureEnv` allocates one of per env. Compiling 64 identical models
    would cost 64x the tune pass to produce 64 identical results.

    Above zero, this is P4: each distinct body is a separate compile + tone pass, so the
    count is a real cost (~1.1 s each) and is capped by `bodies` rather than tied to
    `num_envs`. That cap is not just a speed knob. Distinct bodies are a *variance* knob:
    every env holding its own animal means every minibatch mixes 64 morphs, which is what
    the morph conditioning has to learn from -- but it also means no single body is seen
    often enough early on for the policy to get any of them standing. A pool of ~8 bodies
    over 64 envs gives eight envs per animal and trains visibly faster than 64 distinct
    ones, at the same asymptote. The pool is re-used cyclically so every body gets the same
    number of envs.

    `center` is a fitted morph vector (`out/theta_*.json`): the neighbourhood is drawn
    around the animal you captured rather than around the rig's authored defaults.
    """
    if morph_scale <= 0.0:
        pool = [envmod.build_body(cfg, center)]
    else:
        import random

        from creaturelab.build import load, sample_morph
        params = load(cfg.rig).params
        # Seeded off the env seed so a resumed run rebuilds the same zoo. A different set
        # of bodies after a resume is a change of task in the middle of training, and it
        # shows up as an unexplained step in the learning curve.
        rng = random.Random(seed ^ 0x5EED)
        k = max(1, min(n, bodies if bodies is not None else 8))
        pool = [envmod.build_body(cfg, sample_morph(params, rng, morph_scale, center))
                for _ in range(k)]
        if not quiet:
            w = np.array([b.withers for b in pool])
            print(f"{k} distinct bodies over {n} envs (morph scale {morph_scale:g}"
                  f"{', centred on the fitted morph' if center else ''}): "
                  f"withers {w.min():.3f}-{w.max():.3f} m", flush=True)
    envs = [pool[i % len(pool)] for i in range(n)]
    return envmod.VecCreatureEnv(envs, cfg, seed=seed, auto_reset=auto_reset)


def evaluate(env, ac, norm, pcfg, per_command: bool = False) -> dict:
    """Deterministic rollout over a fixed command grid. Returns mean return and tracking.

    The env passed here must be built with `auto_reset=False`, and that is load-bearing rather
    than tidiness. `VecCreatureEnv._reset_idx` draws a *fresh random command* for every env it
    resets, so under auto-reset the fixed grid assigned below survives only until the first
    animal falls over -- after which the "deterministic evaluation over a fixed command set"
    is quietly scoring a random one. It looked like algorithm noise: consecutive evaluations
    of a steadily improving policy came back 20, 263, 22.

    A separate env also keeps evaluation from disturbing the training rollout, which otherwise
    has to be re-reset afterwards, throwing away `num_envs` partial episodes every time.

    `per_command` additionally returns the *unaggregated* rows. The scalar mean cannot tell a
    policy that tracks every command mediocrely from one that tracks the slow half and ignores
    the fast half, and on this task those are the two things that actually happen -- the second
    is what standing still looks like, since a motionless animal scores ~0.25 against a uniform
    command range by getting the near-zero commands right for free.
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
    rows = {k: np.zeros(n) for k in acc}
    live, steps = np.zeros(n), 0
    for _ in range(env.max_steps):
        a = ac.mean_action(norm(obs, pcfg.obs_clip))
        obs, rew, term, trunc, info = env.step(a)
        ret += rew * alive
        live += alive
        for k in acc:
            contrib = info[k] * alive
            acc[k] += float(np.sum(contrib))
            rows[k] += contrib
        steps += 1
        alive &= ~term          # a fallen env stops contributing and is not restarted
        if not alive.any():
            break
    tot = max(1.0, live.sum())
    out = {f"eval_{k}": v / tot for k, v in acc.items()}
    out.update(eval_return=float(ret.mean()), eval_survived=float(alive.mean()),
               eval_len=float(live.mean()), eval_steps=steps)
    if per_command:
        # Per-env means over that env's own live steps, so a body that fell at step 40 is not
        # averaged as though it had stood there quietly for the remaining 960.
        d = np.maximum(live, 1.0)
        out["per_command"] = {"cmd": grid, "live": live, "alive": alive,
                              **{k: v / d for k, v in rows.items()}}
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
    ap.add_argument("--morph", metavar="FILE", default=None,
                    help="fitted morph vector (out/theta_*.json) to train around")
    ap.add_argument("--set", nargs="*", metavar="NAME=VAL", default=None,
                    help="override individual morph params; wins over --morph")
    ap.add_argument("--morph-scale", type=float, default=0.0,
                    help="randomise bodies this far about the centre (0 = one body)")
    ap.add_argument("--morph-bodies", type=int, default=8,
                    help="distinct bodies to compile, cycled over the envs")
    ap.add_argument("--terrain", action="store_true",
                    help="train on rough ground (P1b): a heightfield re-rolled every "
                         "episode, with its own difficulty curriculum")
    ap.add_argument("--terrain-level", type=float, default=None,
                    help="terrain difficulty 0-1: the curriculum's STARTING difficulty when "
                         "training, and the difficulty to score on with --eval (which is "
                         "otherwise flat). Implies --terrain")
    ap.add_argument("--terrain-kinds", metavar="A,B,C", default=None,
                    help="comma-separated terrain classes to draw from (" +
                         ",".join(terrain.KINDS) + "). Implies --terrain. Training with a "
                         "class held out and then scoring on it with --eval is how P1b's bar "
                         "-- generalising to ground it never saw -- is measured")
    args = ap.parse_args()

    kinds = _parse_kinds(ap, args.terrain_kinds)

    from creaturelab.morph_io import morph_from_args, parse_sets
    sets = parse_sets(args.set)
    center = None
    if args.morph or sets:
        from creaturelab.build import load
        center = morph_from_args(args.rig, args.morph, sets, load(args.rig).params)

    ecfg = envmod.EnvConfig(rig=args.rig,
                            terrain=args.terrain or args.terrain_level is not None
                            or kinds is not None)
    if kinds is not None:
        ecfg = replace(ecfg, terrain_kinds=kinds)

    # Terrain is a property of the TASK, not resumable state, and it is compiled into the
    # model -- so unlike `speed_cap` it cannot be restored after the env is built. Without
    # this, `--resume ckpt` on a terrain run rebuilds a FLAT env (the flag lives in argv, not
    # in the checkpoint), restores a terrain_level onto it, and trains happily on a plane: the
    # task changes mid-run and the only trace is the `terr` column quietly disappearing.
    # Same failure design.md names for the morph zoo, which is why it is an error and not a
    # warning when the two disagree in the direction the user has actually asked for.
    #
    # The class *set* is task too, and it is the half a held-out-class run cannot afford to
    # lose: a run trained on five classes that resumes onto all six has quietly been shown the
    # very terrain its bar is about generalising to, and the resulting number means nothing.
    # It is restorable after the build in a way the flag is not, but it is restored here
    # anyway, so that one block holds everything argv would otherwise silently redefine.
    if args.resume:
        was = ppo.peek(args.resume)
        if not ecfg.terrain and was.get("terrain"):
            ecfg = replace(ecfg, terrain=True)
            print(f"{args.resume} is a terrain run; continuing with terrain on", flush=True)
        if kinds is None and ecfg.terrain and was.get("terrain_kinds"):
            ecfg = replace(ecfg, terrain_kinds=tuple(was["terrain_kinds"]))
    pcfg = ppo.PPOConfig(num_envs=args.envs, horizon=args.horizon, lr=args.lr,
                         seed=args.seed, total_steps=int(args.steps),
                         device=ppo.pick_device(args.device))

    if args.eval:
        return run_eval(args, ecfg, pcfg, center)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = build_envs(ecfg, pcfg.num_envs, pcfg.seed, center=center,
                     morph_scale=args.morph_scale, bodies=args.morph_bodies)
    # A second, separate set of envs used only for scoring. See `evaluate`: it must not
    # auto-reset (that would re-randomise the fixed command grid), and keeping it apart from
    # the training envs means a scoring pass costs the rollout nothing.
    #
    # Evaluation deliberately runs on the CENTRE body only, never on the randomised zoo.
    # The eval score has to be comparable across the whole run and between runs, and a
    # score averaged over 64 different animals moves when the zoo changes -- which would
    # make `best.pt` track "got a lucky draw of bodies" as readily as "got better".
    #
    # For the same reason it runs on FLAT ground even when the training env has terrain
    # (P1b). An evaluation whose task gets harder as the curriculum advances cannot select a
    # checkpoint: the score drops at every promotion, so `best.pt` would freeze at whatever
    # the policy managed on the easiest terrain it ever saw. Flat is the one task that stays
    # fixed for the whole run, it is directly comparable to P1's numbers, and it is exactly
    # the guard P1b is held to -- "flat-ground performance does not regress". To score a
    # checkpoint ON terrain, use `--eval CKPT --terrain-level L`, which is a measurement
    # rather than a selector.
    eval_env = build_envs(replace(ecfg, terrain=False), pcfg.num_envs, pcfg.seed + 1,
                          auto_reset=False, center=center, quiet=True)
    ac = ppo.ActorCritic(env.obs_dim, env.act_dim, pcfg)
    norm = ppo.RunningNorm(env.obs_dim)

    step, best = 0, -np.inf

    def ckpt_extra() -> dict:
        # `speed_cap` is training state exactly as much as the network weights are. A resume
        # that restarts the command curriculum at its initial width hands a competent policy a
        # task it solved 5M steps ago, and the learning curve takes a visible step backwards
        # for reasons entirely internal to the resume. Same class of bug as leaving the
        # observation normaliser out of the checkpoint.
        # `terrain` is in here for a different reason from the rest: not to be restored into
        # the env (it cannot be -- it is compiled into the model) but so a `--resume` can find
        # out what task it is resuming BEFORE it builds one. See `ppo.peek` above.
        return {"best": best, "speed_cap": float(env.speed_cap),
                "terrain": bool(ecfg.terrain),
                "terrain_kinds": list(ecfg.terrain_kinds),
                "terrain_level": float(env.terrain_level)}

    if args.resume:
        step, extra = ppo.load(args.resume, ac, norm)
        best = extra.get("best", -np.inf)
        env.speed_cap = extra.get("speed_cap", env.speed_cap)
        env.terrain_level = extra.get("terrain_level", env.terrain_level)
        print(f"resumed {args.resume} at {step:,} steps (best eval {best:.2f}, "
              f"command cap {env.speed_cap:.2f}, terrain {env.terrain_level:.2f})",
              flush=True)

    # After the resume on purpose: an explicit `--terrain-level` is the user overriding where
    # the curriculum stands, which is the only reason to pass it on a resume at all.
    if args.terrain_level is not None:
        env.terrain_level = args.terrain_level
        print(f"terrain difficulty set to {env.terrain_level:.2f}", flush=True)

    # Said once, at the top of the log, rather than per row: a held-out class is the single
    # fact that decides whether this run's eval number is a generalisation claim or not, and
    # it must be readable from the log file alone months later.
    if ecfg.terrain:
        held = [k for k in terrain.KINDS if k not in ecfg.terrain_kinds]
        print(f"terrain classes: {', '.join(ecfg.terrain_kinds)}"
              + (f"  (held out: {', '.join(held)})" if held else ""), flush=True)

    obs, _ = env.reset(seed=pcfg.seed)
    per_update = pcfg.horizon * pcfg.num_envs
    log_path = out / "log.jsonl"
    t_ck = t0 = time.time()
    upd, step0 = 0, step        # `step0` so `--resume` reports this process's rate, not a
    #                             throughput of 60k/s inherited from the steps it did not run
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

        row = {"step": step, "update": upd,
               "sps": (step - step0) / max(1e-9, time.time() - t0),
               "speed_cap": float(env.speed_cap), "cur_score": float(env.cur_score),
               "cur_bar": float(env.cur_bar), "terrain_level": float(env.terrain_level),
               **ro.stats, **logs}
        if upd % args.eval_every == 0:
            row.update(evaluate(eval_env, ac, norm, pcfg))
            if row["eval_return"] > best:
                best = row["eval_return"]
                ppo.save(out / "best.pt", ac, norm, pcfg, step, ckpt_extra())
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if upd % 10 == 0 or "eval_return" in row:
            ev = (f"  eval {row['eval_return']:7.1f} surv {row['eval_survived']:.2f}"
                  if "eval_return" in row else "")
            say(f"{step:>10,}  ret {row.get('ep_return', float('nan')):7.1f}  "
                f"len {row.get('ep_len', float('nan')):6.1f}  "
                f"rspd {row['r_speed']:.3f}  spd {row['speed']:+.3f}"
                f"/{row['speed_cap']:.2f}  cur {row['cur_score']:.2f}"
                f"/{row['cur_bar']:.2f}"
                + (f"  terr {row['terrain_level']:.1f}" if ecfg.terrain else "")
                + f"  tilt {row['tilt']:5.1f}  "
                f"kl {row['kl']:.4f}  sps {row['sps']:5.0f}{ev}")

        if time.time() - t_ck > args.checkpoint_minutes * 60:
            ppo.save(out / "latest.pt", ac, norm, pcfg, step, ckpt_extra())
            t_ck = time.time()

    ppo.save(out / "latest.pt", ac, norm, pcfg, step, ckpt_extra())
    say(f"done: {step:,} steps ({step - step0:,} this run) in "
        f"{(time.time() - t0) / 60:.1f} min, best eval {best:.2f}")
    env.close()
    eval_env.close()
    return 0


def run_eval(args, ecfg, pcfg, center=None) -> int:
    # Scoring needs `auto_reset=False` (see `evaluate`); the viewer wants the opposite, so a
    # fall puts the animal back on its feet instead of leaving it lying there.
    #
    # `--morph-scale` is ignored here on purpose: evaluating on one body is what makes the
    # per-command table readable. To see the policy on a *different* animal, pass that
    # animal as `--morph` -- which is the P4 claim ("every edit inside the trained
    # neighbourhood is free") in a form you can actually run.
    # An eval is a measurement of one policy on one *stated* task, so the task has to be
    # stated. Terrain without a difficulty is difficulty 0, which is a flat heightfield --
    # the run pays terrain's cost, prints no warning, and hands back a flat-ground table
    # under a heading that says terrain. Ask for the number instead of guessing it.
    if ecfg.terrain and args.terrain_level is None:
        print("--eval on terrain needs an explicit --terrain-level (0-1): without one the "
              "difficulty is 0, which is flat ground wearing a heightfield.", file=sys.stderr)
        return 2

    env = build_envs(ecfg, 1 if args.view else pcfg.num_envs, pcfg.seed,
                     auto_reset=args.view, center=center)
    if args.terrain_level is not None:
        # A fixed difficulty, not the curriculum's: a level that moved during the rollout
        # would make the table unreadable.
        env.terrain_level = args.terrain_level
        print(f"scoring on terrain at difficulty {env.terrain_level:.2f} "
              f"({', '.join(ecfg.terrain_kinds)})")
    ac = ppo.ActorCritic(env.obs_dim, env.act_dim, pcfg)
    norm = ppo.RunningNorm(env.obs_dim)
    step, extra = ppo.load(args.eval, ac, norm)
    print(f"{args.eval}: trained {step:,} steps, best {extra.get('best', float('nan')):.2f}")
    if not args.view:
        res = evaluate(env, ac, norm, pcfg, per_command=True)
        pc = res.pop("per_command")
        for k, v in res.items():
            print(f"  {k:16s} {v:.3f}")
        print(f"\n  {'cmd':>6s} {'speed':>7s} {'error':>7s} {'r_speed':>8s} {'tilt':>6s} "
              f"{'cot':>6s} {'alive s':>8s}")
        for i in range(len(pc["cmd"])):
            c, s = pc["cmd"][i], pc["speed"][i]
            print(f"  {c:6.3f} {s:7.3f} {s - c:+7.3f} {pc['r_speed'][i]:8.3f} "
                  f"{pc['tilt'][i]:6.1f} {pc['cot'][i]:6.2f} "
                  f"{pc['live'][i] / ecfg.control_hz:7.1f}s"
                  + ("" if pc["alive"][i] else "  fell"))
        print("  speeds are Froude (v / sqrt(g*h)); a motionless animal reads cmd 0 perfectly "
              "and\n  everything else not at all, which is what the scalar mean hides")
        env.close()
        return 0

    import mujoco.viewer
    obs, _ = env.reset(seed=0)
    env.command[:] = [0.5, 0.0, 0.0]
    # `env.models[0]`, not `env.bodies[0].model`: with terrain the vec env clones a model per
    # env, and the clone is the one holding the elevation data the resets write.
    with mujoco.viewer.launch_passive(env.models[0], env.datas[0]) as v:
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
