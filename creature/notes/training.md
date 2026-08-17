# Training — the commands, and how to tell whether they worked

*(added 2026-08-12. Every command and every number on this page was run against the tree as it
stands; where a figure is a measurement it says so and says on what.)*

This is stage E of `notes/pipeline.md`, expanded. `pipeline.md` tells you where training sits
between the camera cards and a rendered animal; this page is what you actually type, what the
output means, and — the part that is easy to get wrong — how to distinguish a policy that
learned to walk from one that learned to stand very still and score well.

**Training does not consume captured footage today.** It learns to track a commanded velocity on
the authored rig from a hand-designed reward. Capture feeds it in two places, neither of which
exists yet: the fitted morph vector θ (stage C → `--morph`, which *is* built) and AMP demos
(`--demos`, which is not). See "What is not built yet" at the bottom before planning around it.

---

## The four commands

```bash
cd creature

# 1. train
python tools/train.py --steps 20e6 --out runs/canis

# 2. resume — after a crash, a rate limit, or just closing the laptop
python tools/train.py --resume runs/canis/latest.pt --steps 20e6 --out runs/canis

# 3. score it — the per-command table, and the only honest read of quality
python tools/train.py --eval runs/canis/best.pt

# 4. watch it
python tools/train.py --eval runs/canis/best.pt --view
```

Note `--rig` is a **flag, not a positional**: `python tools/train.py rigs/canis.ftcl` fails with
`unrecognized arguments`. It defaults to `rigs/canis.ftcl`, so you rarely pass it.

`--resume` needs `--out` and `--steps` again — it restores the weights, the optimiser's step
count, the observation normaliser, the best-so-far eval score and the command curriculum's
`speed_cap`, but the run's *destination* comes from the flags, so a resume that forgets
`--steps 20e6` stops at the 2e7 default it was already past and exits immediately.

---

## How long, on what

Measured on this machine (CPU, 2026-08-12), a fresh run with default settings:

| envs | steps/s | 20M steps |
|---|---|---|
| 16 | ~950 | ~5.9 h |
| 64 (default) | ~2 400 | **~2.3 h** |

Throughput climbs for the first few minutes and then settles — the number in the progress line is
a running average since *this process* started, so a resumed run reports its own rate rather than
inheriting the previous process's. Read it after a minute, not off the first line.

`--device auto` picks CUDA if torch sees it. It is worth less here than you would expect: the
bottleneck is MuJoCo stepping 64 environments on the CPU, not the network, and the network is
small. Don't buy a GPU for this.

---

## Reading the progress line

```
   573,440  ret   534.4  len  679.7  rspd 0.593  spd +0.012/0.30  cur 0.51/0.73  tilt   2.4  kl 0.0273  sps  2279
```

| field | is | watch for |
|---|---|---|
| `ret` / `len` | mean episode return and length over episodes that **finished this update** | `nan` means no episode finished in that window — normal once episodes run long, not a bug |
| `rspd` | the speed-tracking reward term, 0–1 | this is the one that has to rise; ~0.5 is roughly "parked" (see below) |
| `spd` | actual mean forward speed (Froude) `/` the curriculum's current cap | **`spd` hovering at ±0.01 is the failure mode.** See the next section |
| `cur` | curriculum score `/` the bar it must beat to widen the cap | `cur` stuck below `cur_bar` for a long stretch means the curriculum has stalled |
| `terr` | terrain difficulty 0–1, **only present with `--terrain`** | the second curriculum axis; see below |
| `tilt` | mean degrees off vertical | rises before a collapse |
| `kl` | policy KL per update | a spike into the tenths is the run coming apart; the PPO loop early-stops on it |
| `sps` | env steps/s, this process | |

A full row per update — including `insane_frac`, `clip_frac`, `entropy`, `v_loss` and the rest —
is appended to `runs/<name>/log.jsonl`, one JSON object per line, which is the thing to plot.
`insane_frac` is the fraction of envs tripping the finite-but-not-physics divergence check
(`known-issues.md` #6); on the default body it should be flat zero.

---

## The one thing you must actually check

**`ret` and `eval_return` cannot tell a policy that walks from a policy that stands still.** Run
`--eval` without `--view` and read the per-command table. This is a real run from this tree, at
573k steps:

```
  eval_return    602.792
  eval_survived    1.000
  eval_len      1000.000

     cmd   speed   error  r_speed   tilt    cot  alive s
   0.000   0.001  +0.001    0.998    1.3   0.09    20.0s
   0.114   0.001  -0.113    0.813    1.7   0.08    20.0s
   0.229   0.001  -0.227    0.437    2.0   0.08    20.0s
   0.343   0.001  -0.342    0.154    2.3   0.08    20.0s
   0.457   0.001  -0.456    0.036    2.5   0.08    20.0s
   0.571   0.001  -0.571    0.005    2.6   0.07    20.0s
   0.686   0.001  -0.685    0.001    2.8   0.07    20.0s
   0.800   0.000  -0.800    0.000    2.8   0.07    20.0s
```

Return 603, survives every episode for the full twenty seconds, tilt under 3°. By every scalar
that run is going beautifully. The `speed` column says the animal has not moved: it is parked,
collecting the near-zero commands for free and ignoring everything else. `cmd 0.000` scores
0.998; `cmd 0.800` scores 0.000.

This is the local optimum the whole design of the reward and the curriculum exists to avoid, and
it is **not** a bug — it is what 573k steps looks like. The default run is 20M. But it is the
reason the per-command table exists, and the reason `--eval` prints it by default. **A healthy
policy has a `speed` column that climbs with `cmd` and an `error` column near zero across the
range.** If, at the end of a full run, `error` is still ≈ `-cmd`, the run failed no matter what
`eval_return` says.

Two supporting reads, both already on screen:

- **`r_speed` ≈ 0.5 in the progress line is the parked score, not half-marks.** A motionless
  animal earns 0.64 of the tracking reward when commands only run to 0.3 Froude and 0.31 when
  they run to 0.8. That is why the curriculum's bar is expressed as a fraction of the gap between
  parked and perfect rather than as an absolute (`EnvConfig.curriculum_margin`) — an absolute bar
  asks an easier question at every widening.
- **`spd` in the progress line is the same signal, live.** `+0.012/0.30` is a parked animal with
  the cap at 0.3. You do not have to wait for an eval to see it.

`--view` is for watching gait quality once the table says it moves. It is not a substitute for
the table: at 1 body and one fixed command of 0.5 Froude, a parked animal and a walking one look
similar for the first second and you will talk yourself into the wrong one.

---

## Checkpoints

Written into `--out`:

| file | is |
|---|---|
| `latest.pt` | rewritten in place every `--checkpoint-minutes` (default 5). The resume target. |
| `best.pt` | the best *evaluated* return so far. Survives a late collapse. |
| `log.jsonl` | one row per update, appended (a resume appends, so the file spans processes). |

Checkpoints are on a **wall-clock** interval rather than an update count deliberately: the thing
they exist to survive is a session ending, and sessions end in minutes, not in updates.

Evaluation runs every `--eval-every` updates (default 20) on a *separate* set of envs built with
`auto_reset=False` and a fixed command grid, so consecutive evals differ only by the policy.
That separation is load-bearing rather than tidy — with auto-reset the fixed grid survives only
until the first animal falls, after which the "deterministic evaluation" is quietly scoring a
random command set. The symptom was consecutive evals of a steadily improving policy reading
20, 263, 22.

---

## Training on rough ground (P1b)

```bash
# terrain on, difficulty starting at 0 and widened by the curriculum
python tools/train.py --terrain --out runs/canis_rough --steps 3e7

# start the curriculum at a specific difficulty (implies --terrain); 0..1
python tools/train.py --terrain-level 0.5 --out runs/canis_rough

# a resume knows it was a terrain run -- you do not have to re-pass --terrain, and if you
# forget it, it will not quietly put the animal back on a plane
python tools/train.py --resume runs/canis_rough/latest.pt --out runs/canis_rough

# score an existing checkpoint on terrain of a chosen difficulty
python tools/train.py --eval runs/canis_rough/best.pt --terrain-level 1.0

# hold a class out of training, then score on exactly that class -- this pair IS P1b's bar
python tools/train.py --terrain-kinds flat,rolling,rubble,steps,slope --out runs/canis_rough
python tools/train.py --eval runs/canis_rough/best.pt --terrain-kinds stairs --terrain-level 1.0
```

`--terrain` swaps the infinite `<geom type="plane">` for a MuJoCo heightfield, regenerated
**every episode** from one of six classes (`flat`, `rolling`, `rubble`, `steps`, `slope`,
`stairs`). Nothing about the observation changes: there is no height map input, no exteroception
of any kind. The policy finds out about the ground through its feet, which is the point — see
`design.md` §"Terrain is a training distribution, not a solver" for why this is a heightfield and
not a foot-placement solver.

The progress line grows a `terr` field, which is the difficulty in 0..1:

```
   573,440  ret  534.4  len  679.7  rspd 0.593  spd +0.012/0.30  cur 0.51/0.73  terr 0.3  ...
```

**Difficulty is a second curriculum axis, sharing the first one's machinery.** Rough ground and a
faster command are both "the task got harder" and both show up in the same tracking reward, so
they share the 256-episode window, the measurement and the bar — and every earned promotion is
spent on whichever axis has progressed less as a fraction of its own range. Advancing speed to
its maximum first would train a flat-ground gallop and then ask it to relearn on rubble.
`terrain_level` is checkpointed alongside `speed_cap`, so a `--resume` does not hand a competent
policy the flat task it solved millions of steps ago.

**Evaluation stays flat even for a terrain run, deliberately.** `best.pt` is selected by
comparing eval returns across a run, and a score measured on ground that gets rougher as the run
proceeds is not comparable with itself — the checkpoint selector would drift with the curriculum
rather than with the policy. To measure the thing you actually trained, use the third command
above: `--eval CKPT --terrain-level L` scores a checkpoint on terrain of a difficulty you name.
Scoring on terrain without naming a difficulty is refused rather than defaulted, because
difficulty 0 is a *flat heightfield* — it would pay terrain's cost and hand back a flat-ground
table under a heading that says terrain.

**`--terrain-kinds` is what makes the bar measurable.** P1b's claim is not "walks on rough
ground" but "walks on rough ground *of a kind it never trained on*", and that needs a class
held out of the training draw and then named at eval time — the last pair of commands above.
The held-out set is printed once at the top of the run's log and travels in the checkpoint, so
a `--resume` that forgets the flag cannot quietly show the policy the one class the run's whole
number depends on it never having seen. A misspelled class is an error, not a smaller set.

**It costs ~26% of throughput**, from MuJoCo's hfield collision geometry and not from anything in
this repo (`known-issues.md` #8 has the attribution, and the two knobs that would buy it back
along with why neither is taken). Budget for it: a 3e7-step terrain run is about a third longer
in wall-clock than the same run on a plane.

The terrain's shape is quoted in **withers heights**, like everything else here — amplitude,
feature size and the patch's own extent all scale with the animal, so the same config means the
same terrain on a terrier and a wolfhound. The knobs are `EnvConfig.terrain_*` in
`creaturelab/env.py` and, as with the reward weights, they are not flags: a run trained on
different terrain is a different task, not a different run.

---

## Training on a captured animal (P4)

Once stage C has produced a fitted morph vector, train *around* it rather than around the rig's
authored defaults:

```bash
# one body: exactly the animal you fitted
python tools/train.py --morph out/theta_rex.json --out runs/rex --steps 3e8

# a neighbourhood of it — this is the one you want
python tools/train.py --morph out/theta_rex.json --morph-scale 0.4 --morph-bodies 8 \
    --out runs/rex --steps 3e8

# poke one parameter without editing the file; --set wins over --morph
python tools/train.py --morph out/theta_rex.json --set limb_gracility=0.7 --out runs/rex_thin
```

`--morph-scale 0` (the default) is the single-body case: one compiled model shared by every env,
which is safe rather than sloppy — the shared object holds the read-only `MjModel`, while the
per-env mutable `MjData` is allocated one per env.

Above zero it compiles `--morph-bodies` **distinct** animals (~1.1 s each) and cycles them over
the envs. The count is capped separately from `--envs` on purpose, and not just for speed:
distinct bodies are a *variance* knob. Every env holding its own animal means every minibatch
mixes morphs, which is what the morph conditioning has to learn from — but it also means no
single body is seen often enough early for the policy to get any of them standing. **~8 bodies
over 64 envs trains visibly faster than 64 distinct ones, at the same asymptote.** The pool is
seeded off `--seed`, so a resumed run rebuilds the same zoo; a different set of bodies after a
resume is a change of task mid-training and shows up as an unexplained step in the curve.

**Evaluation deliberately ignores `--morph-scale` and scores the centre body only.** The eval
score has to be comparable across a run and between runs, and a score averaged over 64 different
animals moves when the zoo changes — which would make `best.pt` track "got a lucky draw of
bodies" as readily as "got better". To see the policy on a *different* animal, pass that animal
as `--morph` at eval time:

```bash
python tools/train.py --eval runs/rex/best.pt --morph out/theta_other.json
```

That command is the P4 claim — "every edit inside the trained neighbourhood is free" — in a form
you can actually run, and it is how you find out whether the neighbourhood was wide enough.

**Known issue, and you will see it.** `--morph-scale > 0` currently trips MuJoCo instability
warnings that the default body never produces:

```
WARNING: Nan, Inf or huge value in QACC at DOF 23. The simulation is unstable.
```

This reproduces on demand (`--morph-scale 0.4 --morph-bodies 4` prints it within a minute) and is
`known-issues.md` #6, still open — the envs are flagged and terminated rather than poisoning the
batch, so the run is not silently corrupt, but the underlying instability (suspected tail chain /
damping, linked to issue 4) has not been fixed. Check `insane_frac` in `log.jsonl` to see how
much of the batch it is costing you.

**Get the morph range right before the first policy gradient.** Anatomy is not an output of
training. Randomise over a *generous* neighbourhood covering every edit you anticipate:
randomisation cost scales with command-space dimensionality, not with how wide each range is, so
generosity is nearly free — whereas a post-hoc edit that leaves the trained region costs a
fine-tune. `todo.md` P6 §"Converting a captured animal" has the argument.

Before committing to a scale, check what fraction of that neighbourhood can even stand:

```bash
python tools/morph_sweep.py rigs/canis.ftcl -n 24 --scale 0.25 0.5 0.75 1.0
python tools/morph_sweep.py rigs/canis.ftcl --morph out/theta_rex.json -n 24 --scale 0.4
```

The second form is the one that matters once you have a fitted animal, and it asks a genuinely
different question from the first: a real animal is likelier than the authored defaults to sit
near a parameter's range edge, where most of its neighbourhood gets clipped.

---

## Tuning knobs, in rough order of usefulness

| flag | default | when to touch it |
|---|---|---|
| `--envs` | 64 | more envs = more throughput and lower gradient variance, up to your core count |
| `--steps` | 2e7 | the run length. 3e8 for a morph-randomised P4 run |
| `--horizon` | 64 | steps per env per update; `--envs × --horizon` is the batch (4096 by default) |
| `--lr` | 3e-4 | lower it if `kl` spikes and `early_stop` fires most updates |
| `--seed` | 0 | also seeds the morph zoo |
| `--eval-every` | 20 | evals cost a full 1000-step rollout on a second env set; raise it if it dominates |
| `--checkpoint-minutes` | 5 | |
| `--device` | auto | see above — this is CPU-bound in MuJoCo |
| `--terrain` | off | rough ground (P1b), with its own curriculum axis; costs ~26% throughput |
| `--terrain-level` | — | the curriculum's starting difficulty 0..1 (and the difficulty `--eval` scores on); implies `--terrain`, and wins over a resumed value |
| `--terrain-kinds` | all six | the classes to draw from, comma-separated; implies `--terrain`. Hold one out to train, name it to `--eval`: that pair is P1b's bar |

The reward weights, the command ranges and the curriculum live in `EnvConfig`
(`creaturelab/env.py`) and are **not** exposed as flags. That is deliberate: they are part of the
task definition, and a run trained under different ones is not comparable. Change them in the
file, and treat it as a new task rather than a new run.

---

## What is not built yet

`--demos out/motion/*.npz` — the AMP discriminator over fitted motion (P2) — **does not exist**.
`ppo.py` was written anticipating "an AMP discriminator sharing the rollout", so the rollout
structure is ready, but the discriminator, its replay buffer and the style-reward mix are not.

Be warned about where the design is thin, because it is not where you would guess. The *fit*
(stage C) is specified down to objective weights, metrics and a cost model. The *training*
additions are not: P2 is four checkboxes plus a paragraph on why AMP beats DeepMimic. Nothing in
this repo says how the discriminator is structured, what its observation is, how the style reward
mixes with the task reward, or how demo frames failing the E_phys gate get down-weighted. P1's
write-up is detailed because it was built; P2's is thin because it has not been. **Expect to
design AMP when you get there, not to implement a plan.**

The intended command, once it exists:

```bash
python tools/train.py --steps 3e8 --out runs/rex \
    --demos out/motion/*.npz  --morph out/theta_rex.json --morph-scale 0.4
```

---

## Cross-references

- **Where training sits in the whole pipeline** — `notes/pipeline.md` stage E.
- **What to shoot, and what each shot unlocks** — `notes/capture.md`.
- **What the reward, curriculum and observation actually are** — `design.md`.
- **Why training happens once and not per-animal** — `todo.md` P6.
- **The QACC instability on randomised bodies** — `known-issues.md` #6.
- **What terrain costs, and the alternative that was deferred** — `known-issues.md` #8.
