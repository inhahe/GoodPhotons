# Creature — known issues and technical debt

Unsolved bugs and deferred work. Fixed entries stay, marked **DONE**, when the failure
mode is subtle enough to be worth not rediscovering.

---

## Open

### 1. One randomised draw in twenty collapses at `--scale 1.0`, mechanism unexplained

`python tools/morph_sweep.py rigs/canis.ftcl -n 60 --scale 1.0` → 57/60 stand. Seed 15 is
the reproducible case:

```
python tools/morph_sweep.py rigs/canis.ftcl -n 1 --seed 15 --scale 1.0 --verbose
```

That draw is a 314 kg animal (`body_mass` 68.6 × `body_scale` 1.66³) on a 0.264 m trunk
with 0.265 m femurs — short-backed, long-legged, very heavy. It **slides steadily
backwards at ~0.25 m/s with all four feet in contact and every joint rigid**, travels
1.29 m, and ends inverted (170° from upright) on its spine and head.

Ruled out, each by direct measurement:

| hypothesis | measurement | verdict |
|---|---|---|
| statically unstable pose | support margin **+126 mm**, tip angle 7.5°, 4 level feet (spread 0.0 mm) | no |
| joint compliance | `relax` raised peak stiffness 400× (4.2e3 → 1.7e6 N·m/rad); settled depth moved 65.7% → 65.4% | no |
| integrator / timestep | dt 0.002 → 0.0005 → 0.0002 gives **bit-identical** results (+792 mm, 170.0°) | no |
| friction cone | statics solve needs μ = 0.000 (reaction is purely vertical); ground provides 0.9 | no |
| self-collision load path | 0 excluded pairs needed; contacts are the 4 paws | no |
| general mass scaling bug | the default rig is flat at 4.8–4.9% sag across a **32×** `body_mass` sweep | no |
| absolute `joint_armature` | replaced by a measured per-joint armature (issue 2, DONE): **bit-identical**, +793 mm and 170.3° before and after | no |

Only mass moves it: the identical morph at `body_mass/4` (78 kg) stands at 4.0% sag. So
it is a threshold effect specific to that geometry, not a systematic scaling error.

Remaining suspect, untested: MuJoCo contact behaviour under a ~3 kN load on four point
contacts — the hind paw's centre descends 37 mm during the slide, which looks like more
than seating.

Low priority — P4 will need a viability filter on randomised draws regardless, and
`tune.support_polygon` already returns the number to filter on. But the mechanism should
be understood before it is filtered away, in case it also affects plausible bodies.

### 2. Independent morph sampling can draw incoherent animals

`sample_morph` perturbs all 26 parameters independently, so `stance_femur` is uncorrelated
with `femur_len`, and `body_scale` with `trunk_len`. At `--scale 1.0` this produces bodies
no animal resembles. Yield is still 93%, but P4 wants a distribution of *plausible* bodies,
not merely standable ones. Options: declare covariance in the `morph` block, or sample a
few latent factors (size, gracility, crouch) that drive the parameters.

Every scale-1.0 collapse other than seed 15 is now of this shape rather than a tuner
failure: seeds 30 and 32 topple to ~50° while sagging only 2–4%, from support margins of
+60 and +18 mm. They are not sprung too softly — they are bodies whose feet are in the
wrong place, and no amount of passive tone fixes a foot in the wrong place.

### 4. `steps_per_cycle_floor`'s integrator branch is factually meaningless, and the ceiling it feeds ignores damping

*(found 2026-08-08, while calibrating `tune.brace`.)* `stiffness_ceiling` derives the
stiffest followable spring as `k = I(2πf)²` with `f = 1/(steps_per_cycle_floor·dt)`, and
`steps_per_cycle_floor` returns **12 for implicit integrators and 30 for Euler**. That branch
does not correspond to anything measurable. On a single hinge, Euler, `implicitfast` and
`implicit` give **bit-identical** spring-response error, because MuJoCo integrates joint
damping implicitly in *all* of them and joint stiffness explicitly in *all* of them — the
integrator choice changes the contact and Coriolis treatment, not the passive spring.

The constant is also mis-framed. It is an **accuracy** budget, not a stability limit:

| steps/cycle | amplitude error | stable? |
|---|---|---|
| 12 | ~6% | yes |
| 8 | ~9% | yes |
| 6 | ~13% | yes |
| 4 | ~22% | yes |
| 3 | — | yes for any ζ ≥ 0.05 |

So 12 buys 6% error, and the "must not be less than" framing implies a cliff that is not
there. It further ignores `damping_ratio` entirely, although ζ is what actually sets where
the stability edge sits.

**Why this matters, measured.** The ceiling binds on real bodies. Of the 8 failures in 24
`--scale 1.0` draws, **five stand at a smaller timestep with no other change** — `brace`
identifies the mode correctly and then asks for more stiffness than 2 ms can integrate:

```
seed  1   dt 0.002 falls (164°)   dt 0.001 falls           dt 0.0005 STANDS (2.8°)
seed  3   dt 0.002 falls (101°)   dt 0.001 STANDS (1.4°)
seed  5   dt 0.002 falls (166°)   dt 0.001 falls           dt 0.0005 STANDS (1.0°)
seed  9   dt 0.002 falls ( 94°)   dt 0.001 STANDS (2.3°)
seed 21   dt 0.002 falls (119°)   dt 0.001 falls           dt 0.0005 STANDS (3.1°)
```

That is exactly what `stiffness_ceiling`'s own docstring says such a case means — "the
timestep is too large, which is a real and reportable fact about the model, not something to
paper over" — so the *behaviour* is honest. The debt is that the constant gating it is partly
fictional.

Not changed yet, deliberately, for two reasons. First, the defensible loosening to 8 does
**not** recover these morphs: seed 3 needs 6 and seed 5 needs 4, and 4 is 22% error, so the
real fix is a smaller default timestep or an implicit spring treatment, not a relaxed
constant. Second, the single-hinge probe that produced the tables above is not validated on a
31-joint articulated body with contacts, which is where it would be applied — **redo that
measurement on the articulated body before touching the constant.**

The other three failures are not this: seed 15 is issue 1, seed 8 sags 84% (issue 2), and
seed 7 survives the shove at 1.02° but fails on sag at 10.8% and gets *worse* at a smaller
timestep — a separate thing, unexplained.

### 5. PPO's KL early-stop is post-hoc, so 93% of updates run exactly one of five epochs

Measured over the first 1749 updates of the 20 M-step P1 run (`runs/canis/log.jsonl`):

| | value |
|---|---|
| `target_kl` | 0.02 |
| median post-epoch KL | 0.056 |
| p90 / max | 0.093 / 0.253 |
| updates completing 1 epoch | 1623 (93%) |
| updates completing 2 | 126 (7%) |
| updates completing 3–5 | **0** |

`ppo.update` measures KL over the whole batch on a fresh pass at the end of each epoch and
breaks if it exceeds `target_kl` — which is honest (an earlier version measured it inside the
minibatch loop and was biased three ways; see design.md §"PPO is written here rather than
imported"). What the honest number now says is that **a single epoch already overshoots the
trust region by ~3×**, so the check can only ever fire after the fact and `epochs: int = 5` is
decorative. The run is doing a fifth of the optimisation its config describes — the same
symptom as the bug that was fixed, arrived at from a different cause.

It is not *breaking* the run: evaluation return climbed past 1200 with every animal surviving
20 s, which is P1's bar. So this is tech debt, not a blocker, and it was deliberately not
changed mid-run.

The proper fix is one of two, and which one is an empirical question that should be settled
with a short A/B at ~1 M steps rather than by argument:

- **Check the full-batch KL after every minibatch, not every epoch**, and break mid-epoch.
  Costs `minibatches` extra forward passes per epoch (cheap against the backward passes) and
  makes the trust region actually bounding. Expect it to stop after ~1–2 minibatches, i.e. a
  much smaller update than today's.
- **Lower `lr` from 3e-4** so an epoch fits inside the trust region and the configured 5
  epochs actually run. Roughly the same total movement per update, spread over more, smaller
  gradient steps on the same data.

Note the coupling already documented in design.md: KL scales as (Δµ/σ)², so this interacts
with `init_log_std` — a policy that has narrowed its σ during training trips the same
`target_kl` on a smaller parameter move, which is part of why the epoch count never recovers
as the run matures.

### 6. Randomised bodies trip MuJoCo instability warnings that the default body never does

Found while wiring `train.py --morph-scale` (distinct bodies per env, P4). Same command,
same step count, same seed, only the bodies differ:

```
python tools/train.py --steps 1.5e4 --envs 8 --horizon 32 --out scraps/trainbase
    ->   0 "Nan, Inf or huge value in QACC" warnings
python tools/train.py --steps 1.5e4 --envs 8 --horizon 32 --morph-scale 0.5 \
                      --morph-bodies 4 --out scraps/trainzoo
    -> 102 warnings, overwhelmingly at DOF 22-26 = fpaw_r_flex and the tail chain
```

**Not** a build-time problem and **not** a stepping problem on its own. Each of these was
measured directly and came back clean:

| probe | result |
|---|---|
| building the same 4 morphed bodies | 0 warnings |
| 4000 steps of uniform random actions, per body | 0 insane steps |
| 6000 steps of *held saturated* (±1) actions, per body | 0 insane |
| the 4-body pool over 8 envs, 1 worker and 8 workers | 0 insane |
| 20 PPO `collect` updates on the same pool | 0 insane, peak speed 1.6 Froude |

So it needs the real PPO action distribution to reproduce, and the last probe above did
not reproduce it — which is the unfinished part of this entry.

**The damage is bounded but real.** `sensing.gather_state`'s detector does work (verified
by forcing `qvel = 1e8`: `warning.number[mjWARN_BADQACC]` survives MuJoCo's own
`mj_resetData`, so `sane` goes False, `env.step` pays −1 and terminates, and `ppo.collect`
masks the row out of the logged means). What it does *not* catch is a large-but-finite
divergence: the zoo run logged `spd +19.5` and `spd -445.2` Froude on rows that were still
`sane`. 445 Froude is ~1100 m/s. That number is clipped out of the observation by
`RATE_CLIP`, and `r_speed` saturates to zero, so the policy sees nothing wrong — but it
goes into `c_energy` at full size and into the run's own progress trace, which is exactly
the "unreadable log" failure that the `sane` mask was added to fix.

Two things to do, in order:

1. ~~**Widen the sanity test to cover finite nonsense.**~~ **DONE.**
   `sensing.flag_divergence(spec, raw)`, called from `env.step` right after the physics pool
   returns, clears `raw.sane` for any env whose root speed exceeds `INSANE_SPEED = 50`
   Froude or whose angular / joint rates exceed `INSANE_RATE = 100` per pendulum period.
   Every bound is in the env's *own* body scales, so it neither penalises a large body nor
   lets a small one away with anything — `test_finite_nonsense_is_judged_per_body` drives
   three morphs to an identical *dimensionless* speed and asserts they agree. 50 Froude is
   deliberately generous: peak locomotion is 2–3 and a free fall through one body length
   reaches ~5, so nothing reachable by an animal is near it. Deliberately outside the thread
   pool, which is GIL-held and serialises the whole vec env; it reads rows the pool already
   copied out, so it is a handful of numpy calls for the entire batch.
   Covered by `test_finite_nonsense_is_insane_too` (both directions: a 3-Froude gallop
   survives, 500 Froude terminates and pays −1 while staying finite).
2. **Then find the mechanism.** *(still open — this is what keeps the entry open.)*
   The DOF distribution points at the tail chain, which
   `sensing.RATE_CLIP`'s own comment already fingers as "light, long and barely damped" —
   the suspicion is that `tune.stiffness_ceiling`, which is derived per joint from
   `I(2πf)²`, is satisfied while the *damping* on those joints is not enough at the morphed
   inertia (see issue 4, which says the ceiling ignores damping — this may be the same bug
   arriving from the other end).

---

### 7. `fit_sequence`'s warm start is unproven, and `sigma_px` is still a placeholder

Two open questions left by the pose fit (`creaturelab/fit.py`), neither of which blocks
anything today but both of which get *harder* to answer once real footage exists.

**(a) The warm start buys 3–4%, not the order of magnitude the design assumed.**
`fit_sequence` carries each frame's solution into the next. The obvious justification —
"fewer LM iterations" — is measurably false: `init_root` already makes a cold start cheap
and independent of the initial guess, so a warm-started frame costs about the same and
sometimes slightly more. The intended justification is *continuity*: a limb occluded for a
run of frames is nearly unconstrained, and re-solving it from scratch each frame lets it
wander inside its null space, which is indistinguishable from motion downstream (it becomes
acceleration in the AMP demos and torque in the E_phys gate). Measured on synthetic clips,
including one with five landmarks hidden in three of four views for the whole clip:

```
                warm     cold (each frame independent)
root     0.0086   0.0090      mean |2nd difference|
joints   0.2615   0.2708
```

3–4%. Too small to build on, so `tests/test_fit.py::test_a_sequence_tracks_the_motion`
deliberately asserts only that the sequence path is not *worse* — an assertion tuned to pass
at one seed would turn this open question into a claim. **What would settle it:** real
footage, where occlusion is longer, correlated between views and not drawn from a uniform;
or a synthetic clip with a deliberately adversarial occlusion schedule (one limb hidden in
*all* views for 20+ frames, which the current generator cannot express because it draws
occlusion i.i.d. per frame). Keep the structure regardless — E_temp and the E_phys gate
operate on a trajectory, not on frames.

**(b) `notes/keypoints.yaml` still says `sigma_px_measured: false`.** Every σ is the 6.0 px
placeholder, so σ currently carries no relative weighting at all — the fit weights all 21
landmarks equally in everything except the Huber width, which comes from `class` instead.
That is why `tools/fit_selftest.py`'s verdict is *relative* (beat free-point triangulation,
stay under the ray-uncertainty floor) rather than an absolute millimetre bar: an absolute bar
would be measuring the placeholder. The real numbers come from the detector's own
cross-validation error per body part, once one is trained;
`tools/keypoints_project.py` prints a note to stderr while the flag is false, and
`tests/test_keypoints.py::test_the_placeholder_sigma_is_flagged_as_a_placeholder` fails if
the flag is flipped without the numbers changing.

---

## Done

### `--help` crashed on two tools, because the console could not encode `θ`  **DONE**

*(found 2026-08-12, while verifying the commands going into `notes/training.md`.)*
`tools/morph_sweep.py --help` and `tools/fit_selftest.py --help` both died with

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b8' in position 1596
```

raised from inside `argparse._print_message`. Their module docstrings refer to the morph vector as
Greek theta, argparse echoes the docstring into the help text, and Windows hands a process's
`sys.stdout` the console's ANSI code page (cp1252) with `errors="strict"`. The tools themselves ran
perfectly; only the *first thing anyone types about them* was guaranteed to fail. The same fault
would have taken down any long run that printed a non-ASCII character in its final summary — after
doing all the work.

Nothing caught it, for two compounding reasons: pytest replaces `sys.stdout` with a UTF-8-capable
capture object, so the bug **cannot exist in-process** under the suite, and every existing test
imported the tool modules rather than executing them. It needed a real subprocess with a narrow
encoding to appear at all.

Fixed by `creaturelab/console.py`'s `use_utf8()`, called by all six tools before argparse. The
tempting fix — deleting the offending characters from the docstrings — was rejected: it leaves the
process still unable to print them, so it regresses silently at the next degree sign, and it makes
the docs worse to route around a bug that is not in the docs. `tests/test_tools_cli.py` pins both
halves: it runs every tool's `--help` in a subprocess under a forced `PYTHONIOENCODING=cp1252` (so
the test reproduces a Windows console on any machine) and asserts the output is *byte-identical* to
the UTF-8 run — which is what fails if someone later strips the characters instead. A final test
asserts some tool's help still contains non-ASCII at all, so the suite cannot quietly go vacuous.

### 3. The default rig was passively unstable in roll, and neither guard could see it  **DONE**

*(found 2026-08-08, while writing P1's env.)* The **unmorphed canis rig** stands
indefinitely from a perfectly symmetric start, and falls over in 3.5–6.5 s under zero
torque from any asymmetry at all. Reproduce it in eight lines — perturb the settled stance
and watch the trunk, motors off throughout:

```python
from creaturelab import env as envmod
from creaturelab.emit_mjcf import place_on_ground
import mujoco, numpy as np
body = envmod.build_body(envmod.EnvConfig(rig="rigs/canis.ftcl")); m, s = body.model, body.spec
d = mujoco.MjData(m); mujoco.mj_resetData(m, d); place_on_ground(m, d)
d.qpos[s.jnt_qposadr] += np.random.default_rng(0).uniform(-.5, .5, m.nu) * (s.jnt_hi - s.jnt_lo) * .02
mujoco.mj_forward(m, d); place_on_ground(m, d)
for k in range(4000):                                  # 8 s, no ctrl ever written
    mujoco.mj_step(m, d)
    print(k * m.opt.timestep, np.degrees(np.arccos(d.xmat[1].reshape(3, 3)[2, 2])))
```

Drop the perturbation line and it stands for 20 s. (The fuller diagnostic probes used to
produce the numbers below — mode shape, stiffness sweep, per-foot loads, the coupled `−∂τ/∂q`
matrix — were written as throwaways in the gitignored `scraps/`; the measurements they
produced are recorded here so they do not need to be rewritten.)

**It is a clean exponential**, not excessive reset noise. Roll e-folds every **0.67 s**
(≈2.6 pendulum periods) and the fall time depends on the perturbation only through its
logarithm:

| `init_joint_noise` | fall time (seeds 0 / 1) | predicted by `t₀ − τ·ln(amp)` |
|---|---|---|
| 0.002 | 5.53 s / 6.35 s | — |
| 0.02  | 3.64 s / 4.56 s | 5.53 − 0.67·ln(10) = 3.99 s |
| 0.05  | 3.55 s / 3.71 s | 3.64 − 0.67·ln(2.5) = 3.03 s |

A 25× larger perturbation buys only 1.95 s, against `τ·ln(25)` = 2.17 s. Pitch stays at
~0.8° the whole time; it is purely the frontal plane.

**The mode is trunk roll on the four abduction springs.** At 0.28° of roll all four
abduction joints have deflected the *same* sign by ~0.21°, i.e. they absorb 74% of it — the
legs stay upright and the trunk rolls over them. Ranked deflection: `hip_l_abduct` 0.229°,
`hip_r_abduct` 0.209°, `shoulder_l_abduct` 0.204°, `shoulder_r_abduct` 0.196°, then the
sagittal joints an order of magnitude behind (all `*_pitch` rms 0.0025°).

**It needs ~2.2× the abduction stiffness it has.** Scaling only the abduction springs
(damping scaled by √ to hold ζ) puts the threshold sharply between 2× and 3×:

| × stiffness | k (N·m/rad) | fell | peak tilt |
|---|---|---|---|
| 1 | 39.8 | 3.64 s | 45° |
| 2 | 79.6 | 9.49 s | 45° |
| 3 | 119.4 | **never (15 s)** | 2.2° |
| 20 | 796 | never | 2.2° |

The two growth rates independently agree with that threshold: `λ² ∝ K_destab − K_restore`
with `λ₁ = 1.49/s` and `λ₂ ≈ 0.575/s` solves to `K_destab = 2.18 · K_restore(1×)`.

**Why `validate.stand_test` cannot see it.** It starts exactly symmetric, so roll stays at
`0.00°` for 19 s of a 20 s run — the mode is never excited, and an antisymmetric instability
is invisible to a symmetric probe no matter how long you watch. `SETTLE_SECONDS` is also
3.0 s, which is *shorter than the fall*. This is the same class of mistake `validate.py`'s
own docstring warns about, one level up.

**Why `tune.measure_buckling` cannot see it, which is the more interesting half.** It
perturbs one joint at a time, so it measures the diagonal of the destabilising stiffness
matrix — but the full matrix does not contain the mode either. Every abduction diagonal
entry is **exactly 0.00 N·m/rad**, and the largest coupled eigenvalue of the symmetrised
`−∂τ/∂q` is 34.65 N·m/rad in a *sagittal* mode (`spine_pitch`, `shoulder_*_flex`,
`scap_*_swing`) with **zero** abduction content. The reason is structural: `_hold_torque`
uses the ground reaction solved from statics and then held **frozen**, and the destabilising
term here is precisely the reaction *redistributing*. Watching it live:

```
   t    roll    y_com | fpaw_l    fpaw_r    hpaw_l    hpaw_r      (Fn, N)
 1.0  -0.262    4.2mm |  50.5      43.1      57.8      54.6
 2.0  -1.049   14.1mm |  61.0      41.6      57.8      45.5
 3.0  -5.434   64.4mm |  86.9      17.8      73.6      26.3
 4.0 -102.8   570.1mm |   0.0       0.0       0.0       0.0   (airborne)
```

The feet barely slide (lateral position 74.8 → 73.0 mm). What moves is the CoM, out towards
the left foot line at ±75–85 mm, unloading the right pair from 51 N nominal to 18 N, until it
crosses the support edge and the body tips about the loaded pair. A frozen reaction cannot
represent load transfer between feet by construction, so **no amount of eigenanalysis of the
current measurement recovers this mode** — the measurement's own assumption excludes it.

**Why the tuner had nothing to size these springs from.** For an abduction joint in a
symmetric stance the static holding torque is zero by symmetry and the single-joint buckling
gradient is zero as shown above, so two of `size_tone`'s four requirements are identically
zero and the spring falls through to the **floor** term — tone proportional to the subtree
weight the joint could carry. design.md already names the reason that term exists ("they
still need tone or the body flops laterally the moment it is disturbed"), which makes this
the one requirement in the table that is a *guess* rather than a measurement of the mode it
exists to prevent. It guesses 2.2× low.

**Consequences at the time.** `env.EnvConfig.init_joint_noise` defaults to 0.05, so every P1
episode started in a state the passive body could not hold, which contradicts the project's own
rule that not-falling-over belongs in the ligaments and not in the policy. P1's bar ("a stable
gait emerges") would have been measured against a body that falls over on its own, so P1 was
blocked on this. `test_env.py::test_passive_body_holds_its_stance` was deliberately written
with zero reset noise and said so — it passed, and would have failed at the default.

Ruled out by direct check: the abduction joints are **not** unsprung (39.651 and 39.975
N·m/rad, in the same range as the sagittal joints), so this is not the zero-stiffness bug it
first looked like.

**The proper fix** is a fifth requirement in `size_tone`, measured the way the module
measures everything else: perturb the settled body in the frontal-plane mode with contacts
**live**, read the resulting roll acceleration, and raise the abduction springs until it
restores with a margin. `stand_test` needs a lateral nudge in the same change, or the fix has
no regression test. Both must be validated against the randomised-morph yield (currently 93%
at scale 1.0) rather than just against the default rig, since the change can only make springs
stiffer and the stiffness ceiling is real.

**Fixed**, essentially along those lines, in four pieces:

- **`tune.tipping_velocity`** gives the stance its own toppling speed `v_c = w·√(g/h)` from
  the energy balance `½mv_c² = mg(√(h²+w²) − h)`, plus the direction to topple in, taken as
  the outward normal of the weakest support-polygon edge. Canis: w = 80.5 mm, h = 459.0 mm,
  **v_c = 0.372 m/s**, direction (+0.020, −1.000). Nothing in it needs the concepts "lateral"
  or "left and right" — the direction falls out of the hull.
- **`validate.stand_test` gained a second phase**: settle, then shove at `NUDGE_FRACTION`
  (0.10) of `v_c` in **both** signs of that axis, and judge on the worst peak tilt. Sized
  against `v_c` rather than the Froude speed `√(gL)` because past `v_c` no stiffness helps at
  all — abduction stiffness at ×3, ×12 and ×100 all end at ~90° tilt — so 0.10 carries 1% of
  the rigid-tipping energy and anything that falls, fell because it *bent*. The fraction sits
  on a measured plateau: 0.05–0.12 give the same verdict on all 48 bodies of the two harder
  morph scales.
- **`tune.brace`** shoves the settled body, identifies which joints carry the mode from the
  joint-space deviation while growth is still linear, and sizes them by secant on
  `λ² = (K_d − x·K₀)/I`, with `posture.buckle_margin` for headroom (no new constant — `K_d`
  here *is* a buckling gradient, for a coordinated mode rather than a single joint). The
  default rig converges in one pass: peak tilt **105.3° → 1.5°**, sag 5.28%, 8 joints at ×2,
  ceiling not binding, `build_tuned` ~1.1 s.
- **Both signs of the shove**, in the tuner as well as the bar. Not symmetry decoration: the
  roll's *joint* response is not antisymmetric, because the side that unloads goes slack
  instead of deflecting, so one direction alone reports one shoulder giving 1.9× its mirror
  and would spring the pair differently.

Three things learned that were not in the plan, each of which had to be measured because it
contradicts an assumption that felt safe:

1. **Stiffness is not monotonically stabilising.** Stiffening all 31 joints ×2 or ×3
   *collapses* the shove test, while stiffening exactly the 4 abduction joints ×3 passes it.
   So an over-broad mode selection actively harms. The selection threshold (0.6 of the largest
   deviation) was set at a plateau — the same joint family over a 130× amplitude range, 0.023°
   to 3.06° — after 0.25 was found to pull in 20 of 31 joints including neck and tail.
2. **Peak tilt saturates and cannot be a progress signal.** Every failing multiplier reports
   ~95°, because past the balance point the body ends up on its side regardless. A divergence
   guard reading it saw 96.8° → 95.6° across a 2× stiffness range, concluded stiffness was not
   helping, and abandoned a morph one secant step from standing (3.5× gives 1.4°). The guard
   now reads `λ²`, which is measured in the linear regime and is the quantity the secant models
   — *and* additionally stops after a fourfold multiplier with the peak unmoved, because `λ²`
   alone keeps falling under rigid tipping (‖dq‖ is a joint-deflection norm, and stiffer joints
   deflect less whether or not the trunk is going over). Without that second half, a body shoved
   at 1.2·v_c ran all six passes and kept **639×** its springs while still landing on its side.
3. **Re-selecting the mode each pass is self-erasing.** A joint is chosen for deflecting, and
   stiffening it is exactly what stops it deflecting, so it drops out of the next read and
   something else replaces it. The set changed identity every pass, which starved the secant of
   two points on one line *and* stopped raising the joints that were the actual fix. The mode is
   now read once, on the unbraced body, and held — which is also what the secant's own model
   requires, since `λ² = (K_d − x·K₀)/I` describes one fixed spring set scaled by one number.

Regression coverage: `test_the_shove_is_what_makes_the_stand_test_able_to_answer` (with
`tune.brace` stubbed out, the rig passes at `nudge_frac=0` and reaches >45° with the shove —
i.e. it reproduces the blind spot on demand), `test_tipping_velocity_is_the_stance_geometry_and_nothing_else`,
`test_brace_stiffens_the_mode_and_not_the_skeleton`, and `test_randomised_morphs_mostly_stand`,
which is now 8/8 and was what surfaced findings 2 and 3.

Cost to randomised yield, honestly: the bar is strictly harder, so scale-1.0 goes 92% (no
shove) → 67% (with it). Most of that gap is issue 4 rather than the tuner.

### `Defaults.joint_armature` and `joint_damping` were absolute literals  **DONE**

`joint_armature = 0.01 kg·m²` and `joint_damping = 0.1 N·m·s/rad` — exactly the
body-specific unit literals `tune.py` exists to eliminate, one layer down, and better
hidden, because nothing about the model looks wrong. The joint is simply heavier than the
bone attached to it.

Measured on the default rig, **unmorphed, with no morph vector involved at all**, the
rig's own `0.008` spanned **0.7% of the spine's inertia and 3790% of the paw's**: the paw
joints were 0.00821 against a true 0.00021, i.e. 97.4% fictitious rotor and 2.6% animal.
It then propagated, because armature is part of the mass matrix every other measurement
reads — `stiffness_ceiling` is `I(2πf)²`, so an inflated `I` licensed 38× more stiffness
than the real limb could follow, and damping is `2ζ√(kI)`, so the same joints came out
heavily overdamped. Both errors were largest exactly where the contact happens.

Fixed by `tune.size_armature`: armature is a dimensionless fraction of each joint's own
measured inertia, which is what a gear ratio physically is (reflected rotor inertia is
`n²·I_rotor`, a roughly fixed share of the load a matched drive is built for). It runs
after `auto_exclude` and before `measure`, since sizing it later would tune the body
against a mass matrix it does not have. `joint_damping` now defaults to `None`, so an
unowned joint gets no arbitrary damping instead of a wrong one.

The ratio was chosen by measurement, not taste: stand yield is **flat across a 40× range**
of it (0.02 → 0.8 all give 100% / 98.3% / ~92% at scales 0.5 / 0.75 / 1.0), which is the
right property for a numerical stabiliser and confirms that what mattered was making the
quantity scale-free, not the number. 0.1 is at or near the best in every column.

Honest cost: scale-1.0 yield went 95% → 93.3% and scale-0.75 100% → 98.3%, two draws in
120. The old figures were partly bought with the fictitious inertia — a 38× overweight
paw is a very effective transient damper — and the draws that now topple are the
incoherent morphs of issue 2, not tuner failures.

### MuJoCo 3.11 API drift — silent wrong answers, no errors  **DONE**

Three changes bit this build, none of which raise:

- **`data.qM` is gone.** The replacement `data.M` is a CSR-like layout, and
  `data.M[model.dof_Madr]` — the idiom that used to give the mass-matrix diagonal — now
  returns off-diagonal entries. It produced *negative* inertias, which only surfaced as
  `ValueError: expected a nonnegative input` from a downstream `sqrt`. Fixed by
  `tune._full_mass_matrix`, which probes both `mj_fullM` signatures (that changed too).
- **`mj_forward` writes `qacc`.** Calling it before `mj_inverse` means asking "what force
  produces the acceleration you already have?" — answer: zero. Every joint torque read
  0.00 and looked like a plausible "all joints unloaded" report. `qacc` must be zeroed
  *after* `mj_forward`.
- **`qfrc_inverse` is the total required force** and does *not* have `xfrc_applied`
  subtracted from it; the documented identity is the opposite. Setting `xfrc_applied` has
  no effect on it at all, so a frozen ground reaction has to be projected into joint space
  by hand with `mj_applyFT`.

### `geom_rbound` is a bounding sphere, not a vertical extent  **DONE**

`place_on_ground` used it, so the feet were seated 3.6 cm above the floor and every
inverse-dynamics torque was measured in free fall (all zeros — free fall is a rigid-body
motion and needs no internal force). It also overstated withers height by most of a
capsule half-length. Fixed by `emit_mjcf.geom_z_extent`, which computes the exact support
function per geom type.

### Buckling measurement returned zero for exactly the buckling joints  **DONE**

Real contacts were still enabled during the perturbation, so MuJoCo's solver reduced the
contact force by whatever external force was pushing the foot up, and the frozen reaction
netted to nothing. Because that cancellation is *exact*, the self-check against the
contact-solved torque also passed — vacuously. Fixed by disabling contacts
(`mjDSBL_CONTACT`) around the perturbation, in a `try/finally`.

### Quaternion pitch folded a 170° flip into "+10°"  **DONE**

`asin(2(q_y q_w − q_x q_z))` saturates, so large rotations read as small ones. A draw that
ended up completely inverted reported a tidy `+10 deg pitch` and was caught only because it
had also sunk most of a metre. Replaced with the angle between the trunk's own up-axis and
world up, unambiguous over the full [0°, 180°] range.

### Tuner and acceptance test disagreed on what "settled" means  **DONE**

`relax` settled for 1.5 s, `stand_test` for 3.0 s. The tuner declared victory on a body
that was still sinking; the extra 1.5 s took it from 6% to 9.2% of withers and a failure
the tuner had no way to see. Both now share `validate.SETTLE_SECONDS`.

### `max_stiffness 4000` in the rig silently under-sprung large morphs  **DONE**

Authored as "integrator guard, not a modelling choice", but a single N·m/rad number is
body-specific: slack for a 21 kg dog, binding for a 140 kg one. It capped exactly the
bodies that needed the most support, and reported a tidy round number while doing it. The
real limit is numerical and per-joint; `tune.stiffness_ceiling` derives it as
`k = I(2πf)²` from the timestep and integrator that `world` already declares.
