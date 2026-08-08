# Creature — TODO

A physically-based animal: articulated skeleton, Hill-type muscle/tendon actuators,
soft-tissue deformation, fur — driven by a **learned** controller rather than keyframes,
fit to real animal motion, and morphable into stylized/fictional creatures **without
losing the learned motion character**.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

**This file holds work items. The rules that constrain the work live in `design.md` §Standing
constraints** — things like "anything that can emerge should emerge", "every control channel must be
a training-time conditioning input", and "the morph space is per-body-plan". Those never get checked
off, so they are not tasks; read them before adding anything here that they would rule out.

---

## The ordering question (read this first)

**Yes — video comes late, deliberately.** Video → reliable 3D animal motion was rated the
*most novel* and the *most likely to disappoint* link in the whole system when this order
was set — an assessment made for **monocular** input. *(Re-rated 2026-08-08: the decided
4-camera rig (capture.md) turns 3D recovery into triangulation rather than monocular
lifting, so the reconstruction half of that risk shrank; what stays risky is keypoint
quality in clutter and anatomy identifiability — see P5 → "Monocular, re-rated". The
ordering below still stands on the yardstick argument, but P5 no longer *needs* to be
last; pull it earlier if logistics allow.)* Everything else is well-trodden. So the plan
front-loads the parts that are known to work, and arrives at video with a working
simulator, a working controller, and — this is the point — **a ground-truth yardstick**:
public dog mocap of the same gaits, so the video pipeline can be scored in centimetres
instead of vibes.

The order is therefore:

```
P0 grammar → MJCF          (cheapest end-to-end proof that the architecture holds)
P1 torque quadruped + PPO  (smoke-test the whole RL loop on a body we trust)
P2 AMP from PUBLIC MOCAP   (derisk imitation with clean data — no cameras yet)
P3 muscles replace torques  (the biomechanics that makes it ours)
P4 morphology conditioning  (the thing that makes stylization free later)
────────────────────────────  everything above is "doing the work"
P5 VIDEO FITTING            ← the research risk, entered with a yardstick
────────────────────────────
P6 stylization / morph transfer
P7 soft tissue + skin + eyes + fur + render through ftrace
P8 live creature viewer     (not strictly ordered — pull it forward the moment
                             morph-space exploration starts costing you time)
P9 human-performance drive  (needs P4's conditioned policy + P7's face to be
                             the interesting version rather than pose retargeting)
P10 layered control         (DESIGN before P2; BUILD alongside P4 — see below)
P11 flight                  (out of the main line; a separate body plan and a
                             separate physics problem — but it scopes P4's claim,
                             so read it before P4 is built)
```

**P10's design cannot wait for its slot.** Every control channel you want at the end — morph vector,
gaze target, style knobs, part-specific goals — must be a conditioning input *during* training or the
policy will ignore it. That is P4's own lesson applied to the whole control interface, and the cost of
discovering it late is a retrain. So decide the interface before P2 spends real compute, even though
most of the implementation lands with P4.

**P8 is deliberately unordered.** It is the one item here that pays back immediately at *any* stage:
26 morph parameters are currently explored by randomising and counting collapses, and a slider plus
a live sag/support readout replaces that with looking. Promote it as soon as it would save a session.

P2 is the one people skip and regret. Fitting a controller to *clean* mocap first means
that when the video version fails, you know it's the video, not the controller.

---

## P0 — The grammar and the MJCF emitter  `[~]`

The load-bearing architectural bet: **one source of truth compiles to several targets.**
The creature layer is a *generator*; it never becomes part of a runtime.

```
              ┌──→ MJCF      → MuJoCo (simulate, train)
  .ftcl ──────┼──→ FTSL      → ftrace (render)
              └──→ (USD)     → art tools
```

- [x] Project skeleton, venv, MuJoCo installed
- [x] `ftcl/` front-end: lexer, schema-driven parser, expression sublanguage, units
- [x] `creaturelab/model.py` — semantic model (Bone/Joint/Site/Muscle/Morph)
- [x] `creaturelab/emit_mjcf.py` — model → MJCF
- [x] `rigs/canis.ftcl` — a real quadruped rig, dog-scaled (25 bones, 31 actuated joints)
- [x] `creaturelab/tune.py` — passive tone measured per morph, not authored. The rig
      declares a `posture` goal; stiffness and damping are measured from the body the
      current morph vector actually produced, then closed-loop corrected against the
      settled pose. See design.md → "Passive tone: declare the goal, measure the gains".
- [x] `creaturelab/validate.py` — the acceptance bar: motors off, 3 s of gravity.
- [x] `tests/test_rig.py` — 28 regression tests, biased towards the silent failures (see
      design.md → "Validation"). `python -m pytest tests/ -q`, ~7 s.
- [x] **Validation:** MuJoCo loads it, it stands under gravity, and sweeping a morph
      param regenerates a *different but still valid* body with no hand-editing.
      - `rig_report.py` — stands across `body_scale` 0.6 → 1.7 (sag 6.9% → 5.2% of
        withers; note it gets *proportionally better* with size, which is exactly what
        scale-correct gains are supposed to do), and across a 32× `body_mass` range at a
        flat 4.8–4.9%.
      - `morph_sweep.py` — full-body randomisation, 60 draws per width:
        **100% stand at scales 0.25 / 0.5; 98.3% at 0.75; 93.3% at scale 1.0.**
        The residual is not a tuner failure: scale 1.0 samples all 26 parameters
        independently, so nothing stops it drawing a 314 kg animal on a 26 cm back. The
        collapses topple to ~50° while sagging only 2–4%, from support margins of +18 and
        +60 mm — feet in the wrong place, which no amount of passive tone fixes. P4
        should either couple stance angles to limb lengths or reject on the measured
        support margin that `tune.support_polygon` already returns.
        (Was 100%/95% before armature was measured rather than declared; the difference is
        two draws in 120, and the old figures were partly bought with a 38× overweight paw
        acting as a transient damper. See known-issues → armature, DONE.)
- [ ] Decide: does FTSL itself need to change, or does the creature layer only ever
      *emit* it? (Current answer: emit only. Keep ftrace static and bit-reproducible.)

### Why not just write MJCF by hand?
Because MJCF has no symbolic morphology. `femur_len` must be **one named parameter** that
simultaneously drives the rig, the RL conditioning vector, the domain-randomisation axis,
and the artist's stylization slider. If limb dimensions are hardcoded numbers scattered
through an XML, P4 and P6 are impossible — you can never recover the parameterisation.
That, and only that, is what justifies a layer above MJCF. Skeleton/joint/muscle
definitions themselves should round-trip to MJCF, not compete with it.

---

## P1 — Torque-actuated quadruped + PPO  `[x]`

Smoke-test the entire loop end to end on a body whose dynamics we trust.

- [x] Gym-style env wrapping the generated model — `creaturelab/env.py`. `VecCreatureEnv` is the
      single implementation and `CreatureEnv` is an N=1 view over it, because the reward and
      termination rules *are* the task and two copies would drift. Batched across envs: the
      thread pool touches `MjData` and nothing else, everything else is one numpy call over all
      N. See design.md §"The vec env is batched because the GIL, not the physics, was the
      bottleneck" — 762 → ~3000 env-steps/s, which is a 2e7-step PPO run in ~2 h not ~7.
- [x] **Proprioceptive observation space — decide this before the first PPO run, not after.**
      *(added 2026-08-06.)* In sim, proprioception is **free and exact** — MuJoCo already gives joint
      angles/velocities, muscle lengths and rates, tendon forces, contacts, body orientation, which
      *is* what spindles, Golgi tendon organs and the vestibular system report. So there is nothing
      to *infer*; the decision is **which signals, in what frame, normalised how**, and it is
      load-bearing:
      - **Global state (world positions, absolute orientation, exact velocities) is the wrong
        choice** even though it trains fastest. It produces a brittle policy that cannot transfer.
      - **Body-local, biologically-available, normalised signals** — muscle length as a fraction of
        rest length, tendon force as a fraction of max, joint angle relative to *its own* limits,
        gravity direction in head frame, foot contact — make the same numbers **mean the same thing
        on a different body**. That is the property P4's morph generalisation actually rests on, so
        this item is a P4 prerequisite disguised as a P1 detail. Getting it wrong is a retrain.
      - **Decided and built** in `creaturelab/sensing.py`: joint angle normalised against *its own*
        limits, joint rate ÷ the body's own pendulum period, efference copy, vestibular (gravity
        direction + angular rate + body-frame velocity in Froude units), per-foot normal force ÷
        body weight, command, morph vector. No world position or absolute orientation anywhere, and
        `test_env.py::test_no_world_frame_channel_leaks_in` bans the names so it stays that way.
        Every rate-like channel saturates at `sensing.RATE_CLIP` — real afferents do, and without
        it an undamped tail segment reached 49.6 against every other channel's ~1.
- [x] **Proprioceptive delay and noise, scaled by body size.** Real conduction latency is 10–40 ms,
      longer for a hind limb than a fore, and >100 ms in a large animal. Training on perfect
      instantaneous state yields superhuman reflexes and a twitchy, over-corrected gait — a genuine
      CG tell, the motor-control analogue of the fixed-pivot knee. **Delay should be derived from
      the morph vector** (limb length ÷ conduction velocity), not authored, so a scaled-up creature
      moves *heavier* for free. Directly serves design.md's "old / exhausted / 40 kg heavier is one
      knob" claim.
      - **Built.** Lag is `central_delay + tree_path_length / conduction_velocity`, measured through
        the kinematic tree to the declared CNS hub, so canis's hind afferents (32 ms) really do
        arrive after its fore (27 ms) and both scale with `body_scale` for free. Stored as
        **fractional** control steps and linearly interpolated between ring-buffer samples —
        rounding to whole steps at 50 Hz collapses 27 and 32 ms to the same number and deletes
        the anatomy the module exists to express, silently.
- [x] PPO baseline, flat ground, forward-velocity reward — `creaturelab/ppo.py`.
      - Own implementation in torch (~350 lines) rather than SB3: SB3's vec-env and rollout
        assumptions are exactly what P2's AMP discriminator and P10's ASE latents have to replace.
      - The three details that decide whether a PPO run is *correct* rather than merely running —
        truncation bootstraps and termination does not, the observation normaliser is part of the
        policy and belongs in the checkpoint, and GAE must not cross an auto-reset — each have an
        assertion in `tests/test_ppo.py` rather than a comment saying they were thought about.
        All three fail silently: the run still trains, the loss still falls, it just arrives
        somewhere worse.
      - **The stock `init_log_std = -0.5` is wrong here by enough to stop the run learning at
        all**, because an action of 1.0 is the *full* motor gear and that gear was sized from the
        joint's static hold torque. At σ = 0.61 episodes last 0.72 s and the energy penalty alone
        is 2.1/step against the 1.3 the tracking terms can pay — the reward is net negative for
        existing, and evaluation return *falls* over 400 k steps. Measured the env directly at
        fixed action noise to find it; −1.6 (σ = 0.20) trains. The reward weights were not the
        problem and were not touched. Table in design.md §"PPO is written here rather than
        imported".
- [x] Termination on fall, action-rate + energy penalties — tilt (the *same* measure as P0's
      acceptance bar, deliberately), height ÷ withers, and an insanity check that reads MuJoCo's
      warning counters rather than `isfinite`, because on a bad qvel MuJoCo resets the body itself
      and hands back a finite default pose. Energy is trapezoid-integrated |τ·q̇| at a measured
      accuracy/throughput trade (`sensing.actuator_work`), and every reward term is dimensionless.
- [x] **Command curriculum, because standing still is a strong local optimum.** With the full
      0–0.8 Froude command range on from the start, over half of every batch is drawn from
      commands where `r_speed = exp(−e²/0.25²)` has already decayed to ~0 — no reward *and no
      gradient* — while the other half pays a standstill ~1.0 for a cost of transport of 0.07
      against the 2.4 a flailing attempt costs. The first full-range run converged to exactly
      that: a `--eval` per-command table showed `speed 0.000` in all twelve rows with every
      animal surviving 20 s. `VecCreatureEnv.speed_cap` now widens only as fast as the policy
      earns it, judged on **tracking reward** (not survival — survival is what standing still is
      already perfect at) over a **window of 256 episodes, step-weighted**. The unwindowed first
      version scored whichever handful of envs happened to finish on the current step and went
      0.30 → 0.80 in 82 k steps, reproducing the standstill it was written to prevent. The cap
      is in the checkpoint, for the same reason the normaliser is.
- [x] **Bar:** a stable gait emerges. It will look bad. That is fine — this step is
      testing the plumbing, not the motion.
      - **Met** (2026-08-08). 20 M steps, 170 min on the CPU, best evaluation return 1224.9.
        The command curriculum reached the full 0–0.8 Froude range at ~3.4 M steps and the
        policy tracks the whole of it: over a 64-point command sweep the worst error is
        −0.040 Froude (5% under) at the top of the range, `r_speed` never drops below 0.93,
        tilt stays at 3–4°, and **every** animal survives the full 20 s at **every** command.
        Cost of transport rises monotonically 0.82 → 2.62 across the sweep, which is the
        physically right shape and is not something the reward asks for directly.
      - The number that matters is the *table*, not the scalar. The pre-curriculum run scored
        a respectable-looking mean and had `speed 0.000` in all twelve of its command rows —
        a perfect standstill. `--eval CKPT` prints the per-command breakdown for exactly this
        reason; a mean over a command grid a motionless animal reads perfectly at one end of
        hides the failure the whole of P1 is trying to detect.
      - ~~Blocked on known-issues #3 (passive roll instability)~~ — **unblocked** (2026-08-08).
        `tune.brace` measures the mode a symmetric settle structurally cannot excite and springs
        it, and `stand_test` shoves the body at 10% of its own tipping velocity so the fix has a
        bar to be held to. The default rig goes from falling over unaided (105° peak tilt) to
        1.5°, so the stance every episode resets into is now one the passive body actually holds.
- [x] Headless training + checkpointing so runs survive between sessions — `tools/train.py`.
      `--resume` picks a run up mid-flight; `--eval CKPT [--view]` scores or watches one.
      - Checkpoints on a **wall-clock** interval, not an update count, because the thing they
        exist to survive is a session ending, and sessions end in minutes rather than in updates.
        `latest.pt` is rewritten in place for resuming, `best.pt` tracks the best evaluated
        return so a late collapse cannot destroy the run's best policy.
      - Evaluation uses its own `auto_reset=False` envs and a fixed command grid. Sharing the
        training envs looked tidy and was wrong: `_reset_idx` draws a fresh *random* command for
        every env it resets, so the "fixed" grid survived only until the first animal fell over
        and consecutive evaluations of a steadily improving policy came back 20, 263, 22 —
        with `best.pt` being selected on that noise.

---

## P2 — AMP imitation from public mocap  `[ ]`

Clean data, no cameras. Derisks the *control* half in isolation.

- [ ] Ingest public dog mocap (Zhang et al., *Mode-Adaptive Neural Networks for Quadruped
      Motion Control* — the standard set in graphics; also used by Peng et al. for
      dog-mocap→robot retargeting)
- [ ] Retarget mocap skeleton → our rig
- [ ] AMP (adversarial motion prior) discriminator + PPO
- [ ] **Bar:** gait is recognisably dog-like, not just stable

**Why AMP over DeepMimic:** DeepMimic wants clean, per-frame phase-aligned reference
trajectories. AMP learns a *style* discriminator from unstructured, unaligned motion —
which is exactly the shape of what noisy video-derived motion will hand us in P5. Choosing
AMP now means P5 doesn't need a new controller.

---

## P3 — Muscles and tendons  `[ ]`

- [ ] `site` / `muscle` / `tendon` blocks in the grammar → MJCF spatial tendons with
      wrapping geometry + Hill-type actuators (MuJoCo has all of this natively)
- [ ] Ligaments as **soft nonlinear end-stops**, not hard clamps
- [ ] Elastic tendon recoil — a large share of quadruped locomotion energy is elastic
      return, not muscle work; without it gait timing is wrong
- [ ] Non-idealised joints where it matters: migrating knee centre-of-rotation, and the
      **floating scapula** (in felids the scapula isn't rigidly articulated to the axial
      skeleton — it rides in a muscular sling, and that is what makes a cat read as a cat)
- [ ] **Cartilage** *(added 2026-08-06 — was missing from this list entirely; zero hits repo-wide)*.
      Two distinct jobs, easy to conflate: (i) **articular cartilage** on joint surfaces, which is
      what actually produces the migrating centre-of-rotation above — model it as compliant contact
      / a shaped constraint surface rather than as a body; (ii) **structural cartilage** (costal,
      nasal, ear, intervertebral discs, the xiphoid) which is load-bearing *geometry* with a
      stiffness between bone and flesh, and matters mostly to P7's deformation, not to control.
      Decide per site which tier it belongs to; do (i) only where it changes the motion.
- [ ] Retrain P2's controller on muscle actuation

---

## P4 — Morphology conditioning  `[ ]`

**Do this before any stylization work, not after.** A policy trained on a fixed body and
then morphed *must* be fine-tuned, and fine-tuning is what destroys the motion character
we spent P2–P3 acquiring. A policy conditioned on the morphology vector from the start,
and randomised over it during training, morphs nearly free within the training
distribution.

- [ ] Morph vector (bone lengths, masses, attachment points, muscle strengths) as a
      first-class policy input
- [ ] Domain randomisation over morph space during training
- [ ] **Bar:** a body never seen in training walks with the same character — **within its body
      plan.** That qualifier is load-bearing and is not hedging: the morph space is *not* one global
      manifold. You cannot interpolate a leg into a wing, because the midpoint body has neither
      working limb and no policy exists across a discontinuous reward landscape. See P11.
- [ ] Prior art: MetaMorph; Shared Modular Policies

---

## P5 — Video fitting  `[ ]`  ← the research risk

*(The checkboxes below are the intent. The load-bearing decisions — the optimiser, the objective
term by term, the keypoint↔rig interface, what a θ evaluation costs — are pinned in **"The
implementation plan"** at the end of this section, added 2026-08-08.)*

- [ ] **Be clear that fitting yields TWO things, and they are consumed by different phases.**
      (i) **The animal's anatomy** — the morph vector θ_animal (bone lengths, proportions, mass
      distribution) of the pre-authored body-plan template that best explains the footage, fit
      jointly with pose from silhouettes + keypoints. This is an *optimisation measured in hours*,
      not RL training, and it is available **before** any policy training starts — which is what
      makes P6's conversion pipeline and P8's anatomy editor work without training twice (see P6).
      (ii) **The motion** — per-frame joint trajectories, which become AMP demonstration data.
      The anatomy estimate is a prerequisite of the motion estimate (reprojection error is
      meaningless through the wrong skeleton), so solve morph first on a few calm clips, then
      freeze it and solve motion per clip.
- [ ] 2D keypoints (DeepLabCut or SLEAP; some hand-labelling unavoidable)
- [ ] Camera intrinsics/extrinsics; fit **directly in the simulator's own joint
      parameterisation** by minimising 2D reprojection error — this sidesteps an entire
      retargeting stage that a separate 3D animal model would force on us
- [ ] Temporal smoothness + physics-plausibility regularisers
- [ ] **The validation that makes this step honest:** run the pipeline on footage of a
      *dog*, compare against the P2 mocap for the same gait, and report a hard error
      number. That says empirically whether the ambitious version is viable, or whether
      video should be demoted to style reference only.
- [ ] Capture rig — **`notes/capture.md`; hardware is now DECIDED: 4× GoPro HERO 12 Black at
      high fps** (user spec ~140 fps; see capture.md → "The decided rig" for the settings that are
      load-bearing: HyperSmooth OFF, native wide FOV with a fisheye camera model fit in calibration
      — never in-camera Linear dewarp — Protune-locked exposure, timecode sync ≠ genlock).
      4 cameras minimum for a quadruped (they self-occlude far worse than humans, and in a messy
      real environment the 4th view is what outvotes a hallucinated keypoint).
      Note that page's central rule: the rig has **two modes that must never share a recording** —
      motion (fast, whole-animal, whatever resolution survives the fps budget) and groom/appearance
      (stills, full sensor, close, controlled light, still subject). Same hardware, opposite settings.

### The implementation plan *(added 2026-08-08)*

*(This subsection exists because the checkboxes above stopped at intent: no optimiser named, no
objective terms, no silhouette source or metric, no keypoint↔rig correspondence, no statement of
what one candidate-anatomy evaluation costs. Those are the load-bearing decisions. Same spirit as
capture.md: decide, and write down why, so the decision can be wrong in public.)*

#### The framing that makes the optimiser question easy: fitting is kinematics, not dynamics

MuJoCo's standard build is not differentiable, and the reflex is to pick between MJX/JAX, finite
differences, and CMA-ES *for the whole problem*. Wrong frame — **nothing in the fit needs to
differentiate the dynamics.** The pose problem is: joint configuration q → site positions
(forward kinematics) → camera projection → residual against 2D evidence. The standard build
already gives the FK map *and its exact analytic Jacobian* — `mj_kinematics` + `mj_jacSite`
(∂ site-position / ∂ qpos) — and the calibrated fisheye projection (capture.md) is closed-form
differentiable. So the solver splits along the anatomy/pose line the section above already drew:

- **Pose, per frame: damped Gauss-Newton (Levenberg–Marquardt)** over ~37 unknowns (the rig's
  ~31 articulation DOFs + the 6-DOF free root), exact Jacobians, warm-started from the previous
  frame's solution; 5–15 iterations/frame. This is standard model-based markerless mocap (the
  same shape as OpenSim IK / Anipose), deliberately boring.
- **Anatomy, outer loop: CMA-ES over the morph vector** (26 params for canis) — the same call
  capture.md already made for the ~20-parameter groom, for the same reason: derivative-free,
  low-dimensional, and the inner objective (converged pose-fit residual as a function of θ) is
  noisy and non-smooth — RANSAC gates flip, silhouette contours change topology. Default
  population λ = 4+⌊3·ln 26⌋ = 13; budget a few hundred generations.
- **MJX is explicitly NOT required.** It becomes interesting only if E_phys (below) is ever
  promoted from a post-hoc gate to an in-loop term needing dynamics gradients. Don't build the
  differentiable version first — that is capture.md's "don't build a differentiable renderer"
  rule making its second appearance.

**The solve runs in two stages per clip:** (1) per-frame LM, temporally chained, on
E_kp + E_sil + E_lim; (2) one batch LM over the whole clip adding E_temp — its normal matrix is
block-tridiagonal in time, so this is cheap — with E_phys then evaluated **once** on the smoothed
trajectory, never inside LM iterations.

#### The objective, term by term (initial weights stated so they can be wrong in public)

- **E_kp — keypoint reprojection** (weight 1, the reference scale). Huber, δ = 2 px, on
  *undistorted keypoints* (capture.md: never undistort frames), scaled by detector confidence.
  A view is gated per-keypoint by capture.md's RANSAC triangulation: an outvoted view
  contributes **zero**, not a residual — a hallucinated keypoint must not pull on the skeleton.
- **E_sil — silhouette** (weight 0.3). *This is the term that carries the anatomy*: keypoints
  constrain the skeleton but say nothing about `trunk_radius`, `limb_gracility`, or the belly —
  girth lives only in the contour, so without E_sil the outer CMA-ES loop is blind to half the
  morph vector. Source: per-view video segmentation (SAM2-class, prompted once per clip per
  view; background subtraction as the fallback in a controlled space). Metric: one-directional
  distance-transform chamfer — per frame/view, precompute the DT of the observed silhouette
  boundary once; sample the projected occluding contours of the model's capsule geoms; each
  sample does a bilinear DT lookup (which is also its analytic gradient). Model→observed
  direction only: robust to segmentation holes and clutter, O(#contour samples), and needs no
  rasteriser and no soft renderer.
- **E_temp — temporal smoothness** (weight 0.05; stage-2 only). Second differences of qpos,
  per-DOF normalised by the authored joint range; first differences on root translation. High fps (capture.md's 140+) is what lets plain second differences do the work
  a motion prior would otherwise be needed for.
- **E_lim — joint-limit barrier** (weight 0.01). Log-barrier on the ranges already authored in
  `canis.ftcl` — those ranges gain their third consumer (after domain-randomisation bounds and
  conditioning normalisation), which is more evidence they were worth authoring once, centrally.
- **E_phys — physics plausibility, as a GATE, not a loop term.** On the stage-2 trajectory:
  (a) implied torques from `mj_inverse` within a strength envelope scaled from tune.py's
  statics (`tau_static`/`tau_ref`); (b) no floor penetration; (c) stance-foot slip below a
  threshold while in contact. Frames/clips failing the gate get flagged or down-weighted as AMP
  demonstrations. Promote to an in-loop penalty only if gating proves too blunt — that is the
  moment MJX earns consideration, not before.
- **Weight calibration is the yardstick's first job.** Before any real footage: replay the P2
  public dog mocap through a *synthetic* copy of the rig — project ground-truth motion through
  the four calibrated camera models, corrupt with measured DLC noise and dropout — and tune the
  four weights until the ground truth is a fixed point of the solve within tolerance. Then
  freeze them. This same replay is the "validation that makes this step honest" checkbox above,
  and it runs before the pipeline ever sees a real dog.

#### Anatomy before motion, and what one θ evaluation actually costs

Per CMA-ES candidate θ: regenerate the model **once** (build + tune.py). tune.py is one-shot
statics — seat the feet, one `mj_inverse`, mass-matrix reads — milliseconds, *not* a closed-loop
or RL process, so it is entirely affordable per candidate (and only E_phys even consults its
output; the kinematic terms need only geometry). `validate.py`'s collapse-counting never runs
inside the fit. Then pose-fit a fixed calm-clip subset (~300 frames: every 4th frame of a few
standing/walking clips), warm-started from the best-so-far member's trajectory. Seconds per
candidate ⇒ 13 × a few hundred generations = the "hours, not RL training" the section above
promises — and that budget holds *because of* the warm starts; cold inner solves would make it
days. Morph first on the calm clips, freeze θ, then solve motion per clip, exactly as already
stated.

**Initialisation is nearly free, and this is where the 4-camera decision pays.** RANSAC-
triangulated keypoints give 3D limb-segment lengths *directly*: θ₀ = per-segment medians over
the calm clips, q₀ = bone-length-aware analytic IK on the triangulated points. CMA-ES polishes;
it does not search blind.

**Identifiability honesty:** `body_mass` and the mass *distribution* are kinematically invisible
— no reprojection or silhouette term sees them; they enter only through the E_phys gate, weakly.
So **weigh the animal** (one number, a bathroom scale) and let the rig's density model
distribute it. Girth comes from E_sil; lengths from triangulation; mass from a scale. Do not
pretend video recovers what it cannot.

#### The keypoint↔rig interface must be authored — and the rig cannot express it yet

`creaturelab` currently has **no site support at all**: zero `site` occurrences in
`emit_mjcf.py`, `schema.py`, or `rigs/canis.ftcl` (checked 2026-08-08). `mj_jacSite` needs
sites. Concrete work items:

- [ ] **Sites in the grammar/schema/emitter**: `site "name" { on <bone>  pos <expr expr expr> }`
      with positions written in morph parameters like every other dimension in the rig — a site
      at the lateral humeral epicondyle must move when `humerus_len` does; a numeric offset goes
      stale under morph, which is this project's founding rule applied to landmarks. Emit as
      MJCF `<site>`; list them in `rig_report.py`.
- [ ] **`notes/keypoints.yaml` — the single source of truth** for the correspondence: DLC/SLEAP
      keypoint name → rig site name, per-keypoint σ_px, class `rigid|soft` (soft — belly,
      mid-tail — gets a wide Huber), schema-versioned. The DLC/SLEAP project config is
      *generated from this file*; the two ends of the interface never restate each other.
- [ ] **canis v1 keypoint set (~21)**: nose, occiput, withers, croup/tail-base, tail-tip, and
      per leg ×4: {shoulder|hip point, elbow|stifle, carpus|hock, paw}.

#### Monocular, re-rated *(this resolves an inconsistency the plan carried for two days)*

The ordering rationale at the top of this file — and design.md's thin-link #2 — were written
when the input was in-the-wild **monocular** video: 2D→3D *lifting* through a learned prior,
ill-posed, the SFV-for-animals gap. The 2026-08-06/07 rig decision quietly changed the problem:
four calibrated 140-fps views make 3D recovery a **triangulation** problem, and multi-view
model-based fitting is well-trodden. What genuinely remains risky in P5: (a) animal keypoint
detector quality in a messy environment, (b) anatomy identifiability (girth/mass — addressed
above), and (c) downstream, whether AMP + muscles can imitate the recovered motion. Not the 3D
lifting. Consequences: **P5 may be pulled earlier** if logistics allow — the P2-first yardstick
argument still holds and costs nothing to keep — and **monocular is restated as a separate,
later ambition**, downstream of P5 rather than P5 itself: the multi-view pipeline mass-produces
exactly the paired (video, 3D-motion) data a monocular lifter would train on. design.md and the
ordering block at the top of this file are updated to match.

---

## P6 — Stylization / morph transfer  `[ ]`

- [ ] Push morph params outside the training distribution (cartoon proportions)
- [ ] AMP discriminator trained on *realistic* motion, applied to the *stylized* body, to
      hold style in place through retargeting
- [ ] Topology changes (extra limbs) — genuinely open

### Converting a *captured* animal — order of operations, and the "train twice" worry
*(added 2026-08-07. The worry, verbatim: "the only way to transform an animal is to include it
into the training stage, and you can't get the bones, joints, muscles, etc. information to change
it in the first place until after you've trained it — so you'd have to train it twice." The worry
rests on one wrong premise, and the rest follows once it is removed.)*

**The premise to remove: anatomy does not come from training.** The bones/joints/muscles of a
captured animal are not an *output* of the expensive RL stage — they are the **morph vector of a
template that existed before any camera rolled**, estimated by P5's *fitting* half (an optimisation,
hours). So the anatomy is on the table, fully editable, **before** the first policy gradient. What
training produces is only the *motion* that drives that anatomy.

The order that makes conversion cost one training run:

```
1. author the body-plan template          (.ftcl rig + morph ranges — exists: canis)
2. capture + FIT  →  θ_animal             (P5's anatomy half; cheap; no RL involved)
3. [optional] GUI-edit anatomy NOW        (P8 editor works pre-training — tune/validate
                                           give physical feedback with no policy at all)
4. train ONCE, conditioned on morph,      (P4: randomise over a NEIGHBOURHOOD that
   randomised around θ_animal              covers θ_animal, every anticipated edit
                                           direction, and a generous margin)
5. GUI-edit after training                (free inside the trained region; the same
                                           policy walks the edited body — that is P4's
                                           entire claim)
6. edits that LEAVE the trained region    (P6 fine-tune: AMP discriminator on the
                                           realistic motion holds the style — a
                                           fine-tune, not a from-scratch retrain)
```

- [ ] **So the real design rule is: decide the morph *ranges* generously before step 4** — you do
      not need to know the exact target creature at training time, only the *region* your edits will
      live in. Randomisation cost scales with the dimensionality of the command space (see P10),
      not with how wide each range is, so generosity is cheap. "Knowing the target during training"
      collapses to "make sure the target is inside the randomised region" — a far weaker and far
      cheaper requirement than it sounds.
- [ ] **Train-twice is the exception path, not the architecture** — it happens only when a
      *post-hoc* edit leaves the trained region, and even then step 6 is a style-anchored fine-tune.
      Budget for one full training run plus fine-tunes, not N full runs.
- [ ] **What genuinely cannot be edited for free: anything not in the morph vector.** A parameter
      that was never symbolic (a hardcoded attachment point, a new joint, a topology change) is a
      *different body plan*, and no amount of conditioning rescues it. This is the standing
      constraint "the morph space is per-body-plan" wearing its practical face — and it is the
      strongest argument for erring on the side of *more* symbolic parameters in the rig.

---

## P7 — Look: soft tissue, fur, render  `[ ]`

Two tiers, deliberately. Flesh FEM inside the control loop is prohibitively slow and buys
the controller nearly nothing.

- [ ] **In-loop:** rigid skeletal sim only
- [ ] **Offline:** skinning + secondary dynamics (mass-spring / quasistatic FEM / learned
      deformer) as a render-time pass
- [ ] Pose → `.ftsl` bake, rendered by ftrace. **Note there is no `creaturelab/emit_ftsl.py` yet** —
      `model.py`'s docstring says "the emitters turn it into MJCF or FTSL", but only `emit_mjcf.py`
      exists. FTSL is y-up and the sim side is z-up (`schema.py`), so the renderer emitter owns
      that conversion.
- [ ] **Skin as a surface, not just a word.** Today "skin" means nothing here: there is no mesh, no
      bind pose, no weights. Needs (a) a skin mesh authored/fitted against the rig, (b) skinning
      weights, (c) sliding over fascia rather than rigidly following bone — the sliding is most of
      what separates a real animal from a CG one.
- [ ] **Eyes** *(added 2026-08-06 — was missing entirely; zero hits repo-wide)*. Small in geometry,
      enormous in read, and they belong in *both* tiers: **control** (gaze as a first-class output —
      saccades, smooth pursuit, vestibulo-ocular reflex stabilising gaze against head bob during
      locomotion; head/neck orientation is downstream of where the animal is looking, so this is not
      cosmetic) and **look** (a layered refractive eye — cornea/aqueous/lens with distinct IORs,
      wet-surface specular, a real caustic on the iris behind the cornea). ftrace's spectral
      dielectric path already renders that tier; the missing part is the anatomy and the gaze
      controller. Also: pupil dilation as a morph/state knob, and a nictitating membrane for species
      that have one.
### Fur — scoped deliberately narrow  *(scoping decided 2026-08-06)*

Fur is the **least novel thing in this project** — thousands of people have shipped it; nobody has
shipped muscle-actuated animal control fit from video. So it stays last and stays small. What
follows is the scope, and just as importantly what is deliberately *excluded*.

- [x] **Gate: a curve primitive in ftrace — SATISFIED 2026-08-07.** ftrace v0.151.0 added the
      `curve` primitive (strands as round-cone chains, CPU **and** CUDA — 74× GPU on the fiber
      workload, guarded by `-checkcurve` incl. an fp32-conditioning fix that a probe measured, not
      guessed). v0.152.0 added the **`fur { on "<object>" … }` groom generator** (`src/fur.h`):
      area-uniform roots over a named sphere/mesh/quad/triangle, closed-form strand shaping
      (`lift`/`jitter`, `direction`+`comb`, `gravity`+`droop`, `curl`, `clump` via nearest-guide
      Voronoi), emitting ordinary `Curve`/`CurveSeg` records so BVH/CPU/CUDA/raster needed zero new
      code. Demo: `scenes/fur_creature.ftsl`, 308 506 strands over a sphere-built animal, CPU↔GPU
      parity 0.18 %. v0.153.0 teaches loom to emit `Strand`. **Remaining renderer-side fur work now
      lives in `../forward raytracer/TODO.md` §P2 (aggregate-BSDF LOD / cost) and §P3 (fiber
      BCSDF)** — improving/optimising fur means working those two items, not building a primitive.
- [ ] **Shading: Yan-style double-cylinder, not plain Marschner.** Marschner was derived for *human
      hair*. Animal fur has a **medulla** — a hollow scattering core — which is why Yan et al.
      (2015/2017) added the TT^s/TRT^s lobes. Plain Marschner on a dog reads as plastic doll hair.
      *(Written down as ftrace TODO §P3; do it there.)*
- [ ] **Inter-fiber multiple scattering (dual scattering, Zinke 2008) is not optional.** Light coats
      are *dominated* by it; white/cream fur without it renders dark and dead. This is the single
      most common "why does my fur look wrong", so budget for it up front rather than bolting it on.
      *(Also ftrace §P3.)*
- [ ] **Antialiasing / LOD is the real technical risk.** A hair is sub-pixel (often 1/5–1/50 of a
      pixel), so in a backward path tracer this appears as **variance**, not jaggies — a ray hits a
      fiber or misses and the two answers differ wildly. You don't antialias fur, you average it,
      expensively. Past some distance individual fibers must give way to an aggregate volumetric
      BSDF, and making that transition not pop is where the effort actually goes.
      **CORRECTED 2026-08-07 — fur is NOT backward-mode-only; the earlier claim here was measured
      and retracted** (ftrace §P2). "Sub-pixel" is a *camera-side* statement; from the **light's**
      side a coat is a dense mat covering a large solid angle — one of the easiest targets in the
      room — and forward modes splat flux, area-averaging every pixel by construction, so their
      per-pixel error follows flux, not geometric density. Measured on `fur_creature.ftsl`: the
      fur-vs-bare-room error penalty is **3.00× forward vs 4.56× backward** — the forward penalty
      is the *smaller*. The aggregate LOD is still worth building, but in forward modes it buys
      **cost**, not variance. Render fur in whichever mode the shot wants.
- [ ] **Procedural groom, driven by the anatomical layer — LOCKED DESIGN DECISION, decided now
      even though the work is late.** *(Status 2026-08-07: the mechanical half now exists — ftrace's
      `fur { }` block **is** a small-parameter-vector procedural groom, which is exactly the shape
      `notes/capture.md`'s analysis-by-synthesis fitting wants. What remains ours is the
      **anatomy→groom mapping**: driving those parameters — especially the direction field — from
      muscle topology / strain / contact history instead of hand-set constants.)* The groom (fiber
      generation, guide curves, clumping, density, length, guard-hair vs underfur populations,
      direction field) must be *parameterised by the anatomy*, never hand-painted. Two reasons, and the second is the load-bearing one:
      1. it's the only genuinely novel part of the fur work — direction following muscle topology,
         clumping from strain and contact history, which is the "we own both layers" payoff;
         **and it makes the groom *fittable from photographs*** — a procedural groom is a ~10–30
         number parameter vector, so capture becomes analysis-by-synthesis on summary statistics
         rather than per-strand reconstruction. See `notes/capture.md`;
      2. **a painted groom does not survive P4's morphing.** Change the body and painted maps are
         stale, which would quietly destroy the thing that justifies this whole architecture.
      Note "sufficiently random yet orderly" is precisely a *correlated*-randomness problem:
      guide curves give order, per-hair jitter gives randomness, and **clumping gives the
      correlation**. Omit clumping and fur reads as carpet. It is also inherently non-stationary —
      whorls, cowlicks, parting lines, belly-vs-back density — and you cannot comb a sphere, so the
      direction field *must* have singularities and where they go is a design decision, not a
      computation. Same missing vocabulary as the texture bullet below (§O3 non-stationary,
      §O4 flow-aligned).
- [ ] **Dynamics: quasi-static deflection only. NO simulated fur.** ("Dynamics" = each hair simulated
      as a stiff segment chain responding to inertia, gravity, wind and body collision — so the coat
      lags, overshoots and settles when the animal moves.) Excluded because:
      - **swing amplitude scales with length.** A 2 cm hair's tip cannot travel more than 2 cm even
        in principle, and with real bending stiffness far less — sub-pixel at normal viewing
        distance. Only long groups (tail plume, ear feathering, a mane) swing visibly.
      - **the killer is statefulness, not CPU.** Simulated fur at frame 500 depends on frames
        0–499, which destroys the embarrassingly-parallel property of frame rendering: no
        distributing frames across machines, no rendering one frame in isolation to test a lighting
        change. That's worse for this project than the compute cost.
      **Do instead:** closed-form deflection as a function of local surface velocity + wind vector,
      evaluated fresh each frame — stateless, parallel, and it buys the "fur leans back when the dog
      runs / ripples in wind" read for almost nothing. Loses lag, overshoot and settle, which is
      exactly the part only long fur shows.
      *If long fur is ever genuinely needed*, the standard escape is: simulate **guide curves only**
      (~10³) in a separate cached pass written to disk, then render statelessly from the cache — a
      pipeline stage, not a flag. Don't do it until a specific shot demands it.
- [ ] Non-stationary texture: curvature/cavity masks, spatially-varying noise parameters,
      domain warping, reaction–diffusion for coat patterning — driven by the *anatomical*
      layer (strain, muscle proximity, contact history). This is the piece nobody has,
      and it's only possible because we own both layers.
      **Renderer-side prerequisites are now written down**: `../forward raytracer/TODO.md` **§O**
      (audit 2026-08-06). ftrace today has value noise + POV's exact Perlin + the fBm/ridged
      multifractals, and a CPU/GPU-identical expression VM that can already *express* spatially
      varying parameters — but it has **no cellular/Worley/Voronoi** (§O1), **no vector noise for
      domain warping** (§O2), and **no reaction–diffusion** (§O6). Those three are exactly this
      bullet's vocabulary, so build them there and this becomes a binding exercise.

---

## P10 — Layered control: from "flee" down to "that foot, there, now"  `[ ]`
*(added 2026-08-06. Do the *design* early — it constrains P1–P4's training — and the
implementation alongside P4, since most of it is conditioning inputs.)*

**The governing principle, and it is P4's lesson generalised.** P4 already says: condition the policy
on the morphology vector *from the start*, because train-then-morph-then-finetune destroys the motion
character. The same is true of **every control channel**. A gaze target, a style knob, a
part-specific goal — each must be present as a conditioning input *during training*, randomised, or
the finished policy will ignore it or fight it. This is one principle, not four features, and getting
it wrong costs a retrain every time. **So the control interface must be designed before P2 trains
anything expensive**, even though it's built later.

### The layers

| level | what you say | mechanism |
|---|---|---|
| 0 | "go there", "flee", "follow that" | path / intent planner above the policy |
| 1 | speed, gait, heading, **wary / exhausted / injured / aggressive** | conditioning knobs |
| 2 | *manner*, blended continuously | learned latent skill space |
| 3 | "look at that while you keep trotting", "favour the left fore" | part-specific goals |
| 4 | "plant the left forefoot **here** at t=1.2 s" | shot-specific hard constraint |

- [ ] **Level 1 — the knobs.** design.md already guessed at these ("menace is probably a knob, not a
      layer"); this is where that gets cashed out. Randomise them during P2/P3 training.
- [ ] **Level 2 — latent skill space.** This is **ASE** (Peng et al. 2022), the direct successor to
      the AMP already chosen in P2 — so the plan is already pointed at it and this is a smaller step
      than it looks. Gives continuous blending between manners and is the natural drive target for
      P9's human performance.
- [ ] **Level 3 — part-specific goals, WITHOUT overriding actuators.** The failure mode to avoid:
      a monolithic policy emits a whole-body action vector, so overriding the head leaves the rest of
      the body acting on stale assumptions, and the creature falls. **Inject the request as a goal in
      the observation + reward and let the policy satisfy it**, adapting the rest of the body itself.
      The policy stays in charge, so it stays stable. (Residual-on-output and per-limb sub-policy
      decomposition are the alternatives; both are more fragile. Try goal-injection first.)
- [ ] **Level 4 — accept that this may not be a policy at all.** "Foot exactly there at exactly that
      frame" is a hard constraint and a policy is a soft thing. Realistic answer is **hybrid**:
      learned controller produces the base motion, then a physics-aware trajectory-optimisation pass
      enforces the shot constraint offline. Write this down now so nobody burns a week trying to make
      the policy hit an exact contact.
### Forward model and planning — **use MuJoCo as the model; do not learn one**

*(refined 2026-08-06.)* The motivating case: balance, and "jump exactly that far". Both need
predicting what happens **if** a set of muscles is fired — a *counterfactual* query, evaluated
without executing, many times, in order to **choose** the action. That is planning/search, not the
narrow delay-compensation use a forward model is usually introduced for.

- [ ] **First, do not build this to get abstraction.** "Say *walk* instead of specifying each leg
      over time" comes free from a **model-free** policy with command inputs — the planning is
      amortised into the weights during training, and runtime is one microsecond forward pass with
      no search. That is exactly level 1 above. Abstraction is not a reason to build a model.
- [ ] **The real payoff is zero-shot goals.** A model-free policy can only pursue goals it was
      *trained* on. A model + planner can pursue one specified for the first time at runtime —
      "clear that 2.3 m gap from this stride phase" — by searching for a takeoff impulse whose
      predicted arc satisfies it. This *is* the "jump just the right length" case and it is a genuine
      capability difference.
- [ ] **Use the simulator as the model. `MuJoCo` already *is* `f(s,a) → s'`, exactly, contact
      included.** Training a net to approximate a simulator we are already running is strictly worse
      unless we need it *faster than real time*. Prior art to follow rather than reinvent:
      **MuJoCo MPC** (DeepMind 2022) — sampling-based predictive control using MuJoCo as its own
      model, in real time. Rolling out the true dynamics avoids all three classic failure modes:
      - **compounding error** — 1% one-step error over a 50-step rollout is garbage; this is what
        kept model-based RL marginal for years;
      - **contact discontinuity** — a foot touches or it does not; smooth nets model that badly and
        the errors land *exactly* at the instants that decide balance and landing;
      - **adversarial exploitation** — a planner searching a learned model *finds its errors*, and
        will cheerfully discover an action sequence the model believes yields a 10 m jump.
- [ ] **Then distil.** Plan offline with MuJoCo-MPC for the hard targeted actions, train the fast
      policy to reproduce them. Keeps the planner's generality and the policy's microsecond runtime —
      which is also what keeps **P9's live-puppeteering latency budget** intact.
- [ ] **Learn a model only if profiled speed demands it**, and treat it as an optimisation with a
      known-good reference (the real sim) to score it against — not as the primary design.
- [ ] **Balance is not planning — keep the two mechanisms separate.** Balance is fast reactive
      feedback; biologically it is *spinal* (reflex arcs, central pattern generators), on a loop far
      faster than deliberation. Targeted jumping is planning. Real motor control layers these —
      spinal reflexes/CPGs for balance and gait rhythm, cerebellum for the forward model and fine
      timing, cortex for planning — which maps onto this section's levels almost directly, and is
      corroborating evidence the layering is right. Implement balance in the policy (fast, reactive,
      trained), planning above it.
- [ ] **Keep the model slightly wrong on purpose.** An animal that occasionally misjudges a gap and
      stumbles reads as alive; one that never does reads as machinery. Same family as P1's
      conduction-delay item — imperfection is character, and it is nearly free.

### Channel inventory — and the discipline that keeps it small

*(2026-08-06, from a brainstorm of "everything an animal might typically do".)*

**That enumeration is a TEST SUITE, not an interface.** The distinction is load-bearing. A list of
behaviours is how you *verify* the interface is sufficient; it is not the interface itself, and
building one channel per listed behaviour produces a command space too large to train.

**The hard constraint nobody states: the conditioning vector is not free.** Every channel must be
randomised over *jointly*, so coverage cost grows with the dimensionality of the command space, not
with the number of behaviours it can express. Hence the discipline:

> **Anything that can emerge should emerge, not be commanded.**

**One mechanism, not N channels.** The policy already actuates every joint. So part-specific control
is **one** generic "goal for body-part-set S" channel (level 3 above) that covers "wag tail", "flick
the left ear", "curve the spine 30° right" and everything unanticipated. Enumerating body parts as
separate channels is how you get a 200-dimensional command space by accident.

**The five categories, which differ in cost by ~3 orders of magnitude:**

| category | examples | mechanism | cost |
|---|---|---|---|
| pose goals | any limb segment, spine curvature, ears, head, tail, jaw opening, claws | one goal-injection channel (L3) | low |
| locomotion params | speed, heading, turn-in-place | ~4-number command vector (L1) | low |
| targeted ballistic | "jump exactly there" | goal + planner (MuJoCo-MPC) | medium |
| affect | happy / pain / sad / angry / sleepy | global modulation vector | low–medium |
| **object interaction** | eat, drink, play-with-X, fight-and-win | **tasks needing a world — NOT control channels** | **enormous** |

- [ ] **Gait must NOT be a command — let it emerge.** Animals select gait by energetics: Hoyt &
      Taylor (1981) showed horses pick the gait minimising metabolic cost at each speed, with sharp
      transition thresholds. Speed command + energy penalty ⇒ walk→trot→gallop emerges *correctly
      placed*. Commanding it spends training coverage and lets you request physically absurd
      speed/gait pairs. Keep an explicit override for stylization only.
- [ ] **Affect is a modulation vector, not a set of poses — and for quadrupeds it is NOT facial.**
      Animal affect lives in **posture and tension**: ear carriage, tail position *and motion
      quality*, body height, weight distribution fore/aft, gaze, piloerection. Implement it as a
      low-dimensional vector conditioning *everything* (posture, gait timing, muscle tension), which
      is also the cheapest way to get design.md's "menace is a knob, not a layer". Dogs are a partial
      exception, interestingly: they have a facial muscle wolves lack (levator anguli oculi medialis,
      Kaminski et al. 2019) for the inner-brow raise, apparently selected for by human attention.
- [ ] **"Be pregnant" is a MORPH, not a channel — and it is the best P4 validation target available.**
      Belly volume + added mass + shifted CoM. If morphology conditioning genuinely works, the wider
      stance, shortened stride and altered balance must emerge **with no new training**. That is a
      falsifiable test of the project's central architectural bet, worth far more than a knob.
- [ ] **Object interaction is out of the control layer entirely.** Eat / drink / play-with-X /
      fight-and-win each need an environment, physical objects, grasp and contact, and in the fight
      case a second agent and an unspecifiable reward. None reuses the others' machinery — that is
      *content*, which this project explicitly scopes out ("we build the system that makes lions
      buildable"). Same reasoning retires sex and birth: no reusable architecture in them.

**Missing from the brainstorm — add to the coverage checklist:**
- [ ] **Gaze.** The largest omission. Gaze *leads* movement; head/neck orientation is downstream of
      where the animal looks. Probably the single most expressive channel. (See P7 eyes.)
- [ ] **Breathing** — visible in the flank, rate/depth coupled to exertion and affect. Its absence is
      a large part of why CG creatures read as dead. Near-free.
- [ ] **Blinking** — same category, near-zero cost, deeply uncanny when missing.
- [ ] **Postural transitions** — lie down, sit, get up, roll over. Genuinely hard (large body
      reorientation, whole-body ground contact) and constantly needed. Absent from the brainstorm.
- [ ] **Terrain.** The brainstorm assumes flat ground. Slopes, uneven footing, stairs, slip-and-recover.
- [ ] **Landing** — the counterpart to jumping; impact absorption is its own skill and is where bad
      physics is most visible.
- [ ] **Idle behaviours** — shake off water, scratch, stretch, yawn. What makes a creature look alive
      *between* actions.
- [ ] **Piloerection** — a state knob on the **fur groom**, connecting P7's look layer to the control
      layer.

### Persistent state: three things, all of which fatigue needs

*(This entry started as "fatigue is accumulating state, not a knob that is set", which was wrong —
it is all three of the following, and the mistake generalises to every slow variable, so it is
written up here rather than as one checklist line.)*

Fatigue splits into three separate objects that are easy to conflate:

1. **Susceptibility — a MORPH parameter (P4).** How fast this creature tires, and under what
   conditions. Muscle fibre-type composition is the physical handle: a sprinter fatigues fast and
   recovers fast, a sustained trotter barely fatigues at that intensity at all. Since P4 transforms
   one creature into another, susceptibility has to ride in the **morph vector** and change with it —
   greyhound→wolf must alter the endurance curve without anyone re-authoring it. Same P4 rule as
   everywhere else: **conditioning input at training time, or the fix is a retrain.**
2. **Level — accumulating state.** Integrates work done, decays with rest. This is the part that must
   *not* be only a knob: an animal that has been galloping for five minutes should be tired with
   nobody setting anything.
3. **Level — a settable knob, with a write port.** **The argument that settles it: the simulation's
   timeline is not the film's timeline.** The animal exits frame fresh and re-enters after an implied
   three-hour chase that was never simulated. There is no run of physics connecting those shots, so
   the state has to be *writable*, not merely reachable by simulating up to it. Accumulation and
   direct setting are not opposed — accumulation is the default, the knob is the seek control.

- [ ] **Give every slow variable a write port.** Fatigue is not special; it is the first instance of
      a class. Hunger, thirst, injury, **wetness** (which is a P7 fur-look input as much as a state),
      body temperature, breath recovery, alertness/arousal, and fatigue all integrate over time and
      all get set out of continuity order by a director. Design the persistent-state block once, as a
      named vector that can be saved, restored, interpolated between shots, and set directly —
      instead of discovering the requirement separately seven times.
- [ ] **Fatigue must be implemented at BOTH levels, or it fails in a recognisable way.**
      - *Actuator level*: fatigue reduces available force from the Hill-type actuators. Physically
        real, and consequences follow for free.
      - *Conditioning level*: fatigue is in the policy's observation/conditioning vector, so it
        knows it is tired and changes strategy — shorter stride, lower head carriage, more ground
        contact time.
      - Only the first ⇒ the policy is startled by its own weakness and simply falls over. Only the
        second ⇒ theatrical tiredness with no physical consequence, which is what hand-animation
        already does badly. Both ⇒ an animal that is *actually* weaker and *knows* it.
- [ ] **Gait downgrade should then emerge, not be authored.** Fatigue lowers available force, the
      energy penalty that already selects gait (Hoyt & Taylor, above) re-optimises against the new
      cost landscape, and gallop→trot→walk should fall out of the same mechanism that produced the
      upward transitions. If it does, that is strong evidence the emergence argument is real and not
      a story told about a hand-tuned result. **Worth testing explicitly as a P10 milestone.**
- [ ] **Fatigue is also the honest resolution of the muscle-redundancy problem** noted at the end of
      this section: recovering activations from captured kinematics is underdetermined and needs an
      effort/fatigue criterion to pick a solution. The same fatigue model therefore serves both the
      control layer and the capture-inference layer — a reason to build it early rather than treat it
      as set dressing.

**Why this section exists.** The user's framing was "simulate the kinesthetic sense so the creature is
easy to control abstractly". The sensing half of that is free in simulation and needs no net; the
*abstraction* half is real and is this section. Note also that capture data cannot supply the sensing
half regardless — proprioception is unobservable from outside, and capture yields **kinematics, not
activations** (recovering muscle forces from joint angles is the underdetermined muscle-redundancy
problem, and needs an effort/fatigue criterion to resolve — OpenSim static optimisation / CMC).

---

## P8 — Live creature viewer: see and pose the knobs interactively  `[ ]`
*(added 2026-08-06 — asked for explicitly, and absent from the plan until now.)*

**What exists:** `tools/ftcl_build.py --view` opens **MuJoCo's** viewer on the generated model —
useful for checking physics, useless for judging *look*, and it has no knobs. `rig_report.py` and
`morph_sweep.py` are headless and numeric: they answer "does it stand?" in a table, and 26 morph
parameters are currently explored by *randomising and counting collapses*, never by a human moving
a slider and watching. That is a real gap — the whole justification for the `.ftcl` layer (see "Why
not just write MJCF by hand?") is that morphology is **symbolic and named**, and nobody has ever
seen those names as controls.

- [ ] **Slider-per-morph-param panel**, live: move `femur_len`, rebuild, re-tune, re-settle, redraw.
      The rebuild path is already fast and already correct (`build` → `tune` → `validate`); this is
      a UI over machinery that exists.
- [ ] **Show the tuner's verdict inline** — sag %, support margin, trunk tilt, per-joint buckling —
      so a slider that walks the body out of the feasible set says so *while you drag it*, instead
      of showing up as a collapse statistic 60 draws later. `tune.support_polygon` already returns
      the number.
- [ ] **Which renderer?** Two tiers, and they are not competing:
      - **near-term:** MuJoCo's own viewer + a param panel. Cheap, immediate, good enough to explore
        morph space and to catch known-issue #2 (incoherent draws — a 314 kg animal on a 26 cm back)
        by eye rather than by statistics.
      - **the one actually asked for:** drive **ftrace's `-raster-gpu`** so the creature is seen in
        the real renderer with real materials. This needs P7's `emit_ftsl.py` first, and then it is
        largely a *solved integration*: ftrace already has a resident `-serve` mode, an interactive
        `-explore`/`-fly` loop, and — most relevantly — the loom viewer's **`LoomBridge`** pattern
        (spawn a Python process, hold a newline-delimited-JSON channel, latest-wins on a one-slot
        job so a continuous drag costs one rebuild, adopt results on whatever frame they land).
        `creaturelab` sits in exactly loom's position in that architecture. Read
        `../forward raytracer/src/viewer_gui.cpp` (the **Live (loom)** panel) before designing this;
        do not invent a second bridge.
- [ ] **Playback, not just posing** — scrub/play a trajectory (a settled fall, later a trained gait)
      rather than only static poses. Note the loom viewer has the *same* gap and it is written up as
      §F8 there; the pacing lesson (bake-rate-paced play vs. prebaked play) transfers directly.
- [ ] **Bar:** a person can find a good-looking, physically-valid animal by dragging, in one sitting,
      without reading a table.

### The anatomy-transform editor — turn a *captured* animal into another animal by dragging
*(added 2026-08-07 — asked for explicitly.)* The slider panel above is already this editor's core;
three additions turn it from "explore morph space" into "transform this dog":

- [ ] **Load a fitted animal as the working point.** Open θ_animal from P5's fitting stage and edit
      *from* it, rather than from the template default. "Editing bones, joints, muscles and
      everything else" then means editing the symbolic parameters — which is why P6's closing rule
      (err on the side of more symbolic params; muscle attachments and strengths included, via P3's
      grammar) is a prerequisite of this editor, not a nicety.
- [ ] **Show the trained region on every slider.** Each parameter renders its P4 randomisation
      interval: inside = the existing policy drives the edited body *right now*; outside = flagged
      "leaves trained region — P6 fine-tune before this moves". This is the "train twice" worry
      (see P6) converted into a visible, per-edit cost indicator instead of a surprise after a
      wasted training run.
- [ ] **Two lifecycle modes, same UI.** *Pre-training:* edits get physical feedback only
      (tune/validate — sag, support, buckling), useful for step 3 of P6's pipeline. *Post-training:*
      the policy runs live while you drag, so you see the edited body *move*, which is the real
      acceptance test of a morph edit.
- [ ] **Bar:** load a fitted dog; drag it to jackal proportions and the trained policy trots on
      unchanged; drag it toward giraffe proportions and the UI says which sliders left the trained
      region *before* any compute is spent.

### The timeline editor — a shot's knobs over time, on a rasterized strip
*(added 2026-08-07 — asked for explicitly.)* Not a pose editor: **the tracks hold command channels,
and the policy synthesizes the motion under them.** This is design.md's directability section made
concrete — you direct the inputs, sim owns the consequences, and the UI is honest about causality.

- [ ] **Layout.** A rasterized preview strip (thumbnails — MuJoCo offscreen at first, ftrace
      `-raster-gpu` once P7's `emit_ftsl.py` exists) over one track per channel P10 defines:
      level-1 knobs (speed, heading), the affect vector, gaze target, level-3 part goals as spans,
      persistent-state write ports (fatigue, wetness, …) as step/ramp events — the "seek control"
      from P10's state section gets its UI here — and level-4 hard constraints as markers.
- [ ] **Causality is the design center.** An edit at time t changes *nothing before t* and
      invalidates *everything after it* — physics propagates forward only. Implementation: periodic
      sim-state checkpoints along the timeline; an edit re-simulates from the nearest checkpoint
      ≤ t as a **latest-wins one-slot job** (the LoomBridge pattern a third time), repainting the
      strip as frames land. A drag on a knob curve costs one re-sim, not one per mouse event.
- [ ] **Level-4 markers trigger the offline pass.** "Left forefoot *here* at t=1.2 s" is not a knob
      — placing such a marker schedules the MuJoCo-MPC trajectory-optimisation pass (P10 level 4)
      over its span, and the strip renders that span distinctly until it is solved.
- [ ] **Bar:** scrub a trained gallop; drag the speed curve down across two seconds and write
      fatigue = 0.8 at the cut; the re-simmed strip shows the animal breaking to a heavy, short
      trot — nobody posed anything.

---

## P9 — Human-performance drive: puppeteer the creature  `[ ]`
*(added 2026-08-06 — asked for explicitly, and genuinely not in the plan. Zero hits repo-wide for
`mocap`, `facial`, `expression`, `retarget`, `blendshape`, `FACS`.)*

**This is a different axis from P5, and conflating them would be a mistake.** P5 is *animal video →
learned controller*: offline, one-way, its output is a **policy**. P9 is *live human performance →
creature state*: real-time, interactive, its output is a **pose/knob stream**. P5 makes the creature
move like an animal; P9 lets a person act through it. They share a keypoint front-end and almost
nothing else.

- [ ] **Decide the target of the drive first — this is the whole design question.** Three options,
      increasingly interesting and increasingly hard:
      1. **Pose retargeting** — human joint angles → creature joint angles. Straightforward,
        well-trodden, and *throws away everything this project is for*: it bypasses the muscles and
        the learned controller, so the creature moves like a costumed human. Useful as a baseline
        and as a debugging harness, not as the goal.
      2. **Knob drive** — the performance sets *high-level* variables the policy already consumes
        (heading, speed, gait, gaze target, posture/tension, effort). The controller still produces
        the motion, so it stays biomechanically honest and stays *animal*. **This is the one that
        fits the architecture**, and it is why P4 (morphology conditioning) and a well-chosen
        observation space matter: the drive is only as expressive as the policy's inputs.
      3. **Style/latent drive** — map performance into an AMP latent so the performer supplies
        *manner* (skulking, wary, exhausted) rather than pose. Open research, but the natural
        endpoint of choosing AMP in P2.
- [ ] **Body capture.** Bias to **markerless multi-camera** (calibrated rig + triangulated 2D
      keypoints) over marker suits: it shares its whole front-end with P5's animal pipeline, needs no
      hardware the project doesn't already want, and P5 already budgets a ≥4-camera ≥120 fps rig
      (`notes/capture.md`). Marker-based Vicon/OptiTrack is more accurate but is a second pipeline
      for one purpose. **Reuse P5's rig and calibration — do not build a parallel one.**
- [ ] **Facial capture and a face rig — note the face does not exist at all yet.** `canis.ftcl` is
      25 bones of locomotor skeleton: there is no skull articulation beyond the jaw, no facial
      musculature, no ear/brow/lip/nostril controls, and no eyes (see P7). So "control facial
      expression" is currently blocked on *building a face*, not on capture. Order of work:
      1. facial anatomy in the grammar (jaw, ears, brow, lips, nostrils, eyelids, tongue) as
         muscle-driven controls, not blendshapes — blendshapes would fork the actuation model and
         would not morph with P4;
      2. **cross-species mapping is the hard part** — a human smile has no canine referent. Do not
         retarget geometry; retarget *intent* (arousal, valence, attention, threat) onto the
         creature's own species-appropriate display. This is authored mapping plus taste, not a
         solved algorithm, and it is where the character will actually live;
      3. only then capture (monocular face tracking is mature and cheap; a head-mounted camera is
         the production answer).
- [ ] **Latency budget.** "Live puppeteering" means the loop capture → solve → policy → sim → render
      must close in tens of milliseconds. That is a hard constraint on the controller's inference
      cost and it should be measured early, not discovered at the end.
- [ ] **Bar:** a person moves, and a *dog* moves — recognisably driven, recognisably still a dog.

---

## P11 — Flight  `[ ]`  ← breaks a different layer than everything above

*(2026-08-06. Asked as "what if we wanted to simulate a flying animal?". Written down because the
answer turned out to scope P4's central claim, which is worth knowing before P4 is built.)*

**The headline: almost everything in this file transfers, and the one thing that doesn't is the
physics engine.** For a walker, contact is the hard part and MuJoCo is excellent at contact. For a
flyer, contact barely matters and *the fluid is everything* — and MuJoCo's fluid model is a
quasi-steady per-geom ellipsoid approximation (blunt/slender/angular drag, Kutta lift, Magnus). It
has no wake, no circulation history, no leading-edge vortex. Flapping flight lift is *dominated* by
unsteady mechanisms, so the engine is not merely inaccurate here, it is missing the mechanism.

### The training budget picks the aerodynamics model — this is not a free choice

RL needs on the order of 10⁸–10⁹ simulation steps. That sets a hard per-step aero budget of roughly
**O(100 µs)**, which eliminates the entire high-fidelity end of the menu before quality is even
discussed:

| approach | per-step cost | verdict |
|---|---|---|
| CFD (Navier–Stokes, moving boundary) | ~1 s–min | **10⁶× over budget.** A single training run would take years. Not a fidelity trade-off — an impossibility. |
| **Blade-element / BEMT** | ~10 µs at 20 strips/wing | **The answer.** Strip the wing, compute local relative airflow, look up Cl/Cd(α), apply force + torque per strip via `xfrc_applied`. Pennycuick-style, standard in the biomechanics literature. |
| learned CFD surrogate | ~100 µs | Plausible *later*, but you must run the CFD first to have training data, so it is not a way to avoid the CFD. |

- [ ] **Write the blade-element model as an external force layer over MuJoCo**, not as a MuJoCo
      feature. Per strip: relative wind (body velocity + flapping velocity + ambient field − induced
      velocity), angle of attack, Cl/Cd polar, force at the strip's centre of pressure. This is small,
      self-contained, and testable against published lift/power curves.
- [ ] **Know what quasi-steady costs you.** It is decent (order tens of percent) for *bird* flight
      and hopeless for insect flight — quasi-steady famously underpredicts insect lift, which is the
      origin of the "bumblebees can't fly" folklore. Dickinson et al. (1999) patch it with rotational
      circulation + wake-capture terms; Ellington et al. (1996) identified the leading-edge vortex as
      the missing lift. **So: birds and bats are in scope, insects are a different project.**
- [ ] **Reynolds number is a scope boundary, not a parameter.** Re ~10² (insect) to ~10⁵ (large bird)
      is a qualitative change in the flow regime, not a coefficient to interpolate. One aero model
      does not span it.

### Glide before flap — and the reason is architectural, not incremental caution

**Gliding is the correct first target**, because a glider is a *continuous deformation of the
existing quadruped*: flying squirrel, colugo, sugar glider, Draco. A patagium is a membrane between
existing limbs — it is a morph parameter on the body plan already in `canis.ftcl`. And gliding is
aerodynamically the easy case: attached flow, quasi-steady, no unsteady mechanisms required, so
blade-element is not an approximation there so much as the actual right model.

That gliding evolved independently *many* times in mammals is itself the evidence for the claim:
evolution reached it repeatedly by incremental change, which is what "lies on a continuous manifold"
means. Powered flapping flight evolved ~4 times, ever.

- [ ] **Glider milestone:** patagium as a morph parameter, blade-element aero, and a controlled
      descent that trades height for distance. Exercises the entire aerodynamic pipeline without
      touching flapping, on a body the project already has.

### Flight is a SEPARATE morph manifold — this scopes P4's claim

You cannot interpolate between a leg and a wing. The midpoint creature has neither working legs nor
working wings, *and no policy exists for it* because the reward landscape across that path is
discontinuous. So the morph vector is **not one global space; it is per-body-plan**, and P4's bar
should read "a body never seen in training walks with the same character **within its body plan**".

That is still the right claim and still a strong one — but the unqualified version is false, and it
is much cheaper to learn that now than to discover it when a morph sweep produces an unflyable
chimera and it reads as a bug.

- [ ] **New failure mode: morph vectors can produce creatures that cannot fly at all**, in a way
      terrestrial morphs cannot produce creatures that cannot walk. Static margin (CoM vs
      aerodynamic centre), wing loading, and dihedral decide controllability outright. P4 morph
      sweeps need a **flight-viability check**, analogous to what `tune.py` already does for cost.
- [ ] **The mass ceiling is physics, not a bug.** Power required scales ≈ m^1.17 while power
      available scales ≈ m^0.67, so powered flight has a ceiling near ~15–20 kg (the heaviest flying
      birds sit right there; the far heavier extinct flyers were almost certainly soarers). **Scale
      the morph up and powered flight *should* fail** — and if soaring takes over on its own, that is
      the emergence argument validating itself again.

### Control: no equilibrium, and a limit cycle instead of a pose

- Terrestrial control has a stable reference — the ground — and a **zero-action equilibrium**: an
  animal can stand still and think. A flyer has neither. Stop flying and you fall. Everything is
  underactuated in all 6 DOF at once, continuously.
- **Flapping is a limit cycle, not a pose sequence.** The control problem is *shaping an
  oscillation*, which is exactly what P1's CPG/spinal layer is for — so that layer becomes **more**
  load-bearing here, not less. This is the strongest argument in the file for building P1's reflex
  tier properly rather than letting the policy do everything.
- [ ] **Takeoff and landing are the hard parts, and they are where flight meets contact again.**
      Takeoff is a power spike (the *legs* do much of it — birds leap first). Perching is a precision
      landing on a small, possibly moving target with an unforgiving failure mode. Cruise is the easy
      middle.
- [ ] **flap → glide → soar is the aerial gait ladder — and it should fall out of the SAME mechanism
      as gallop → trot → walk.** Flight is enormously expensive and the pectoralis fatigues; the
      energy criterion that selects terrestrial gait, plus the fatigue model above, should select the
      aerial one. If commanding it is ever needed, the emergence argument has failed somewhere.

### A wind field is infrastructure, and it pays for itself twice

- [ ] **Add a spatially-varying ambient velocity field** that the aero model samples. It is cheap,
      and it is the prerequisite for thermals, ridge lift, gusts, ground effect and therefore for
      soaring at all. **The same field drives P7's quasi-static fur/feather wind deflection** — so
      one piece of infrastructure serves both the physics layer and the look layer. (Note this is
      *not* the "object interaction is content" category from P10: a velocity field is a few numbers
      per point, not a world of graspable objects.)

### Feathers are a new tissue — closer to fur than to skin, but not the same problem

- [ ] Feathers are **not** the curve primitive from the raytracer's §P. A feather is hierarchical:
      rachis (a stiff curve) → barbs (curves) → barbules. At distance the vane is a *surface* with a
      strongly anisotropic BSDF; barbs only need resolving up close. That is a real LOD story and a
      different one from hair's.
- [ ] Counts are far friendlier than fur — ~1.5–3 k feathers on a songbird, ~25 k on a swan, versus
      millions of hairs. But each one is *individually visible and individually posed*, so the
      statistical treatment that saves fur is unavailable. **Fur is a texture problem; feathers are a
      posing problem.**
- [ ] Wing area changes drastically between downstroke and upstroke (flexion, primary separation),
      and feathers slide over one another. The wing is a **deforming surface**, not a rigid linkage —
      and for bats it is worse still: an elastic membrane with muscles embedded *in* it.
- [ ] **Iridescence is where ftrace's spectral machinery finally earns its keep.** Structural colour
      in feathers is a multilayer/photonic-crystal effect (hummingbird gorget: stacked melanosome
      platelets; peacock barbule: a 2-D photonic crystal) — and ftrace *already* has `ThinFilm`,
      `Multilayer`, `Grating` and `Layered` material types and is spectral end-to-end. An RGB
      renderer cannot do this correctly at all. Nearly free capability, high visual payoff.

### What the capture rig cannot give us

`notes/capture.md` is built for a ground animal at ~3 m in a fixed volume, and flight defeats it on
three counts: the subject **leaves the volume**; wingbeat is 5–50 Hz (pigeon ~8 Hz, hummingbird
50–80 Hz) so 120 fps yields ~15 samples per pigeon beat and nothing usable for a hummingbird; and
joint-centre triangulation is **ill-posed on a wing**, where the bones are buried under a much larger
deforming feather surface.

- [ ] Accept that **flight reference motion must come from published biomechanics / wind-tunnel work,
      not from our own rig** — or go reference-free. Reference-free is *more* plausible for flight
      than for walking, interestingly, because the aerodynamic objective constrains the solution so
      much harder than the walking objective does.

### What transfers unchanged (most of it)

P10's layered control is untouched — level 1 just becomes "fly there at speed v". The persistent-state
block, fatigue, breathing (coupled to a much larger exertion range), gaze, and the affect vector all
work as written. **Gaze is a standout: bird head stabilisation is famously precise and visually
iconic, and it falls straight out of "hold gaze fixed" as a goal** with no new machinery. And the
legs do not go away — perching, takeoff and walking still need them, so flight is *additive* to the
quadruped work rather than a replacement for it.

---

## Cross-cutting

- [x] `design.md` kept current (it is the architecture record) — ongoing, not a milestone
- [x] `known-issues.md` for bugs/tech debt — 3 open, 6 resolved
- [x] Version control: `git init`ed at the close of P0; `out/`, `runs/`, `*.pt` ignored
- [ ] Determinism: seed everything; training runs must be re-runnable
- [ ] Literature sweep before assuming any link is unclaimed — this field moves fast and
      video→animal-motion is actively worked on

## Explicitly NOT doing
- Building a *lion*. That's a content problem — artists, years. We're building the system
  that makes lions buildable.
- Reimplementing anything in the "already exists" list in `design.md`.
