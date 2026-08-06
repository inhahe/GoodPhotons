# Creature — TODO

A physically-based animal: articulated skeleton, Hill-type muscle/tendon actuators,
soft-tissue deformation, fur — driven by a **learned** controller rather than keyframes,
fit to real animal motion, and morphable into stylized/fictional creatures **without
losing the learned motion character**.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## The ordering question (read this first)

**Yes — video comes late, deliberately.** Monocular video → reliable 3D animal motion is
simultaneously the *most novel* and the *most likely to disappoint* link in the whole
system. Everything else is well-trodden. So the plan front-loads the parts that are known
to work, and arrives at video with a working simulator, a working controller, and — this
is the point — **a ground-truth yardstick**: public dog mocap of the same gaits, so the
video pipeline can be scored in centimetres instead of vibes.

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
P7 soft tissue + fur + render through ftrace
```

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
- [x] **Validation:** MuJoCo loads it, it stands under gravity, and sweeping a morph
      param regenerates a *different but still valid* body with no hand-editing.
      - `rig_report.py` — stands across `body_scale` 0.6 → 1.7 (sag 6.9% → 5.2% of
        withers; note it gets *proportionally better* with size, which is exactly what
        scale-correct gains are supposed to do), and across a 32× `body_mass` range at a
        flat 4.8–4.9%.
      - `morph_sweep.py` — full-body randomisation, 60 draws per width:
        **100% stand at scales 0.25 / 0.5 / 0.75; 95% at scale 1.0.**
        The residual 5% is not a tuner failure: scale 1.0 samples all 26 parameters
        independently, so nothing stops it drawing a 314 kg animal on a 26 cm back. P4
        should either couple stance angles to limb lengths or reject on the measured
        support margin that `tune.support_polygon` already returns.
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

## P1 — Torque-actuated quadruped + PPO  `[ ]`

Smoke-test the entire loop end to end on a body whose dynamics we trust.

- [ ] Gym-style env wrapping the generated model
- [ ] PPO baseline, flat ground, forward-velocity reward
- [ ] Termination on fall, action-rate + energy penalties
- [ ] **Bar:** a stable gait emerges. It will look bad. That is fine — this step is
      testing the plumbing, not the motion.
- [ ] Headless training + checkpointing so runs survive between sessions

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
- [ ] **Bar:** a body never seen in training walks with the same character
- [ ] Prior art: MetaMorph; Shared Modular Policies

---

## P5 — Video fitting  `[ ]`  ← the research risk

- [ ] 2D keypoints (DeepLabCut or SLEAP; some hand-labelling unavoidable)
- [ ] Camera intrinsics/extrinsics; fit **directly in the simulator's own joint
      parameterisation** by minimising 2D reprojection error — this sidesteps an entire
      retargeting stage that a separate 3D animal model would force on us
- [ ] Temporal smoothness + physics-plausibility regularisers
- [ ] **The validation that makes this step honest:** run the pipeline on footage of a
      *dog*, compare against the P2 mocap for the same gait, and report a hard error
      number. That says empirically whether the ambitious version is viable, or whether
      video should be demoted to style reference only.
- [ ] Capture rig (see `notes/capture.md`) — 4 cameras minimum for a quadruped, ≥120fps

---

## P6 — Stylization / morph transfer  `[ ]`

- [ ] Push morph params outside the training distribution (cartoon proportions)
- [ ] AMP discriminator trained on *realistic* motion, applied to the *stylized* body, to
      hold style in place through retargeting
- [ ] Topology changes (extra limbs) — genuinely open

---

## P7 — Look: soft tissue, fur, render  `[ ]`

Two tiers, deliberately. Flesh FEM inside the control loop is prohibitively slow and buys
the controller nearly nothing.

- [ ] **In-loop:** rigid skeletal sim only
- [ ] **Offline:** skinning + secondary dynamics (mass-spring / quasistatic FEM / learned
      deformer) as a render-time pass
- [ ] Pose → `.ftsl` bake, rendered by ftrace
- [ ] Fur: fiber BCSDF (Marschner / d'Eon / Yan) + dual scattering — a natural extension
      of ftrace's existing spectral + volume machinery rather than a new dependency
- [ ] Non-stationary texture: curvature/cavity masks, spatially-varying noise parameters,
      domain warping, reaction–diffusion for coat patterning — driven by the *anatomical*
      layer (strain, muscle proximity, contact history). This is the piece nobody has,
      and it's only possible because we own both layers.

---

## Cross-cutting

- [ ] `design.md` kept current (it is the architecture record)
- [ ] `known-issues.md` for bugs/tech debt
- [ ] Determinism: seed everything; training runs must be re-runnable
- [ ] Literature sweep before assuming any link is unclaimed — this field moves fast and
      video→animal-motion is actively worked on

## Explicitly NOT doing
- Building a *lion*. That's a content problem — artists, years. We're building the system
  that makes lions buildable.
- Reimplementing anything in the "already exists" list in `design.md`.
