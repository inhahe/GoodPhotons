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
P7 soft tissue + skin + eyes + fur + render through ftrace
P8 live creature viewer     (not strictly ordered — pull it forward the moment
                             morph-space exploration starts costing you time)
P9 human-performance drive  (needs P4's conditioned policy + P7's face to be
                             the interesting version rather than pose retargeting)
P10 layered control         (DESIGN before P2; BUILD alongside P4 — see below)
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

## P1 — Torque-actuated quadruped + PPO  `[ ]`

Smoke-test the entire loop end to end on a body whose dynamics we trust.

- [ ] Gym-style env wrapping the generated model
- [ ] **Proprioceptive observation space — decide this before the first PPO run, not after.**
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
- [ ] **Proprioceptive delay and noise, scaled by body size.** Real conduction latency is 10–40 ms,
      longer for a hind limb than a fore, and >100 ms in a large animal. Training on perfect
      instantaneous state yields superhuman reflexes and a twitchy, over-corrected gait — a genuine
      CG tell, the motor-control analogue of the fixed-pivot knee. **Delay should be derived from
      the morph vector** (limb length ÷ conduction velocity), not authored, so a scaled-up creature
      moves *heavier* for free. Directly serves design.md's "old / exhausted / 40 kg heavier is one
      knob" claim.
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
- [ ] Capture rig — **`notes/capture.md` now exists** (it was a dangling reference until 2026-08-06).
      4 cameras minimum for a quadruped (they self-occlude far worse than humans), ≥120 fps.
      Note that page's central rule: the rig has **two modes that must never share a recording** —
      motion (fast, whole-animal, whatever resolution survives the fps budget) and groom/appearance
      (stills, full sensor, close, controlled light, still subject). Same hardware, opposite settings.

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

- [ ] **Gate: a curve primitive in ftrace.** Verified 2026-08-06 — ftrace has **none**: zero hits for
      `fiber`, `ribbon`, `bezier`, `b-spline`, and all four `hair` matches are the English idiom
      ("a hair negative"). Without one, fur must be triangle ribbons at ~64 tris/hair, i.e. 10⁸–10⁹
      triangles for a dog (~1–10M hairs × 8–32 segments). Not viable. Curve primitive + its own BVH
      is the entry ticket and nothing else starts until it exists.
- [ ] **Shading: Yan-style double-cylinder, not plain Marschner.** Marschner was derived for *human
      hair*. Animal fur has a **medulla** — a hollow scattering core — which is why Yan et al.
      (2015/2017) added the TT^s/TRT^s lobes. Plain Marschner on a dog reads as plastic doll hair.
- [ ] **Inter-fiber multiple scattering (dual scattering, Zinke 2008) is not optional.** Light coats
      are *dominated* by it; white/cream fur without it renders dark and dead. This is the single
      most common "why does my fur look wrong", so budget for it up front rather than bolting it on.
- [ ] **Antialiasing / LOD is the real technical risk.** A hair is sub-pixel (often 1/5–1/50 of a
      pixel), so in a path tracer this appears as **variance**, not jaggies — a ray hits a fiber or
      misses and the two answers differ wildly. You don't antialias fur, you average it, expensively.
      Past some distance individual fibers must give way to an aggregate volumetric BSDF, and making
      that transition not pop is where the effort actually goes.
      **Backward mode only** (`-mode L`): a forward/photon tracer is the wrong vehicle, because
      photons cast from a light almost never usefully hit sub-pixel fibers.
- [ ] **Procedural groom, driven by the anatomical layer — LOCKED DESIGN DECISION, decided now
      even though the work is late.** The groom (fiber generation, guide curves, clumping, density,
      length, guard-hair vs underfur populations, direction field) must be *parameterised by the
      anatomy*, never hand-painted. Two reasons, and the second is the load-bearing one:
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
- [ ] **A learned forward model** — state + motor command → predicted next sensory state. This is the
      one place a small net genuinely belongs (see P1's proprioception items: the net is for
      *predicting*, not for *sensing*, since sensing is exact in sim). It is what lets a controller
      act through the delays added in P1, it is standard model-based-RL machinery, and its learned
      representation is the substrate levels 0–2 operate on.

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
