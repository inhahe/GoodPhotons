# Creature — design

## What this is

A system for building physically-simulated animals whose motion is **learned** rather than
keyframed: articulated skeleton → muscle/tendon actuators → soft tissue → fur, driven by a
neural controller fit to real animal movement, and morphable into stylized or fictional
creatures while preserving the learned motion character.

The goal is **not** to build a specific creature. A convincing lion is a content problem
(artists, years). The system that makes lions buildable is an architecture problem, and
architecture is where a small effort can beat a large one.

## Where the novelty actually is

Most components already exist and should not be reinvented:

| exists | representative work |
|---|---|
| musculoskeletal animal models | OpenSim equine/canine models |
| muscle-actuated learned control | Lee et al. 2019 (346-muscle human); MyoSuite |
| physics quadruped control from mocap | DeepMimic; Mode-Adaptive NN; AMP; Peng dog→robot |
| physics control from video (**human**) | SFV (Peng et al. 2018) |
| 3D animal reconstruction from images | SMAL family, BITE, LASSIE, MagicPony, 3D-Fauna |
| morphology-conditioned policies | MetaMorph; Shared Modular Policies |
| fur / soft tissue | Marschner–d'Eon–Yan fiber BSDFs; Ziva; Weta Tissue |

Three links are genuinely thin, and they are where effort should go:

1. **Muscle-actuated control for *animals*, at scale.** The impressive muscle-RL work is
   almost entirely human. Animal musculoskeletal models exist, but they were built for
   biomechanical *analysis*, not to sit inside a massively-parallel RL loop.
2. **Physics-based control fit from in-the-wild monocular *animal* video.** SFV did this
   for humans — helped enormously by SMPL (a canonical parametric body) and strong human
   pose estimators. Animals have neither.
3. **Style-preserving morphology transfer for learned controllers.** Especially under
   muscle actuation, where changing the body changes the *actuators themselves*.

Framed differently: VFX owns world-class muscle/tissue/fur but keyframes or
performance-captures its creatures; robotics/RL owns learned quadruped locomotion but on
rigid torque-actuated robots with no flesh and no biological fidelity. This project sits
in that gap.

**The novelty map and the risk map are the same map.** Video fitting is both the most
novel link and the most likely to disappoint. That is not a coincidence, and it dictates
the staging in `todo.md`: build the well-trodden 80% fast, then push on the hard parts —
and arrive at video already holding clean mocap of the same gaits as a yardstick.

## Architecture

### The central decision: the creature layer is a *generator*, not a runtime

FTSL (the raytracer's scene language, `../forward raytracer`) bakes everything at load
time — group transforms collapse into world-space triangles, the BVH builds once,
materials upload once. That is exactly why ftrace is fast and bit-reproducible. Threading
simulation state and time through it would contaminate a well-tuned renderer with concerns
that don't belong to it.

So nothing here runs inside ftrace. The creature layer *compiles*:

```
                        ┌──→ MJCF   → MuJoCo   (simulate, train)
   rig.ftcl  ──────────┼──→ FTSL   → ftrace   (render)
   + morph vector       └──→ USD    → art tools (later)
```

FTSL becomes a compiled *output* format, which is what it is already good at. What is
shared with FTSL is the **front-end** — one lexer, one block/statement syntax, one
expression evaluator, one units system, one named-reference registry. It stays a single
language family with multiple profiles and targets, not one mega-grammar.

### Why a layer above MJCF at all (the NIH check)

MJCF already describes bodies, joints, spatial tendons with wrapping, and Hill-type
muscles. Inventing a competing format would be pure NIH. The layer earns its place for
exactly one reason:

> **Symbolic morphology.** `femur_len` must be *one named parameter* that simultaneously
> drives the rig geometry, the RL conditioning vector, the domain-randomisation axis, and
> the artist's stylization slider.

MJCF has no such concept — dimensions are literals scattered through XML, and once
flattened the parameterisation is unrecoverable. That single property is what makes P4
(morphology conditioning) and P6 (stylization) possible at all. Everything else about the
rig round-trips to MJCF rather than competing with it.

### A skeleton is FTSL's group tree with the collapse deferred

This is not a foreign concept bolted on. FTSL already has a hierarchical transform tree
with composition; it just bakes it. A skeleton is the same structure with baking deferred
and the joints given DOF and limits. Bones map close to one-to-one onto MuJoCo's
body/joint tree, which is what makes the MJCF emitter cheap.

### Layer stack

| layer | in the control loop? | notes |
|---|---|---|
| skeleton (bones, joints, ligaments) | yes | defines the configuration space |
| muscles / tendons | yes | actuators *within* that space |
| soft tissue (flesh, fat, skin slide) | **no** | offline render pass — FEM in-loop is prohibitive and buys the controller nothing |
| fur / surface | **no** | offline |

### Why the skeleton is the foundational layer, not the muscles

1. **It defines the configuration space.** Joints set the DOF; everything downstream
   operates inside it.
2. **It is the transmission.** A muscle's effect is entirely determined by its
   origin/insertion and how the moment arm changes through the range of motion. Same
   contractile force, different attachment geometry, completely different motion — you
   cannot even define the muscle model without bone geometry first.
3. **Real joints are not the idealisations rigs use, and that is where the "CG creature"
   tell lives.** A knee is not a hinge: the femoral condyles roll *and* slide on the
   tibial plateau, so the instantaneous centre of rotation migrates through flexion. Rig
   it as a fixed pivot and you get the characteristic CG knee.
4. **It is the most reusable layer**, because comparative anatomy is homologous. A
   parameterised quadruped limb topology generalises across felids far better than any
   muscle model or texture will.

Add ligaments as soft nonlinear end-stops and tendons as springs — much of quadruped
locomotion energy is elastic recoil, not muscle work — and the *passive* skeletal layer
does most of the characteristic work before a single muscle fires.

### Passive tone: declare the goal, measure the gains

A bare skeleton folds up. Real ones don't, and not because the animal is thinking about
it — ligaments, joint capsules and resting muscle tone carry a standing quadruped with the
nervous system barely involved. So the rig needs passive stiffness before any policy can
be trained: a fully floppy 31-DOF body makes PPO spend its whole budget discovering "don't
collapse" instead of "walk".

The obvious implementation — `stiffness 150` on each joint — is the exact mistake this
project exists to avoid, and it is the same mistake as `mass` scaling densities one layer
up. Stiffness has units of N·m/rad, so it scales as mass × length² × time⁻², and *every*
one of `body_scale`, `body_mass`, a bone length or a stance angle invalidates the literal.
Nothing reports this. A mis-sprung model still loads, still simulates, still renders as a
plausible dog; you find out as a policy that mysteriously fails on large morphs.

So the rig declares the **goal** in a `posture` block — "no joint may give more than 4°
under the standing load", plus a tone floor, a buckling margin and a damping ratio — and
`creaturelab/tune.py` measures what that costs on the body the current morph vector
actually produced. Four requirements, per joint, take the max:

| requirement | why it exists |
|---|---|
| **static** — hold the measured standing torque within `sag` | the obvious one, and on its own it is badly insufficient |
| **floor** — tone proportional to the subtree weight the joint *could* carry | a dog standing square needs zero torque at every yaw and abduction joint by symmetry, but they still need tone or the body flops laterally the moment it is disturbed |
| **buckle** — beat the rate at which the required torque grows as the joint deflects | a standing quadruped stacks stifle and shoulder near the ground-reaction line (that is *good* design — it is why a horse can sleep standing), so those joints hold almost no torque and sizing them from that torque gives them almost no spring. The load through them is compressive, so any deflection moves the load line off the joint. Euler buckling. For the stifle this term is ~50× the static one |
| **ceiling** — never exceed what the declared timestep and integrator can integrate | derived per-joint from `k = I(2πf)²`, *not* authored |

Damping is set as a fraction of critical against each joint's own effective inertia read
from the mass matrix, `c = ζ·2√(kI)`. A damping literal is doubly wrong under
randomisation: it misses both the inertia change and the stiffness change caused by the
same morph edit.

**Everything above is a prediction, and predictions are not the deliverable.** The
linearisation is systematically optimistic — it asks each joint "what if *you* deflect?"
while holding the others fixed, and a limb does not fold that way. Hip, stifle and hock
give together, the trunk descends, load redistributes between fore and hind feet, and the
CoM migrates until it leaves the support polygon and the animal tips. Measured on
randomised morphs, the linear model predicted 4° and the body settled to 18°. So the
prediction is only the initial guess: `tune.relax` then **simulates the settle, reads the
deflection each joint actually reached, and scales its spring by the ratio it missed by**,
iterating until the settled body meets the goal. That closes the loop on the only number
anyone cares about — where the body ends up — rather than on a model of it.

Two things this pass must do that are easy to omit:

- **A per-joint budget does not bound the whole-body drop.** Thirty-one joints each sagging
  a legal 3.9° put the chest on the floor while every joint reports itself within budget,
  and there is nothing left for the per-joint rule to stiffen. The settled *height* is
  therefore a goal in its own right, and needs no new constant: the body may sink by the
  same fraction of its own height that its joints may rotate in radians.
- **Stop when stiffening stops paying.** Not every collapse is a compliance problem — a
  body whose CoM leaves its support polygon topples about its feet no matter how rigid its
  joints are. Chasing that ran one 314 kg draw to 2.3×10⁶ N·m/rad over eight rounds of a
  loop that was never going to converge. Diverging quietly is worse than failing.

**Armature is measured too, and for the same reason.** Reflected rotor inertia is
`n²·I_rotor`, so the body-independent quantity is the *fraction* of the load it represents
— a drive matched to a heavier limb puts a bigger motor behind a similar gear ratio.
Declaring it as `kg·m²` is the same units mistake as a stiffness literal and hides better,
because nothing in the model looks wrong: the joint is simply heavier than the bone
attached to it. On this rig, unmorphed, one `0.008 kg·m²` default was **0.7% of the
spine's own inertia and 3790% of the paw's** — the paw joints were 97.4% rotor and 2.6%
animal. Because armature is part of the mass matrix, that propagated into everything read
from it: `stiffness_ceiling = I(2πf)²` licensed 38× more stiffness than the real limb could
follow, and `c = ζ·2√(kI)` overdamped the same joints — both worst exactly where the foot
meets the ground. `tune.size_armature` therefore measures each joint's true inertia (with
armature at zero, which is what the model already has if nothing set it) and applies a
dimensionless `joint_armature_ratio`. It runs after `auto_exclude` and *before* `measure`,
since sizing it afterwards would tune the body against a mass matrix it does not have.

Ordering, then, is not incidental — each pass changes the model the next one measures:

```
auto_exclude   →  size_armature  →  measure  →  size_tone  →  relax
(load path)       (mass matrix)     (torques)   (predict)     (correct)
```

Two rig bugs were found by measurement that were invisible to loading, simulating and
eyeballing, and both had been silently corrupting every torque reported before they were
fixed:

1. **Self-collision load path.** Reference poses legitimately overlap — a humerus is
   *inside* the body outline — and MuJoCo happily resolved that overlap as contact. The dog
   was standing on its own thorax–humerus interpenetration at 6× body weight, with the feet
   carrying 44 N of a 206 N animal. Fixed by `tune.auto_exclude`, which detects
   overlapping-by-design pairs per build rather than from a hand-written list — hand-listing
   cannot survive randomisation.
2. **Soft-contact support shortfall.** MuJoCo's contacts are compliant, so at a 1 mm
   seating depth the solver supplied only 92 N under that same 206 N dog and charged the
   missing 114 N to the floating base, where nobody looks. Every joint torque measured that
   way is the torque for a body *partly held up by a crane*. Fixed by solving the ground
   reaction from statics instead of reading it out of the solver.

The statics solve has its own trap worth recording: with two or more feet, wrench balance
is six equations in 3N unknowns, so it is underdetermined and `lstsq` returns an exact
solution *always* — including solutions where a foot pulls *down* on the ground to stop the
creature toppling. A residual check therefore never fires and a morph that falls flat on
its face reports a clean bill of health. The actual criterion is geometric (is the CoM's
ground projection inside the convex hull of the contact points), and it is checked
separately in `tune.support_polygon`.

**Validation.** `tools/rig_report.py` reports the whole measurement; `creaturelab/validate.py`
runs the acceptance bar — motors off, three seconds of gravity, does it still look like an
animal standing up. Motors *off* is the point: a body that needs its controller to avoid
collapsing has pushed the job of not falling over into the policy, where it costs training
budget forever, instead of into the ligaments, where real animals put it. The tuner and the
acceptance test share one `SETTLE_SECONDS`; when they disagreed (1.5 s vs 3 s) the tuner
declared victory on a body that was still sinking.

Sanity checks the report performs because they are invisible by eye in a 25-body tree:
left/right torque symmetry (an asymmetric rig teaches an asymmetric gait), whether each
foot actually reaches the ground in the reference pose, and steps-per-oscillation against
the creature's own declared integrator.

`tests/test_rig.py` is the regression suite, and it is deliberately biased towards the
failures that **do not raise**. Every serious bug found while building this layer returned
a plausible number rather than an error: feet seated 3.6 cm above the floor reported every
joint torque as `0.00` (free fall needs no internal force), a body flipped 170° reported
`+10 deg pitch`, an underdetermined statics solve had a foot pulling downwards, and a
disabled-contact self-check passed vacuously because the quantity it compared was exactly
zero on both sides. A test suite that only asserts "it didn't throw" would have caught none
of them, so the assertions are on measured physical quantities with known-correct values —
foot levelness, ground-reaction balance and sign, sag as a fraction of withers held across
a 32× mass range, tilt over the full [0°, 180°] range.

`validate.trunk_tilt` is a named function rather than a line inside `stand_test` because P1
needs the same measure: "has the creature fallen over" is the termination condition of every
locomotion episode, and if it differs from the acceptance test's notion a policy can learn
to satisfy one and not the other.

### Textures: non-stationarity, not randomness

Procedural noise (Perlin/Worley/fBm) is **stationary** — statistically identical
everywhere. Real surfaces are the residue of a process: dirt where water ran, wear where
hands touched, patina following exposure. Uniformly-random is perceptually distinguishable
from process-generated, and that difference is what reads as "sterile".

Full causal simulation is a deep problem and mostly unnecessary. The affordable middle
ground is **non-stationarity** — noise whose statistics vary over the surface:

- curvature- and cavity/AO-driven masks (grime in concavities, wear on convexities) — a
  pure geometry proxy for "what got rained on and rubbed", nearly free, captures a large
  share of the causal result
- modulating noise frequency/amplitude/lacunarity by low-frequency fields
- domain warping (warp the noise input by another noise field) — very cheap, destroys the
  uniform-random look immediately
- gravity/flow projection for streaking and drip history
- reaction–diffusion where the true causal process happens to be cheap — Turing
  morphogenesis is the actual mechanism behind spots, stripes and rosettes

The part only this project can do: **drive all of it from the anatomical layer** — skin
strain, muscle proximity, contact history, in the creature's own coordinate frame.

Separately: natural materials occupy a surprisingly narrow albedo gamut, and artists
routinely author physically impossible saturation. Constraining to a measured plausible
gamut is concretely fixable and underexploited — and adjacent to ftrace's own
RGB→spectral upsampling work.

### Directability (why physics-first pipelines have died before)

The failure mode is not philosophical. In a keyframe system, "change frame 340" changes
frame 340. In a dynamically-consistent system you cannot move a foot without the balance
solution changing, and the correction propagates both forward *and backward* in time. The
note doesn't stay local. Mitigable via trajectory optimisation (re-solve with the edit as
a constraint) or a corrective layer over a reduced-space controller — but it is a real
cost, not a misunderstanding on the artists' part.

Design consequence: **every layer must be independently inspectable and overridable, with
sim as a default that can be overridden anywhere, never as a mandate.** The VFX middleware
graveyard is full of technically excellent systems that died on integration and artist
control, not on simulation quality. The bar is not "does the solver converge", it's "can a
TD debug this at 2am three days before delivery".

## Repo layout

```
ftcl/            language front-end: lexer, schema-driven parser, expressions, units
creaturelab/     semantic model (Bone/Joint/Site/Muscle/Morph), builder, emitters
rigs/            .ftcl creature definitions
tools/           CLIs
tests/           pytest
out/             generated MJCF/FTSL, checkpoints, logs  (git-ignored)
notes/           capture rig notes, literature
```

## Environment

- Python 3.14 in `.venv`, MuJoCo 3.11, PyTorch (CUDA), RTX 4090.
