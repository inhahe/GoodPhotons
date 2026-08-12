# Creature — design

## What this is

A system for building physically-simulated animals whose motion is **learned** rather than
keyframed: articulated skeleton → muscle/tendon actuators → respiration → soft tissue → fur,
driven by a neural controller fit to real animal movement, and morphable into stylized or
fictional creatures while preserving the learned motion character.

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
2. **Physics-based control fit from real *animal* video.** SFV did this for humans —
   helped enormously by SMPL (a canonical parametric body) and strong human pose
   estimators. Animals have neither, so this project authors its own parametric body (the
   rig) and fits it. *(Re-rated 2026-08-08: the decided 4-camera capture rig makes the 3D
   recovery a triangulation problem, not the ill-posed monocular lifting this bullet was
   written about — the residual risk is keypoint quality and anatomy identifiability, not
   reconstruction; see todo.md P5 → "The implementation plan". In-the-wild monocular
   remains the stretch ambition, downstream of P5: the multi-view pipeline produces the
   paired (video, 3D-motion) data a monocular lifter would train on.)*
3. **Style-preserving morphology transfer for learned controllers.** Especially under
   muscle actuation, where changing the body changes the *actuators themselves*.

Framed differently: VFX owns world-class muscle/tissue/fur but keyframes or
performance-captures its creatures; robotics/RL owns learned quadruped locomotion but on
rigid torque-actuated robots with no flesh and no biological fidelity. This project sits
in that gap.

**The novelty map and the risk map are the same map.** Video fitting is both the most
novel link and the most likely to disappoint. That is not a coincidence, and it dictates
the staging in `todo.md`: build the well-trodden 80% fast, then push on the hard parts —
and arrive at video already holding clean mocap of the same gaits as a yardstick. *(The
4-camera rig decision has since shrunk the reconstruction half of that risk — the staging
survives on the yardstick argument; todo.md P5 carries the re-rating.)*

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
| respiration (lungs, diaphragm, ribcage) | **partly** | the *oscillation* is offline; the **aerobic budget** is in the loop, because it is what makes sustained speed cost something. See below |
| soft tissue (flesh, fat, skin slide) | **no** | offline render pass — FEM in-loop is prohibitive and buys the controller nothing |
| fur / surface | **no** | offline |
| sound (voice, contacts, breath) | **no** | offline, but *event-driven by in-loop state* — exhale slots, contact impulses, effort — which is exactly what a hand-placed foley pass cannot see. See §"Sound is an output of the same systems" |

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

### Sites: the rig's half of the keypoint interface

A **site** is a named, massless landmark fixed to a bone — no mass, no inertia, no
collision geometry. Bones already say where the skeleton is; sites say where the *outside
world's* named anatomical points are on it, which is a different question with a different
consumer. The fit needs "where is the withers on this body?" as a differentiable function
of the morph vector; the renderer and the annotator need it as a name.

Three properties make this worth a first-class concept rather than a convention:

1. **A site is an `Expr` over morph params, exactly like every other geometric quantity.**
   This is the rig's founding rule applied to landmarks, and it is not cosmetic here: a
   landmark written as a literal is correct for exactly one body, and on any other body its
   bias is *indistinguishable inside E_kp from a bad fit*. The optimiser would trade a
   mis-placed withers against a wrong trunk length and report convergence.
2. **Sites are provably outside the physics.** `tests/test_keypoints.py`'s sibling in
   `test_rig.py` strips every `<site>` element from the emitted MJCF and compiles both
   versions: `body_mass`, `body_inertia`, `body_ipos`, `body_iquat` and `dof_M0` are
   bit-identical. That is what makes it safe to add 21 landmarks to a rig a trained policy
   already depends on — the dynamics cannot have changed.
3. **MuJoCo already differentiates them.** `mj_jacSite` gives ∂(site world position)/∂q
   directly, which is the entire Jacobian E_kp needs. Without it the fit would need a
   hand-written kinematic chain that could drift from the model; with it, the reprojection
   term is a few lines.

Site names are **global**, not per-bone, and the builder rejects duplicates. The reason is
the consumer: `notes/keypoints.yaml` refers to a site by name alone, because an annotator
naming a landmark should not have to know which bone the rigger hung it on.

`notes/keypoints.yaml` is the other half — the single source of truth for the
detector↔rig correspondence, and the *only* place it is written down. The DLC/SLEAP
project config is **generated** from it (`tools/keypoints_project.py --emit dlc`), never
maintained alongside it. An interface whose two ends each restate the mapping has two
places to be wrong and they disagree silently: the detector emits `l_elbow`, the fit looks
for `elbow_l`, that landmark contributes nothing, and the only symptom is that one limb
fits slightly worse than the other. Nothing raises. `tools/keypoints_project.py --check`
points the same comparison at a config that has already been edited in the DLC GUI, which
is the one remaining way drift can be introduced.

Two fields in that file are load-bearing beyond the name mapping. `class: rigid|soft`
selects the Huber width — a soft landmark's residual is dominated by real tissue sliding
that a rigid-body model *cannot* represent, so treating it as rigid lets a wandering
marker drag the whole skeleton. And `guide` is not documentation: every joint site sits at
the **joint centre**, not at the palpable prominence a person instinctively clicks (the
point of the hock is centimetres caudal to the hock's axis), so without the guide text
every frame carries a constant offset — and a constant offset is precisely what the
anatomy fit absorbs by making the bone the wrong length.

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
| **floor** — tone proportional to the subtree weight the joint *could* carry | a dog standing square needs zero torque at every yaw and abduction joint by symmetry, but they still need tone or the body flops laterally the moment it is disturbed. Note that *how much* tone is a guess here — a plausible proportionality, not a measurement. `tune.brace`, below, is what replaced the guess |
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

**A symmetric settle cannot see an antisymmetric mode — ever.** Not "is unlikely to": the
default rig is passively unstable in roll with a 0.67 s e-folding time, and it sat at
*exactly* 0.00° of roll for 19 of 20 simulated seconds, because a perfectly left-right
symmetric body released from a perfectly symmetric pose has nothing to roll towards. The
instability was real, the release was clean, and `stand_test` therefore reported a body that
would fall over the instant a policy breathed on it as STANDS. `measure_buckling` was blind
to it too, for an unrelated and equally structural reason: the destabilising term *is* the
ground reaction redistributing between left and right as the mass moves out over one foot
line, and it runs under a frozen reaction, so every abduction joint's entry is exactly 0.00.
Three separate instruments, three blind spots, one shared cause — none of them ever moved
the body sideways.

So `tune.brace` moves it sideways, and `stand_test` gained the same shove as a second phase.
The disturbance is sized against the stance's own **tipping velocity**
`v_c = w·√(g/h)` (`tune.tipping_velocity`, from `½mv_c² = mg(√(h²+w²) − h)`), not against the
Froude speed `√(gL)`: past `v_c` the body rotates about a foot as one rigid piece and *no*
stiffness helps — measured, with abduction stiffness at ×3, ×12 and ×100 all ending at ~90°
tilt — so a Froude-quoted nudge measures stance width rather than passive stability. As a
fraction of `v_c` the shove means the same thing on every morph: `NUDGE_FRACTION = 0.10`
carries 1% of the energy needed to tip the body over rigidly, so anything that falls, fell
because it *bent*.

Three findings from this pass are worth keeping, because each one contradicts an assumption
that felt safe:

- **Stiffness is not monotonically stabilising.** Stiffening all 31 joints ×2 or ×3
  *collapses* the shove test, while stiffening exactly the four abduction joints ×3 passes
  it. So the mode has to be identified, not blanketed — `brace` reads the joint-space
  deviation while growth is still linear (abduction joints come back at 0.21° against 0.0025°
  for the sagittal ones, an 84× separation) and stiffens only the carriers. Nothing in it
  knows the word "abduction", or "lateral", or "left".
- **Peak tilt saturates, so it cannot be a progress signal.** Once a body is past its balance
  point it ends up on its side regardless, and every failing multiplier reports the same
  ~95°. On one morph the peak moved 96.8° → 95.6° across a 2× stiffness range and then fell
  off a cliff to 1.4°. A divergence guard reading that metric concluded stiffness was not
  helping and gave up one step short of a body that stands. The growth rate `λ` does not
  saturate — it is measured on the way up, in the linear regime — and it is also the quantity
  the sizing model is written about, so the guard reads `λ²` instead. `λ²` alone is not enough
  either: it keeps falling even when the trunk is tipping rigidly, because `‖dq‖` is a
  joint-deflection norm and stiffer joints deflect less either way, so the guard also stops
  once the mode's springs have been multiplied fourfold with the peak unmoved.
- **Selecting the mode repeatedly is self-erasing.** A joint is chosen for deflecting, and
  stiffening it is precisely what stops it deflecting, so it drops out of the very next read
  and something else takes its place. The set changed identity every pass, which both starved
  the secant of two points on one line and stopped raising the joints that were actually the
  fix. The mode is therefore read **once**, on the unbraced body, and held.

**How much** stiffness is then a two-point solve rather than a search. For a mode with modal
inertia `I`, destabilising stiffness `K_d` and restoring `x·K₀`, `λ² = (K_d − x·K₀)/I` is
linear in `x`, so a secant on two measured `(x, λ²)` pairs locates marginal stability with no
modal inertia, no mass matrix and no mode-shape normalisation to get wrong. `posture.buckle_margin`
supplies the headroom over marginal, because `K_d` here *is* a buckling gradient — for a
coordinated mode instead of a single joint — so no new constant is needed. Verified by hand
before it was written: canis grows at 1.49/s as tuned and 0.575/s at twice the abduction
stiffness, predicting marginal stability at 2.18× against a measured threshold between 2×
and 3×. In practice the default rig converges in one pass (peak tilt 105° → 1.5°).

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
auto_exclude  →  size_armature  →  measure  →  size_tone  →  relax     →  brace
(load path)      (mass matrix)     (torques)   (predict)     (correct)    (disturb)
```

`brace` is last because a shoved body and a sagging body want opposite corrections applied
to different joints, and the shove is only meaningful once the creature is standing where
it will actually stand. The order is safe in the direction it is used and not in the other:
bracing costs sag nothing measurable (5.36% of withers before, 5.43% after).

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
runs the acceptance bar — motors off, gravity, **a shove**, does it still look like an animal
standing up. Motors *off* is the point: a body that needs its controller to avoid collapsing
has pushed the job of not falling over into the policy, where it costs training budget
forever, instead of into the ligaments, where real animals put it.

The shove is not robustness testing bolted on top of the stand test; it is what makes the
stand test able to answer its own question, for the reason above. It runs as a second phase
after the settle, over the weakest edge of the body's own support polygon, in **both**
directions — a body that holds together shoved left and folds shoved right has not passed
anything, and testing both is also what keeps a symmetric rig's tuned gains symmetric, since
the roll's joint response is *not* antisymmetric (the side that unloads goes slack instead of
deflecting, so one direction alone reports one shoulder giving 1.9× its mirror). The verdict
is taken on the worst peak tilt reached during recovery, not the final one: a body that swings
out to 40° and happens to come back has not passed anything a policy could rely on.

The tuner and the acceptance test share one `SETTLE_SECONDS`; when they disagreed (1.5 s vs
3 s) the tuner declared victory on a body that was still sinking. `NUDGE_FRACTION` is shared
for the same reason and it is the stronger case — a tuner that braced against a gentler shove
than the bar applies would certify bodies the bar then rejects, and one that braced against a
harsher shove would spend stiffness the bar never asked for. Both constants are set at
measured *plateaus* rather than chosen for roundness: 0.05–0.12 give the same verdict on all
48 bodies of the two harder morph scales, which is the property a constant like this needs.
The calibration also records why the phase exists at all — with the shove disabled, the bar
passes a body that falls over on its own.

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

### Breathing is mechanically coupled to gait, so it is not a render effect

*(added 2026-08-08.)* The obvious place to put breathing is the offline look pass, next to fur
— a periodic swell of the flank, applied after the physics is done. That is where it will end
up *visually*, and it is still the wrong model, for three reasons that are each measurable.

**1. At a gallop, breathing is driven by locomotion, not by the animal.** Bramble & Carrier
(1983) found running quadrupeds lock to a **1:1 stride-to-breath ratio**, and the mechanism is
mechanical rather than neural: the visceral mass slides fore-and-aft in the abdomen and drives
the diaphragm like a piston, while forelimb impact loads the thorax. So the breath *phase* is
not free to be authored — it is a consequence of the gait, and a decorative sine wave applied
afterwards will drift out of phase with the footfalls in exactly the way a real animal's does
not. (Humans are the odd ones out here, with flexible 4:1/3:1/2:1 ratios; that flexibility is
part of the endurance-running story and is a *human* trait, not a mammalian one. Getting this
backwards would put human breathing on a dog.)

**2. The aerobic budget is what makes sustained speed cost something.** The energy penalty in
P1's reward is instantaneous mechanical cost of transport, which has no notion of a debt. A
real animal has a sustainable aerobic ceiling and can exceed it briefly by going anaerobic,
after which it must repay — and that repayment is *the* reason a cheetah's sprint is measured
in seconds. This belongs in the loop, as a slow state variable feeding the fatigue model
(todo.md §"Persistent state"), and it is the honest source of the gait-downgrade behaviour that
section wants to emerge rather than be authored. It also fits the existing rule that anything
with a rate is a morph parameter: aerobic capacity and lung volume are properties of the body.

**3. Its parameters are allometric, so they come free from the morph vector.** Lung volume
scales roughly with mass while metabolic rate scales as M^0.75, so respiratory frequency goes
as about M^−0.25 — Stahl (1967) gives f ≈ 53.5·M^−0.26 breaths/min and a tidal volume near
7.7 mL/kg. A bigger animal breathes *slower and deeper*, and that is derived, not authored, in
exactly the way `sensing`'s conduction delays and pendulum periods already are. This is the
same "40 kg heavier is one knob" claim, applied to a system that would otherwise be a
hand-tuned constant per creature.

**One correction worth writing down before it is coded wrong: in dogs, panting is
thermoregulation, not gas exchange.** It is shallow, moves mostly dead-space air so it does not
blow off CO₂ into alkalosis, and it runs at the respiratory system's mechanical resonant
frequency (~5–6 Hz) to minimise the work of breathing. The natural implementation — tie panting
rate to oxygen demand — is therefore wrong: it is tied to *body temperature*, which is a
different slow variable with a different time constant, and a dog that stops panting the instant
it stops running reads as fake. Respiration and thermoregulation are two systems that share one
airway.

What this buys on the look side is disproportionate to its cost. A resting animal's *only*
motion is breathing, and its absence is a large part of why a still CG creature reads as dead —
the flank rise and fall, nostril flare, the ribcage moving under fur (which the fur groom
already deforms with skin strain), and, in cold air, visible breath, which ftrace's
participating media can already render.

### Sound is an output of the same systems, not a foley pass

*(added 2026-08-09 — sound was entirely absent from this document until then.)* The default
pipeline — a sound designer drops barks and footsteps on the timeline after the render — fails
the way keyframed breathing fails, and for the same reason: **vocalisation is respiration.**
Nearly all mammal vocalisation is egressive — air driven out through the larynx — so a call is
an event on the exhale half of the very cycle the section above just pinned to the gait. The
consequences chain directly off decisions already made:

- **Timing is inherited, not authored.** At a gallop breathing locks 1:1 to the stride, so the
  voice does too — a dog barking mid-chase barks in stride rhythm *for free*, because the
  behaviour layer only requests ("vocalize, type T, intensity a") and the respiration layer
  resolves the request to the next legal exhale slot (or steals the exhale for urgent calls,
  visibly costing a breath in the flank). Panting is the degenerate case: the audible pant *is*
  the ~5–6 Hz mechanical oscillation — one airway, one rate, nothing separate to drift. The cat
  purr is the one common exception (continuous through both phases via ~25–30 Hz laryngeal
  gating, Remmers & Gautier 1972) and is modelled as a *state* with an explicit gate bypass,
  not an event.
- **The voice is allometric, like everything else with a rate.** Formant dispersion is set by
  vocal-tract length (ΔF ≈ c/2·VTL — Fitch 1997), and VTL tracks skull scale, which the morph
  vector already carries; F0 follows M^−0.4 with wide honest scatter (Bowling et al. 2017). So
  the *default* voice is derived from θ, and the scatter is an explicit override knob — which is
  precisely what lets a fictional creature's voice be a choice while the default stays
  plausible. Same claim as conduction delays and breath rate: 40 kg heavier is one knob, and
  the voice deepens by itself.
- **Synthesis is analysis-resynthesis, not simulation.** Articulatory (physical) voice
  synthesis is ruled out by the cost model — decades of unsolved realism problems for human
  speech alone. Recorded calls from the capture animal (capture.md §Audio — recorded from the
  first session onward, because capture cannot be re-run) are decomposed with a WORLD-class
  vocoder into F0 / spectral envelope / aperiodicity; every morph and effort edit is
  closed-form in that parameter space.
- **Contacts export foley events.** MuJoCo already computes every footfall and body impact; the
  audio deliverable is dry stems (voice, contacts, breath) plus the raw event track
  (t, bodies, impulse, material class). Propagation, reverb and mixing belong to the scene and
  the listener, not the creature — the contract ends at the dry stems, the way the render
  contract ends at the image.

Work items: todo.md **P12**; capture protocol: `notes/capture.md` §Audio.

### The training environment: only signals a nerve could carry

`creaturelab/sensing.py` owns the observation contract and `creaturelab/env.py` the task.
The split is not cosmetic: the observation vector is a **cross-morph interface**, so it has
to be derivable from the body rather than authored alongside the reward.

**Everything is in the body's own units, via dynamic similarity.** Divide every length by
withers height `L`, every time by the pendulum period `T = √(L/g)`, speed by `V = √(gL)`,
force by body weight `W = mg`. Then `joint_rate = 0.8` and `contact = 0.5` mean the same
physical situation on a 14 kg body and a 300 kg one, and so does a reward weight. That is
the property P4's morph generalisation rests on, and it is cheaper to get right now than to
retrain for later. `V·T == L` is asserted in the tests, because if the three scales ever
stop being consistent every channel is quietly in a different unit system than its weight.

**No channel may carry information a nerve could not.** World position, absolute
orientation and exact world velocity all train *faster* and produce a policy that cannot
transfer. So the vector is: joint angle normalised against *its own* limits, joint rate,
efference copy of the last action, vestibular (gravity direction in the root frame, angular
rate, body-frame linear velocity), per-foot normal force ÷ body weight, the command, and the
morph vector. Root height exists in `RawState` because termination needs it, and is
deliberately *not* a channel — `test_no_world_frame_channel_leaks_in` bans the names, since
this is exactly the kind of thing that gets added during a debugging session and stays.

**Rate channels saturate**, at `sensing.RATE_CLIP`. Real afferents do, and the alternative
is not "unbounded in theory" but unbounded in fact: 200 steps of ±0.3 random actuation drove
one tail joint's rate channel to 49.6 while every other channel sat near 1. No exception, no
NaN — just one input at eight times the scale of the rest, which is the input the first
layer organises itself around.

**Conduction delay is measured through the tree, not authored.** Each channel's lag is
`central_delay + path_length_to_the_CNS_hub / conduction_velocity`, so a hind-paw spindle
(32 ms) really does arrive after a fore-paw one (27 ms), the vestibular signal is the
fastest thing the animal has, and scaling the body scales its reflexes for free — which is
what makes design.md's "40 kg heavier is one knob" claim true rather than aspirational. The
lags are stored as **fractional** control steps and interpolated between ring-buffer
samples. Rounding to whole steps at 50 Hz maps 27 ms and 32 ms to the same integer and
deletes the entire fore/hind asymmetry the module exists to express, while still training.

### The vec env is batched because the GIL, not the physics, was the bottleneck

The first version stepped N envs on a thread pool with each env's normalisation, delay,
reward and termination running inside its own worker. It managed 762 env-steps/s, which
makes a 2×10⁷-step PPO smoke test an overnight job. Root-causing it by measurement rather
than by intuition changed the design:

- `mj_step` releases the GIL, so the physics genuinely parallelises — ~343 µs per env-step,
  scaling 3.9× across 16 threads.
- The ~500 µs of surrounding numpy per env holds the GIL and does not scale at all.
- Worse, what costs is the **number of GIL hand-offs**, not the serial fraction.
  Interleaving 58 µs of Python between physics steps — a 15% serial fraction — cut a
  physics-only loop from 7900 to 1800 env-steps/s. Amdahl predicts nothing like that.
- `ThreadPoolExecutor.map` is itself GIL-held Python: a Future, a condition variable and a
  queue entry per task, ~10–18 µs each.

Ruled out with data, so they don't get re-proposed: solver and integrator changes (`mj_step`
is 36 µs; CG buys 1.31×, not the 2× an earlier measurement on a loaded machine suggested,
and every configuration passes `stand_test` identically), `sys.setswitchinterval`,
free-threaded Python (not installed), and multiprocess workers (~2900/s — the Windows pipe
barrier eats the gain).

So the rule is: **the thread pool touches `MjData` and nothing else.** Workers write raw
state into rows of a shared `RawState`, one task per *worker* over a contiguous span of envs
so dispatch is O(workers) not O(N), and every other operation — normalisation, the delay
lines, the noise, all five reward terms, termination — is a single numpy call across all N.
Result: **762 → ~3000 env-steps/s**, i.e. 318 µs per env-step against a 343 µs single-core
physics floor. There is nothing left in this layer; the next lever is the physics itself.

Two consequences worth stating because they constrain later work:

- **`VecCreatureEnv` is the only implementation.** `CreatureEnv` is an N=1 view over it. The
  reward and termination rules *are* the definition of the task, and two copies of them
  drift the moment either is tuned.
- **Anything `info` hands out must be a buffer the auto-reset cannot touch.** `info` returns
  the env's own arrays to avoid a per-step allocation, and the auto-reset runs at the end of
  the same `step`, so `command` and the sanity flag are snapshotted. Reading them live was a
  real bug of the shape this project keeps producing: only the rows that *finished* were
  wrong — precisely the rows a logger or an AMP buffer reads — and the wrong values were a
  valid command and `sane=True`, so nothing raised.

**Energy is a measured trade, not a default.** The energy penalty needs `∫|τ·q̇|` over the
control step, and each sample costs a GIL hand-off inside the physics. Both sides were
measured: 1 sample/step under-reports by 22% on an episode total (systematically, and worst
exactly when the animal is flailing and the penalty matters most), 5 samples costs 1.2% and
27% of throughput, 10 is exact. Five is the default and `energy_samples` is the knob. The
naive rectangle estimator is no better than a 1-sample trapezoid, so the trapezoid is free.

**"The integrator blew up" is not `isfinite`.** On a bad qpos/qvel/qacc MuJoCo warns and
calls `mj_resetData` *itself*, so the step returns a perfectly finite state that happens to
be the default pose at the origin at rest. That is worse than a NaN: a NaN crashes the
optimiser and leaves a traceback, whereas a silent mid-episode teleport just teaches the
value function that some state transitions to a fresh standing pose for free. The detector
is therefore MuJoCo's own warning counter, against a per-env baseline that the reset clears.

### PPO is written here rather than imported

`creaturelab/ppo.py` is ~350 lines of PPO. The obvious alternative is Stable-Baselines3, and
the reason it was rejected is not that PPO is hard — it is that **the parts of it P2 and P10
have to reach into are exactly the parts SB3 owns**. P2's AMP discriminator adds a second
reward term computed from a replay buffer of reference motion, inside the rollout loop; P10's
ASE latents add a per-episode conditioning variable that has to be sampled at reset and fed to
both the policy and the discriminator. Both are surgery on `collect`. A vendored 350-line
implementation is cheaper to modify than a subclass fighting a framework, and — the real
argument — every line of it is a line whose behaviour this project can *state*, which the
sections below do.

Three details decide whether a PPO run is correct, and none of them announce themselves. A run
with any of them broken still trains, still shows a falling loss and a rising return, and
simply arrives somewhere worse:

- **Truncation is not termination.** A terminal state has no future and its value target is
  the reward alone. A state cut off by the time limit has a perfectly good future the rollout
  stopped watching, and dropping its value teaches the critic that surviving to the end of a
  20 s episode is worth as little as falling over — a lie told at the end of every *successful*
  episode and never at the end of a failed one. `collect` bootstraps `V(final_obs)` on `trunc`
  rows only; `advantages` therefore needs no `(1 − term)` factor because `next_val` already
  carries the rule.
- **The observation normaliser is part of the policy.** A checkpoint without it loads cleanly,
  is bit-identical in its weights, and behaves untrained, because it is being fed observations
  in units it has never seen. It goes in the checkpoint next to the optimiser state.
- **GAE must not cross an auto-reset.** The `(1 − done)` on the recursion is a separate thing
  from the bootstrap above, and is what stops an advantage propagating backwards from a fresh
  episode into the one that ended. At ~3% done-fraction per step early in training this is
  most of the rollout, not an edge case.

**The default initial action std was wrong by enough to stop the run learning.** The usual
continuous-control default is `log_std = −0.5`, i.e. σ = 0.61 of the action range. Here an
action of 1.0 is the *full* motor gear, and that gear was sized from the joint's static
holding torque — so σ = 0.61 is a 60%-of-hold-torque white-noise shove on each of 31 joints,
50 times a second. Measured on the untrained policy:

| σ | episode length | cost of transport | mean reward |
|---|---|---|---|
| 0.000 | 20.00 s | 0.00 | +0.657 |
| 0.100 | 2.90 s | 0.58 | +0.524 |
| 0.202 | 1.98 s | 1.96 | +0.446 |
| 0.400 | 0.92 s | 16.65 | +0.235 |
| 0.607 | 0.72 s | 106.35 | **−0.035** |

Two failures compound at 0.607. Episodes end in 0.72 s, so a 64-step horizon holds almost no
post-transient behaviour; and `c_energy` reaches 106, which against `w_energy = 0.02` is a
2.1/step penalty against the 1.3 the tracking terms pay at their theoretical maximum — the
reward is net negative for existing. A 400 k-step run at −0.5 has its evaluation return *fall*,
54 → 19. At −1.6 it reaches 579 with every animal surviving the full 20 s. The point worth
keeping: this was invisible in the learning curve, which looked like slow progress, and only
became obvious once the env was measured directly at fixed action noise. **The reward weights
were not the problem and were not touched.**

**KL has to be measured on the whole batch, on a fresh pass.** Read inside the minibatch loop,
it measures the policy on the very samples whose gradient it just stepped along, and keeping
only the last minibatch's value estimates the epoch's KL from `n/minibatches` samples. At the
smoke-test batch size those two biases together read 0.05–0.11 against a 0.02 target and
early-stopped after one epoch on almost every update — the run was quietly doing a fifth of the
optimisation it was configured for. One extra forward pass per epoch buys the honest number.
Note the coupling this exposes: KL scales as (Δµ/σ)², so a *narrower* policy trips the same
target sooner and throttles its own optimisation, which is why σ = 0.10 trains worse than
σ = 0.20 despite surviving longer at initialisation.

**Evaluation runs on its own envs, and that is load-bearing.** `evaluate` assigns a fixed grid
of commanded speeds so that consecutive evaluations differ only by the policy. Under
auto-reset, `_reset_idx` draws a *fresh random command* for every env it resets, so the fixed
grid survives only until the first animal falls over — after which "deterministic evaluation
over a fixed command set" is quietly scoring a random one. It presented as algorithm noise:
consecutive evaluations of a steadily improving policy came back 20, 263, 22, and `best.pt`
was being selected on it. The evaluation env is therefore built with `auto_reset=False`, which
also stops a scoring pass from costing the training rollout `num_envs` partial episodes.

**Diagnostics are masked by `sane`; the training signal is not.** An env whose integrator
diverged has a well-defined training outcome (`env.step` pays it −1 and terminates it) and an
undefined *measured* speed, read out of a diverged `qvel`. One such env logged 806 Froude units
and made the run's own progress trace unreadable. An unphysical number does not belong in a
mean.

**The device is the CPU, and that is a measurement.** The nets are 135→256→256→31. Acting is
one forward pass on a `(64, 135)` batch per control step — far too small to amortise a
host-device round trip — and MuJoCo on the CPU is the bottleneck by a wide margin either way.

### The command curriculum: standing still is a strong local optimum

The tracking reward is `r_speed = exp(−e²/speed_tol²)` with `speed_tol = 0.25`, and commands
are drawn up to 0.8 Froude. Measured on the untrained policy, that kernel decays to nothing
well inside the command range:

| commanded | 0.00 | 0.15 | 0.29 | 0.44 | 0.58 | 0.73 | 0.80 |
|---|---|---|---|---|---|---|---|
| `r_speed` at a standstill | 0.99 | 0.71 | 0.26 | 0.05 | 0.005 | 0.001 | 0.000 |

Above ~0.44 there is no reward *and no gradient*: an animal that cannot yet move is paid the
same 0.000 whether it leans forward or falls backward, so more than half of every batch is
drawn from a region that teaches nothing. What the batch does contain is the other half, where
standing still scores ~1.0, never terminates, and costs a cost-of-transport of 0.07 against the
2.4 a flailing attempt at locomotion pays. The first full-range run converged to exactly that
and stayed there — a `--eval` per-command table showed `speed 0.000` in all twelve command
rows, tilt under 2°, every animal surviving the full 20 s. It was not a failure to learn. It
had learned the best policy available to it inside the reward it was given.

So `VecCreatureEnv` widens the commanded-speed bound only as fast as the policy earns it:
`speed_cap` starts at `curriculum_start = 0.3`, and each promotion adds `curriculum_step`
until it reaches the configured `speed_range`. Two things about *how* it is judged matter more
than the schedule itself:

- **The bar is tracking reward, not survival.** Survival is what standing still is already
  perfect at, so promoting on episode length or total return promotes the local optimum.
  `_ep_track` accumulates `r_speed` alone.
- **It is judged over a window of `curriculum_window` episodes, step-weighted.** The first
  version scored whichever handful of envs happened to finish on the current step, which ran
  the promotion test dozens of times per rollout on samples of one to five episodes — drawn
  from a population selected precisely for having ended. It went 0.30 to the full 0.80 in 82 k
  steps, before the animal could stand, and reproduced the standstill it was written to
  prevent. Promotion is a claim about the policy, and a claim about a population cannot be
  made from a sample selected by the thing being measured.

The bar itself is **relative, not absolute**, and that is the third thing that had to be got
right. A parked animal collects 0.64 of the tracking reward when commands run to 0.3 Froude and
0.31 when they run to 0.8, so a fixed number means "a little better than standing" at the start
and "near perfect" later — backwards, since wider commands are the harder task.
`_standstill_score` computes the parked baseline in closed form (the command components are
drawn independently and uniformly, so the expectation of `exp(−(u²+w²)/tol²)` factorises into
two `erf` differences, with `stand_fraction` mixed in separately) and the bar sits
`curriculum_margin` of the way from there to perfect. `speed_cap` is a property whose setter
recomputes the bar, so the one place the cap is written from outside — `train.py` restoring it
on `--resume` — cannot leave a stale bar behind.

`_advance_curriculum` is called only when `auto_reset` is on, so an evaluation env — handed a
fixed command grid and never training — cannot advance a curriculum it is not part of. The cap
lives in the checkpoint alongside the weights and the normaliser, for the same reason the
normaliser does: a resume that restarts the curriculum at its initial width hands a competent
policy a task it solved millions of steps ago, and the learning curve takes a visible step
backwards for reasons entirely internal to the resume.

**Result.** 20 M steps in 170 minutes on the CPU. The cap holds at 0.30 for 1.1 M steps while
tracking climbs 0.40 → 0.83, then widens on earned promotions and reaches the full range at
~3.4 M. The finished policy tracks the whole of it: over a 64-point command sweep the worst
error is −0.040 Froude at the top, `r_speed` never falls below 0.93, tilt stays at 3–4°, and
every animal survives the full 20 s at every command. Cost of transport rises monotonically
0.82 → 2.62 across the sweep — the physically right shape, and not something the reward asks
for directly.

**Report the table, not the mean.** The pre-curriculum run's scalar evaluation return looked
respectable and its per-command breakdown was `speed 0.000` in all twelve rows. A mean over a
command grid that a motionless animal reads perfectly at one end of will hide a total failure
to locomote, which is the one thing P1 exists to detect, so `train.py --eval CKPT` prints the
breakdown and says so underneath it.

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

## Standing constraints

*Decisions that are never "done", extracted here because they lived in `todo.md` and a checklist is
the wrong place for a rule. Each one is cheap to honour now and expensive to retrofit; several were
reached from different directions and only afterwards recognised as the same argument.*

### The recurring argument: the cost model picks the algorithm, before quality is discussed

This settled three unrelated decisions, which is why it is stated first rather than three times:

| decision | the budget | what it eliminated |
|---|---|---|
| fur groom is **procedural**, not simulated or captured per-strand | per-frame render cost × millions of fibers | per-strand capture and reconstruction |
| fur responds **quasi-statically** to wind, with no dynamics | dynamics makes rendering *stateful in time*, killing per-frame parallelism | any integrated hair sim |
| aerodynamics is **blade-element**, not CFD | ~10⁸ RL steps ⇒ ~100 µs/step | CFD, by ~10⁶× — not a fidelity trade, an impossibility |

**Consequence: compute the budget first.** In every case the budget had exactly one survivor, and
arguing fidelity beforehand would have been wasted. When a new subsystem is proposed, the first
question is its per-step or per-frame cost against the loop it sits in.

### Control

- **Anything that can emerge should emerge, not be commanded.** Conditioning inputs must be
  randomised *jointly*, so training cost scales with the dimensionality of the command space, not
  with the behaviours it can express. Enumerating behaviours as channels is how a command space
  becomes untrainable by accident. Worked example: gait is not commanded — speed plus an energy
  penalty reproduces the walk/trot/gallop transitions *and* puts them at the right speeds.
- **Every control channel must be a conditioning input at training time.** Morph vector, gaze,
  affect, style, part goals, fatigue. A channel added afterwards is ignored by the policy, and the
  fix is a retrain. This is P4's lesson generalised to the whole interface, and it is why the control
  interface must be *designed* long before it is built.
- **Part-specific control is one generic mechanism, not N channels** — a goal for a body-part set,
  injected as a goal rather than overriding actuators. Overriding actuators discards the balance
  solution that makes the motion physical. **The positive half of this, which matters more than the
  prohibition: putting the request in the *observation* is what makes anticipation possible.** A real
  animal lifting a hind foot shifts its weight into the remaining support triangle *before* the lift
  — feedforward, not a reaction to the resulting wobble (Belen'kii et al. 1967; Massion 1992: raise
  a human arm and the postural leg muscles fire ~100 ms ahead of the prime mover). A policy that
  reads the goal a moment before it must act can spend that time loading the other three legs.
  Override deletes exactly this: the rest of the body finds out by feeling the balance error
  afterwards, which is a condition no animal is ever in — the nearest real equivalent is having
  your leg yanked out from under you, and that *does* put people on the floor. So the request is not
  "don't touch the actuators" for tidiness; it is that the observation path is the anticipation
  path.
- **Plan with the simulator; do not learn an approximation of it.** MuJoCo is already an accurate,
  fast, differentiable-enough forward model. A learned dynamics model would be a lossy copy of
  something we own outright.

### How a knob acquires meaning — measurement, examples, or neither

*(added 2026-08-09, from the question "the policy starts knowing nothing about the animal, so how
does it know what to map to 'run', 'be angry', 'lift your back right foot'?" The three examples
turn out to use three different mechanisms, and knowing which one a proposed knob needs is the
first thing to establish about it.)*

A conditioning channel is meaningless at initialisation. It acquires meaning exactly two ways —
**a measurement, or a set of examples** — and there is no third. Most of the design's job is
arranging for as many channels as possible to be the first kind.

**Grounded by measurement.** The reward *is* the definition. "Run at 0.5 Froude" means nothing to
the net at init; it means something because `r_speed = exp(−(v − cmd)²/tol²)` pays only when the
measured velocity matches the number in the channel. The channel and the measurement are two
halves of one statement, and no anatomical prior is involved. P1 is the existence proof: at 1.1 M
steps the command channel was decorative and the animal stood still; by 3.4 M it tracked all 64
command rows to within 0.04 Froude. Speed, heading, body height, gaze target and part goals are
all this kind. Note there is no naming problem here and no language layer — "back right foot" is
an index into the rig, the goal is `(part index, target)` in the observation, and the policy
learns "when channel *k* carries a target, moving body-*k* there is what pays".

**Grounded by examples.** "Angry" has no sensor, so the reward cannot be written. You do not
define it; you point at clips of it and let AMP's discriminator learn "did this motion come from
that set?". The knob is the *label attached to the clip set*, and the definition is extensional
rather than intensional — the human labels examples and never writes a specification. This is
precisely why P2 chose AMP over DeepMimic: DeepMimic imitates a specific phase-aligned trajectory,
AMP imitates a *manner* from unaligned clips. ASE (P10 level 2) then replaces N discrete labels
with a latent space in which the labels are anchors.

**Neither — delete the knob.** The preferred outcome where it is available, and the first bullet
of this section is the worked example: there is no gait channel, because speed plus an energy
penalty puts walk/trot/gallop at the right speeds by itself. The mapping problem for a gait knob
is not solved, it is deleted. Level 0 intents ("flee", "go there") are the same shape — a planner
*above* the policy that emits speed and heading, so they ground out in measurement transitively.

Three consequences worth acting on:

- **Decompose an affect knob into measured channels before handing the remainder to a
  discriminator.** Quadruped affect is posture and tension (see "Textures"/P10): ear carriage is a
  joint angle, body height is measurable, fore/aft weight distribution falls out of contact forces,
  gaze is a target, piloerection is a fur-groom knob. All measurable. The residue that genuinely
  needs examples is motion *quality* — jerk, tail-whip sharpness, co-contraction — and it is
  smaller than "angry" makes it sound. This is not tidiness: a measured channel is dimensionless
  and transfers across morphs, whereas a discriminator trained on dog clips is tied to dog-shaped
  bodies, so every channel moved from examples to measurement is a P4 liability removed.
- **An example-grounded knob can only be randomised as jointly as its data.** "Randomise
  conditioning inputs jointly" (first bullet) is cheap for measured channels and *not free* here:
  if every clip labelled angry is also fast, the discriminator cannot separate the two and the
  anger knob silently becomes a second speed knob. Mocap of an angry-but-slow dog may not exist.
  Mitigations: label along the factored channels rather than with one word; hold out a combination
  and test it; check the discriminator cannot predict speed from the residual. This is the failure
  mode that looks fine in a single-knob demo and falls apart in combination.
- **The two kinds verify differently, and the asymmetry is permanent.** A measured knob gets a
  64-row table with a worst-case error. An example-grounded knob's only test is a human looking at
  it, so it can be wrong or entangled without anything reporting so. Weight demo evidence
  accordingly.

### Morphology

- **Anatomy comes from *fitting*, not from training — so conversion costs one training run.**
  A captured animal's bones/joints/muscles are the morph vector of a pre-authored template,
  estimated by P5's fitting half (an optimisation, hours, no RL). That vector exists — and is
  editable — *before* policy training starts. Train once, conditioned on morph, randomised over a
  **generous neighbourhood** of the fitted vector; every later edit inside that region is free, and
  edits outside it are style-anchored fine-tunes (P6), not from-scratch retrains. The practical
  rule: you never need to know the exact target creature at training time, only to choose the
  randomisation *ranges* generously — and range width is nearly free, since training cost scales
  with the command space's dimensionality, not its extent.
- **The morph space is per-body-plan, not one global manifold.** You cannot interpolate a leg into a
  wing: the midpoint body has neither working limb, and no policy exists across a discontinuous
  reward landscape. Morph claims must always be qualified by body plan.
- **Morphs can produce non-viable creatures, and viability is plan-specific.** Terrestrial morphs
  essentially always walk; flight morphs can be aerodynamically uncontrollable (static margin, wing
  loading) or over the mass ceiling for powered flight. Each body plan needs its own viability check
  alongside `tune.py`'s cost measurement.
- **Physical impossibility is a correct result, not a bug** — a scaled-up flyer *should* fail at
  powered flight. The test is whether the fallback behaviour emerges rather than whether the failure
  is suppressed.

### State

- **Every slow variable needs a write port.** Fatigue, hunger, injury, wetness, temperature,
  alertness. The decisive reason: **the simulation's timeline is not the film's timeline** — an
  animal re-enters frame after an implied three-hour chase that was never simulated, so state must be
  *writable*, not merely reachable by running physics up to it. Accumulation is the default; the knob
  is the seek control. Design this as one named, saveable, interpolatable vector rather than
  discovering the requirement separately per variable.
- **Anything with a rate is also a morph parameter.** How fast a given creature tires is a property
  of that creature and must ride in the morph vector, or it goes stale the moment the body changes.

### Scope

- **Content is out; architecture is in.** Object interaction (eat, drink, play-with-X, fight) needs a
  world, graspable objects, contact and — for fighting — a second agent and an unspecifiable reward.
  None of it reuses the others' machinery, which is the test: *if it generalises it is architecture,
  if each instance is bespoke it is content.* A wind field passes that test (a few numbers per point,
  serving both aerodynamics and fur deflection); a lion does not.

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

`notes/capture.md` is the **capture protocol** and is operational, not just rationale: the two-mode
rule (motion and groom never share a recording), the 4× GoPro HERO 12 decision and the settings that
silently corrupt geometry, camera placement with the depth-error argument for 90° spacing,
capture-volume sizing per gait, calibration and sync as on-site checklists, a table mapping *what you
shoot* → *which morph knob it makes identifiable*, and a ~2-hour session shot list with its
thermal/storage/power limits. Read it before a shoot; the fitting side it feeds is todo.md **P5**.

`notes/pipeline.md` is the other half of the same story: **the command sequence** from four camera
cards to a trained policy, stage by stage, with every stage marked `EXISTS` / `PARTLY EXISTS` /
`TO BUILD` / `EXTERNAL`. Its headline finding is worth repeating here — **nothing in this repo reads
a video file today.** Stages D–F (rig build, train, eval) exist, and stage C's *pose* solve now
exists too (`creaturelab/fit.py`, exercised end-to-end by `tools/fit_selftest.py` against synthetic
cameras — see §"The pose fit" below); ingest/calibrate, 2D keypoints and the anatomy search over θ
are still unbuilt (todo.md P5), and `tools/train.py` currently trains from
scratch on the authored rig against a hand-designed reward, consuming no captured data at all. The
commands it lists for those stages are *specifications*, written so the tools have a target and so a
capture session is not shot against a pipeline whose shape is undecided. It also names the four file
formats worth freezing early — `session/calib.json`, `session/kp2d.h5`, `session/masks.h5`,
`out/theta_animal.json` — and ends with the four things on the critical path that need no footage at
all.

`notes/training.md` is stage E expanded into a reference: every `train.py` flag, what each field of
the progress line means, the checkpoint/resume contract, and the P4 morph-randomisation commands
with the argument for ~8 bodies over 64 envs. Its load-bearing section is **"the one thing you must
actually check"** — a real run from this tree that scored `eval_return` 603, survived every episode
and held under 3° of tilt, while moving at 0.001 Froude against commands up to 0.8. It had learned
to stand still and collect the near-zero commands for free. That is exactly the local optimum the
reward and the curriculum are both designed around (§"The command curriculum" above), and **no
scalar in the log distinguishes it from success** — only the per-command table from `--eval` does,
which is why that table is printed by default rather than hidden behind a flag.

`notes/keypoints.yaml` is the detector↔rig correspondence (see §"Sites" above): 21 canis landmarks,
each mapping an annotator-facing `label` to a rig `site`, with a per-landmark noise sigma, a
rigid/soft class that sets the Huber width, and the labelling instruction that keeps the annotator
clicking joint centres. `tests/test_keypoints.py` fails if it and `rigs/canis.ftcl` ever drift apart,
in either direction — a keypoint naming a site the rig lacks *and* a rig site no keypoint covers,
the second of which is otherwise entirely silent.

`out/theta_<animal>.json` (`creaturelab/morph_io.py`) is the narrow waist between the two halves of
the project: stage C's fit produces one, stages D–F consume one, and neither end knows anything about
the other beyond this file. It is deliberately boring — a name→number map, a schema version, and
enough provenance (`rig`, `creature`, `source`) to answer "where did this dog come from?" six months
later. A θ with no record of which rig it was fit against is actively dangerous, because parameter
names are per-rig: applied to a different rig it either errors (good) or, if the names happen to
overlap, silently builds a chimera. Unknown parameter names are a hard error naming the *file*;
missing ones stay at the rig defaults and are reported, which is what makes a partial θ ("just the
leg lengths I measured") useful. `--morph FILE` on `ftcl_build.py`, `rig_report.py`, `morph_sweep.py`
and `train.py` all read it, with `--set NAME=VAL` overriding it so you can load the fitted animal and
then poke one parameter.

**Every tool calls `creaturelab/console.py`'s `use_utf8()` before parsing arguments**, and the
reason is worth one line so it is not "cleaned up" later. These docstrings talk about θ, use
en-dashes and print degree signs; Windows gives a process's `sys.stdout` the console's ANSI code
page (cp1252) with `errors="strict"`, so argparse echoing a docstring into help text raised
`UnicodeEncodeError` from inside `_print_message`. `morph_sweep.py --help` and `fit_selftest.py
--help` both died that way while the tools themselves ran fine — the first thing anyone types about
a tool was the one thing guaranteed to fail. The fix widens the stream rather than narrowing the
docs, because stripping the characters would leave the process still unable to print them and would
regress silently at the next degree sign; `tests/test_tools_cli.py` pins both halves by running
every tool's `--help` in a subprocess under a forced `PYTHONIOENCODING=cp1252` and asserting the
output is byte-identical to the UTF-8 run. It is a called function rather than an import side
effect because `creaturelab` is a library and must not reach out and mutate its host's streams.

`session/calib.json` (`creaturelab/camera.py`) is the second frozen format, and the one with the
least margin for error: **a calibration that is wrong is not detectable from the footage
afterwards**, and every later stage — including the E_phys floor gate — reads it and believes it. So
the schema carries its own verification numbers (`rms_px`, plus `drift_px`, the closing re-shoot's
disagreement with the opening one that capture.md makes mandatory), and `load_calib` refuses rather
than warns: a non-orthonormal `R` (a scaled extrinsic silently rescales the whole animal, which the
morph fit then absorbs into `body_scale`), a reflection, a duplicate camera name, a `pinhole` model
carrying distortion coefficients, or a file with only one camera — from which a 3D pose is not
observable at all, though the fit will happily run and return one.

The camera model is `cv::fisheye`, and **keypoints are undistorted, never frames** (capture.md).
That splits into two directions used by different callers, which is deliberate: `project()` is the
full nonlinear map and is used only to manufacture synthetic views, while the fit runs on
`undistort()` → `project_linear()`, leaving the objective a clean pinhole whose Jacobian is four
entries of a 2×3 instead of the derivative of an 8th-order polynomial. Because the self-test
generates through one path and fits through the other, a disagreement between the distortion model
and its inverse surfaces as recovery error instead of cancelling out.

### The pose fit: `creaturelab/fit.py`

E_kp — reprojection error minimised **directly in the simulator's own `qpos`** (todo.md P5). The
unknown is the same vector the policy is later trained on, so there is no retargeting stage, no
correspondence between two skeletons to maintain, and no way for the fitted motion to be unreachable
by the body that has to reproduce it. Levenberg–Marquardt with Marquardt scaling (`λ·diag(JᵀJ)`, not
`λI`, or a shared damping penalises root translation in metres and a toe joint in radians equally
and in practice freezes the root); the Jacobian analytic through `mj_jacSite`; the free joint stepped
with `mj_integratePos` so the root quaternion stays on the unit sphere; joint limits imposed by
projection rather than a barrier, because a barrier's gradient blows up exactly where LM most needs
a large step. The Huber width comes from `keypoints.yaml`'s `class` (2 px rigid / 8 px soft) — which
is what makes `class` load-bearing rather than documentation — and observations below
`PoseFitter.CONF_FLOOR` are *rejected*, not down-weighted, because a detector emits a coordinate for
an occluded part too and it is typically the centroid of the animal.

`triangulate` + `init_root` place the root by Kabsch-aligning the model's landmarks onto free
triangulated points before LM starts. Measured, this buys **start-independence and bounded cost, not
accuracy**: the fitted pose becomes bit-identical from initial guesses 0 m, 1 m and 3 m off, at a
constant ~30 LM iterations, against 38 → 50 and rising without it. A fit whose answer moves when the
initial guess moves is not a measurement, and on real footage nothing would notice.

**Two things this deliberately is not.** It is not the anatomy search — θ is fixed here; CMA-ES over
the morph vector wraps this loop in `tools/fit_anatomy.py` (unbuilt). And it is not the whole
objective: E_sil (the only source of girth), E_temp, E_lim as a barrier and the E_phys gate are all
still to come. See `notes/pipeline.md` stage C.

`tools/fit_selftest.py` is the yardstick, and it runs **before any footage exists**: build a body,
pose it, project its landmarks through a synthetic four-GoPro ring, corrupt the result the way a real
detector does (per-landmark σ, occlusion, gross outliers, left/right swaps), throw the truth away and
measure what comes back. Its verdict is deliberately *relative* — beat free-point triangulation of the
same observations, and stay under the ray-uncertainty floor — because `keypoints.yaml` still says
`sigma_px_measured: false`, so any absolute millimetre bar would be measuring the placeholder.
Beating unskeletoned triangulation is the sharper test anyway: it asks whether the skeleton is
*adding* information, and a wrong correspondence or a camera-convention error fails it while leaving
the reprojection number looking fine (a broken fit drives its own reprojection to zero by contorting
the body). With clean observations recovery is exact to 0.0 mm; with the default corruption it is
~24 mm against ~34 mm unskeletoned. **What it cannot tell you:** a synthetic pose is by construction
inside the rig's reachable set, so this measures the optimiser, not the model — whether a real dog is
representable by `canis.ftcl` needs public P2 mocap replayed through the same cameras (`--mocap`,
still to build).

## Environment

- Python 3.14 in `.venv`, MuJoCo 3.11, PyTorch (CUDA), RTX 4090.
