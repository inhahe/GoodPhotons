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

---

## Done

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
