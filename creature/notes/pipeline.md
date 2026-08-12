# From camera cards to a trained policy — the command pipeline

*(Created 2026-08-12, because `notes/capture.md` now says what to shoot and `todo.md` P5 says how
the fit works, and neither says **what you actually type**. This page is the command sequence, end
to end, with every step marked with its real status.)*

**Looking for the training commands specifically? They live in `notes/training.md`** — this page
summarises them at stage E, that one is the reference.

**Read the status column before planning a session.** Only stages **D–F** exist today, plus stage
C's *pose solve* (`creaturelab/fit.py`, `tools/fit_selftest.py`). Nothing in
this repo currently reads a video file, and that is not a gap in the docs — it is unbuilt code
(todo.md P5 is `[ ]`). The commands below for stages A–C are therefore **specifications**, written
now so that the tools have a target and so a capture session is not shot against a pipeline whose
shape is undecided. They are marked `TO BUILD` and they will change; the ones marked `EXISTS` are
verified working and copy-pasteable.

Everything runs from `creature/` with that directory's own venv:

```
cd "D:\visual studio projects\forward raytracer\creature"
.venv/Scripts/python.exe tools/<tool>.py ...
```

---

## The shape of it

```
   cards            A. ingest      B. 2D observations    C. fit          D. rig        E. train        F. use
  ────────         ───────────     ────────────────     ────────       ────────      ──────────     ─────────
  4x .mp4   ──►  calibrate + sync ──►  DLC / SLEAP  ──►  anatomy  ──►  ftcl_build ──►  train.py  ──►  eval/view
  4x .wav        session/calib.json    kp2d.h5          theta_animal   rig_report      + AMP demos     FTSL bake
                 session/sync.json     ↑ joints         + motion .npz  morph_sweep     + morph rand
                                       SAM2 masks.h5
                                       ↑ girth
                   TO BUILD              EXTERNAL       PARTLY BUILT      EXISTS       PARTLY EXISTS   PARTLY
                                                       (pose solve yes,
                                                        anatomy/motion no)
```

**Stage B produces two things, not one, and the second is easy to forget.** Keypoints pin the
*skeleton*; they say nothing whatsoever about `trunk_radius`, `limb_gracility` or the belly. Girth
lives **only** in the contour, so the silhouette masks are not a refinement — without them the outer
CMA-ES loop is blind to half the morph vector (todo.md P5 §"The objective, term by term": E_sil "is
the term that carries the anatomy"). A pipeline that ships keypoints and skips masks fits a skeleton
and then guesses the animal's shape.

The four arrows that carry data are the four file formats worth freezing early:
`session/calib.json` (camera models + extrinsics + ground plane), `session/kp2d.h5` (per-view 2D
keypoint tracks + confidences), `session/masks.h5` (per-view binary silhouettes), and
`out/theta_animal.json` (the 26-number morph vector). Everything else is internal.

---

## Stage A — ingest, calibrate, sync   `TO BUILD`

```bash
# Lay out a session directory from the four cards. Verifies every take has all four
# views, finds the clap/flash, checks timecode agreement, and refuses on a missing view.
python tools/session_ingest.py  --cards E:/ F:/ G:/ H:/  --out sessions/2026-08-20_rex
#   -> sessions/.../takes/<take>/cam{0..3}.mp4, cam{0..3}.wav, sync.json, session.yaml

# Fisheye intrinsics per camera per mode + extrinsics + ground plane, from the ChArUco
# footage (capture.md "Calibration - the on-site procedure"). --verify is the step that
# turns a bad calibration into a detected one rather than a silent corruption.
python tools/calibrate.py  --session sessions/2026-08-20_rex  --board charuco_7x5_60mm  --verify
#   -> sessions/.../calib.json   (cv::fisheye K/D per camera, R|t per camera, floor plane,
#                                 metric scale, reprojection RMS, and the closing re-shoot's
#                                 disagreement with the opening one)
```

**Why it is its own tool and not a notebook:** `calib.json` is consumed by every later stage and by
the E_phys floor gate. It needs a schema version and a verification number attached, because a
calibration that is wrong is *not detectable from the footage afterwards*.

**Reuse, don't write from scratch:** OpenCV `cv::fisheye`/omnidir for intrinsics, and Anipose's
wand/board bundle adjustment is the well-trodden path for extrinsics. capture.md already commits to
the fisheye model and to undistorting *keypoints*, never frames.

## Stage B — 2D observations: keypoints **and** silhouettes   `EXTERNAL + TO BUILD glue`

### B1 — keypoints (the skeleton)

```bash
# The correspondence file is the single source of truth; the detector project is GENERATED
# from it so the two ends of the interface never restate each other (todo.md P5).
python tools/keypoints_project.py  --spec notes/keypoints.yaml  --emit dlc  --out dlc/rex
#   (--emit sleap for the SLEAP path)

#  ... then DLC's or SLEAP's own labelling / training / inference commands, unchanged.
#  Label frames FROM THE REAL ENVIRONMENT, including occluded and blurred ones
#  (capture.md: the messy environment is a keypoint problem, not a rig problem).

python tools/keypoints_import.py  --session sessions/2026-08-20_rex  --from dlc/rex/output
#   -> sessions/.../kp2d.h5   (per take, per view: T x K x 3  [x, y, confidence])
```

**Blocked on a prerequisite that is not video:** `notes/keypoints.yaml` cannot be written until the
rig can express landmark **sites**, and `creaturelab` has *no site support at all* today — zero
`site` occurrences in `schema.py`, `emit_mjcf.py` or `rigs/canis.ftcl`. That work item is already
listed in todo.md P5 §"The keypoint↔rig interface must be authored". It is the first thing on the
critical path, it needs no footage, and it can be done today.

### B2 — silhouettes (the girth)   `EXTERNAL (SAM2) + TO BUILD glue`

```bash
# One prompt per clip per view - a few clicks on the animal in the first frame - then
# SAM2 propagates through the clip. Background subtraction is the fallback in a
# controlled space. Only the passes shot against an uncluttered backdrop need this
# (capture.md's "silhouette-friendly passes" row).
python tools/masks_import.py  --session sessions/2026-08-20_rex  --from sam2/rex \
    --clips stand_* walk_*
#   -> sessions/.../masks.h5   (per take, per view: T x H x W binary, RLE-packed)

# Precompute the distance transform of each observed silhouette BOUNDARY once, because
# the fit samples it a few hundred times per frame per candidate theta and a per-
# evaluation DT would dominate the whole CMA-ES budget.
python tools/masks_dt.py  --session sessions/2026-08-20_rex
#   -> sessions/.../masks_dt.h5   (float16 DT + its bilinear gradient)
```

**Why this is cheap and why it needs no renderer.** E_sil is a *one-directional* (model→observed)
chamfer: sample the projected occluding contours of the model's capsule geoms, do a bilinear lookup
into the precomputed DT, and the lookup is its own analytic gradient. That is robust to segmentation
holes and clutter, is O(#contour samples), and — the load-bearing part — needs **no rasteriser and
no differentiable renderer**. This is why "don't build a soft renderer" survives contact with the
silhouette term.

**The correction it will need:** fur inflates the silhouette, so E_sil measures the *coat*, not the
body. `trunk_radius` and `limb_gracility` come out biased fat by roughly the coat depth, and
capture.md flags this. Decide deliberately whether to subtract a coat allowance or to accept that θ
describes the groomed animal.

## Stage C — fit: anatomy, then motion   `PARTLY BUILT` — this is P5, the research risk

The **pose solve exists** (`creaturelab/camera.py` + `creaturelab/fit.py`, exercised by
`tools/fit_selftest.py`); the **anatomy search and the motion pass do not**. Step 0 below runs
today in its synthetic form and is worth running before every session.

```bash
# 0. THE YARDSTICK, and it runs BEFORE any real footage exists.
#
#    RUNS TODAY: generate poses on the rig itself, project them through a synthetic copy of
#    the four calibrated cameras, corrupt them, and check the fit recovers them -- and beats
#    free-point triangulation of the same data, which is the sharper test, because a broken
#    fit drives its own reprojection error to zero by contorting the body.
python tools/fit_selftest.py                                    # synthetic rig, default corruption
python tools/fit_selftest.py --calib sessions/.../calib.json    # against the REAL cameras
python tools/fit_selftest.py --morph out/theta_rex.json --occlusion 0.3 --outliers 0.05 --verbose
#   Caveat, and it is the important one: a synthetic pose is by construction inside the rig's
#   reachable set, so a pass says the solver, cameras, Jacobian and keypoint correspondence
#   agree -- NOT that a real dog is representable by canis.ftcl.
#
#    TO BUILD: replay public P2 mocap through those same cameras (that is the part the tool
#    is named for -- it is what tests the MODEL rather than the OPTIMISER), and tune the
#    objective weights. Weight tuning waits on E_sil: today there is only one term to weight.
python tools/fit_selftest.py  --mocap data/p2_dog  --calib sessions/.../calib.json  --tune-weights
#   -> notes/fit_weights.json   (then frozen)

# 1. ANATOMY: CMA-ES over the 26-param morph vector, LM pose-fit inside.  TO BUILD
#    Uses only the CALM clips - standing square and slow walk - and only ~300 frames.
python tools/fit_anatomy.py  --session sessions/2026-08-20_rex  --rig rigs/canis.ftcl \
    --clips stand_* walk_*  --mass 24.5kg  -o out/theta_rex.json
#   hours, not RL training. --mass is not optional: body_mass is kinematically invisible
#   and comes from a bathroom scale (capture.md, todo.md P5 §Identifiability honesty).

# 2. MOTION: theta frozen, per-clip LM + batch smoothing, E_phys as a gate.  TO BUILD
#    (the per-frame LM this needs is `fit.PoseFitter` / `fit.fit_sequence`, which exist)
python tools/fit_motion.py  --session sessions/2026-08-20_rex  --theta out/theta_rex.json \
    --clips trot_*  -o out/motion/
#   -> out/motion/trot_03.npz  (qpos/qvel per frame + per-frame gate flags)

# 3. Is any of it trustworthy? Per-param identifiability, reprojection RMS, gate pass rate.
python tools/fit_report.py  out/theta_rex.json  out/motion/
```

Order is not negotiable: **anatomy first, then freeze it, then motion.** Reprojection error through
the wrong skeleton is meaningless. todo.md P5 §"Anatomy before motion" has the cost model.

`fit_anatomy.py` reads **both** stage-B products — `kp2d.h5` *and* `masks_dt.h5` — plus `calib.json`
and the one number video cannot give it. What each input is actually responsible for:

| the objective term | weight | what it is the only source of | input |
|---|---|---|---|
| **E_kp** reprojection (Huber δ=2 px, ×confidence) | 1 (reference scale) | bone **lengths**, joint angles | `kp2d.h5` |
| **E_sil** chamfer on the DT | 0.3 | **girth** — `trunk_radius`, `limb_gracility`, belly | `masks_dt.h5` |
| **E_temp** 2nd differences of qpos | 0.05, stage 2 only | — (regulariser; 140 fps is what lets plain 2nd differences replace a motion prior) | — |
| **E_lim** log-barrier on authored ranges | 0.01 | — (keeps θ legal) | `canis.ftcl` |
| **E_phys** `mj_inverse` torque envelope, floor penetration, stance slip | **a gate, not a loop term** | flags bad frames as AMP demos | `tune.py` statics |
| — | — | **`body_mass` and mass distribution: kinematically invisible.** No term sees them. | **a bathroom scale** |

Two facts about the inner loop that decide whether the budget is hours or days, both from P5:

- **Initialisation is nearly free, and it is where the 4-camera decision pays.** RANSAC-triangulated
  keypoints give 3D limb-segment lengths *directly*, so θ₀ = per-segment medians over the calm clips
  and q₀ = bone-length-aware analytic IK on the triangulated points. CMA-ES polishes; it never
  searches blind.
- **Warm starts are load-bearing, not an optimisation.** Per candidate θ: rebuild the model once
  (build + `tune.py`, one-shot statics — milliseconds, *not* a closed-loop process), then pose-fit a
  fixed ~300-frame subset warm-started from the best-so-far member's trajectory. Seconds per
  candidate ⇒ 13 × a few hundred generations = hours. Cold inner solves make the same fit take days.
  `validate.py`'s collapse-counting never runs inside the fit.

## Stage D — turn the fit into a concrete rig   `EXISTS`

```bash
# Compile, settle in MuJoCo, and check it stands.
python tools/ftcl_build.py rigs/canis.ftcl -o out/canis.xml --check
python tools/ftcl_build.py rigs/canis.ftcl --set body_scale=1.4 limb_gracility=0.7
python tools/ftcl_build.py rigs/canis.ftcl --view

# Measure it rather than guessing: static joint torques with the feet in contact,
# left/right symmetry, where the standing pose actually puts the feet.
python tools/rig_report.py rigs/canis.ftcl
python tools/rig_report.py rigs/canis.ftcl --set body_scale=1.6

# Before training across a morph distribution: what fraction of sampled bodies stand up?
python tools/morph_sweep.py rigs/canis.ftcl -n 24 --scale 0.25 0.5 0.75 1.0

# ...and the version of that question that actually matters, once a fit exists: sweep the
# same centre you intend to train on, because a fitted theta near a range edge has most of
# its neighbourhood clipped, which the defaults' neighbourhood does not.
python tools/morph_sweep.py rigs/canis.ftcl --morph out/theta_rex.json -n 24 --scale 0.25 0.5
```

**`--morph out/theta_rex.json` now exists** on `ftcl_build.py`, `rig_report.py`, `morph_sweep.py`
and `train.py`, with `--set NAME=VAL` overriding individual entries, so a fitted animal is loaded
as a file rather than as twenty-six `--set` pairs typed by hand.

## Stage E — train   `PARTLY EXISTS`

> **The full command reference for this stage is `notes/training.md`** — every flag, how to read
> the progress line, how to tell a walking policy from one that learned to stand still and score
> well, and the P4 morph-randomisation commands. What follows is the summary.

### What runs today (P1: torque-actuated quadruped + PPO, hand-designed reward)

```bash
python tools/train.py --steps 20e6 --out runs/canis            # ~2.3 h on CPU at 64 envs
python tools/train.py --resume runs/canis/latest.pt --steps 20e6 --out runs/canis
python tools/train.py --eval runs/canis/best.pt                # per-command table
python tools/train.py --eval runs/canis/best.pt --view         # watch it in the MuJoCo viewer

# P4: train over a neighbourhood of a fitted animal rather than the authored defaults
python tools/train.py --morph out/theta_rex.json --morph-scale 0.4 --morph-bodies 8 \
    --out runs/rex --steps 3e8
```

Note `--rig` is a flag, not a positional. Useful flags: `--envs 64 --horizon 64 --lr 3e-4
--device auto --checkpoint-minutes 5 --eval-every 20`. Checkpoints are wall-clock-interval,
`latest.pt` for resuming and `best.pt` for the best evaluated return; the command curriculum's
`speed_cap` is checkpointed too, so a resume does not restart the curriculum.

**This does not use captured data at all.** It learns to track a commanded Froude-number velocity
from a hand-designed reward on the authored rig. It is the substrate the captured data plugs into,
not a consumer of it.

**And the scalar score is not the read.** A 573k-step run in this tree scored `eval_return` 603,
survived 100% of episodes and held under 3° of tilt — while moving at 0.001 Froude against
commands up to 0.8. It had learned to stand still and collect the near-zero commands. Always read
the per-command table from `--eval`; `training.md` has that table and what a healthy one looks
like.

### What has to be added, and how much of it is already plumbed

| addition | phase | status of the plumbing |
|---|---|---|
| `--demos out/motion/*.npz` — AMP discriminator over the fitted motion | P2 | **not started.** `ppo.py` was written anticipating "an AMP discriminator sharing the rollout", so the rollout structure is ready; the discriminator, its replay buffer and the style-reward mix are not. |
| `--morph out/theta_rex.json --morph-scale 0.4` — randomise bodies per env | P4 | **DONE.** `train.py` compiles `--morph-bodies` distinct animals (default 8) and cycles them over the envs; the zoo is seeded off `--seed` so a resume rebuilds it identically, and evaluation deliberately scores the centre body only. Uncovered `known-issues.md` #6, still open. |

**A warning about where the design detail is thin, and it is not where you would guess.** The *fit*
is specified down to weights, metrics and a cost model (todo.md P5 §"The implementation plan"). The
*training* additions are not: P2 is four checkboxes plus a paragraph on why AMP beats DeepMimic, and
P4 is five. There is nowhere in this repo that says how the discriminator is structured, what its
observation is, how the style reward mixes with the task reward, or how demo frames failing the
E_phys gate get down-weighted. P1's write-up is detailed because it was *built*; P2/P4's is thin
because they have not been. Expect to design AMP when you get there rather than to implement a plan.

The intended command, once AMP exists (everything but `--demos` runs today):

```bash
python tools/train.py --steps 3e8 --out runs/rex \
    --demos out/motion/*.npz  --morph out/theta_rex.json --morph-scale 0.4
```

**Train once, and get the morph range right before you do.** todo.md P6 §"Converting a captured
animal": anatomy is *not* an output of training, so edit it before the first policy gradient and
randomise over a **generous neighbourhood** of θ that covers every edit you anticipate. Randomisation
cost scales with command-space dimensionality, not with how wide each range is — generosity is
cheap, and a post-hoc edit that leaves the trained region costs a fine-tune (P6 step 6).

## Stage F — use the result   `PARTLY EXISTS`

```bash
python tools/train.py --eval runs/rex/best.pt --view      # EXISTS
```

`TO BUILD`: `creaturelab/emit_ftsl.py` — the pose → `.ftsl` bake that lets ftrace render it.
`model.py`'s docstring already claims "the emitters turn it into MJCF or FTSL" but only
`emit_mjcf.py` exists, and the emitter owns the z-up (sim) → y-up (FTSL) conversion. Also P8's live
viewer and P7's groom.

---

## What you can actually do today, in order

Nothing below needs a camera, and all of it is on the critical path:

1. ~~**Sites in the grammar/schema/emitter**, then `notes/keypoints.yaml`~~ — **DONE.** 21 canis
   landmarks, loaded by `creaturelab/keypoints.py`, with `tools/keypoints_project.py` generating the
   DLC/SLEAP project from it so the two ends of the interface never restate each other, and
   `tests/test_keypoints.py` failing if the file and the rig drift apart in *either* direction.
2. ~~**`--morph <file>`**~~ — **DONE**, on `ftcl_build.py`, `rig_report.py`, `morph_sweep.py` and
   `train.py`, with `--set NAME=VAL` overriding it.
3. **`tools/fit_selftest.py`** — **PARTLY DONE.** The camera model (`creaturelab/camera.py`), the
   pose solve (`creaturelab/fit.py`) and the round-trip driver all exist and pass: clean observations
   are recovered exactly, and the default corruption comes back at ~24 mm against ~34 mm for
   free-point triangulation of the same data. **What is left is the part the name refers to** —
   replaying *public P2 mocap* through the synthetic cameras (`--mocap`, currently a clean refusal).
   Until that lands, the tool measures the **optimiser**, not the **model**: a synthetic pose is by
   construction inside the rig's reachable set, so a pass says the solver, cameras, Jacobian and
   keypoint correspondence agree — not that a real dog is representable by `canis.ftcl`. The
   objective-weight tuning also has to wait for E_sil, since there is only one term to weight today.
4. ~~**Distinct bodies per env in `train.py`**~~ — **DONE**, via `--morph-scale` / `--morph-bodies`.
   Uncovered known-issues #6 (randomised bodies trip MuJoCo instability warnings the default body
   never does), which is still open.

Finishing (3) before shooting is the single highest-value ordering decision on this page: it is the
only way to find out that the fit is broken *before* an animal, an owner and a two-hour session have
been spent on footage the pipeline cannot use. Note that the half already built would not have caught
"a real dog does not fit this rig" — it caught a stale-Jacobian bug and an initialiser whose claimed
benefit evaporated once the Jacobian was fixed, both of which would otherwise have been diagnosed on
real footage as a bad calibration.

---

## Cross-references

- **The training commands in full** — `notes/training.md` (stage E, expanded).
- **What to shoot, where the cameras go, how long a session takes** — `notes/capture.md`.
- **How the fit works** (optimiser, objective term by term, what one θ evaluation costs, the
  identifiability honesty) — `todo.md` **P5** §"The implementation plan".
- **Why training happens once and not twice** — `todo.md` **P6** §"Converting a captured animal".
- **What the trainer's reward and curriculum actually are** — `design.md`.
