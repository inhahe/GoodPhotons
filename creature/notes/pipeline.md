# From camera cards to a trained policy — the command pipeline

*(Created 2026-08-12, because `notes/capture.md` now says what to shoot and `todo.md` P5 says how
the fit works, and neither says **what you actually type**. This page is the command sequence, end
to end, with every step marked with its real status.)*

**Read the status column before planning a session.** Only stages **D–F** exist today. Nothing in
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
   cards            A. ingest        B. 2D keypoints      C. fit          D. rig        E. train        F. use
  ────────         ───────────       ─────────────       ────────       ────────      ──────────     ─────────
  4x .mp4   ──►  calibrate + sync ──►  DLC / SLEAP   ──►  anatomy  ──►  ftcl_build ──►  train.py  ──►  eval/view
  4x .wav        session/calib.json    kp2d.h5           theta_animal   rig_report      + AMP demos     FTSL bake
                 session/sync.json                       + motion .npz  morph_sweep     + morph rand
                   TO BUILD              EXTERNAL          TO BUILD        EXISTS       PARTLY EXISTS   PARTLY
```

The three arrows that carry the data are the three file formats worth freezing early:
`session/calib.json` (camera models + extrinsics + ground plane), `session/kp2d.h5` (per-view 2D
keypoint tracks + confidences), and `out/theta_animal.json` (the 26-number morph vector). Everything
else is internal.

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

## Stage B — 2D keypoints   `EXTERNAL (DLC / SLEAP) + TO BUILD glue`

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

## Stage C — fit: anatomy, then motion   `TO BUILD` — this is P5, the research risk

```bash
# 0. THE YARDSTICK, and it runs BEFORE any real footage exists. Project the P2 public dog
#    mocap through a synthetic copy of the four calibrated cameras, corrupt it with measured
#    detector noise, and tune the four objective weights until ground truth is a fixed point.
python tools/fit_selftest.py  --mocap data/p2_dog  --calib sessions/.../calib.json  --tune-weights
#   -> notes/fit_weights.json   (then frozen)

# 1. ANATOMY: CMA-ES over the 26-param morph vector, LM pose-fit inside.
#    Uses only the CALM clips - standing square and slow walk - and only ~300 frames.
python tools/fit_anatomy.py  --session sessions/2026-08-20_rex  --rig rigs/canis.ftcl \
    --clips stand_* walk_*  --mass 24.5kg  -o out/theta_rex.json
#   hours, not RL training. --mass is not optional: body_mass is kinematically invisible
#   and comes from a bathroom scale (capture.md, todo.md P5 §Identifiability honesty).

# 2. MOTION: theta frozen, per-clip LM + batch smoothing, E_phys as a gate.
python tools/fit_motion.py  --session sessions/2026-08-20_rex  --theta out/theta_rex.json \
    --clips trot_*  -o out/motion/
#   -> out/motion/trot_03.npz  (qpos/qvel per frame + per-frame gate flags)

# 3. Is any of it trustworthy? Per-param identifiability, reprojection RMS, gate pass rate.
python tools/fit_report.py  out/theta_rex.json  out/motion/
```

Order is not negotiable: **anatomy first, then freeze it, then motion.** Reprojection error through
the wrong skeleton is meaningless. todo.md P5 §"Anatomy before motion" has the cost model.

## Stage D — turn the fit into a concrete rig   `EXISTS` (one small addition wanted)

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
```

`TO BUILD`, and it is small: **`--morph out/theta_rex.json`** on `ftcl_build.py`, `rig_report.py`
and `train.py`, so a fitted animal is loaded as a file instead of twenty-six `--set` pairs typed by
hand.

## Stage E — train   `PARTLY EXISTS`

### What runs today (P1: torque-actuated quadruped + PPO, hand-designed reward)

```bash
python tools/train.py --steps 20e6 --out runs/canis            # ~170 min on CPU for 20M steps
python tools/train.py --resume runs/canis/latest.pt            # picks up mid-run, incl. curriculum
python tools/train.py --eval runs/canis/best.pt                # per-command table
python tools/train.py --eval runs/canis/best.pt --view         # watch it in the MuJoCo viewer
```

Useful flags: `--envs 64 --horizon 64 --lr 3e-4 --device auto --checkpoint-minutes 5
--eval-every 20`. Checkpoints are wall-clock-interval, `latest.pt` for resuming and `best.pt` for
the best evaluated return; the command curriculum's `speed_cap` is checkpointed too, so a resume
does not restart the curriculum.

**This does not use captured data at all.** It learns to track a commanded Froude-number velocity
from a hand-designed reward on the authored rig. It is the substrate the captured data plugs into,
not a consumer of it.

### What has to be added, and how much of it is already plumbed

| addition | phase | status of the plumbing |
|---|---|---|
| `--demos out/motion/*.npz` — AMP discriminator over the fitted motion | P2 | **not started.** `ppo.py` was written anticipating "an AMP discriminator sharing the rollout", so the rollout structure is ready; the discriminator, its replay buffer and the style-reward mix are not. |
| `--morph-center out/theta_rex.json --morph-scale 0.4` — randomise bodies per env | P4 | **mostly plumbed.** `build_body(cfg, morph=...)` already accepts a morph dict, and `sensing.py` already rides `morph_norm` in the observation "from day one". What is missing is only that `train.py` builds `[body] * n` — one shared body — instead of a list of distinct ones. |

The intended command, once both exist:

```bash
python tools/train.py --steps 3e8 --out runs/rex \
    --demos out/motion/*.npz  --morph-center out/theta_rex.json --morph-scale 0.4
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

1. **Sites in the grammar/schema/emitter**, then `notes/keypoints.yaml` (~21 landmarks for canis).
   This is the first hard blocker for everything downstream, and it needs no footage.
2. **`--morph <file>`** on `ftcl_build.py` / `rig_report.py` / `train.py`.
3. **`tools/fit_selftest.py`** — the synthetic-camera replay of P2 mocap. It is both the objective
   weight calibration and the honest answer to "does this pipeline work at all", and it runs against
   *public* mocap, before a single frame of the real animal is shot.
4. **Distinct bodies per env in `train.py`** — the P4 randomisation loop, given the conditioning
   channel already exists.

Doing (3) before shooting is the single highest-value ordering decision on this page: it is the only
way to find out that the fit is broken *before* an animal, an owner and a two-hour session have been
spent on footage the pipeline cannot use.

---

## Cross-references

- **What to shoot, where the cameras go, how long a session takes** — `notes/capture.md`.
- **How the fit works** (optimiser, objective term by term, what one θ evaluation costs, the
  identifiability honesty) — `todo.md` **P5** §"The implementation plan".
- **Why training happens once and not twice** — `todo.md` **P6** §"Converting a captured animal".
- **What the trainer's reward and curriculum actually are** — `design.md`.
