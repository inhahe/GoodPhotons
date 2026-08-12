"""Does the fit recover a pose it is *known* to be able to represent? Run this before shooting.

`notes/pipeline.md` calls this "the single highest-value ordering decision on the page", and
the argument is about cost, not elegance: the only way to find out the fit is broken *before*
an animal, an owner and a two-hour session have been spent producing footage the pipeline
cannot use. It needs no camera, no calibration shoot and no footage.

**The experiment.** Take a body the rig can build. Put it in a pose the rig can hold. Project
its ~21 landmark sites through a synthetic copy of the four calibrated GoPros, using the FULL
fisheye model. Corrupt the result exactly the way a real detector does -- Gaussian jitter at
each landmark's own σ, occlusion when a view cannot see the part, and the occasional gross
mislabel. Throw the true pose away, hand the fit a deliberately bad initial guess, and measure
how much of the truth comes back.

Because the answer is known, every number here is an *error*, not a diagnostic to interpret:

    landmark 3D recovery   how far the fitted landmarks are from where they really were
    root position / yaw    the six numbers no joint can compensate for
    reprojection RMS       what the fit thinks it achieved -- the number a real session
                           reports, and the one to distrust when it disagrees with the above

**The gap this leaves, stated plainly.** A synthetic pose is by construction inside the rig's
reachable set, so this measures the *optimiser*, not the *model*. It cannot tell you whether a
real dog's motion is representable by `canis.ftcl` at all -- that is what replaying public P2
mocap through the same cameras is for (`--mocap`), and it is the next thing to build. A clean
report here means "the solver, the cameras, the Jacobian and the keypoint correspondence all
agree"; it does not yet mean "this pipeline works on a dog".

    python tools/fit_selftest.py rigs/canis.ftcl
    python tools/fit_selftest.py rigs/canis.ftcl --frames 40 --views 4 --occlusion 0.25
    python tools/fit_selftest.py rigs/canis.ftcl --noise-scale 3 --outliers 0.05 --verbose
    python tools/fit_selftest.py rigs/canis.ftcl --morph out/theta_rex.json
    python tools/fit_selftest.py rigs/canis.ftcl --save-calib scraps/calib.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creaturelab.console import use_utf8  # noqa: E402

use_utf8()   # before argparse: --help is the thing a cp1252 console cannot print

import numpy as np                                     # noqa: E402

from ftcl.errors import FtclError                      # noqa: E402
from creaturelab.build import load                     # noqa: E402
from creaturelab.camera import save_calib, synthetic_rig, load_calib   # noqa: E402
from creaturelab.emit_mjcf import place_on_ground, to_mjcf   # noqa: E402
from creaturelab.fit import PoseFitter, triangulate    # noqa: E402
from creaturelab.keypoints import load_keypoints       # noqa: E402
from creaturelab.morph_io import morph_from_args, parse_sets   # noqa: E402
from creaturelab.tune import build_tuned               # noqa: E402


def make_truth(model, data, kps, n: int, rng, *, spread: float = 0.35):
    """`n` poses the body can actually hold, and where its landmarks are in each.

    Deliberately *not* a physics rollout. A rollout gives poses that are correlated in time and
    concentrated near the stance the passive tone settles into, which flatters the fit twice
    over: the initial guess is nearly right and the joints barely move. Independent draws
    across a fraction of each joint's authored range are a harder and more honest test of the
    optimiser, and they are still legal poses by construction.

    `spread` is that fraction. It is *not* 1.0 on purpose: a pose with every joint
    independently at an extreme is a pose no animal adopts, and failing on one would say
    nothing about the pipeline.
    """
    import mujoco
    place_on_ground(model, data, 0.0)
    rest = data.qpos.copy()

    hinges = [j for j in range(model.njnt)
              if model.jnt_type[j] in (2, 3) and model.jnt_limited[j]]
    adr = np.array([model.jnt_qposadr[j] for j in hinges], dtype=int)
    lo, hi = model.jnt_range[hinges, 0].copy(), model.jnt_range[hinges, 1].copy()

    qs = np.empty((n, model.nq))
    X = np.empty((n, len(kps), 3))
    sids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, k.site) for k in kps],
                    dtype=np.int32)
    for i in range(n):
        q = rest.copy()
        mid = q[adr]
        half = spread * 0.5 * (hi - lo)
        q[adr] = np.clip(mid + rng.uniform(-1.0, 1.0, adr.size) * half, lo, hi)
        # Root: move the animal around the capture volume and spin it, because a fit that only
        # ever sees the dog at the origin facing +x is a fit whose root DOFs were never tested.
        q[0:2] = rng.uniform(-0.7, 0.7, 2)
        q[2] = rest[2] + rng.uniform(-0.05, 0.15)
        yaw, pitch = rng.uniform(-math.pi, math.pi), rng.uniform(-0.12, 0.12)
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), \
            math.sin(pitch / 2)
        q[3:7] = [cy * cp, cy * sp, sy * sp, sy * cp]     # yaw about z composed with pitch
        q[3:7] /= np.linalg.norm(q[3:7])
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        qs[i], X[i] = q, data.site_xpos[sids]
    return qs, X, sids


def observe(calib, kps, X, rng, *, noise_scale: float, occlusion: float, outliers: float,
            swap: float):
    """Turn ground-truth 3D landmarks into what a detector would have emitted.

    Every corruption here is one a real DLC/SLEAP output has, and each is included because
    leaving it out makes the self-test pass for the wrong reason:

    * **σ jitter**, per landmark, from `keypoints.yaml`. Soft landmarks are noisier *and* wider
      in the Huber, and this is what checks the two are being applied to the same landmark.
    * **Occlusion.** A four-camera ring around a dog means a near-side limb hides the far-side
      one in roughly every frame. Occluded parts come back with a low confidence, not with a
      missing coordinate -- the detector always emits *something*, and `PoseFitter.CONF_FLOOR`
      is what has to reject it.
    * **Gross outliers.** A detector that is confidently wrong is the failure mode robust loss
      exists for. Without any, the Huber width is untested and δ could be anything.
    * **Left/right swaps**, separately from generic outliers, because they are the
      characteristic failure on a symmetric quadruped and they are *not* uniformly distributed
      noise -- a swap puts a confident, plausible-looking point on the wrong leg.
    """
    T, P, V = X.shape[0], len(kps), len(calib)
    uv = np.stack([c.project(X) for c in calib], axis=1)          # (T, V, P, 2) distorted px
    conf = rng.uniform(0.75, 0.99, (T, V, P))

    sigma = np.array([k.sigma_px for k in kps]) * noise_scale
    uv = uv + rng.normal(0.0, 1.0, uv.shape) * sigma[None, None, :, None]

    # Out of frame is an occlusion the detector never even reports.
    for v, c in enumerate(calib):
        w, h = c.size
        oob = ~np.isfinite(uv[:, v]).all(axis=-1)
        oob |= (uv[:, v, :, 0] < 0) | (uv[:, v, :, 0] >= w)
        oob |= (uv[:, v, :, 1] < 0) | (uv[:, v, :, 1] >= h)
        conf[:, v][oob] = 0.05
        uv[:, v][oob] = np.array([w / 2.0, h / 2.0])              # the centroid prior

    hidden = rng.random((T, V, P)) < occlusion
    conf[hidden] = rng.uniform(0.02, 0.3, int(hidden.sum()))
    uv[hidden] += rng.normal(0.0, 40.0, (int(hidden.sum()), 2))

    bad = (rng.random((T, V, P)) < outliers) & ~hidden
    uv[bad] += rng.normal(0.0, 90.0, (int(bad.sum()), 2))

    # Swaps act on PAIRS, so build the partner map from the labels themselves.
    pairs = []
    for i, k in enumerate(kps):
        other = (k.label[:-2] + "_r") if k.label.endswith("_l") else None
        j = kps.by_label.get(other) if other else None
        if j is not None:
            pairs.append((i, kps.labels.index(other)))
    for i, j in pairs:
        m = rng.random((T, V)) < swap
        if np.any(m):
            a, b = uv[m][:, i].copy(), uv[m][:, j].copy()
            block = uv[m]
            block[:, i], block[:, j] = b, a
            uv[m] = block
    return uv, conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig", nargs="?", default="rigs/canis.ftcl")
    ap.add_argument("--morph", metavar="FILE", help="fit against a specific body")
    ap.add_argument("--set", nargs="*", metavar="NAME=VAL", default=[])
    ap.add_argument("--keypoints", default=None, help="override notes/keypoints.yaml")
    ap.add_argument("--calib", default=None,
                    help="a real calib.json; default is a synthetic four-GoPro ring")
    ap.add_argument("--save-calib", metavar="FILE", help="write the synthetic rig out")
    ap.add_argument("--mocap", metavar="DIR",
                    help="(NOT BUILT) replay public P2 dog mocap instead of synthetic poses")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="multiply every landmark's sigma; 3 is a bad detector")
    ap.add_argument("--occlusion", type=float, default=0.20, help="P(part hidden in a view)")
    ap.add_argument("--outliers", type=float, default=0.02, help="P(confidently wrong)")
    ap.add_argument("--swap", type=float, default=0.01, help="P(left/right swapped)")
    ap.add_argument("--spread", type=float, default=0.35,
                    help="fraction of each joint's range the truth poses use")
    ap.add_argument("--guess", type=float, default=0.25,
                    help="how wrong the initial guess is: metres of root offset")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--verbose", action="store_true", help="per-frame and per-landmark detail")
    args = ap.parse_args()

    if args.mocap:
        print("error: --mocap is not built yet. The P2 importer is the next step; today this "
              "tool measures the OPTIMISER against synthetic poses, which cannot tell you "
              "whether a real dog is representable by the rig. See the module docstring.",
              file=sys.stderr)
        return 2

    import mujoco
    rng = np.random.default_rng(args.seed)

    try:
        params = load(args.rig).params
        morph = morph_from_args(args.rig, args.morph, parse_sets(args.set), params)
        creature, _ = build_tuned(args.rig, morph)
    except (FtclError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    data = mujoco.MjData(model)

    kps = load_keypoints(args.keypoints)
    problems = []
    from creaturelab.keypoints import check_against_rig
    problems = check_against_rig(kps, creature)
    if problems:
        print("error: the keypoint file and the rig disagree:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    calib = load_calib(args.calib) if args.calib else synthetic_rig(args.views, seed=args.seed)
    if args.save_calib:
        save_calib(args.save_calib, calib, session="fit_selftest synthetic")
        print(f"synthetic calibration -> {args.save_calib}")

    print(f"{creature.name}: {len(kps)} landmarks, {len(calib)} views, {args.frames} frames, "
          f"nq {model.nq} nv {model.nv}")
    print(f"corruption: sigma x{args.noise_scale:g}, {args.occlusion*100:.0f}% occluded, "
          f"{args.outliers*100:.0f}% outliers, {args.swap*100:.0f}% L/R swaps")

    qs, X_true, sids = make_truth(model, data, kps, args.frames, rng, spread=args.spread)
    obs, conf = observe(calib, kps, X_true, rng, noise_scale=args.noise_scale,
                        occlusion=args.occlusion, outliers=args.outliers, swap=args.swap)

    # A deliberately wrong initial guess. The rest pose at the origin is what a real clip's
    # first frame gets, and offsetting the root on top of that is what checks LM is doing the
    # work rather than the initialiser.
    place_on_ground(model, data, 0.0)
    q0 = data.qpos.copy()
    q0[0:2] += rng.normal(0.0, args.guess, 2)

    # Every frame is fit INDEPENDENTLY and from the same bad guess. `make_truth` draws poses
    # independently, so they are not a clip and warm-starting across them would be measuring a
    # sequence that does not exist -- and would hide precisely the cold-start failure this tool
    # is here to catch. `fit_sequence`'s warm start is exercised separately, in tests/test_fit.py.
    fitter = PoseFitter(model, data, kps, calib)
    und = np.stack([fitter.undistort(obs[t]) for t in range(args.frames)], axis=0)
    qfit = np.empty((args.frames, model.nq))
    results = []
    for t in range(args.frames):
        res = fitter.fit(und[t], conf[t], q0=q0, iters=args.iters, undistorted=True)
        qfit[t] = res.qpos
        results.append(res)

    # --- how much of the truth came back --------------------------------------------------
    X_fit = np.empty_like(X_true)
    for t in range(args.frames):
        data.qpos[:] = qfit[t]
        mujoco.mj_kinematics(model, data)
        X_fit[t] = data.site_xpos[sids]
    d3 = np.linalg.norm(X_fit - X_true, axis=2)                    # (T, P) metres
    root = np.linalg.norm(qfit[:, 0:3] - qs[:, 0:3], axis=1)
    # Quaternion angle between truth and fit, the sign-agnostic way.
    dot = np.abs(np.einsum("ij,ij->i", qfit[:, 3:7], qs[:, 3:7]))
    ang = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    rms = np.array([r.rms_px for r in results])
    conv = sum(r.converged for r in results)

    print(f"\nfit: {conv}/{args.frames} converged, "
          f"{np.mean([r.iters for r in results]):.0f} LM iterations mean")
    print(f"  reprojection      {np.nanmean(rms):6.2f} px RMS   "
          f"(worst frame {np.nanmax(rms):6.2f})")
    print(f"  landmark 3D       {np.mean(d3)*1000:6.1f} mm mean  "
          f"median {np.median(d3)*1000:6.1f}  p95 {np.percentile(d3, 95)*1000:6.1f}  "
          f"worst {d3.max()*1000:6.1f}")
    print(f"  root position     {np.mean(root)*1000:6.1f} mm mean  "
          f"worst {root.max()*1000:6.1f}")
    print(f"  root orientation  {np.mean(ang):6.2f} deg mean  worst {ang.max():6.2f}")

    if args.verbose:
        print("\n  per landmark (mean 3D error, mean reprojection):")
        per_px = np.nanmean(np.array([r.per_kp_px for r in results]), axis=0)
        order = np.argsort(-d3.mean(axis=0))
        for i in order:
            k = kps.keypoints[i]
            print(f"    {k.label:<14s} {k.cls:<5s} sigma {k.sigma_px:4.1f}  "
                  f"{d3[:, i].mean()*1000:6.1f} mm   {per_px[i]:6.2f} px")
        print("\n  per frame:")
        for t, r in enumerate(results):
            print(f"    {t:3d}  {r}   3D {d3[t].mean()*1000:6.1f} mm  "
                  f"root {root[t]*1000:6.1f} mm  {ang[t]:5.2f} deg")

    # --- the baseline the fit has to beat -------------------------------------------------
    # Free-point triangulation of the SAME corrupted observations: each landmark placed by its
    # own rays, with no skeleton at all. This, not a round number of millimetres, is the honest
    # bar. An absolute bar cannot be set without knowing the detector's real sigma, and
    # `keypoints.yaml` says plainly that it does not (`sigma_px_measured: false`) -- so any
    # fixed millimetre threshold here would be measuring the placeholder, not the pipeline.
    #
    # What the comparison actually asks is the right question anyway: does the skeleton ADD
    # information? It pools 21 landmarks into 37 DOFs and forbids configurations the body
    # cannot hold, so it should beat independent points comfortably. If it does not, the
    # skeleton is fighting the data -- a wrong correspondence, a wrong bone length, or a
    # camera convention error all show up exactly that way, and none of them show up in the
    # reprojection number, which a broken fit can drive to zero by contorting the body.
    tri_err = []
    for t in range(args.frames):
        pts, okmask = triangulate(calib, und[t], conf[t], conf_floor=PoseFitter.CONF_FLOOR)
        if np.any(okmask):
            tri_err.append(np.linalg.norm(pts[okmask] - X_true[t][okmask], axis=1))
    tri = np.concatenate(tri_err) if tri_err else np.array([np.nan])

    # And what the observations physically carry: one pixel of jitter is depth/f metres of
    # transverse ray uncertainty, so this is the floor no estimator gets under.
    depth = np.mean([np.linalg.norm(X_true.reshape(-1, 3) - c.center, axis=1).mean()
                     for c in calib])
    floor = float(np.mean([np.mean([k.sigma_px for k in kps]) * args.noise_scale * depth / c.fx
                           for c in calib]))

    med, med_tri = float(np.median(d3)), float(np.median(tri))
    print(f"\n  free-point triangulation of the same data: {med_tri*1000:6.1f} mm median")
    print(f"  ray-uncertainty floor at {depth:.2f} m:      {floor*1000:6.1f} mm "
          f"(sigma {np.mean([k.sigma_px for k in kps]):.1f} px x{args.noise_scale:g}"
          f"{'' if kps.sigma_px_measured else ', UNMEASURED placeholder'})")

    # Root orientation is reported but deliberately NOT part of the verdict. The spine is
    # articulated, so a rotation of the root body can be traded against the spine joints for
    # nearly the same landmark positions: a few degrees of root error with correct landmarks is
    # a gauge freedom, not a mistake. Failing on it would be failing the rig for being
    # articulated. It is the per-parameter identifiability report (`fit_report.py`) that has to
    # deal with this properly.
    ok = (conv >= 0.9 * args.frames
          and (not np.isfinite(med_tri) or med <= med_tri)
          and med <= max(3.0 * floor, 1e-4))
    print(f"\n{'PASS' if ok else 'FAIL'}: landmark recovery {med*1000:.1f} mm median "
          f"-- vs {med_tri*1000:.1f} mm unskeletoned and a {floor*1000:.1f} mm ray floor; "
          f"{conv}/{args.frames} converged")
    if not ok:
        print("  This is the failure the tool exists to find EARLY. Before blaming the "
              "optimiser, check the keypoint<->site correspondence and the camera "
              "convention -- both fail this way, and neither shows up in the reprojection "
              "number. Re-run with --noise-scale 0 --occlusion 0 --outliers 0 --swap 0: "
              "that case must come back at 0.0 mm exactly, and if it does not, the bug is "
              "in the geometry rather than in the robustness.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
