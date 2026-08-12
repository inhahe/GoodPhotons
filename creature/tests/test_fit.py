"""Regression tests for the camera model and the keypoint pose fit.

Same bias as the rest of the suite: aimed at the failures that return a plausible number
instead of raising. Multi-view geometry is unusually rich in those. A camera whose +y points
up instead of down produces a picture that is merely upside down, which looks like a plausible
mounting. An extrinsic that has picked up a scale makes the animal come out uniformly the wrong
size, which the morph fit then absorbs into `body_scale`. A Jacobian computed from a stale
`cdof` is still often a descent direction, so LM still converges -- just slowly, and somewhere
else. And a fit with a broken correspondence drives its own reprojection error to zero by
contorting the body, so the number it reports looks *better* than a correct fit's.

Every one of those is caught here by comparing against something independently known: an
analytic derivative against finite differences, a projection against its own inverse, and a
fitted pose against the ground truth it was generated from.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mujoco = pytest.importorskip("mujoco")

from creaturelab import camera as cammod                          # noqa: E402
from creaturelab.camera import (CalibError, Calibration, Camera,  # noqa: E402
                                load_calib, look_at, save_calib, synthetic_rig)
from creaturelab.emit_mjcf import place_on_ground, to_mjcf        # noqa: E402
from creaturelab.fit import PoseFitter, fit_sequence, triangulate  # noqa: E402
from creaturelab.keypoints import load_keypoints                  # noqa: E402
from creaturelab.tune import build_tuned                          # noqa: E402

RIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rigs", "canis.ftcl")


@pytest.fixture(scope="module")
def kps():
    return load_keypoints()


@pytest.fixture(scope="module")
def rig():
    creature, _ = build_tuned(RIG, {})
    model = mujoco.MjModel.from_xml_string(to_mjcf(creature))
    return model


@pytest.fixture
def data(rig):
    d = mujoco.MjData(rig)
    place_on_ground(rig, d, 0.0)
    return d


@pytest.fixture(scope="module")
def calib():
    return synthetic_rig(4, seed=3)


@pytest.fixture(scope="module")
def cloud():
    """A blob of points in front of every camera, standing in for landmarks."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-0.6, 0.6, 40), rng.uniform(-0.6, 0.6, 40),
                            rng.uniform(0.1, 0.9, 40)])


# ------------------------------------------------------------------------------ the camera
def test_undistort_inverts_project(calib, cloud):
    """The two directions of the fisheye model must actually be inverses.

    They are used by *different callers* -- synthetic views project, the fit undistorts -- so
    a disagreement between them never appears as an inconsistency at any single call site. It
    appears as a fit that cannot get under a few pixels and looks like a bad calibration.
    """
    for cam in calib:
        d = cam.undistort(cam.project(cloud))
        assert np.allclose(d, cam.project_linear(cloud), atol=1e-7), cam.name


def test_distortion_is_not_a_no_op(calib):
    """A round trip that passes trivially would also pass with D ignored entirely.

    Measured out at the EDGE of frame on purpose. The capture volume sits within ~10 degrees of
    each camera's axis, where a fourth-order distortion moves a point by well under a pixel --
    so a test using the landmark cloud would be satisfied by a no-op undistort and would go on
    passing after someone deleted the model. The whole reason `capture.md` insists on native
    wide with a fisheye fit, rather than in-camera Linear dewarp, is what happens off-axis.
    """
    for cam in calib:
        # A fan of directions in the camera's own frame, out to 50 degrees off axis.
        ang = np.radians(np.array([5.0, 20.0, 35.0, 50.0]))
        dirs = np.stack([np.sin(ang), np.zeros_like(ang), np.cos(ang)], axis=1)
        X = cam.center + dirs @ cam.R                      # R.T @ d, batched
        moved = np.linalg.norm(cam.project(X) - cam.project_linear(X), axis=1)
        assert moved.max() > 20.0, f"{cam.name}: distortion moves nothing at the frame edge " \
                                   f"({moved.max():.2f} px); the round trip above is vacuous"


def test_an_ideal_fisheye_is_not_a_pinhole():
    """`D == 0` does NOT make a fisheye a pinhole -- it makes it equidistant.

    Keying the undistort branch off `any(D)` rather than off `model` is an easy simplification
    that silently turns a legal ideal-fisheye calibration into a pinhole one, and the error is
    zero on the optical axis and tens of pixels at the edge of frame -- i.e. invisible in any
    test that keeps the subject centred.
    """
    cam = synthetic_rig(2, D=(0.0, 0.0, 0.0, 0.0))[0]
    X = cam.center + cam.R.T @ np.array([2.0, 0.0, 2.0])       # 45 deg off axis
    assert not np.allclose(cam.project(X), cam.project_linear(X), atol=1.0)
    assert np.allclose(cam.undistort(cam.project(X)), cam.project_linear(X), atol=1e-7)


def test_pixel_axes_point_the_way_opencv_says(calib):
    """+v is DOWN and +u is right, in the world, for every camera in the ring.

    Get this wrong and the picture is upside down, the fit still converges (to a dog standing
    on the ceiling in a mirrored world), and nothing raises anywhere.
    """
    for cam in calib:
        base = np.array([0.0, 0.0, 0.45])
        up = cam.project_linear(base + np.array([0.0, 0.0, 0.3]))
        assert up[1] < cam.project_linear(base)[1], f"{cam.name}: +z world is not up in frame"
        # The camera's own right axis (R's first row) must move u in the + direction.
        right = cam.project_linear(base + cam.R[0] * 0.3)
        assert right[0] > cam.project_linear(base)[0], f"{cam.name}: +u is not camera-right"


def test_points_behind_the_camera_have_no_projection(calib):
    """Not a pixel on the wrong side of the frame -- NaN.

    A point behind the pupil reflects through the centre of projection and lands somewhere
    perfectly plausible. During a fit that is a residual pointing the wrong way, and LM will
    happily chase it straight through the camera.
    """
    cam = calib[0]
    behind = cam.center - cam.R[2] * 1.0
    assert np.all(np.isnan(cam.project_linear(behind)))
    assert np.all(np.isnan(cam.project(behind)))


def test_d_uv_d_X_matches_finite_differences(calib, cloud):
    """The first link of the fit's chain rule, against the only thing that can check it."""
    h = 1e-6
    for cam in calib:
        J = cam.d_uv_d_X(cloud)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            fd = (cam.project_linear(cloud + step) - cam.project_linear(cloud - step)) / (2 * h)
            assert np.allclose(J[:, :, axis], fd, atol=1e-4, rtol=1e-4), \
                f"{cam.name} axis {axis}"


def test_camera_centre_and_extrinsics_agree(calib):
    for cam in calib:
        assert np.allclose(cam.to_camera(cam.center), 0.0, atol=1e-9)


def test_look_at_refuses_a_degenerate_up():
    with pytest.raises(CalibError):
        look_at((0, 0, 3), (0, 0, 0))          # straight down, with up = +z


def test_calib_round_trips_through_json(tmp_path, calib, cloud):
    p = str(tmp_path / "calib.json")
    save_calib(p, calib, session="unit")
    back = load_calib(p)
    assert back.names == calib.names and back.session == "unit"
    for a, b in zip(calib, back):
        assert np.allclose(a.project(cloud), b.project(cloud), atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("break_it, msg", [
    (lambda d: d.update(schema="creature/calib@99"), "schema"),
    (lambda d: d.__setitem__("cameras", d["cameras"][:1]), "camera"),
    (lambda d: d["cameras"][0].__setitem__("R", [[2, 0, 0], [0, 2, 0], [0, 0, 2]]), "rotation"),
    (lambda d: d["cameras"][0].__setitem__("R", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
     "reflection"),
    (lambda d: d["cameras"][1].__setitem__("name", d["cameras"][0]["name"]), "called"),
    (lambda d: d["cameras"][0].__setitem__("model", "pinhole"), "pinhole"),
    (lambda d: d["cameras"][0].__setitem__("D", [0.0, 0.0]), "coefficients"),
    (lambda d: d["cameras"][0].__setitem__("t", [0.0, 0.0]), "components"),
])
def test_a_calibration_that_cannot_be_trusted_is_refused(tmp_path, calib, break_it, msg):
    """A wrong calibration is NOT detectable from the footage afterwards, so it is detected here.

    The single-camera case is in this list on purpose: one view is a perfectly well-formed
    file, and the fit will run on it, converge, and return a pose that is unobservable in depth.
    """
    import json
    p = str(tmp_path / "calib.json")
    save_calib(p, calib)
    with open(p) as fh:
        doc = json.load(fh)
    break_it(doc)
    with open(p, "w") as fh:
        json.dump(doc, fh)
    with pytest.raises(CalibError) as e:
        load_calib(p)
    assert msg in str(e.value)


# ------------------------------------------------------------------------- triangulation
def test_triangulation_recovers_clean_points(calib, cloud):
    uv = np.stack([c.undistort(c.project(cloud)) for c in calib], axis=0)
    conf = np.ones(uv.shape[:2])
    pts, ok = triangulate(calib, uv, conf)
    assert ok.all()
    assert np.abs(pts - cloud).max() < 1e-6


def test_a_point_seen_once_is_returned_invalid(calib, cloud):
    """Rather than placed at an arbitrary depth along its single ray."""
    uv = np.stack([c.undistort(c.project(cloud)) for c in calib], axis=0)
    conf = np.ones(uv.shape[:2])
    conf[1:, 0] = 0.0                                  # landmark 0 seen by cam0 only
    pts, ok = triangulate(calib, uv, conf)
    assert not ok[0] and np.all(np.isnan(pts[0]))
    assert ok[1:].all()


def test_triangulation_ignores_low_confidence(calib, cloud):
    uv = np.stack([c.undistort(c.project(cloud)) for c in calib], axis=0)
    conf = np.ones(uv.shape[:2])
    uv[3] += 400.0                                     # cam3 is nonsense ...
    conf[3] = 0.1                                      # ... and says so
    pts, ok = triangulate(calib, uv, conf)
    assert ok.all() and np.abs(pts - cloud).max() < 1e-6


# --------------------------------------------------------------------------- the pose fit
def _observe(calib, model, data, fitter, q):
    """Perfect observations of the pose `q`: (V, P, 2) undistorted px, (V, P) confidences."""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    X = data.site_xpos[fitter.site_ids]
    uv = np.stack([c.undistort(c.project(X)) for c in calib], axis=0)
    return uv, np.ones(uv.shape[:2])


def _pose(model, data, rng, spread=0.3, yaw=0.0):
    place_on_ground(model, data, 0.0)
    q = data.qpos.copy()
    j = [k for k in range(model.njnt) if model.jnt_type[k] in (2, 3) and model.jnt_limited[k]]
    adr = np.array([model.jnt_qposadr[k] for k in j])
    lo, hi = model.jnt_range[j, 0], model.jnt_range[j, 1]
    q[adr] = np.clip(q[adr] + rng.uniform(-1, 1, adr.size) * spread * 0.5 * (hi - lo), lo, hi)
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return q


def test_the_jacobian_matches_finite_differences(rig, data, kps, calib):
    """The whole fit is this matrix. Everything else is `numpy.linalg.solve`.

    Differenced in the same VELOCITY space LM steps in, via `mj_integratePos`, because that is
    what the Jacobian is a derivative with respect to -- differencing `qpos` directly would
    disagree on the four quaternion components for reasons that have nothing to do with a bug.

    This is the test that catches the missing `mj_comPos`: `mj_jacSite` reads `cdof`, which
    `mj_kinematics` does not write, so without it the Jacobian is the previous configuration's.
    """
    rng = np.random.default_rng(1)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, yaw=0.7)
    obs, conf = _observe(calib, rig, data, fitter, q)

    data.qpos[:] = q
    _, J, _ = fitter.residual(obs, conf, jac=True)
    h = 1e-6
    step = np.zeros(rig.nv)
    for dof in rng.choice(rig.nv, 8, replace=False):
        step[:] = 0.0
        step[dof] = h
        data.qpos[:] = q
        mujoco.mj_integratePos(rig, data.qpos, step, 1.0)
        rp, _, _ = fitter.residual(obs, conf, jac=False)
        step[dof] = -h
        data.qpos[:] = q
        mujoco.mj_integratePos(rig, data.qpos, step, 1.0)
        rm, _, _ = fitter.residual(obs, conf, jac=False)
        fd = (rp - rm) / (2 * h)
        # Rows for observations that are not used are identically zero in both.
        scale = max(np.abs(fd).max(), 1e-9)
        assert np.abs(J[:, dof] - fd).max() / scale < 2e-3, f"dof {dof}"


def test_a_perfect_observation_is_fit_exactly(rig, data, kps, calib):
    """No noise, no occlusion: the answer is known and the fit must find it.

    This is the check that the camera model, the correspondence, the Jacobian and the
    quaternion handling all agree. Any one of them being wrong leaves a residual floor here,
    and every one of them is invisible on real footage, where a few pixels of error is normal.
    """
    rng = np.random.default_rng(2)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, yaw=1.9)
    obs, conf = _observe(calib, rig, data, fitter, q)

    place_on_ground(rig, data, 0.0)
    res = fitter.fit(obs, conf, q0=data.qpos.copy(), undistorted=True, iters=200)
    assert res.converged and res.rms_px < 1e-3, str(res)

    data.qpos[:] = res.qpos
    mujoco.mj_kinematics(rig, data)
    got = data.site_xpos[fitter.site_ids].copy()
    data.qpos[:] = q
    mujoco.mj_kinematics(rig, data)
    assert np.linalg.norm(got - data.site_xpos[fitter.site_ids], axis=1).max() < 1e-4


def test_the_root_initialiser_makes_the_answer_independent_of_the_guess(rig, data, kps, calib):
    """A fit whose answer moves when the initial guess moves is not a measurement.

    This is the property `init_root` is actually kept for -- see its docstring, which used to
    claim accuracy instead and was wrong. On real footage there is no ground truth to notice
    start-dependence against, so it has to be pinned here: three wildly different starting
    positions must produce the *same* pose, and the initialised fit must also be cheaper than
    the uninitialised one, whose cost grows with how wrong the guess was.
    """
    rng = np.random.default_rng(3)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, spread=0.2, yaw=2.6)
    obs, conf = _observe(calib, rig, data, fitter, q)
    obs += rng.normal(0.0, 4.0, obs.shape)
    truth = _sites(rig, data, fitter, q)

    place_on_ground(rig, data, 0.0)
    base = data.qpos.copy()
    offsets = (0.0, 1.0, 3.0)
    poses, iters_on, iters_off = [], [], []
    for off in offsets:
        q0 = base.copy()
        q0[0:2] += off
        on = fitter.fit(obs, conf, q0=q0.copy(), undistorted=True, iters=300, init_root=True)
        off_ = fitter.fit(obs, conf, q0=q0.copy(), undistorted=True, iters=300, init_root=False)
        poses.append(_sites(rig, data, fitter, on.qpos))
        iters_on.append(on.iters)
        iters_off.append(off_.iters)

    for p in poses[1:]:
        assert np.abs(p - poses[0]).max() < 1e-9, "the answer depends on the initial guess"
    assert np.median(np.linalg.norm(poses[0] - truth, axis=1)) < 0.05
    # Constant cost, versus a cost that grows with how wrong the guess was. Note the
    # uninitialised fit is CHEAPER from a good guess -- it skips the triangulation. Asserting
    # `on < off` everywhere would be asserting something untrue; what is true is that the
    # initialised cost does not depend on the guess and the uninitialised cost does.
    assert len(set(iters_on)) == 1, f"initialised cost varies with the guess: {iters_on}"
    assert iters_off[-1] > iters_off[0], \
        f"uninitialised cost did not grow with a worse guess: {iters_off}"
    assert max(iters_on) < max(iters_off), f"{iters_on} vs {iters_off}"


def test_low_confidence_garbage_is_not_evidence(rig, data, kps, calib):
    """The detector emits a coordinate for an occluded part too -- typically the centroid.

    Weighting those in at low confidence is not conservative, it is a systematic pull toward
    the middle of the animal, so `CONF_FLOOR` rejects rather than down-weights. The garbage
    here is 300 px off, which would be impossible to miss if it were being used.
    """
    rng = np.random.default_rng(4)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, yaw=-0.8)
    obs, conf = _observe(calib, rig, data, fitter, q)
    obs[2] += 300.0
    conf[2] = 0.1
    res = fitter.fit(obs, conf, q0=q.copy(), undistorted=True, iters=200)
    assert res.rms_px < 1e-3, str(res)
    assert res.n_obs == 3 * len(kps), "the rejected view still contributed observations"


def test_the_huber_keeps_one_confident_mistake_from_moving_the_body(rig, data, kps, calib):
    """A gross outlier at FULL confidence is what robust loss is for.

    Least squares would spread a 200 px error over every joint; the Huber caps its pull at
    delta. The bar is deliberately loose -- the point is that the body stays put, not that the
    outlier is free.
    """
    rng = np.random.default_rng(5)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, yaw=0.4)
    obs, conf = _observe(calib, rig, data, fitter, q)

    data.qpos[:] = q
    mujoco.mj_kinematics(rig, data)
    truth = data.site_xpos[fitter.site_ids].copy()

    obs[1, 4] += 200.0                                    # one landmark, one view, confident
    res = fitter.fit(obs, conf, q0=q.copy(), undistorted=True, iters=200)
    data.qpos[:] = res.qpos
    mujoco.mj_kinematics(rig, data)
    moved = np.linalg.norm(data.site_xpos[fitter.site_ids] - truth, axis=1)
    assert np.median(moved) < 0.005, f"one bad point moved the whole body: {moved.max():.3f} m"


def test_the_soft_landmarks_are_given_the_wider_huber(rig, data, kps, calib):
    """`class` in keypoints.yaml has to reach the loss, or the field is documentation.

    A soft landmark slides over bone by centimetres, which no rigid-body model can represent;
    fit it at the rigid width and it drags the skeleton. Checked at the fitter, not at the
    loader, because the loader agreeing with the file proves nothing about what the fit uses.
    """
    fitter = PoseFitter(rig, data, kps, calib)
    soft = np.array([k.is_soft for k in kps])
    assert soft.any() and not soft.all(), "the rig has no mix of classes left to test"
    assert fitter.huber[soft].min() > fitter.huber[~soft].max()


def test_a_missing_site_is_named_not_ignored(rig, data, kps, calib):
    """The keypoint<->rig drift failure, at the one place it can still be caught cheaply."""
    from creaturelab.keypoints import Keypoint, KeypointSet
    bad = KeypointSet(rig=kps.rig, creature=kps.creature,
                      keypoints=list(kps.keypoints) + [
                          Keypoint("nose_tip_2", "no_such_site", "rigid", 6.0, 2.0, "x")],
                      skeleton=[], sigma_px_default=kps.sigma_px_default,
                      sigma_px_measured=False)
    with pytest.raises(ValueError, match="no_such_site"):
        PoseFitter(rig, data, bad, calib)


def test_too_few_observations_is_reported_not_guessed(rig, data, kps, calib):
    """An under-determined frame must come back flagged, not wearing a convergence badge."""
    rng = np.random.default_rng(6)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng)
    obs, conf = _observe(calib, rig, data, fitter, q)
    conf[:] = 0.0
    res = fitter.fit(obs, conf, q0=q.copy(), undistorted=True)
    assert not res.converged and res.n_obs == 0 and math.isnan(res.rms_px)


def test_the_fit_stays_inside_the_authored_joint_ranges(rig, data, kps, calib):
    """A fitted pose the body cannot hold is not a pose; it is a number that looks like one.

    Every consumer downstream -- the AMP demonstrations, `mj_inverse` in the E_phys gate --
    assumes the trajectory is legal, and none of them check.
    """
    rng = np.random.default_rng(7)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, spread=0.9, yaw=2.4)
    obs, conf = _observe(calib, rig, data, fitter, q)
    obs += rng.normal(0.0, 8.0, obs.shape)                # push it hard enough to want to leave
    res = fitter.fit(obs, conf, q0=q.copy(), undistorted=True, iters=200)
    j = [k for k in range(rig.njnt) if rig.jnt_type[k] in (2, 3) and rig.jnt_limited[k]]
    adr = np.array([rig.jnt_qposadr[k] for k in j])
    assert np.all(res.qpos[adr] >= rig.jnt_range[j, 0] - 1e-9)
    assert np.all(res.qpos[adr] <= rig.jnt_range[j, 1] + 1e-9)


def test_a_sequence_tracks_the_motion(rig, data, kps, calib):
    """A clip fit must recover the motion and converge on every frame.

    Deliberately does NOT assert that the warm start is smoother or cheaper than solving each
    frame independently. Both were measured and both are, at best, marginal on this synthetic
    data -- see `fit_sequence`'s docstring and known-issues #7. Writing an assertion that
    happens to pass at one seed would turn an open question into a claim.

    What IS asserted is what the sequence path has to get right to be usable at all: it tracks
    5 cm of travel from the pixels alone, and it does not end up *worse* than independent
    solving, which is how a warm-start bug (handing a failed frame's pose forward) shows up.
    """
    rng = np.random.default_rng(8)
    fitter = PoseFitter(rig, data, kps, calib)
    q = _pose(rig, data, rng, spread=0.2, yaw=1.2)
    qs, obs, conf = [], [], []
    for t in range(6):
        qt = q.copy()
        qt[0] += 0.01 * t                                 # a slow walk across the volume
        qs.append(qt)
        # `fit_sequence` undistorts internally, so hand it DISTORTED pixels.
        obs.append(np.stack([cam.project(_sites(rig, data, fitter, qt)) for cam in calib]))
        conf.append(np.ones((len(calib), len(kps))))
    obs, conf = np.asarray(obs), np.asarray(conf)
    obs += rng.normal(0.0, 3.0, obs.shape)                # or triangulation is exact and the
    #                                                       cold start is as cheap as a warm one

    place_on_ground(rig, data, 0.0)
    q0 = data.qpos.copy()
    out, results = fit_sequence(fitter, obs, conf, q0=q0.copy(), iters=60, first_iters=300)
    assert all(r.converged for r in results), [str(r) for r in results]
    # It tracked the motion: 5 cm of travel recovered from the pixels alone.
    assert np.abs(out[:, 0] - np.array([qt[0] for qt in qs])).max() < 0.02

    # The same frames, each solved cold and independently: the trajectory a fit without a warm
    # start would produce. The true motion is a constant 1 cm per frame, so every bit of second
    # difference in either trajectory is estimator jitter and nothing else.
    cold = np.stack([fitter.fit(fitter.undistort(obs[t]), conf[t], q0=q0.copy(), iters=300,
                                undistorted=True).qpos for t in range(len(obs))])

    def jitter(traj):
        return float(np.abs(np.diff(traj[:, 0:3], n=2, axis=0)).mean())

    assert jitter(out) < 1.5 * jitter(cold), \
        f"warm-started trajectory jitters {jitter(out)*1000:.2f} mm/frame^2 against " \
        f"{jitter(cold)*1000:.2f} solved independently -- the warm start is making it worse, " \
        f"which is what handing a FAILED frame's pose forward looks like"


def _sites(model, data, fitter, q):
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[fitter.site_ids].copy()
