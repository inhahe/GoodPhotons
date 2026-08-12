"""E_kp: fit the simulator's own `qpos` to multi-view 2D keypoints.

This is the inner loop of stage C. `todo.md` P5 makes the load-bearing choice: fit **directly
in the simulator's joint parameterisation**, by minimising 2D reprojection error, rather than
fitting some separate 3D animal model and retargeting afterwards. Everything here follows from
that. The unknown is `qpos` -- the same vector the policy will later be trained on -- so there
is no retargeting stage, no correspondence between two skeletons to maintain, and no way for
the fitted motion to be unreachable by the body that has to reproduce it.

The three pieces, and why each is the boring choice:

* **The Jacobian is analytic, via `mj_jacSite`.** MuJoCo will hand back d(site position)/dq for
  free, so the only thing left to write is d(pixel)/d(world point), which is four entries of a
  2x3. Finite-differencing 37 DOFs instead would cost 37 extra `mj_kinematics` calls per
  iteration and would still be less accurate. `notes/keypoints.yaml`'s sites exist precisely so
  this call has something to point at.
* **Levenberg-Marquardt, on the normal equations.** nv is ~37 and the residual is a few hundred
  rows; `J^T J` is a 37x37 solve that costs nothing next to the kinematics. LM rather than
  Gauss-Newton because a keypoint fit *will* be started from a bad guess (the first frame of a
  clip has no previous frame to warm-start from) and Gauss-Newton diverges there.
* **The free joint is stepped with `mj_integratePos`, never by adding to `qpos`.** The root
  orientation is a unit quaternion: nq is 38 and nv is 37, and the LM step lives in the
  37-dimensional *velocity* space. Adding a 3-vector into a 4-component quaternion is the
  single most common way to write a pose fitter that almost works -- it drifts off the unit
  sphere, MuJoCo renormalises silently, and the result is a rotation that lags the translation.

**Huber, and where its width comes from.** δ is in pixels and comes from `keypoints.yaml`'s
`huber_delta_px`, per landmark *class*: 2 px for a rigid landmark, 8 px for a soft one. That
asymmetry is the whole reason `class` is in the correspondence file. A soft landmark's residual
is dominated by tissue sliding over bone, which a rigid-body model cannot represent at all; fit
it at the rigid width and a marker wandering over the shoulder blade drags the entire skeleton
after it. σ, separately, is the detector's own noise and only sets the relative weight.

**What is NOT here.** E_sil (the silhouette chamfer, which is the only source of girth), E_temp,
E_lim as a barrier, and the E_phys gate. This module is E_kp and the pose solve; the anatomy
search over θ that wraps it belongs in `tools/fit_anatomy.py`. See `notes/pipeline.md` stage C
for the objective term by term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Calibration
from .keypoints import KeypointSet


def triangulate(calib: Calibration, uv: np.ndarray, conf: np.ndarray, *,
                conf_floor: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Linear DLT: `(V, P, 2)` UNDISTORTED pixels -> `(P, 3)` world points, plus a valid mask.

    This is not part of the objective and is never iterated -- it exists only to produce an
    initial guess. It is a *free-3D-point* triangulation: it ignores the skeleton entirely and
    lets every landmark go wherever the rays say, which is exactly what makes it useful as an
    initialiser and useless as an answer. The fit's job is to explain those points with a body.

    Two views minimum per point, and a point seen by fewer is returned invalid rather than
    guessed at -- a landmark triangulated from one ray lands at an arbitrary depth, and an
    initialiser that includes it will drag the whole rigid alignment after it.
    """
    V, P = len(calib), uv.shape[1]
    Pm = [c.K @ np.hstack([c.R, c.t[:, None]]) for c in calib]      # (3, 4) each
    out = np.full((P, 3), np.nan)
    ok = np.zeros(P, dtype=bool)
    for p in range(P):
        rows = []
        for v in range(V):
            if conf[v, p] < conf_floor or not np.isfinite(uv[v, p]).all():
                continue
            u, w = uv[v, p]
            rows.append(u * Pm[v][2] - Pm[v][0])
            rows.append(w * Pm[v][2] - Pm[v][1])
        if len(rows) < 4:                                           # < 2 views
            continue
        _, _, Vt = np.linalg.svd(np.asarray(rows))
        h = Vt[-1]
        if abs(h[3]) < 1e-12:
            continue                                                # a point at infinity
        out[p] = h[:3] / h[3]
        ok[p] = True
    return out, ok


def _kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid transform taking `A` onto `B`: returns (R, cA, cB) with `b ~= R(a - cA) + cB`.

    No scale: the body's size is θ's job, and letting the initialiser absorb it would hand the
    anatomy search a starting point that already explains the wrong thing. The reflection guard
    is not optional -- a noisy near-planar set of landmarks (a dog seen side-on) can make the
    raw SVD prefer a mirror, which reads downstream as a dog whose legs are on the wrong side.
    """
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ D @ U.T, cA, cB


@dataclass
class FitResult:
    """What one pose solve did. Kept even when it failed -- a fit that stopped early is data."""
    qpos: np.ndarray
    rms_px: float               # RMS reprojection error over the observations that were used
    max_px: float               # worst single landmark, which is what spots a swapped label
    iters: int
    converged: bool
    n_obs: int                  # (view, landmark) pairs with a confidence above the floor
    cost: float = 0.0           # the robustified objective, for callers that compare fits
    per_kp_px: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __str__(self) -> str:
        ok = "converged" if self.converged else f"STOPPED at {self.iters}"
        return (f"{ok}: {self.rms_px:.2f} px RMS, worst {self.max_px:.2f} px, "
                f"{self.n_obs} observations")


class PoseFitter:
    """Solve `qpos` from one frame of multi-view 2D keypoints. Reusable across frames.

    Holds the index tables (which site each landmark is, which σ and δ it carries) so a
    sequence fit does not rebuild them 20 000 times, and holds the scratch Jacobian so the
    inner loop allocates nothing.
    """

    #: Below this confidence a detector output is not evidence. DLC/SLEAP emit a coordinate for
    #: every frame whether or not the part is visible, and an occluded landmark's "position" is
    #: wherever the network's prior put it -- typically dead centre of the animal. Including
    #: those at low weight is not conservative, it is a systematic pull toward the centroid.
    CONF_FLOOR = 0.35

    def __init__(self, model, data, kps: KeypointSet, calib: Calibration, *,
                 free_dofs: np.ndarray | None = None):
        import mujoco
        self._mj = mujoco
        self.m, self.d = model, data
        self.kps, self.calib = kps, calib
        self.nv = int(model.nv)

        ids = []
        for kp in kps:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, kp.site)
            if sid < 0:
                raise ValueError(f"keypoint {kp.label!r} names site {kp.site!r}, which this "
                                 f"model does not have -- run "
                                 f"`python tools/keypoints_project.py --check`")
            ids.append(sid)
        self.site_ids = np.asarray(ids, dtype=np.int32)
        self.sigma = np.array([kp.sigma_px for kp in kps], dtype=float)
        self.huber = np.array([kp.huber_px for kp in kps], dtype=float)

        #: Which DOFs the solve is allowed to move. Locking the joints and fitting the six root
        #: DOFs alone is the standard way to get a usable initial guess out of a bad one, and
        #: it is also how a caller freezes a limb whose landmarks are all occluded this frame.
        self.free_dofs = (np.arange(self.nv) if free_dofs is None
                          else np.asarray(free_dofs, dtype=int))

        self._jacp = np.zeros((3, self.nv))
        self._n_view = len(calib)
        self._n_kp = len(kps)

    # -- observations ----------------------------------------------------------------------
    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """(V, P, 2) distorted detector pixels -> the pinhole pixels the fit works in.

        Do this once per frame, not once per iteration: the observations do not change as q
        does, and `camera.undistort` is a Newton solve.
        """
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (self._n_view, self._n_kp, 2):
            raise ValueError(f"observations are {uv.shape}, expected "
                             f"{(self._n_view, self._n_kp, 2)} = (views, landmarks, 2)")
        return np.stack([c.undistort(uv[v]) for v, c in enumerate(self.calib)], axis=0)

    # -- the initial guess -----------------------------------------------------------------
    def init_root(self, obs: np.ndarray, conf: np.ndarray, *,
                  undistorted: bool = False) -> bool:
        """Place the root by rigidly aligning the current pose's landmarks onto triangulated ones.

        **What this actually buys, measured rather than assumed.** On `fit_selftest`'s
        four-camera ring, 21 landmarks, 20% occlusion and 2% outliers, with the initial root
        offset by 0 m, 1 m and 3 m:

            offset      landmark recovery        LM iterations
            any         25.6 mm  (identical)     29.4     with init_root
            0 m         25.6 mm                  37.9     without
            1 m         25.8 mm                  42.8     without
            3 m         25.7 mm                  49.5     without

        So it is worth having for two reasons, and NEITHER of them is accuracy:

        1. **The answer stops depending on where you started.** The three initialised rows are
           bit-identical because the triangulated placement discards `q0`'s root entirely. A
           fit whose result moves when the initial guess moves is not a measurement, and on
           real footage there is no ground truth to notice it against.
        2. **Cost grows with how wrong the guess is, and this flattens it** -- 30 iterations
           regardless, against 38-50 and rising.

        An earlier version of this docstring claimed LM could not escape a 180-degree yaw error
        without help. That was true, and it was a symptom of the missing `mj_comPos` in
        `residual` (see the note there), not of the geometry. With a correct Jacobian LM does
        walk out of a backwards start. The claim is left here as a warning: it was measured
        against a broken build and believed for exactly as long as nobody re-measured it.

        Returns False and leaves `qpos` alone when too few landmarks triangulated -- the caller
        then still has a defined starting pose, just not a good one.
        """
        mj, m, d = self._mj, self.m, self.d
        if m.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
            return False                                  # nothing to place
        obs = np.asarray(obs, float) if undistorted else self.undistort(obs)
        pts, ok = triangulate(self.calib, obs, np.asarray(conf, float),
                              conf_floor=self.CONF_FLOOR)
        # Kabsch on 3 points is exactly determined and therefore fits the noise; demand enough
        # that the alignment is actually over-constrained.
        if int(ok.sum()) < 5:
            return False

        mj.mj_kinematics(m, d)
        A = d.site_xpos[self.site_ids][ok]
        B = pts[ok]
        # One robustness pass. A single badly triangulated landmark -- one swapped left/right
        # label is enough -- shifts the centroid and tilts the rotation, and because Kabsch is
        # least-squares it spreads that error over every other point rather than isolating it.
        for _ in range(2):
            R, cA, cB = _kabsch(A, B)
            resid = np.linalg.norm((A - cA) @ R.T + cB - B, axis=1)
            keep = resid <= max(3.0 * np.median(resid), 1e-3)
            if keep.all() or keep.sum() < 5:
                break
            A, B = A[keep], B[keep]

        quat = np.empty(4)
        mj.mju_mat2Quat(quat, R.reshape(9))
        root_q = np.empty(4)
        mj.mju_mulQuat(root_q, quat, d.qpos[3:7].copy())
        d.qpos[0:3] = R @ (d.qpos[0:3] - cA) + cB
        d.qpos[3:7] = root_q
        return True

    # -- the objective ---------------------------------------------------------------------
    def residual(self, obs: np.ndarray, conf: np.ndarray, *, jac: bool = False):
        """Robustified, weighted reprojection residual at the CURRENT `data.qpos`.

        Returns `(r, J, err_px)` with `r` flat `(2*V*P,)` and `J` `(2*V*P, len(free_dofs))`
        (or None). `err_px` is `(V, P)` raw pixel distance, NaN where the observation was not
        used -- the caller reports on that, and it is how a mislabelled landmark is found.

        Rows with no usable observation are ZEROED rather than dropped, so the residual keeps a
        fixed shape across frames and iterations. That matters more than it looks: an LM whose
        residual vector changes length between the trial step and the accept test is comparing
        two different objectives, and it will accept steps that made things worse.
        """
        mj, m, d = self._mj, self.m, self.d
        mj.mj_kinematics(m, d)
        # `mj_comPos`, and it is NOT optional. `mj_kinematics` refreshes `site_xpos`, so the
        # residual alone looks perfectly correct without it -- but `mj_jacSite` is built from
        # `cdof` and `subtree_com`, which only `mj_comPos` writes. Skip it and every Jacobian is
        # the one for the PREVIOUS configuration: LM still descends (the residual is right, and
        # a stale Jacobian is often still a descent direction), just slowly and into the wrong
        # place, and it stalls a few iterations in with a large residual and no error anywhere.
        # Symptom, for the next person: a zero-noise self-test that converges to ~15 px.
        mj.mj_comPos(m, d)
        X = d.site_xpos[self.site_ids]                          # (P, 3) world
        V, P = self._n_view, self._n_kp
        cols = self.free_dofs

        r = np.zeros((V, P, 2))
        J = np.zeros((V, P, 2, cols.size)) if jac else None
        err = np.full((V, P), np.nan)

        # d(site)/dq is per landmark and shared across views, so it is computed once here and
        # reused for all four cameras. With 21 landmarks that is 21 `mj_jacSite` calls per
        # iteration instead of 84.
        jacs = None
        if jac:
            jacs = np.empty((P, 3, self.nv))
            for p, sid in enumerate(self.site_ids):
                mj.mj_jacSite(m, d, self._jacp, None, int(sid))
                jacs[p] = self._jacp

        for v, cam in enumerate(self.calib):
            uv = cam.project_linear(X)                          # (P, 2)
            good = (conf[v] >= self.CONF_FLOOR) & np.isfinite(uv).all(axis=1) \
                & np.isfinite(obs[v]).all(axis=1)
            if not np.any(good):
                continue
            diff = uv - obs[v]                                  # (P, 2) pixels
            e = np.hypot(diff[:, 0], diff[:, 1])
            err[v, good] = e[good]
            # Huber as an IRLS weight, in the sqrt (residual) form the LM solver wants:
            # cost = sum(w^2 * |diff|^2) reproduces the Huber loss when w = sqrt(delta/e).
            w = np.where(e > self.huber, np.sqrt(self.huber / np.maximum(e, 1e-9)), 1.0)
            w = np.where(good, w * conf[v] / self.sigma, 0.0)    # (P,)
            r[v] = diff * w[:, None]
            if jac:
                # (P,2,3) @ (P,3,nv) -> (P,2,nv), then weight and select the free columns.
                J[v] = (cam.d_uv_d_X(X) @ jacs)[:, :, cols] * w[:, None, None]

        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        if jac:
            J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).reshape(r.size, cols.size)
        return r, J, err

    # -- the solve -------------------------------------------------------------------------
    def fit(self, obs: np.ndarray, conf: np.ndarray, *, q0: np.ndarray | None = None,
            iters: int = 60, tol: float = 1e-5, lam0: float = 1e-3,
            undistorted: bool = False, init_root: bool = True) -> FitResult:
        """Levenberg-Marquardt on `qpos`, starting from `q0` (default: whatever `data` holds).

        `undistorted=True` says the caller has already run `undistort` -- which a sequence fit
        should, once for the whole clip. `init_root=False` skips the triangulated placement,
        which is what a warm-started frame in the middle of a clip wants: the previous frame is
        7 ms away and is a far better guess than triangulation, and re-placing the root from
        noisy free points every frame would inject exactly the jitter E_temp then has to remove.
        """
        mj, m, d = self._mj, self.m, self.d
        if q0 is not None:
            d.qpos[:] = q0
        obs = np.asarray(obs, float) if undistorted else self.undistort(obs)
        conf = np.asarray(conf, float)
        if conf.shape != (self._n_view, self._n_kp):
            raise ValueError(f"confidences are {conf.shape}, expected "
                             f"{(self._n_view, self._n_kp)}")
        if init_root:
            self.init_root(obs, conf, undistorted=True)

        lam = lam0
        r, J, err = self.residual(obs, conf, jac=True)
        cost = float(r @ r)
        n_obs = int(np.isfinite(err).sum())
        if n_obs < 6:
            # Six DOFs of root alone need three points in two views; below that the normal
            # equations are singular and LM would return the initial guess wearing a
            # convergence flag.
            return FitResult(d.qpos.copy(), float("nan"), float("nan"), 0, False, n_obs, cost)

        step = np.zeros(self.nv)
        qbest = d.qpos.copy()
        # `lam0 * mean(diag)` would be scale-dependent; Marquardt scaling below handles it.
        it = 0
        converged = False
        for it in range(1, iters + 1):
            A = J.T @ J
            g = J.T @ r
            # Marquardt's scaling (lambda * diag(A), not lambda * I) so the damping is in the
            # units of each DOF. A shared identity damping penalises the root translation, in
            # metres, and a toe joint, in radians, by the same amount -- which in practice
            # freezes the root and lets the toes absorb everything.
            diag = np.maximum(np.diag(A), 1e-12)
            for _ in range(12):
                try:
                    delta = np.linalg.solve(A + lam * np.diag(diag), -g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                step[:] = 0.0
                step[self.free_dofs] = delta
                d.qpos[:] = qbest
                mj.mj_integratePos(m, d.qpos, step, 1.0)
                self._clamp_to_limits()
                r_try, _, _ = self.residual(obs, conf, jac=False)
                cost_try = float(r_try @ r_try)
                if cost_try < cost:
                    break
                lam *= 10.0
            else:
                # No damping in twelve decades of lambda buys an improvement. That is what a
                # stationary point looks like from inside LM, so it is CONVERGENCE, not
                # failure -- and it is the branch a clean fit exits through, because a residual
                # already at machine zero cannot be improved either. Reporting it as
                # `converged=False` would make a perfect fit indistinguishable from a stall;
                # whether the stationary point is any good is what `rms_px` is for.
                d.qpos[:] = qbest
                converged = True
                break

            rel = (cost - cost_try) / max(cost, 1e-30)
            qbest = d.qpos.copy()
            cost = cost_try
            lam = max(lam * 0.3, 1e-12)
            r, J, err = self.residual(obs, conf, jac=True)
            if rel < tol:
                converged = True
                break
        # Falling out of the loop means the iteration cap was hit with the cost still moving.
        # That one genuinely is not convergence.

        d.qpos[:] = qbest
        _, _, err = self.residual(obs, conf, jac=False)
        used = err[np.isfinite(err)]
        with np.errstate(invalid="ignore"):
            # A landmark occluded in EVERY view is an all-NaN column, and its mean is honestly
            # NaN; numpy is right to warn and the caller is expected to use nan-aware reducers.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_kp = (np.nanmean(err, axis=0) if used.size
                          else np.full(self._n_kp, np.nan))
        return FitResult(qbest.copy(),
                         float(np.sqrt(np.mean(used ** 2))) if used.size else float("nan"),
                         float(used.max()) if used.size else float("nan"),
                         it, converged, int(used.size), cost, per_kp)

    def _clamp_to_limits(self) -> None:
        """Project `qpos` back inside the authored joint ranges after every trial step.

        `notes/pipeline.md` lists E_lim as a log-barrier at weight 0.01, and that is right for
        the *anatomy* search, where θ must stay legal for the barrier's whole trajectory. For
        the pose solve a projection is both cheaper and better behaved: a barrier's gradient
        blows up exactly where LM most needs a large step (a limb pinned against its limit by a
        bad initial guess), whereas clamping just stops. The limits are the rig's authored
        ranges, so a clamped fit is still a pose the body can hold.
        """
        m, d = self.m, self.d
        lim = self._limited
        if lim is None:
            return
        adr, lo, hi = lim
        d.qpos[adr] = np.clip(d.qpos[adr], lo, hi)

    @property
    def _limited(self):
        if not hasattr(self, "_limited_cache"):
            m = self.m
            idx = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)]
            if idx:
                adr = np.array([m.jnt_qposadr[j] for j in idx], dtype=int)
                rng = m.jnt_range[idx]
                self._limited_cache = (adr, rng[:, 0].copy(), rng[:, 1].copy())
            else:
                self._limited_cache = None
        return self._limited_cache


def fit_sequence(fitter: PoseFitter, obs: np.ndarray, conf: np.ndarray, *,
                 q0: np.ndarray | None = None, iters: int = 60,
                 first_iters: int = 300) -> tuple[np.ndarray, list[FitResult]]:
    """Fit a whole clip, warm-starting each frame from the previous one.

    `obs` is `(T, V, P, 2)` distorted pixels, `conf` is `(T, V, P)`. At 140 fps consecutive
    frames are ~7 ms apart, so the previous solution is a very good initial guess.

    **How much the warm start is worth is currently an OPEN QUESTION -- see known-issues #7.**
    The obvious assumption is "warm start = fewer iterations", and that is measurably false
    here: `init_root` already makes a cold start cheap and start-independent, so a warm-started
    frame costs about the same, sometimes slightly more. The intended benefit is continuity --
    a limb occluded for ten frames is nearly unconstrained, and re-solving it from scratch
    every frame lets it wander inside its null space, which is indistinguishable from motion to
    everything downstream (it becomes acceleration in the AMP demonstrations and torque in the
    E_phys gate). But measured on synthetic clips, including one with five landmarks hidden in
    three of four views for the whole clip, the smoothing is only 3-4%: 0.0086 vs 0.0090 on the
    root, 0.2615 vs 0.2708 on the joints. That is too small to build on.

    The structure is kept because it is the right shape regardless -- E_temp and the E_phys
    gate operate on a trajectory, not on frames -- but do not cite the warm start as the reason
    a clip fit is affordable until it has been measured against real footage, where occlusion
    is longer, correlated between views, and not drawn from a uniform.

    The FIRST frame gets `first_iters` because it is the only one with no warm start, and a
    clip whose first frame lands in a local minimum drags that minimum through every frame
    after it.

    Note what this deliberately does NOT do: no temporal smoothing (E_temp) and no gating. Both
    are stage-2 concerns that operate on the whole trajectory at once, and folding them in here
    would make each frame's answer depend on frames that have not been fit yet.
    """
    obs = np.asarray(obs, float)
    T = obs.shape[0]
    und = np.stack([fitter.undistort(obs[t]) for t in range(T)], axis=0)
    out = np.empty((T, fitter.m.nq))
    results: list[FitResult] = []
    q, warm = q0, False
    for t in range(T):
        res = fitter.fit(und[t], conf[t], q0=q, iters=first_iters if not warm else iters,
                         undistorted=True, init_root=not warm)
        out[t] = res.qpos
        results.append(res)
        # Warm-start from the last GOOD frame, not simply the last one: a frame the fit failed
        # on has a qpos that is wherever LM gave up, and handing that forward turns one bad
        # frame into a bad tail.
        if np.isfinite(res.rms_px):
            q, warm = res.qpos, True
    return out, results
