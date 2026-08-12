"""`session/calib.json`: the camera models the fit projects through.

This is the second of `notes/pipeline.md`'s four frozen formats, and the one with the least
margin for error. A calibration that is *wrong* is not detectable from the footage afterwards
-- every later stage, including the E_phys floor gate, reads this file and believes it. So the
schema carries its own verification numbers (`rms_px`, and the closing re-shoot's disagreement
with the opening one) rather than leaving them in a notebook that is thrown away.

**The model is `cv::fisheye`, and keypoints are undistorted -- never frames.** `notes/capture.md`
commits to both. The consequence for this module is that there are two directions and they are
used by different callers:

* `Camera.project(X)` is the **full nonlinear** world -> distorted-pixel map. Only two things
  need it: `fit_selftest.py`, which manufactures synthetic observations, and any overlay that
  draws the model back onto a raw frame.
* `Camera.undistort(uv)` -> `Camera.project_linear(X)` is what the **fit** runs on. Undistorting
  the ~21 observed points once per frame costs nothing and leaves the objective a clean pinhole,
  so the Jacobian is three lines of algebra instead of the derivative of an 8th-order
  polynomial. Getting that derivative subtly wrong is a bug that looks exactly like a bad
  calibration, and this side-steps it entirely.

That split is also why `fit_selftest.py` is a real test of this file and not a tautology: it
generates through `project` and fits through `undistort`, so any disagreement between the
distortion model and its inverse shows up as recovery error rather than cancelling out.

Conventions, stated once because every sign error in multi-view geometry comes from leaving
them implicit:

* World is the **simulator's** frame: metres, **z up**, the floor at z = 0. (FTSL's y-up
  conversion belongs to the emitter, not here.)
* `R`, `t` are **world -> camera**: `X_cam = R @ X_world + t`. The camera looks down its own
  +z; `centre = -R.T @ t`.
* Pixels are (u, v) with v **down**, origin at the top-left corner of the top-left pixel, so a
  pixel centre is at a half-integer. That is OpenCV's convention and therefore the detector's.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "creature/calib@1"

#: Newton on the fisheye's scalar angle map. `cv::fisheye::undistortPoints` uses a fixed 10
#: iterations of a non-contracting fixed-point scheme that quietly diverges past ~120 deg of
#: field; a HERO 12 at native wide is well inside that, but the whole point of undistorting
#: keypoints rather than frames is that we can afford to be exact here. Newton on a monotone
#: scalar converges in 3-4 for any angle the lens can actually see.
_UNDISTORT_ITERS = 12
_UNDISTORT_TOL = 1e-12


class CalibError(ValueError):
    """A calibration that cannot be trusted. Never downgraded to a warning: see the header."""


@dataclass(frozen=True)
class Camera:
    """One calibrated view. Immutable, because half the pipeline holds a reference to it."""
    name: str
    K: np.ndarray                 # (3, 3) intrinsics
    D: np.ndarray                 # (4,) cv::fisheye k1..k4, or all-zero for a pinhole
    R: np.ndarray                 # (3, 3) world -> camera rotation
    t: np.ndarray                 # (3,)   world -> camera translation
    size: tuple[int, int]         # (width, height) in pixels
    model: str = "fisheye"
    rms_px: float = float("nan")  # this camera's own calibration residual

    # --- derived, cached once -------------------------------------------------------------
    @property
    def fx(self) -> float: return float(self.K[0, 0])

    @property
    def fy(self) -> float: return float(self.K[1, 1])

    @property
    def cx(self) -> float: return float(self.K[0, 2])

    @property
    def cy(self) -> float: return float(self.K[1, 2])

    @property
    def center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t

    @property
    def is_fisheye(self) -> bool:
        """`model` decides, not `D`. A fisheye with all-zero coefficients is still an
        *equidistant* projection (r = f*theta), which is not a pinhole and differs from one by
        tens of pixels at the edge of a 118-degree field -- so keying the branch off `any(D)`
        would turn a perfectly legal ideal-fisheye calibration into a silent pinhole."""
        return self.model == "fisheye"

    # --- world -> camera ------------------------------------------------------------------
    def to_camera(self, X: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 3) camera points."""
        X = np.asarray(X, dtype=float)
        return X @ self.R.T + self.t

    def project_linear(self, X: np.ndarray) -> np.ndarray:
        """Pinhole projection, ignoring distortion. This is what the FIT uses.

        Points at or behind the pupil have no projection at all; they come back as NaN rather
        than as a plausible pixel on the wrong side of the image, because a keypoint that
        silently reflects through the centre of projection is a residual the optimiser will
        happily chase.
        """
        Xc = self.to_camera(X)
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = Xc[..., 0] / z, Xc[..., 1] / z
        uv = np.stack([self.fx * a + self.cx, self.fy * b + self.cy], axis=-1)
        return np.where((z > 1e-6)[..., None], uv, np.nan)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Full `cv::fisheye` projection: world -> DISTORTED pixels. Synthetic views only."""
        if not self.is_fisheye:
            return self.project_linear(X)
        Xc = self.to_camera(X)
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = Xc[..., 0] / z, Xc[..., 1] / z
            r = np.hypot(a, b)
            theta = np.arctan(r)
            scale = np.where(r > 1e-12, self._theta_d(theta) / np.where(r > 1e-12, r, 1.0), 1.0)
            uv = np.stack([self.fx * scale * a + self.cx, self.fy * scale * b + self.cy],
                          axis=-1)
        return np.where((z > 1e-6)[..., None], uv, np.nan)

    def _theta_d(self, theta: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4 = self.D
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _dtheta_d(self, theta: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4 = self.D
        t2 = theta * theta
        return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """DISTORTED pixels -> the pixels an ideal pinhole with the same K would have seen.

        The fit's observations pass through here exactly once, at load. Newton on the scalar
        angle map -- see `_UNDISTORT_ITERS`. NaNs (an unlabelled keypoint) pass through
        untouched.
        """
        uv = np.asarray(uv, dtype=float)
        if not self.is_fisheye:
            return uv.copy()
        a = (uv[..., 0] - self.cx) / self.fx
        b = (uv[..., 1] - self.cy) / self.fy
        theta_d = np.hypot(a, b)
        theta = theta_d.copy()                       # exact when D == 0
        if np.any(self.D):
            for _ in range(_UNDISTORT_ITERS):
                err = self._theta_d(theta) - theta_d
                if np.all(np.abs(err[np.isfinite(err)]) < _UNDISTORT_TOL):
                    break
                theta = theta - err / np.maximum(self._dtheta_d(theta), 1e-9)
        # theta is the true angle from the axis; the undistorted normalised radius is tan(theta).
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(theta_d > 1e-12, np.tan(theta) / np.where(theta_d > 1e-12,
                                                                      theta_d, 1.0), 1.0)
        return np.stack([self.fx * scale * a + self.cx, self.fy * scale * b + self.cy], axis=-1)

    def d_uv_d_X(self, X: np.ndarray) -> np.ndarray:
        """d(pinhole pixel) / d(world point): (..., 2, 3). The fit's chain rule, first link.

        Pairs with `project_linear`, NOT with `project` -- see the module header for why the
        fit never differentiates the distortion.
        """
        Xc = self.to_camera(X)
        x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
        iz = 1.0 / z
        # d(u,v)/d(Xc)
        J = np.zeros(Xc.shape[:-1] + (2, 3), dtype=float)
        J[..., 0, 0] = self.fx * iz
        J[..., 0, 2] = -self.fx * x * iz * iz
        J[..., 1, 1] = self.fy * iz
        J[..., 1, 2] = -self.fy * y * iz * iz
        return J @ self.R                       # ... and d(Xc)/d(Xw) = R


@dataclass
class Calibration:
    """The whole `calib.json`: the cameras, the floor, and the numbers that say to trust it."""
    cameras: list[Camera]
    floor_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    floor_offset: float = 0.0
    rms_px: float = float("nan")
    #: Disagreement between the opening and closing calibration shoots, in pixels. capture.md
    #: makes the closing re-shoot mandatory precisely because drift over a two-hour session is
    #: invisible otherwise; recording the number here is what lets a later stage refuse.
    drift_px: float = float("nan")
    session: str = ""
    path: str = ""

    def __len__(self) -> int: return len(self.cameras)

    def __iter__(self): return iter(self.cameras)

    def __getitem__(self, i): return self.cameras[i]

    @property
    def names(self) -> list[str]: return [c.name for c in self.cameras]

    def project(self, X: np.ndarray) -> np.ndarray:
        """(P, 3) world points -> (V, P, 2) distorted pixels, one plane per view."""
        return np.stack([c.project(X) for c in self.cameras], axis=0)


def _mat3(v, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.shape == (9,):
        a = a.reshape(3, 3)
    if a.shape != (3, 3):
        raise CalibError(f"{what}: expected a 3x3 matrix, got shape {a.shape}")
    return a


def _check_rotation(R: np.ndarray, what: str) -> np.ndarray:
    """A near-rotation is silently accepted by every downstream matmul and by nothing else.

    An extrinsic that has picked up a scale (a bundle adjustment that was not constrained, a
    hand-edited matrix) makes the animal come out uniformly the wrong size, which the morph
    fit then absorbs into `body_scale`. The result is a plausible dog of the wrong height and
    no error anywhere, so this is checked on load rather than trusted.
    """
    err = np.abs(R @ R.T - np.eye(3)).max()
    if err > 1e-6:
        raise CalibError(f"{what}: R is not a rotation (|RR^T - I| = {err:.2e}); a scaled or "
                         f"skewed extrinsic silently rescales the whole animal")
    if np.linalg.det(R) < 0:
        raise CalibError(f"{what}: R is a reflection (det < 0); the world is mirrored")
    return R


def load_calib(path: str) -> Calibration:
    """Read and validate a `calib.json`. Every problem raises -- see the module header."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    schema = doc.get("schema")
    if schema != SCHEMA:
        raise CalibError(f"{path}: schema is {schema!r}, this build reads {SCHEMA!r}")
    raw = doc.get("cameras")
    if not isinstance(raw, list) or not raw:
        raise CalibError(f"{path}: no cameras")

    cams, seen = [], set()
    for i, c in enumerate(raw):
        name = str(c.get("name", f"cam{i}"))
        if name in seen:
            raise CalibError(f"{path}: two cameras are both called {name!r}")
        seen.add(name)
        model = c.get("model", "fisheye")
        if model not in ("fisheye", "pinhole"):
            raise CalibError(f"{path}: {name}: unknown camera model {model!r}")
        D = np.asarray(c.get("D", [0.0] * 4), dtype=float).ravel()
        if D.size != 4:
            raise CalibError(f"{path}: {name}: D has {D.size} coefficients, cv::fisheye has 4")
        if model == "pinhole" and np.any(D):
            raise CalibError(f"{path}: {name}: model is 'pinhole' but D is non-zero")
        size = c.get("size")
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise CalibError(f"{path}: {name}: size must be [width, height]")
        K = _mat3(c["K"], f"{path}: {name}: K")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CalibError(f"{path}: {name}: non-positive focal length")
        R = _check_rotation(_mat3(c["R"], f"{path}: {name}: R"), f"{path}: {name}")
        t = np.asarray(c["t"], dtype=float).ravel()
        if t.size != 3:
            raise CalibError(f"{path}: {name}: t has {t.size} components, expected 3")
        cams.append(Camera(name=name, K=K, D=D, R=R, t=t,
                           size=(int(size[0]), int(size[1])), model=model,
                           rms_px=float(c.get("rms_px", float("nan")))))

    if len(cams) < 2:
        raise CalibError(f"{path}: {len(cams)} camera(s); a 3D pose is not observable from one "
                         f"view and the fit would run happily and return nonsense")
    floor = doc.get("floor", {})
    n = np.asarray(floor.get("normal", [0.0, 0.0, 1.0]), dtype=float).ravel()
    if n.size != 3 or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9:
        raise CalibError(f"{path}: floor.normal is not a usable direction")
    return Calibration(cameras=cams, floor_normal=n / np.linalg.norm(n),
                       floor_offset=float(floor.get("offset", 0.0)),
                       rms_px=float(doc.get("rms_px", float("nan"))),
                       drift_px=float(doc.get("drift_px", float("nan"))),
                       session=str(doc.get("session", "")), path=path)


def save_calib(path: str, calib: Calibration, *, session: str | None = None) -> None:
    doc = {
        "schema": SCHEMA,
        "session": session if session is not None else calib.session,
        "rms_px": calib.rms_px,
        "drift_px": calib.drift_px,
        "floor": {"normal": [float(x) for x in calib.floor_normal],
                  "offset": float(calib.floor_offset)},
        "cameras": [{"name": c.name, "model": c.model,
                     "size": [int(c.size[0]), int(c.size[1])],
                     "K": [[float(x) for x in row] for row in c.K],
                     "D": [float(x) for x in c.D],
                     "R": [[float(x) for x in row] for row in c.R],
                     "t": [float(x) for x in c.t],
                     "rms_px": float(c.rms_px)} for c in calib.cameras],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Build a world->camera (R, t) for a camera at `eye` pointing at `target`.

    OpenCV convention: camera +z forward, +x right, +y DOWN. The last part is the one that is
    easy to get wrong and produces a picture that is merely upside down -- which looks like a
    plausible camera mounting rather than like a bug.
    """
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    nr = np.linalg.norm(r)
    if nr < 1e-9:
        raise CalibError("look_at: the view direction is parallel to `up`")
    r /= nr
    d = np.cross(f, r)                       # +y down = f x right
    R = np.stack([r, d, f], axis=0)          # rows are the camera axes in world coords
    return R, -R @ eye


def synthetic_rig(n: int = 4, *, radius: float = 4.0, height: float = 1.6,
                  target=(0.0, 0.0, 0.45), size=(1920, 1080), hfov_deg: float = 118.0,
                  D=(-0.02, 0.004, -0.001, 0.0002), seed: int = 0) -> Calibration:
    """The four-GoPro ring of `notes/capture.md`, as a `Calibration` with no session behind it.

    This exists so `tools/fit_selftest.py` can run today, with no footage and no calibration
    shoot -- which is the entire argument for building the self-test first. The defaults are
    the decided rig: four HERO 12s at 1920x1080, native wide (~118 deg horizontal), ringed
    around a capture volume at roughly chest height, with a mild barrel distortion so the
    undistortion path is genuinely exercised rather than reducing to the identity.

    Cameras are placed on an arc rather than a full circle by default only in the sense that
    `n` evenly spaced views on a circle is what four cameras around a run gives you; the
    asymmetric `seed` jitter breaks the perfect symmetry that would otherwise make the
    self-test's conditioning unrepresentatively good.
    """
    rng = np.random.default_rng(seed)
    w, h = int(size[0]), int(size[1])
    # For an equidistant fisheye, r = f*theta, so the half-field maps to half the width.
    f = (w / 2.0) / math.radians(hfov_deg / 2.0)
    cams = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n + float(rng.normal(0.0, 0.05))
        eye = np.array([radius * math.cos(ang), radius * math.sin(ang),
                        height + float(rng.normal(0.0, 0.08))])
        R, t = look_at(eye, np.asarray(target, float) + rng.normal(0.0, 0.03, 3))
        K = np.array([[f, 0.0, w / 2.0 - 0.5], [0.0, f, h / 2.0 - 0.5], [0.0, 0.0, 1.0]])
        cams.append(Camera(name=f"cam{i}", K=K, D=np.asarray(D, float), R=R, t=t,
                           size=(w, h), model="fisheye", rms_px=0.0))
    return Calibration(cameras=cams, rms_px=0.0, drift_px=0.0, session="synthetic")
