"""Regression check: the guide pass's depth, position and motion vectors (ROADMAP 11 stage C).

    D:/python314/python.exe -u tools/check_aovs.py [--out <dir>] [--exe ftrace.exe]

  * `-write-aovs` writes <out>.albedo, .normal, .depth, .position, .prevposition, .motion and
    .motionfwd.pfm, without -oidn.
  * Depth and position agree with each other and with the camera: every pixel that meets a
    surface has a position that projects back onto its own centre, and a depth that is that
    position's distance along the camera's axis.
  * A ball sliding 0.4 m a frame across a still box, seen by a still camera: at the ball's
    pixels the vector a frame back is the projection of the hit point carried 0.4 m back, minus
    the pixel's centre, and the one a frame forward the same 0.4 m ahead, and its prevposition is
    the hit point 0.4 m back; a floor pixel far from the ball holds still (0, and where it sits).
  * The middle frame of a camera path sees the still box move with the neighbouring frames'
    cameras: each pixel's backward vector is its world position projected by the previous frame's
    camera, minus the pixel's centre (and forward, by the next).
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_motion import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
W, H = 80, 60
EYE, LOOK, UP, FOV = (0.0, 1.0, 4.2), (0.0, 0.5, 0.0), (0.0, 1.0, 0.0), 40.0


def box(extra, camera):
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "gold"  { type diffuse reflect rgb 0.8 0.6 0.2 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
        extra,
        camera,
    ]) + NL


def still_camera(eye=EYE):
    return ('camera "c" { eye %r %r %r  look_at %r %r %r  up 0 1 0  fov_y %r  film { res %d %d } }'
            % (eye + LOOK + (FOV, W, H)))


class Cam:
    """ftrace's rectilinear camera (Camera::lookAt, projectF)."""
    def __init__(self, eye, look=LOOK, up=UP, fov=FOV):
        self.eye = np.array(eye, float)
        w = np.array(look, float) - self.eye
        self.w = w / np.linalg.norm(w)
        u = np.cross(self.w, np.array(up, float))
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.u, self.w)
        self.ty = np.tan(np.radians(fov) / 2)
        self.tx = self.ty * W / H

    def project(self, p):
        d = np.asarray(p, float) - self.eye
        cz = d @ self.w
        ix = (d @ self.u / cz) / self.tx
        iy = (d @ self.v / cz) / self.ty
        return np.stack([(ix * 0.5 + 0.5) * W, (iy * 0.5 + 0.5) * H], -1)


def render(exe, path, png, extra=()):
    for suf in (".pfm", ".albedo.pfm", ".normal.pfm", ".depth.pfm", ".position.pfm", ".prevposition.pfm",
                ".motion.pfm", ".motionfwd.pfm"):
        f = png[:-4] + suf
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", path, "-mode", "R", "-device", "cpu", "-spp", "1", "-no-meter", "-write-aovs",
           "-o", png] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    stem = png[:-4]
    # read_pfm gives rows top first; the film's rows count from the bottom, so flip back.
    aov = {k: read_pfm(stem + "." + k + ".pfm")[::-1]
           for k in ("depth", "position", "prevposition", "motion", "motionfwd")}
    return aov, all(os.path.exists(stem + s) for s in (".albedo.pfm", ".normal.pfm"))


def centres():
    ys, xs = np.mgrid[0:H, 0:W]
    return np.stack([xs + 0.5, ys + 0.5], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_aovs"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-62s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""))

    # --- a ball sliding 0.4 m a frame, a still camera ---
    ball = ("group { motion { key { translate -0.2 0.5 0 } key { translate 0.2 0.5 0 } }  "
            "sphere { center 0 0 0  radius 0.3  material gold } }")
    p = os.path.join(out, "slide.ftsl")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(box(ball, still_camera()))
    aov, guides = render(exe, p, os.path.join(out, "slide.png"))
    verdict("-write-aovs writes the seven images without -oidn", guides)
    cam = Cam(EYE)
    depth, pos = aov["depth"][..., 0], aov["position"]
    hit = depth > 0
    back = cam.project(pos[hit])
    err = np.abs(back - centres()[hit]).max()
    verdict("each position projects back onto its own pixel", err < 1e-3, "worst %.2g px" % err)
    derr = np.abs(((pos[hit] - cam.eye) @ cam.w) - depth[hit]).max()
    verdict("... at the depth along the camera's axis", derr < 1e-4, "worst %.2g m" % derr)

    c = cam.project([0.0, 0.5, 0.0]).astype(int)                   # the ball's centre at the instant
    on = [(c[1] + dy, c[0] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    worst = 0.0
    for (y, x) in on:
        p0 = pos[y, x]
        want_b = cam.project(p0 - np.array([0.4, 0, 0])) - np.array([x + 0.5, y + 0.5])
        want_f = cam.project(p0 + np.array([0.4, 0, 0])) - np.array([x + 0.5, y + 0.5])
        worst = max(worst, np.abs(aov["motion"][y, x, :2] - want_b).max(),
                    np.abs(aov["motionfwd"][y, x, :2] - want_f).max())
    verdict("the ball's vectors a frame back and forward", worst < 1e-3, "worst %.2g px" % worst)
    pe = max(np.abs(aov["prevposition"][y, x] - (pos[y, x] - np.array([0.4, 0, 0]))).max() for (y, x) in on)
    verdict("... and where it sat a frame earlier", pe < 1e-6, "worst %.2g m" % pe)
    fl = (3, 3)                                                    # floor, bottom left, far from it
    still = np.abs(aov["motion"][fl][:2]).max() + np.abs(aov["motionfwd"][fl][:2]).max()
    still += np.abs(aov["prevposition"][fl] - pos[fl]).max()
    verdict("... and the still floor's are zero (it sat where it sits)", still < 1e-6, "%.2g" % still)

    # --- a camera path's middle frame ---
    path = ('camera_path "fly" { look_at %r %r %r  up 0 1 0  fov_y %r  frames 3  film { res %d %d }'
            % (LOOK + (FOV, W, H)) + NL + "key 0 -0.3 1.0 4.2" + NL + "key 1 0.3 1.0 4.2" + NL + "}")
    p = os.path.join(out, "path.ftsl")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(box("", path))
    aov, _ = render(exe, p, os.path.join(out, "path.png"), ["-camera", "fly1"])
    hit = aov["depth"][..., 0] > 0
    pos = aov["position"][hit]
    cen = centres()[hit]
    eb = np.abs(Cam((-0.3, 1.0, 4.2)).project(pos) - cen - aov["motion"][hit][:, :2]).max()
    ef = np.abs(Cam((0.3, 1.0, 4.2)).project(pos) - cen - aov["motionfwd"][hit][:, :2]).max()
    verdict("a path frame's vectors follow its neighbours' cameras", max(eb, ef) < 1e-3,
            "worst %.2g px back, %.2g px forward" % (eb, ef))

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
