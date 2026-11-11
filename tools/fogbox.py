"""The scene and harness GPU mode J's small regression checks share (not a check itself).

An open-fronted unit box -- floor, ceiling, back and two side walls, grey -- with a 0.3 m square
area light just under the ceiling, a medium given by the caller, and cameras looking in through
the open side from z = 2.7. check_wf_resume, check_gpu_mixed_modes and check_jgpu_one_beam build
theirs from it. Also an HDR image reader and the comparison the wavefront checks hold their
images to.
"""
import os

import numpy as np

NL = chr(10)

WALLS = [
    "scene { units meters }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}"]

# Heterogeneous: its transmittance estimates draw random numbers, which some checks need.
FOG = NL.join([
    "medium {",
    "    sigma_a 0.3   sigma_s 1.2   g 0.3",
    "    bounds { min 0 0 0   max 1 1 1 }",
    '    density "0.4 + 0.6*saturate(1.2 - 2.0*sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2))"',
    "}"])


def camera(name="cam", mode=None, res=48):
    """A camera looking into the box through its open side; `mode` its own render mode, if any."""
    return NL.join(['camera "%s" {' % name] + (["    mode %s" % mode] if mode else []) + [
        "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
        "    film { res %d %d }" % (res, res),
        "}"])


def scene(medium=FOG, cameras=None):
    """The box with `medium` (an FTSL medium block) and `cameras` (camera blocks; one by default)."""
    return NL.join(WALLS + [medium] + (cameras or [camera()])) + NL


def write(directory, name, text):
    """Write a scene into `directory` (made if missing); returns its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


class Checks:
    """Named pass/fail lines, printed as they come, and the verdict at the end."""

    def __init__(self, width=56):
        self.width, self.fails = width, []

    def __call__(self, name, ok, detail=""):
        print("%-*s %s  %s" % (self.width, name, "ok  " if ok else "FAIL", detail), flush=True)
        if not ok:
            self.fails.append(name)

    def verdict(self, log_tail=""):
        """Print PASS, or the failures (after `log_tail`, if any); returns the exit code."""
        if self.fails:
            if log_tail:
                print(log_tail)
            print("FAILED: " + ", ".join(self.fails))
            return 1
        print("PASS")
        return 0


# Two images that differ only in the order of the film's atomic sums agree to about float32's
# 1.2e-7 of a pixel: a sum in another order can round the other way.
ORDER_TOL = 1e-5


def load_pfm(path):
    """A PFM (ftrace's -hdr output) as an (h, w, channels) float64 array: 3 channels for `PF`,
    1 for `Pf`. Rows stay in the file's order (PFM's bottom-up)."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError("%s: not a PFM (%r)" % (path, kind))
        nc = 3 if kind == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        s = float(f.readline())
        data = np.frombuffer(f.read(w * h * nc * 4), dtype="<f4" if s < 0 else ">f4")
        return data.reshape(h, w, nc).astype(np.float64)


def image_diff(x, y):
    """The largest difference between two images as a share of the pixel, and how many values
    differ. A NaN in either makes the first NaN, which fails any `<` test."""
    rel = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-12)
    return float(rel.max()), int((x != y).sum())
