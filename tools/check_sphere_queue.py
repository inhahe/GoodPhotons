"""Regression check: GPU mode B's queued glass-sphere connections render what the inline ones render.

    python tools/check_sphere_queue.py [--out png/check_sphere_queue] [--exe ftrace.exe]

GPU-SPHERE-CONNECT-SLOW. launchForward queues each photon vertex's connections through a glass sphere
(DSphReq: the sphere, the vertex, its splat records with their hit, its wavelengths, its stream) and
kSphConnect runs them after each wave of photons, a warp's lanes together, where inline each ran in the
photon's own thread (FTRACE_NOSPHQ=1). Under per-photon seeding (FTRACE_GPU_PHOTONSEED=1) and with no
medium, whose transmittance is the only draw a connection takes, the two trace the same photons and make
the same connections, and only the order of the film's sums differs: images within ORDER_TOL. Each case,
GPU mode B, the queue's image against the inline one's:

  - a glass ball in a lit room, seen from outside (check_ball_images' room);
  - the eye inside the ball, off its centre (its chains past the critical angle);
  - the ball over a glossy floor, a vertex's glossy splat record copied into the request;
  - the ball over a coated floor (a clear coat over a diffuse body), its coat record copied;
  - the first case again with the queue shrunk to 2000 requests (FTRACE_SPHQCAP): many waves, and
    requests past the queue connected inline (FTRACE_WAVE_DEBUG's line a wave says both happened).

Each queued run must show the queue used (its wave lines). Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fogbox  # noqa: E402
from pfm import ORDER_TOL, image_diff, read_pfm  # noqa: E402
from check_ball_images import BALL, INSIDE_OFF, OUTSIDE, ROOM  # noqa: E402

WAVE = re.compile(r"\[wave\] B sphere connections: wave (\d+), (\d+) photons, (\d+) queued, (\d+) inline")
PHOTONS = "2000000"


def floor_of(material_lines, name):
    """ROOM with its floor in `name`, the materials `material_lines` added."""
    lines = ROOM.split(NL)
    floor = "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }"
    assert floor in lines
    lines = [ln.replace("material white", "material " + name) if ln == floor else ln for ln in lines]
    return NL.join(lines[:1] + material_lines + lines[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_sphere_queue"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    check = fogbox.Checks(62)
    logs = []

    def scene(name, *parts):
        path = os.path.join(d, name + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(parts) + NL)
        return path

    glossy = floor_of(['material "gfloor" { type glossy reflect 0.6 roughness 0.3 }'], "gfloor")
    coated = floor_of(['material "body" { type diffuse reflect 0.7 }',
                       'material "cfloor" { type layered  coat { reflectance fresnel  ior 1.5  roughness 0 }'
                       '  layer "body" 1.0 }'], "cfloor")
    cases = [
        ("outside", scene("outside", ROOM, BALL, OUTSIDE), {}),
        ("inside", scene("inside", ROOM, BALL, INSIDE_OFF), {}),
        ("glossy", scene("glossy", glossy, BALL, OUTSIDE), {}),
        ("coated", scene("coated", coated, BALL, OUTSIDE), {}),
        ("waves", os.path.join(d, "outside.ftsl"), {"FTRACE_SPHQCAP": "2000"}),
    ]

    def run(tag, path, env):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", path, "-mode", "B", "-device", "gpu", "-n", PHOTONS, "-max-bounce", "64",
               "-seed", "3", "-hdr", "-no-meter", "-window-min", "-o", png]
        full = dict(os.environ, FTRACE_GPU_PHOTONSEED="1", FTRACE_WAVE_DEBUG="1", **env)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=full)
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (read_pfm(pfm, top_down=False) if os.path.exists(pfm) else None)

    for name, path, env in cases:
        rc0, log0, inline = run(name + "_inline", path, dict(env, FTRACE_NOSPHQ="1"))
        rc1, log1, queued = run(name + "_queued", path, env)
        waves = [tuple(int(g) for g in m.groups()) for m in WAVE.finditer(log1)]
        queuedN = sum(w[2] for w in waves)
        inlineN = sum(w[3] for w in waves)
        check(name + ": the inline connections render, no queue",
              rc0 == 0 and inline is not None and not WAVE.search(log0), "rc %d" % rc0)
        if name == "waves":
            check(name + ": the shrunk queue splits the photons into waves and turns requests away inline",
                  rc1 == 0 and len(waves) >= 4 and queuedN > 0 and inlineN > 0,
                  "rc %d, %d waves, %d queued, %d inline" % (rc1, len(waves), queuedN, inlineN))
        else:
            check(name + ": the queue renders, its connections queued",
                  rc1 == 0 and queued is not None and queuedN > 0,
                  "rc %d, %d waves, %d queued, %d inline" % (rc1, len(waves), queuedN, inlineN))
        if inline is not None and queued is not None:
            r, n = image_diff(inline, queued)
            check(name + ": ...the image the inline connections'", r < ORDER_TOL,
                  "largest change %.2g of the pixel, %d values differ" % (r, n))
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
