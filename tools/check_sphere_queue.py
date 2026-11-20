"""Regression check: GPU mode B's queued glass-sphere connections render what the inline ones render.

    python tools/check_sphere_queue.py [--out png/check_sphere_queue] [--exe ftrace.exe]

GPU-SPHERE-CONNECT-SLOW. launchForward queues each photon vertex's connections through a glass sphere
(DSphReq: the sphere, the vertex, its splat records with their hit, its wavelengths, its stream) and
kSphConnect runs them after each wave of photons, a warp's lanes together, where inline each ran in the
photon's own thread (FTRACE_NOSPHQ=1). Under per-photon seeding (FTRACE_GPU_PHOTONSEED=1) the two trace
the same photons and make the same connections, each drawing from a stream of its own (a medium's
transmittance is all a connection draws), and only the order of the film's sums differs: images within
ORDER_TOL. Each case,
GPU mode B, the queue's image against the inline one's:

  - a glass ball in a lit room, seen from outside (check_ball_images' room);
  - the eye inside the ball, off its centre (its chains past the critical angle);
  - the ball over a glossy floor, a vertex's glossy splat record copied into the request;
  - the ball over a coated floor (a clear coat over a diffuse body), its coat record copied;
  - the first case again with the queue shrunk to 2000 requests (FTRACE_SPHQCAP): many waves, and
    requests past the queue connected inline (FTRACE_WAVE_DEBUG's line a wave says both happened);
  - two cameras through a homogeneous fog with `-beams`: one shared flight whose beam gather splats
    each camera's in-scatter points to a one-camera slice of the set, the fog seen through the ball
    a volume vertex's connection (each camera's image against its own; the fog's transmittance is
    analytic, so the connections still draw nothing);
  - the ball sliding across the shutter (a moving group, the camera's shutter open): a request
    keeps its path's time;
  - a dispersive (SF10) ball under `-heroc 4`: a bundle's wavelengths split into groups that see the
    ball each with its own index, each group a request;
  - the fog thinning with height (a density field): its transmittance draws, from the connection's
    own stream whether it is queued or not, so the images still agree to the bit.

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
    fog = 'medium { sigma_t 0.4  albedo 0.9  g 0.2  bounds { min -2 0 -2  max 2 3 2 } }'
    second = 'camera "d" { eye -0.8 1.55 1.6  look_at 0 1.5 1.85  up 0 1 0  fov_y 16  film { res 96 72 } }'
    shutter = OUTSIDE.replace("film {", "shutter_open -0.5  shutter_close 0.5  film {")
    assert shutter != OUTSIDE
    slide = ("group { motion { key { translate -0.06 0 0 }  key { translate 0.06 0 0 } }  %s }" % BALL)
    sf10 = 'material "sf10" { type dielectric  ior glass:SF10 }'
    hazy = ('medium { sigma_t 0.6  albedo 0.9  g 0.2  density "0.2 + 0.8*saturate(1.0 - y/3.0)"  '
            'density_max 1  bounds { min -2 0 -2  max 2 3 2 } }')
    cases = [
        ("outside", scene("outside", ROOM, BALL, OUTSIDE), {}, [], [""]),
        ("inside", scene("inside", ROOM, BALL, INSIDE_OFF), {}, [], [""]),
        ("glossy", scene("glossy", glossy, BALL, OUTSIDE), {}, [], [""]),
        ("coated", scene("coated", coated, BALL, OUTSIDE), {}, [], [""]),
        ("waves", os.path.join(d, "outside.ftsl"), {"FTRACE_SPHQCAP": "2000"}, [], [""]),
        ("beams", scene("beams", ROOM, fog, BALL, OUTSIDE, second), {}, ["-beams"], ["_c", "_d"]),
        ("motion", scene("motion", ROOM.replace("scene { units meters }",
                                                "scene { units meters  motion_interval -0.5 0.5 }"),
                         slide, shutter), {}, [], [""]),
        ("hero", scene("hero", ROOM, sf10, BALL.replace("material glass", "material sf10"), OUTSIDE), {},
         ["-heroc", "4"], [""]),
        ("hetero", scene("hetero", ROOM, hazy, BALL, OUTSIDE), {}, [], [""]),
    ]

    def run(tag, path, env, extra, cams):
        """One render; its images, one a camera (`cams`: what each camera's file adds to the stem)."""
        png = os.path.join(d, tag + ".png")
        pfms = [png[:-4] + c + ".pfm" for c in cams]
        for stale in [png] + pfms:
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", path, "-mode", "B", "-device", "gpu", "-n", PHOTONS, "-max-bounce", "64",
               "-seed", "3", "-hdr", "-no-meter", "-window-min", "-o", png] + extra
        full = dict(os.environ, FTRACE_GPU_PHOTONSEED="1", FTRACE_WAVE_DEBUG="1", **env)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=full)
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        imgs = [read_pfm(f, top_down=False) if os.path.exists(f) else None for f in pfms]
        return p.returncode, log, imgs

    for name, path, env, extra, cams in cases:
        rc0, log0, inline = run(name + "_inline", path, dict(env, FTRACE_NOSPHQ="1"), extra, cams)
        rc1, log1, queued = run(name + "_queued", path, env, extra, cams)
        waves = [tuple(int(g) for g in m.groups()) for m in WAVE.finditer(log1)]
        queuedN = sum(w[2] for w in waves)
        inlineN = sum(w[3] for w in waves)
        check(name + ": the inline connections render, no queue",
              rc0 == 0 and all(x is not None for x in inline) and not WAVE.search(log0), "rc %d" % rc0)
        if name == "waves":
            check(name + ": the shrunk queue splits the photons into waves and turns requests away inline",
                  rc1 == 0 and all(x is not None for x in queued) and len(waves) >= 4 and queuedN > 0 and
                  inlineN > 0,
                  "rc %d, %d waves, %d queued, %d inline" % (rc1, len(waves), queuedN, inlineN))
        else:
            check(name + ": the queue renders, its connections queued",
                  rc1 == 0 and all(x is not None for x in queued) and queuedN > 0,
                  "rc %d, %d waves, %d queued, %d inline" % (rc1, len(waves), queuedN, inlineN))
        for c, a0, a1 in zip(cams, inline, queued):
            if a0 is not None and a1 is not None:
                r, n = image_diff(a0, a1)
                check(name + c + ": ...the image the inline connections'", r < ORDER_TOL,
                      "largest change %.2g of the pixel, %d values differ" % (r, n))
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
