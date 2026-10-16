"""Regression check: an AIMED area light -- a quad whose `normal` leans out of its plane -- shines
along that normal while the rectangle keeps its own orientation (0.384.0), in every strategy.

    D:/python314/python.exe -u tools/check_tilted_light.py [--out <dir>] [--exe ftrace.exe]
                                                           [--spp N] [--n N] [--no-gpu]
                                                           [--modes RWDUSMBAP]

The model (scene.h, Emitter::aimCos): the radiance leaving the patch toward a direction w is
Le * cos_a / cos_g, with cos_a = w . normal (the aim) and cos_g = w . face (the rectangle's own
normal, on the side the aim leans to), and zero unless both are positive. So the patch's
intensity per unit area is Le * cos_a, a Lambertian lobe about the aim cut off at the face's
plane, and its flux is Le * area * pi * (1 + cos t) / 2 for a lean of t.

1. Analytic, on a floor. A 0.2 x 0.2 light 1 m above a diffuse floor, seen from above, rendered
   twice at the same `power`: normal straight down (flat), and leaning 45 degrees toward +x
   (aimed). The floor sees nothing but the light, so each pixel's ratio aimed / flat is the
   ratio of the two direct irradiances, which this check integrates over the light itself:

       R = pi / aimNorm * Int cos_a cos_r / d^2 dA  /  Int cos_g cos_r / d^2 dA.

   The flat render times R is the prediction of the aimed one, pixel by pixel; a tile of a 4 x 4
   grid fails when the two differ by more than 5 standard errors (two seeds each) AND --tol
   (2 %). And the floor past the aim's horizon (x < -1.3: the light still faces it, but leans
   away from it) must stay black, under 0.5 % of what the flat light puts there.
   Pixels the light's back covers, even partly, are left out.

2. Fog. The aimed light over a box of fog on a floor, the same 4 x 4 tiles compared between
   every strategy and mode R on the host: the volume's light sampling and the forward tracer's
   births are separate code from the surface paths part 1 measures. Modes W (single scattering
   only, so meant to differ), U and S (surfaces only) and A (a thin-lens sensor, on its own
   exposure scale) sit this part out.

Every mode in --modes runs on the host and (unless --no-gpu) the device. Before 0.384.0 the
patch measured a light by its face on the camera side and by the authored normal on the light
side, so a leaning normal rendered as a flat light whose power was spread differently: part 1
fails there by tens of percent. Exits 1 on any failure.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)
RES = (64, 48)
EYE = np.array([0.3, 4.0, -1.2])
LOOK = np.array([0.3, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
FOV_Y = 50.0
LIGHT_O = np.array([-0.1, 1.0, -0.1])     # origin, u and v of the light quad
LIGHT_U = np.array([0.2, 0.0, 0.0])
LIGHT_V = np.array([0.0, 0.0, 0.2])
FACE = np.array([0.0, -1.0, 0.0])
LEAN = math.radians(45.0)
AIM = np.array([math.sin(LEAN), -math.cos(LEAN), 0.0])
DARK_X = -1.3                             # floor beyond the aim's horizon, with a margin
MODE_ARGS = {"M": ["-beams"]}             # the fog needs the beam map in mode M
# Not compared in fog: mode W gathers a medium's single scattering only, so it is meant to differ
# from mode R there; modes U and S refuse media; and mode A images through a thin lens onto a
# sensor (its own exposure scale, with depth of field), so it is compared only with itself.
NO_FOG = "WUSA"


def light_block(normal):
    return ('light area { origin %g %g %g  u %g %g %g  v %g %g %g  normal %s  power 40  '
            'spd preset:d65 }' % (tuple(LIGHT_O) + tuple(LIGHT_U) + tuple(LIGHT_V) + (normal,)))


def floor_scene(normal):
    return [
        'camera "c" { eye %g %g %g  look_at %g %g %g  up %g %g %g  fov_y %g  film { res %d %d } }'
        % (tuple(EYE) + tuple(LOOK) + tuple(UP) + (FOV_Y,) + RES),
        'material "floor" { type diffuse  reflect 0.6 }',
        'quad { origin -4 0 -4  u 0 0 8  v 8 0 0  material floor }',
        light_block(normal),
    ]


def fog_scene(normal):
    return [
        'camera "c" { eye 0 0.5 -3.2  look_at 0 0.35 0  up 0 1 0  fov_y 34  film { res 64 48 } }',
        'material "floor" { type diffuse  reflect 0.6 }',
        'quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material floor }',
        'medium { sigma_t 0.8  albedo 0.8  g 0.2  bounds { min -1.2 0 -0.8  max 1.2 0.8 0.8 } }',
        light_block(normal),
    ]


def floor_points():
    """Per pixel (row-major as the PFM is stored): the floor point the pixel's centre ray meets,
    and whether the light's quad covers any of the pixel (a ray through its centre, an edge
    midpoint or a corner meets it first). Mirrors Camera::lookAt and genRay (camera.h)."""
    w_px, h_px = RES
    w = LOOK - EYE
    w /= np.linalg.norm(w)
    u = np.cross(w, UP)
    u /= np.linalg.norm(u)
    v = np.cross(u, w)
    ty = math.tan(math.radians(FOV_Y) / 2)
    tx = ty * w_px / h_px

    def rays(fx, fy):
        sx = 2.0 * (np.arange(w_px) + fx) / w_px - 1.0
        sy = 2.0 * (np.arange(h_px) + fy) / h_px - 1.0
        return (w[None, None, :] + u[None, None, :] * (sx[None, :, None] * tx)
                + v[None, None, :] * (sy[:, None, None] * ty))

    d = rays(0.5, 0.5)
    p = EYE + d * (-EYE[1] / d[..., 1])[..., None]
    blocked = np.zeros((h_px, w_px), dtype=bool)
    for fx in (0.0, 0.5, 1.0):
        for fy in (0.0, 0.5, 1.0):
            dd = rays(fx, fy)
            q = EYE + dd * ((LIGHT_O[1] - EYE[1]) / dd[..., 1])[..., None] - LIGHT_O
            a = q @ LIGHT_U / LIGHT_U.dot(LIGHT_U)
            b = q @ LIGHT_V / LIGHT_V.dot(LIGHT_V)
            blocked |= (a >= -0.02) & (a <= 1.02) & (b >= -0.02) & (b <= 1.02)
    return p, blocked


def irradiance(p, cos_emit, k=24):
    """Direct irradiance at floor points `p` (..., 3) from the light, per unit radiance scale:
    the integral over the quad of cos_emit(w) * cos_r / d^2, by a k x k midpoint rule."""
    e = np.zeros(p.shape[:-1])
    da = np.linalg.norm(np.cross(LIGHT_U, LIGHT_V)) / (k * k)
    for i in range(k):
        for j in range(k):
            y = LIGHT_O + LIGHT_U * ((i + 0.5) / k) + LIGHT_V * ((j + 0.5) / k)
            r = p - y
            d2 = (r * r).sum(-1)
            wdir = r / np.sqrt(d2)[..., None]
            cos_r = np.maximum(-wdir[..., 1], 0.0)      # floor normal +y, light above it
            e += cos_emit(wdir) * cos_r / d2 * da
    return e


def predicted_ratio():
    """Per pixel, the aimed render over the flat one (0 where both are black), the pixels the
    light's back hides, and the floor past the aim's horizon."""
    p, blocked = floor_points()
    aim_norm = math.pi * (1.0 + AIM.dot(FACE)) / 2.0           # pi (1 + cos t) / 2

    def flat(wd):
        return np.maximum(wd @ FACE, 0.0)

    def aimed(wd):
        return np.where(wd @ FACE > 0.0, np.maximum(wd @ AIM, 0.0), 0.0)

    ef, ea = irradiance(p, flat), irradiance(p, aimed)
    ratio = math.pi / aim_norm * ea / np.maximum(ef, 1e-300)
    ratio[blocked] = 0.0
    return ratio, blocked, (p[..., 0] < DARK_X) & ~blocked


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_tilted_light"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--n", type=int, default=20000000,
                    help="photons: for the forward modes, and in all (over its passes) for mode S")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--modes", default="RWDUSMBAP")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    normals = {"flat": "%g %g %g" % tuple(FACE), "aimed": "%.8f %.8f %.8f" % tuple(AIM)}
    for name, normal in normals.items():
        for kind, build in (("floor", floor_scene), ("fog", fog_scene)):
            with open(os.path.join(d, "%s_%s.ftsl" % (kind, name)), "w", encoding="utf-8",
                      newline="") as f:
                f.write(NL.join(build(normal)) + NL)

    ratio, blocked, dark = predicted_ratio()
    keep = ~blocked
    ok = True

    def report(good, what):
        nonlocal ok
        ok &= good
        print("%s: %s" % ("ok" if good else "FAIL", what), flush=True)

    def worst(r, z):
        if z.max() <= 0.0:
            return "every tile identical"
        i, j = np.unravel_index(int(np.argmax(z)), z.shape)
        sig = "noise-free" if z[i, j] > 1e6 else "%.1f sigma" % z[i, j]
        return "worst tile x%.4f (%s)" % (r[i, j], sig)

    devices = ["cpu"] + ([] if a.no_gpu else ["gpu"])
    fog_ref = None
    for dev in devices:
        for m in a.modes:
            label = "mode %s, %s" % (m, "device" if dev == "gpu" else "host")
            tag = "%s%s" % (m, "g" if dev == "gpu" else "")
            # Mode S takes -n per pass (and -spp passes): the same photon total as the others.
            args = ["-mode", m, "-n", str(a.n // a.spp if m == "S" else a.n)]
            flat = render_seeds(a.exe, d, "floor_flat.ftsl", "floor_flat_" + tag, args, dev, a.spp)
            aimd = render_seeds(a.exe, d, "floor_aimed.ftsl", "floor_aimed_" + tag, args, dev,
                                a.spp)
            pred = [f * ratio for f in flat]
            r, z, bad = tile_verdict(tile_stats(*[x * keep for x in aimd]), tile_stats(*pred),
                                     a.tol)
            fr = sum((x * keep).sum() for x in aimd) / max(sum(x.sum() for x in pred), 1e-30)
            report(not bad.any(), "%s, floor: aimed / predicted, frame x%.4f; %s"
                   % (label, fr, worst(r, z)))
            lit = sum(x[dark].mean() for x in flat)
            leak = sum(x[dark].mean() for x in aimd) / max(lit, 1e-30)
            report(lit > 0.0 and leak < 0.005,
                   "%s, floor past the aim's horizon: %.3f %% of the flat light" % (label, 100 * leak))

            if m in NO_FOG:
                continue
            fog = render_seeds(a.exe, d, "fog_aimed.ftsl", "fog_aimed_" + tag,
                               args + MODE_ARGS.get(m, []), dev, a.spp)
            if fog_ref is None:
                fog_ref = tile_stats(*fog)          # mode R on the host, the first case run
                continue
            r, z, bad = tile_verdict(tile_stats(*fog), fog_ref, a.tol)
            report(not bad.any(), "%s, fog: against mode %s on the host; %s"
                   % (label, a.modes[0], worst(r, z)))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
