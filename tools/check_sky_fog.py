"""Regression check: the sky seen through a fog, in the forward modes and mode M, against mode R -- and
mode R's own sky through a fog that absorbs only, against its transmittance worked out in closed form.

    python tools/check_sky_fog.py [--out png/check_sky_fog] [--exe ftrace.exe] [--no-gpu] [--spp 256] [--modes BAMP]
                                  [--fogs absorb,scatter,field_absorb,ground,expfog,far_expfog,haze,
                                          sun_haze,ball_edge,sun_sky,clear,mirror_ball]

A uniform sky (`light env { spd 1.0 }`) over a floor, seen through a fog. The floor is 10 m across, small
enough that the forward modes' env photons, spread over the scene's bounding sphere, cross the fog often;
the `far_` cases film a 1 m one from 20 m, which puts the camera far outside the scene's sphere. The fogs:
a box 2 m deep absorbing only (`albedo 0`) and scattering (`albedo 0.5`); three absorbing density fields
with no bounds -- one with a floor, a ground fog thinning to nothing above 2 m, and an exponential one
(scale height 20 m) whose optical depth past eight scene radii is a third of the whole -- and, for mode P
alone, a scattering haze with no bounds, whose sky pixels its still composite takes from the backward
layer -- as it does under a sun alone (`sun_haze`), where those pixels hold the haze's glow; a ball of fog
against the sky (`ball_edge`, compared pixel by pixel too: mode P's still composite draws the sky its
forward pixels' samples see past the ball's outline); and no fog at all, a sun
in the frame over the sky (`sun_sky`: its disc, which the composite draws in its sky pass); and the
clear sky alone (`clear`, mode P: drawn exactly, so two seeds' sky are the same, the forward layer's
fitted scale no part of it); and a mirror ball against the sky (`mirror_ball`, mode P pixel by pixel:
a pixel part of which shows the ball is its backward layer's, or the ball has a dark rim). (An unbounded
medium that scatters is no case for the other modes: they light one only near the scene,
warnHazeOnlyNearScene, where mode R lights all of it.)

The frame's top quarter is sky alone, so it holds the sky's transmittance and the fog's in-scatter. Each
mode on the CPU and the GPU against mode R on the CPU, that quarter in X, Y and Z over 4 x 4 tiles:

  - mode B, whose sky pass (main.cpp prepareSky, skySeenThroughXYZ) takes the media along the ray;
  - mode A through a physical lens, the same pass's lens samples;
  - mode M with -beams, whose camera ray takes the media's extinction whether or not a beam was stored
    (a medium that scatters nothing stores none);
  - mode P, whose still composite takes the sky, and the sky seen through a bounded medium, from its
    forward layer (classifyComposite), the sky pass drawing it there.

The density fields hold an unbounded field's far end (UNBOUNDED-FIELD-FAR, Medium::farSplit): mode R on
both devices, divided by its clear sky, must read exp(-sigma_t * F / d_y) per pixel, F the density's
integral above the eye and d_y the ray's rise -- 0 for the field with a floor, whose optical depth has
no end, and for the exponential fog that much with its far part counted (the far end's coarse tail), as
with the camera 20 m out (the far sphere holds the cameras).

Exits 1 on any failure.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import SRGB_TO_XYZ, render_seeds, tile_stats, tile_verdict  # noqa: E402

TOL = 0.03
W, H = 64, 48
FOV_Y = 30.0
SKY_ROWS = H // 4        # the top quarter: sky alone
SCENE = [
    'camera "c" { eye %s  look_at %s  up 0 1 0  fov_y %g  %s film { res %d %d } }',
    'material "white" { type diffuse  reflect 0.7 }',
    '%s',
    '%s',
    '%s',
]
RIGS = {   # name -> (eye, look_at, floor): the same pitch, the camera inside or far outside the scene
    "near": ((0, 0.6, -2.5), (0, 0.9, 0), "quad { origin -5 0 -5   u 0 0 10   v 10 0 0   material white }"),
    "far": ((0, 0.6, -20), (0, 3.0, 0), "quad { origin -0.5 0 -0.5   u 0 0 1   v 1 0 0   material white }"),
}
PHYS = "focus 4.0  lens { preset achromat  focal 50  fstop 2.8 }"
BOX = "medium { sigma_t 0.5  albedo %s  g 0.0  bounds { min -5 0 -1  max 5 6 1 } }"
FIELD = 'medium { sigma_t 0.06  albedo %s  density "0.3 + 0.7*saturate(1.0 - y/8.0)"  density_max 1 }'
GROUND = 'medium { sigma_t 0.1  albedo 0.0  density "saturate(1.0 - y/2.0)"  density_max 1 }'
EXPFOG = 'medium { sigma_t 0.01  albedo 0.0  density "exp(-max(y, 0.0)/20.0)"  density_max 1 }'


def depth_above(fog, y0):
    """sigma_t times the density's integral over y from y0 to infinity (inf where it has none)."""
    if fog == "field_absorb":
        return math.inf
    if fog == "ground":   # 1 - y/2 integrated from y0 to y = 2, where it ends
        return 0.1 * ((2.0 - y0) - (4.0 - y0 * y0) / 4.0) if y0 < 2.0 else 0.0
    if fog in ("expfog", "far_expfog"):
        return 0.01 * 20.0 * math.exp(-y0 / 20.0)
    return None


ENV = "light env { spd 1.0 }"
SUN = "light sun { dir 0.3 1 0.2  angle 0.53  spd preset:d65 }"
SUN_IN_VIEW = "light sun { dir 0.1 0.33 1  angle 0.53  spd preset:d65 }"   # in the sky rows
HAZE = "medium { sigma_t 0.04  albedo 0.5 }"
# A ball of fog in the sky's rows, its outline crossing their pixels at every offset.
BALL = "medium { sigma_t 0.5  albedo 0.0  g 0.0  bounds { center 0 1.68 0.83  radius 0.5 } }"
PIXEL_FOGS = ("ball_edge", "mirror_ball")   # also pixel by pixel: a one-pixel outline hides in a tile's mean
EXACT_FOGS = ("clear",)       # a sky drawn exactly: each seed's the same
# A mirror ball against the sky, mode P pixel by pixel: a forward pixel part of which shows it takes the
# backward layer's (SkyView::specSeen), or the ball has a dark rim.
MIRROR_BALL = NL.join(['material "chrome" { type mirror  reflect 0.9 }',
                       "sphere { center 0 1.68 0.83  radius 0.5  material chrome }"])
PIXEL_TOL = 0.25              # ...off by more than this (5 sigma of the noisiest pixel's two seeds)
FOGS = [   # (name, the rig, the medium, the modes that take it, the light)
    ("absorb", "near", BOX % "0.0", "BAMP", ENV),
    ("scatter", "near", BOX % "0.5", "BAMP", ENV),
    ("field_absorb", "near", FIELD % "0.0", "BAMP", ENV),
    ("ground", "near", GROUND, "BAMP", ENV),
    ("expfog", "near", EXPFOG, "BAMP", ENV),
    ("far_expfog", "far", EXPFOG, "BAMP", ENV),
    ("haze", "near", HAZE, "P", ENV),
    ("sun_haze", "near", HAZE, "P", SUN),
    ("ball_edge", "near", BALL, "BP", ENV),
    ("sun_sky", "near", "", "BP", ENV + NL + SUN_IN_VIEW),
    ("clear", "near", "", "P", ENV),
    ("mirror_ball", "near", MIRROR_BALL, "P", ENV),
]
MODES = [   # (tag, the camera's extra statement, the mode's arguments)
    ("B", "", ["-mode", "B", "-n", "4000000"]),
    ("A", PHYS, ["-mode", "A", "-n", "4000000"]),
    ("M", "", ["-mode", "M", "-beams", "-n", "400000"]),
    ("P", "", ["-mode", "P", "-n", "4000000"]),
]


def sky_transmittance(rig, fog, sub=4):
    """Each sky pixel's transmittance through `fog` from the rig's pinhole, averaged over sub x sub
    points of the pixel; rows bottom-up, as the PFM's are."""
    eye, look, _ = RIGS[rig]
    e, f = np.array(eye, float), np.array(look, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    th = math.tan(math.radians(FOV_Y) / 2.0)
    out = np.zeros((SKY_ROWS, W))
    for j in range(SKY_ROWS):              # j: the row from the top
        for i in range(W):
            acc = 0.0
            for a in range(sub):
                for b in range(sub):
                    x = (2.0 * (i + (a + 0.5) / sub) / W - 1.0) * th * W / H
                    y = (1.0 - 2.0 * (j + (b + 0.5) / sub) / H) * th
                    d = f + x * r + y * u
                    dy = d[1] / np.linalg.norm(d)
                    tau = depth_above(fog, e[1])
                    acc += math.exp(-tau / dy) if math.isfinite(tau) else 0.0
            out[SKY_ROWS - 1 - j, i] = acc / (sub * sub)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_sky_fog"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--fogs", default="", help="comma-separated fog names (default: all)")
    ap.add_argument("--modes", default="BAMP", help="the modes compared with mode R (default BAMP; "
                    "'-' for none, the closed-form checks alone)")
    a = ap.parse_args()
    names = [f[0] for f in FOGS]
    want = [w for w in a.fogs.split(",") if w] or names
    if [w for w in want if w not in names]:
        ap.error("the fogs are %s" % ", ".join(names))
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]

    cache = {}

    def sky(name, rig, cam, medium, light, args, dev):
        """Each seed's render, in XYZ, cut to the sky's rows (the PFM stores rows bottom-up)."""
        key = (name, rig, cam, medium, light, args[1], dev)
        if key in cache:
            return cache[key]
        eye, look, floor = RIGS[rig]
        lines = list(SCENE)
        lines[0] = lines[0] % ("%g %g %g" % eye, "%g %g %g" % look, FOV_Y, cam, W, H)
        lines[2] = floor
        lines[3] = light
        lines[4] = medium
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        tag = name + "_" + args[1] + ("g" if dev == "gpu" else "")
        cache[key] = [(x @ SRGB_TO_XYZ.T)[-SKY_ROWS:] for x in
                      render_seeds(exe, d, name + ".ftsl", tag, args, dev, a.spp, rgb=True)]
        return cache[key]

    ok = True
    for fog, rig, medium, modes, light in FOGS:
        if fog not in want:
            continue
        for tag, cam, margs in MODES:
            if tag not in modes or tag not in a.modes:
                continue
            # Mode A's lens makes a scene of its own; the others share one, and its mode R render.
            name = "%s_%s" % (fog, "A" if tag == "A" else "pin")
            rr = sky(name, rig, cam, medium, light, ["-mode", "R"], "cpu")
            for dev in devs:
                bb = sky(name, rig, cam, medium, light, margs, dev)
                good, parts = True, []
                for c, cname in enumerate("XYZ"):
                    ratio, z, bad = tile_verdict(tile_stats(bb[0][..., c], bb[1][..., c]),
                                                 tile_stats(rr[0][..., c], rr[1][..., c]), TOL)
                    good &= not bad.any()
                    parts.append("%s worst x%.3g (%.1f sigma)" % (cname, ratio.flat[int(z.argmax())], z.max()))
                if fog in EXACT_FOGS:   # the two seeds' sky, Y: the same (mode P's fitted scale no part of it)
                    q = bb[1][..., 1] / np.maximum(bb[0][..., 1], 1e-30)
                    apart = float(np.abs(q - 1.0).max())
                    good &= apart < 1e-5
                    parts.append("the seeds' sky apart by %.2g" % apart)
                if fog in PIXEL_FOGS:   # each pixel's Y
                    mb = 0.5 * (bb[0][..., 1] + bb[1][..., 1])
                    mr = 0.5 * (rr[0][..., 1] + rr[1][..., 1])
                    q = mb / np.maximum(mr, 1e-30)
                    off = np.where(mr > 0.02 * mr.max(), np.abs(q - 1.0), 0.0)
                    good &= not (off > PIXEL_TOL).any()
                    parts.append("pixels worst x%.3g (%d off by %d %%)" % (q.flat[int(off.argmax())],
                                 int((off > PIXEL_TOL).sum()), int(100 * PIXEL_TOL)))
                ok &= good
                print("%s: %s fog, mode %s%s / mode R: %s" % ("ok" if good else "FAIL", fog, tag,
                      " (device)" if dev == "gpu" else "", ", ".join(parts)), flush=True)
        if depth_above(fog, 0.0) is None:
            continue
        # Mode R's own sky through the field against its closed form, per 4 x 4 tile of Y.
        clear = sky("clear_" + rig, rig, "", "", light, ["-mode", "R"], "cpu")
        c0 = 0.5 * (clear[0][..., 1] + clear[1][..., 1])
        want_t = sky_transmittance(rig, fog)
        for dev in devs:
            rr = sky("%s_pin" % fog, rig, "", medium, light, ["-mode", "R"], dev)
            got = 0.5 * (rr[0][..., 1] + rr[1][..., 1])
            th, tw = SKY_ROWS // 4, W // 4
            worst = 0.0
            for ty in range(4):
                for tx in range(4):
                    s = (slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw))
                    t_got = got[s].sum() / c0[s].sum()
                    t_want = (want_t[s] * c0[s]).sum() / c0[s].sum()
                    worst = max(worst, abs(t_got - t_want))
            good = worst < 0.02
            ok &= good
            print("%s: %s fog, mode R%s's sky transmittance / closed form: worst tile off by %.4f"
                  % ("ok" if good else "FAIL", fog, " (device)" if dev == "gpu" else "", worst), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
