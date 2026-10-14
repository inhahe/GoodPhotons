"""Regression check: camera-side transport scales radiance by (n_from / n_to)^2 at a refraction.

    D:/python314/python.exe -u tools/check_radiance_eta2.py [--out <dir>] [--exe ftrace.exe]
                                                            [--spp N] [--no-gpu]

Radiance over n^2 is what a ray conserves across a refraction, so a walk that carries radiance --
the camera side of every mode -- multiplies its throughput by (n_from / n_to)^2 there, and one that
carries flux -- the forward tracer, BDPT's light walk -- does not (Renderer::refractOrReflect,
RADIANCE-ETA2). Going into a body and out again it cancels; ending inside one, it does not. Until
0.378.1 no camera-side walk applied it. The scenes:
  1. a lamp inside a solid glass block (n 1.5) lighting the floor below it: modes R and D against
     mode B (the forward tracer, right by construction), on the floor, which the camera sees
     directly -- R read x2.25 (= n^2): its light reaches the lamp through one interface; and the
     same block made of a thin-film-coated substrate, and of a multilayer stack (mode R against
     B), whose interfaces carry the factor through their own code;
  1b. a spherical lamp inside the glass block, seen through it: mode W (the deterministic
     preview) against mode R over the middle of the frame, where both see the lamp straight
     through the front face;
  2. a diffuse ball over a diffuse pool floor, both inside a closed water tank (n 1.333), lit from
     above and seen from outside: modes M, S, U and D against mode R, which is right there with
     or without the factor, every one of its paths crossing in and out. M read x1.8 (= n^2) on
     the ball, its photons (flux) meeting a camera walk that had gone in once, and S and U the
     same through their photon merges. Mode D is unmoved here -- its connections inside the water
     share these paths with s = 0 ones that cross out to the lamp -- and is compared for
     consistency;
  3. the same tank with the lamp INSIDE the water, above the ball: every camera path crosses in
     once and never out, so every strategy of a camera-side mode carries the factor, and mode D
     (and U) against mode R read x1.8 without their own. (R is pinned by the first scene.)
  4. the tank lit from above with nested dielectrics: the water (`priority 1`) holds a glass ball
     (`priority 2`) around a smaller diffuse one, so the camera's second interface refracts
     water to glass, (1.333 / 1.5)^2 by the priority rule: modes M and S against mode R -- a
     factor that ignored the exterior index (refracting by it all the same) would put M and S
     x0.56 off on the inner ball, while R, crossing out again, would not notice; and the same
     with the glass ball thin-film-coated (4b), whose interface takes the water's index too.
In the tank mode M also runs with its final gather (-pmfg 4), a walk of its own. (Not in the nested
scene: there the gather sees the glass ball's caustic, too heavy-tailed for two seeds.)
Modes D and U run at mode R's depth (-max-bounce 32): light trapped in the water by total internal
reflection makes long paths, and at its own default of 8 D reads x0.94 -- as R does at 8. Mode U
merges with a quarter of the default radius (-pmradiusfrac 0.005): at the default, its merges on the
pool floor seen in the water's surface read up to 19 % low near the floor's edges (a photon map's
radius bias, gone at the smaller radius). Then, unless --no-gpu, the device against the host,
each device render required to say it ran on the GPU: mode R on the glass, thin-film and
multilayer blocks, and the fast RGB walk (-rgb, device only) against the spectral one there; mode
W on the lamp seen in glass; modes M (both ways) and S on the tank, M over the ball (the floor in
front of the tank carries a separate device / host difference, known-issues GPU-M-FLOOR); modes D
and U on the submerged lamp. Every render's outputs are deleted before it runs, so a run that
writes nothing cannot be scored on an earlier run's image. A tile fails when the two differ by
more than 5 standard errors (two seeds each) AND by more than the pair's tolerance. Exits 1 on
any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)


def box(x0, y0, z0, x1, y1, z1, mat):
    """An axis-aligned box as six quads whose u x v points out."""
    w, h, dz = x1 - x0, y1 - y0, z1 - z0
    faces = [((x0, y0, z0), (0, h, 0), (w, 0, 0)), ((x0, y0, z1), (w, 0, 0), (0, h, 0)),
             ((x0, y0, z0), (0, 0, dz), (0, h, 0)), ((x1, y0, z0), (0, h, 0), (0, 0, dz)),
             ((x0, y0, z0), (w, 0, 0), (0, 0, dz)), ((x0, y1, z0), (0, 0, dz), (w, 0, 0))]
    f = lambda v: "%g %g %g" % v  # noqa: E731
    return ['quad { origin %s  u %s  v %s  material %s }' % (f(o), f(u), f(v), mat)
            for o, u, v in faces]


# 1. A lamp in a glass block above the floor; the camera looks at the floor beneath and beside
#    it, low enough that the block stays out of view (mode B cannot see a glass-first pixel).
LAMP_IN_GLASS = [
    'camera "c" { eye 0 0.35 -2.5  look_at 0 0 0.3  up 0 1 0  fov_y 20  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "glass" { type dielectric  ior 1.5 }',
    'quad { origin -4 0 -4   u 0 0 8   v 8 0 0   material floor }',
    'light area { origin -0.15 0.75 -0.15  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0  power 100  '
    'spd preset:d65 }',
] + box(-0.5, 0.5, -0.5, 0.5, 1.0, 0.5, "glass")

# 1. ... and the block a coated substrate: a thin film, and a two-layer stack.
FILM = 'material "glass" { type thinfilm  ior 1.5  film_ior 1.3  film_thickness 300 }'
STACK = 'material "glass" { type multilayer  ior 1.5  layer 2.3 0.0 60  layer 1.38 0.0 100 }'
LAMP_IN_FILM = [FILM if ln.startswith('material "glass"') else ln for ln in LAMP_IN_GLASS]
LAMP_IN_STACK = [STACK if ln.startswith('material "glass"') else ln for ln in LAMP_IN_GLASS]

# 1b. A spherical lamp in the glass block, seen straight through its front face.
GLASS_SEEN = [
    'camera "c" { eye 0 0.75 -2.5  look_at 0 0.75 0  up 0 1 0  fov_y 20  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "glass" { type dielectric  ior 1.5 }',
    'quad { origin -4 0 -4   u 0 0 8   v 8 0 0   material floor }',
    'light sphere { center 0 0.75 0  radius 0.1  power 20  spd preset:d65 }',
] + box(-0.5, 0.5, -0.5, 0.5, 1.0, 0.5, "glass")

# 2. A diffuse ball over a diffuse pool floor in a closed water tank, a lamp above it, the camera
#    outside.
TANK_LAMP = ('light area { origin -0.4 2.0 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 200  '
             'spd preset:d65 }')
TANK = [
    'camera "c" { eye 0 0.55 -2.4  look_at 0 0.45 0  up 0 1 0  fov_y 36  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.5 }',
    'material "white" { type diffuse  reflect 0.8 }',
    'material "water" { type dielectric  ior 1.333 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'sphere { center 0 0.4 0  radius 0.25  material white }',
    'quad { origin -0.69 0.05 -0.69   u 0 0 1.38   v 1.38 0 0   material white }',
    TANK_LAMP,
] + box(-0.7, 0.01, -0.7, 0.7, 0.9, 0.7, "water")

# 3. The tank with its lamp under the water's surface, above the ball.
SUBMERGED = [ln for ln in TANK if ln != TANK_LAMP] + [
    'light area { origin -0.15 0.8 -0.15  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0  power 50  '
    'spd preset:d65 }']

# 4. The tank with ranked dielectrics: the water holds a glass ball around a smaller diffuse one --
#    and (4b) the same with the ball a thin-film-coated one, ranked the same way.
NESTED = [{'material "water" { type dielectric  ior 1.333 }':
           'material "water" { type dielectric  ior 1.333  priority 1 }',
           'sphere { center 0 0.4 0  radius 0.25  material white }':
           'sphere { center 0 0.4 0  radius 0.2  material white }'}.get(ln, ln) for ln in TANK] + [
    'material "glass" { type dielectric  ior 1.5  priority 2 }',
    'sphere { center 0 0.4 0  radius 0.3  material glass }']
NESTED_COAT = [('material "glass" { type thinfilm  ior 1.5  film_ior 1.3  film_thickness 300  '
                'priority 2 }') if ln.startswith('material "glass"') else ln for ln in NESTED]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_radiance_eta2"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    for name, lines in (("lamp_in_glass.ftsl", LAMP_IN_GLASS), ("lamp_in_film.ftsl", LAMP_IN_FILM),
                        ("lamp_in_stack.ftsl", LAMP_IN_STACK), ("glass_seen.ftsl", GLASS_SEEN),
                        ("tank.ftsl", TANK), ("submerged.ftsl", SUBMERGED),
                        ("nested.ftsl", NESTED), ("nested_coat.ftsl", NESTED_COAT)):
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)

    def render(tag, scene, args, device="cpu", spp=None):
        """Two seeds of one configuration; a device render must say it ran on the GPU (mode M
        logs `[gpu]` lines, modes U and S name the device they run on -- `VCM/UPS on`, `SPPM
        on` -- and the others print a `-> GPU` device line)."""
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            pfm = os.path.join(d, png[:-4] + ".pfm")
            for f in (os.path.join(d, png), pfm):       # never score an earlier run's image
                if os.path.exists(f):
                    os.remove(f)
            cmd = [a.exe, "-in", scene, "-device", device, "-seed", str(seed), "-hdr", "-no-meter",
                   "-spp", str(spp or a.spp), "-o", png] + args
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            log = p.stdout + p.stderr
            if device == "gpu" and not any(t in log for t in ("-> GPU", "[gpu]", "VCM/UPS on ",
                                                               "SPPM on ")):
                raise SystemExit("did not run on the GPU: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    def ball(imgs):
        """The tank frame's middle: the ball and the water around it."""
        h, w = imgs[0].shape
        return [im[h // 4:3 * h // 4, w // 3:2 * w // 3] for im in imgs]

    def middle(imgs):
        """The 16 x 16 pixels at the frame's centre: the lamp seen through the glass."""
        h, w = imgs[0].shape
        return [im[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] for im in imgs]

    deep = ["-max-bounce", "32"]
    modeD = ["-mode", "D"] + deep
    modeU = ["-mode", "U", "-pmradiusfrac", "0.005"] + deep
    modeM = ["-mode", "M", "-n", "20000000"]
    modeMfg = ["-mode", "M", "-n", "20000000", "-pmfg", "4"]     # at a quarter of the spp
    sppFg = max(a.spp // 4, 1)
    modeS = ["-mode", "S", "-n", "2000000"]

    lamp = "lamp_in_glass.ftsl"
    b = render("lamp_B", lamp, ["-mode", "B", "-n", "100000000"])
    r = render("lamp_R", lamp, ["-mode", "R"])
    pairs = [("lamp in glass: mode R / mode B", r, b, 0.03),
             ("lamp in glass: mode D / mode B", render("lamp_D", lamp, ["-mode", "D"]), b, 0.03)]
    coats = {}
    for scene, label in (("lamp_in_film.ftsl", "thin-film block"),
                         ("lamp_in_stack.ftsl", "multilayer block")):
        tag = scene[8:-5]
        coats[scene] = render(tag + "_R", scene, ["-mode", "R"])
        pairs.append(("lamp in a %s: mode R / mode B" % label, coats[scene],
                      render(tag + "_B", scene, ["-mode", "B", "-n", "100000000"]), 0.03))
    seen = "glass_seen.ftsl"
    ws = render("seen_W", seen, ["-mode", "W"])
    pairs.append(("lamp seen in glass: mode W / mode R, the middle", middle(ws),
                  middle(render("seen_R", seen, ["-mode", "R"])), 0.03))
    tank = "tank.ftsl"
    rt = render("tank_R", tank, ["-mode", "R"])
    mt = render("tank_M", tank, modeM)
    mft = render("tank_Mfg", tank, modeMfg, spp=sppFg)
    st = render("tank_S", tank, modeS)
    pairs += [("tank: mode M / mode R", mt, rt, 0.05),
              ("tank: mode M -pmfg 4 / mode R", mft, rt, 0.05),
              ("tank: mode S / mode R", st, rt, 0.05),
              ("tank: mode U / mode R", render("tank_U", tank, modeU), rt, 0.05),
              ("tank: mode D / mode R", render("tank_D", tank, modeD), rt, 0.05)]
    nest = "nested.ftsl"
    rn = render("nest_R", nest, ["-mode", "R"])
    pairs += [("nested: mode M / mode R", render("nest_M", nest, modeM), rn, 0.05),
              ("nested: mode S / mode R", render("nest_S", nest, modeS), rn, 0.05)]
    coat = "nested_coat.ftsl"
    rc = render("ncoat_R", coat, ["-mode", "R"])
    pairs += [("nested coat: mode M / mode R", render("ncoat_M", coat, modeM), rc, 0.05),
              ("nested coat: mode S / mode R", render("ncoat_S", coat, modeS), rc, 0.05)]
    sub = "submerged.ftsl"
    rs = render("sub_R", sub, ["-mode", "R"])
    ds = render("sub_D", sub, modeD)
    us = render("sub_U", sub, modeU)
    pairs += [("submerged lamp: mode D / mode R", ds, rs, 0.05),
              ("submerged lamp: mode U / mode R", us, rs, 0.05)]
    if not a.no_gpu:
        rg = render("lamp_Rg", lamp, ["-mode", "R"], "gpu")
        pairs += [("lamp in glass: mode R, device / host", rg, r, 0.03),
                  ("lamp in glass: -rgb / spectral, device",
                   render("lamp_rgb", lamp, ["-mode", "R", "-rgb"], "gpu"), rg, 0.03),
                  ("lamp seen in glass: mode W, device / host",
                   render("seen_Wg", seen, ["-mode", "W"], "gpu"), ws, 0.01)]
        for scene, label in (("lamp_in_film.ftsl", "thin-film block"),
                             ("lamp_in_stack.ftsl", "multilayer block")):
            tag = scene[8:-5]
            pairs.append(("lamp in a %s: mode R, device / host" % label,
                          render(tag + "_Rg", scene, ["-mode", "R"], "gpu"), coats[scene], 0.03))
        pairs += [("tank: mode M on the ball, device / host",
                   ball(render("tank_Mg", tank, modeM, "gpu")), ball(mt), 0.05),
                  ("tank: mode M -pmfg 4 on the ball, device / host",
                   ball(render("tank_Mfgg", tank, modeMfg, "gpu", spp=sppFg)), ball(mft), 0.05),
                  ("tank: mode S, device / host", render("tank_Sg", tank, modeS, "gpu"), st, 0.05),
                  ("submerged lamp: mode D, device / host",
                   render("sub_Dg", sub, modeD, "gpu"), ds, 0.05),
                  ("submerged lamp: mode U, device / host",
                   render("sub_Ug", sub, modeU, "gpu"), us, 0.05)]
    ok = True
    for name, x1, x0, tol in pairs:
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), tol)
        good = not bad.any()
        ok &= good
        print("%s: %s; worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, ratio.flat[int(z.argmax())], z.max()), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
