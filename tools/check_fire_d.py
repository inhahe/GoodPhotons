"""Regression check: mode D renders a fire as mode R does.

    D:/python314/python.exe -u tools/check_fire_d.py [--out <dir>] [--exe ftrace.exe] [--spp N]

A medium with a `temperature` grid emits j(x, lambda) = Medium::emissionAt per unit length. Mode D
(BDPT) takes it as one more light with all three of an area light's strategies (bdpt.h, the fires
note): light subpaths born at points of it, the s=1 connection to a point of it, and, for s=0,
the emission each camera-walk segment gathers at stratified points -- weighted against the
connections, and whole where none reaches. Writes check_fire_r's fire (a soot `density` blob, a
hot `temperature` core; the soot scatters) lighting a diffuse floor and wall and a glossy sphere,
the fire the only light, and compares mode D with mode R (an independent estimator, itself
checked against the forward tracer by check_fire_r), two seeds each:
  1. the scene as it is, both at -heroc 4;
  2. mode D at one wavelength per path (-heroc 1) against the bundle (-heroc 4);
  3. the wall made a mirror: the fire, and the soot its glow lights, seen in it -- paths with a
     delta vertex next to the camera;
  4. a physical lens on the camera: its vertex is delta, so no light subpath reaches the film,
     and the soot seen directly has no light-tracing strategy (single wavelength);
  5. a glass pane between the fire and the camera, through the same lens: the floor in front of
     it is lit only through the glass, where no connection passes, and seen through the lens,
     where no light subpath lands -- the s=0 points the eye walk gathers past the pane are the
     only strategy left;
  6. the pane through a pinhole: there light subpaths through the pane reach the film too, and
     share those paths with the s=0 points;
  7. the fire inside a solid glass ball: every path from it crosses one interface, the camera
     side carrying radiance across it at (n_from / n_to)^2 and the light side flux
     (known-issues RADIANCE-ETA2);
  8. an area lamp beside the fire, about as bright on the frame: the one configuration in which
     mode D picks between a fire and an emitter, so the fire's share of the light pick
     (bdptPickLight, bdptPickPdf) enters every weight.
A tile fails when the two differ by more than 5 standard errors (from the seed pairs) AND by more
than 2 % (3 % behind the lens); the frame, by more than 5 standard errors AND 1 %. Exits 1 on any
failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_fire_r import SCENE, write_fire  # noqa: E402
from check_glossy_nee_media import read_pfm, tile_stats, tile_verdict  # noqa: E402

NL = chr(10)
LENS = 'lens { preset achromat  focal 50  fstop 2.8  glass BK7 }'


# A glass pane, x -2..2, y 0.002..1.5, z -1.0..-0.9, as six quads whose u x v points out.
PANE = ['material "glass" { type dielectric  ior 1.5 }'] + [
    'quad { origin %s  u %s  v %s  material glass }' % q for q in (
        ("-2 0.002 -1.0", "0 1.498 0", "4 0 0"),       # front, -z
        ("-2 0.002 -0.9", "4 0 0", "0 1.498 0"),       # back, +z
        ("-2 0.002 -1.0", "0 0 0.1", "0 1.498 0"),     # left, -x
        ("2 0.002 -1.0", "0 1.498 0", "0 0 0.1"),      # right, +x
        ("-2 0.002 -1.0", "4 0 0", "0 0 0.1"),         # bottom, -y
        ("-2 1.5 -1.0", "0 0 0.1", "4 0 0"))]          # top, +y


# The fire's box is x, z -0.5..0.5, y 0..1; its soot fills the ball inscribed in it.
BALL = ['material "glass" { type dielectric  ior 1.5 }',
        'sphere { center 0 0.5 0  radius 0.5  material glass }']
# An area lamp off to the side, about as bright on the frame as the fire (its glow is faint).
LAMP = ('light area { origin 0.9 1.6 -1.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0  power 6e-5  '
        'spd preset:d65 }')


def variants():
    """The mirror, lens, glass-pane, glass-ball and lamp variants of check_fire_r's scene
    (name -> lines)."""
    mirror = [l.replace('material "wall"  { type diffuse  reflect rgb 0.55 0.6 0.7 }',
                        'material "wall"  { type mirror  reflect 0.9 }') for l in SCENE]
    lens = [l.replace("film { res 96 72 } }", "film { res 96 72 }  %s }" % LENS) for l in SCENE]
    pane = SCENE + PANE
    pane_lens = lens + PANE
    for v in (mirror, lens):
        assert v != SCENE
    return {"fire_mirror.ftsl": mirror, "fire_lens.ftsl": lens, "fire_pane.ftsl": pane,
            "fire_pane_lens.ftsl": pane_lens, "fire_ball.ftsl": SCENE + BALL,
            "fire_lamp.ftsl": SCENE + [LAMP]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_fire_d"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    write_fire(d)
    for name, lines in variants().items():
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)

    def render(tag, args, scene="fire.ftsl"):
        imgs = []
        for seed in (1, 2):
            png = "%s_s%d.png" % (tag, seed)
            for f in (png, png[:-4] + ".pfm"):     # never score an earlier run's image
                if os.path.exists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
            cmd = [a.exe, "-in", scene, "-device", "cpu", "-seed", str(seed), "-hdr",
                   "-no-meter", "-spp", str(a.spp), "-o", png] + args
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    d4 = render("d_c4", ["-mode", "D", "-heroc", "4"])
    pairs = [("mode D / mode R", d4, render("r_c4", ["-mode", "R", "-heroc", "4"]), 0.02),
             ("mode D -heroc 4 / -heroc 1", d4, render("d_c1", ["-mode", "D", "-heroc", "1"]), 0.02)]
    for scene, label, tol in (("fire_mirror.ftsl", "a mirror wall", 0.02),
                              ("fire_lens.ftsl", "a physical lens", 0.03),
                              ("fire_pane_lens.ftsl", "a glass pane, through the lens", 0.03),
                              ("fire_pane.ftsl", "a glass pane, through a pinhole", 0.02),
                              ("fire_ball.ftsl", "inside a glass ball", 0.02),
                              ("fire_lamp.ftsl", "with an area lamp", 0.02)):
        tag = scene[:-5]
        pairs.append(("mode D / mode R, " + label, render("d_" + tag, ["-mode", "D"], scene),
                      render("r_" + tag, ["-mode", "R"], scene), tol))
    ok = True
    for name, x1, x0, tol in pairs:
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), tol)
        fr, fz, fbad = tile_verdict(tile_stats(*x1, 1), tile_stats(*x0, 1), 0.01)
        good = not bad.any() and not fbad.any()
        ok &= good
        print("%s: %s, frame x%.4f (%.1f sigma); worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, fr[0, 0], fz[0, 0], ratio.flat[int(z.argmax())], z.max()),
            flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
