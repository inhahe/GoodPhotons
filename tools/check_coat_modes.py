"""Regression check: coated bodies -- modes M and S render one as a refracting body, in its own
colour, and an opaque one is no medium.

    D:/python314/python.exe -u tools/check_coat_modes.py [--out <dir>] [--exe ftrace.exe]
                                                         [--spp N] [--no-gpu]

A thin-film or multilayer body with a clear substrate is a dielectric body with a coating: it
reflects with the coating's interference reflectance -- its whole colour -- and refracts the rest
(Renderer::interfaceStep). Until 0.378.5 the camera walks of modes M and S bounced off one as off
a half mirror (the material's unset `reflect` slot), so a soap bubble or a dichroic ball rendered
grey and opaque there. This renders such a bubble and a dichroic ball, each around a diffuse core,
over a floor and in front of a wall, and requires against mode R, per tile:
  * mode M, each colour channel alike (its spectral gather re-expresses the coating's coin at
    every photon's wavelength -- Renderer::stepFactorAt);
  * mode S, the sum of the channels (mode S folds no spectral factor per photon for any
    material: known-issues SPPM-SPECTRAL);
  * unless --no-gpu, both on the device against the host.
And an OPAQUE coated ball -- a stack with an absorbing layer -- in a water tank that carries a
`priority`: it is never entered, so it is no medium, and the priority it is given cannot matter.
Mode R (host, and device unless --no-gpu) and mode D (host) must render it the same ranked below
the water as above it. (Until 0.379.1 the priority rule suppressed it when it ranked no higher than
the water, and the ball vanished.)
And three scenes where what a walk can carry depends on its wavelength, mode M (host,
and device unless --no-gpu) against mode R per colour channel -- mode M's gather re-expresses its
walk at every photon's wavelength, which a walk can do only for branches its own wavelength allows
(SpecThr::mulCoin, known-issues MODE-M-FOLD-SUPPORT):
  * that tank, framed close on a thin-film ball ranked as the water whose substrate absorbs above
    595 nm: where it is clear a ray passes straight through, where it absorbs the coat reflects.
    Mode R's tank is noisy (no light sampling reaches the lamp from under the water), so its
    reference is the device's at 8192 spp (the host's at 1024 with --no-gpu). Until 0.381.2 mode
    M read x0.50 in blue over the ball;
  * a half mirror at 45 degrees that reflects red and passes cyan, reflecting a lit white wall
    and, above the horizon, passing nothing but the dark: until 0.381.2 mode M took its coin at the
    walk's wavelength for every photon, so the reflection came out in the mirror's average colour;
  * a red gel filter, transmitting nothing below 595 nm, between the camera and a lit white wall:
    a weight, not a coin, but a walk drawn where it is zero carries nothing at any wavelength, the
    same support problem -- until 0.381.4 red behind it read at the share of walk wavelengths
    above 595 nm; and a green band-pass gel (0 below 440 and above 600 nm, 0.9 from 500 to 560, linear
    between): zero at all three wavelengths mode M's carrier once probed for flatness, so until
    0.381.7 it folded as a flat zero and the wall behind it rendered black. Its edges are ramps over
    several carrier bins, which both backends' spectral tables hold exactly (the device's are 5 nm
    apart and interpolated, so a step reaches it as a 5 nm ramp, wherever it falls); their kinks
    sit inside carrier bins, which leaves about 8 % in blue (SPECGATHER-BIN-RESOLUTION), so this
    pair allows 10 %.
  The others put their spectral edges on a boundary of mode M's 24 carrier bins (595 nm: 360 +
  12 x 470 / 24). The carrier takes each factor at its bin's centre for every photon in the bin
  (known-issues SPECGATHER-BIN-RESOLUTION); an edge inside a bin is another error, and this check
  is about the coins.
A tile fails when the two differ by more than 5 standard errors (two seeds each) AND by more
than 5 % (10 % for the green gel, above). Exits 1 on any failure.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import render_seeds, tile_stats, tile_verdict  # noqa: E402
from check_radiance_eta2 import TANK  # noqa: E402

NL = chr(10)
SCENE = [
    'camera "c" { eye 0 0.45 -2.2  look_at 0 0.35 0  up 0 1 0  fov_y 34  film { res 64 48 } }',
    'material "floor" { type diffuse  reflect 0.6 }',
    'material "wall"  { type diffuse  reflect rgb 0.7 0.6 0.5 }',
    'material "core"  { type diffuse  reflect rgb 0.3 0.5 0.8 }',
    'material "bubble" { type thinfilm  ior 1.33  film_ior 1.28  film_thickness 380 }',
    'material "dichroic" { type multilayer  ior 1.5  layer 2.3 0.0 60  layer 1.38 0.0 100'
    '  layer 2.3 0.0 60  layer 1.38 0.0 100 }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material floor }',
    'quad { origin -3 0 1.2  u 6 0 0   v 0 3 0   material wall }',
    'sphere { center -0.4 0.3 0  radius 0.28  material bubble }',
    'sphere { center -0.4 0.3 0  radius 0.14  material core }',
    'sphere { center 0.4 0.3 0  radius 0.28  material dichroic }',
    'sphere { center 0.4 0.3 0  radius 0.14  material core }',
    'light area { origin -0.5 2.0 -1.0  u 1.0 0 0  v 0 0 1.0  normal 0 -1 0  power 120  '
    'spd preset:d65 }',
]


def opaque_tank(priority):
    """check_radiance_eta2's water tank, the water ranked 1 and its ball an opaque Bragg stack
    ranked `priority`."""
    out = []
    for ln in TANK:
        if ln.startswith('material "water"'):
            out += ['material "water" { type dielectric  ior 1.333  priority 1 }',
                    'material "beetle" { type multilayer  ior 1.5  layer 1.56 0.0 120  '
                    'layer 1.0 0.5 80  layer 1.56 0.0 120  priority %d }' % priority]
            continue
        out.append(ln.replace("radius 0.25  material white", "radius 0.25  material beetle"))
    return out


def partial_tank():
    """The water tank framed close on its ball, a thin-film body ranked as the water is, on a
    substrate that absorbs above 595 nm only: where it is clear the priority rule suppresses the
    surface (inside the water it is no boundary at all), and where it absorbs the coat is opaque,
    reflecting its interference colour."""
    out = []
    for ln in TANK:
        if ln.startswith('camera'):
            out.append('camera "c" { eye 0 0.45 -1.5  look_at 0 0.4 0  up 0 1 0  fov_y 26  '
                       'film { res 64 48 } }')
        elif ln.startswith('material "water"'):
            out += ['material "water" { type dielectric  ior 1.333  priority 1 }',
                    'material "redcoat" { type thinfilm  ior 1.5  film_ior 1.38  film_thickness 300'
                    '  substrate_k table { 380:0 594.99:0 595.01:2 780:2 }  priority 1 }']
        else:
            out.append(ln.replace("radius 0.25  material white", "radius 0.25  material redcoat"))
    return out


FILTER = [
    'camera "c" { eye 0 0.6 -2.5  look_at 0 0.6 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
    'material "white" { type diffuse  reflect 0.7 }',
    'material "redgel" { type filter  transmit table { 380:0 594.99:0 595.01:0.9 780:0.9 } }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material white }',
    'quad { origin -3 0 1.5  u 6 0 0   v 0 3 0   material white }',
    'quad { origin -0.6 0.05 -0.5  u 1.2 0 0  v 0 1.2 0  material redgel }',
    'light area { origin -0.5 2.4 0.3  u 1 0 0  v 0 0 1  normal 0 -1 0  power 150  '
    'spd preset:d65 }',
]


GREEN_GEL = [ln.replace('material "redgel" { type filter  transmit table { 380:0 594.99:0 595.01:0.9 780:0.9 } }',
                        'material "redgel" { type filter  transmit table { 380:0 440:0 500:0.9 '
                        '560:0.9 600:0 780:0 } }') for ln in FILTER]
assert GREEN_GEL != FILTER


HALF_MIRROR = [
    'camera "c" { eye 0 0.6 -2.5  look_at 0 0.6 0  up 0 1 0  fov_y 30  film { res 64 48 } }',
    'material "white" { type diffuse  reflect 0.7 }',
    'material "redmirror" { type halfmirror  reflect table { 380:0.05 594.99:0.05 595.01:0.9 '
    '780:0.9 } }',
    'quad { origin -3 0 -3   u 0 0 6   v 6 0 0   material white }',
    'quad { origin 1.5 0 -3  u 0 3 0   v 0 0 6   material white }',
    'quad { origin -0.6 0.05 -0.6  u 1.2 0 1.2  v 0 1.2 0  material redmirror }',
    'light area { origin -0.5 2.4 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  power 150  '
    'spd preset:d65 }',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_coat_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "coats.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(SCENE) + NL)
    for pri in (1, 2):
        with open(os.path.join(d, "opaque_p%d.ftsl" % pri), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(opaque_tank(pri)) + NL)
    for name, lines in (("partial", partial_tank()), ("halfmirror", HALF_MIRROR), ("filter", FILTER),
                        ("greengel", GREEN_GEL)):
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)

    def render(tag, args, device="cpu"):
        """Two seeds of one configuration, as RGB images (render_seeds)."""
        return render_seeds(a.exe, d, "coats.ftsl", tag, args, device, a.spp, rgb=True)

    modeM = ["-mode", "M", "-n", "20000000"]
    modeS = ["-mode", "S", "-n", "2000000"]
    r = render("R", ["-mode", "R"])
    m = render("M", modeM)
    s = render("S", modeS)
    pairs = [("mode M / mode R, channel %d" % c, [x[..., c] for x in m], [x[..., c] for x in r])
             for c in range(3)]
    pairs.append(("mode S / mode R, luminance", [x.sum(-1) for x in s], [x.sum(-1) for x in r]))
    if not a.no_gpu:
        mg = render("Mg", modeM, "gpu")
        sg = render("Sg", modeS, "gpu")
        pairs += [("mode M device / host, channel %d" % c, [x[..., c] for x in mg],
                   [x[..., c] for x in m]) for c in range(3)]
        pairs.append(("mode S device / host, luminance", [x.sum(-1) for x in sg],
                      [x.sum(-1) for x in s]))
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]
    for mode, dev in [("R", x) for x in devs] + [("D", "cpu")]:
        tag = "opaque_%s%s" % (mode, "g" if dev == "gpu" else "")
        lo, hi = [render_seeds(a.exe, d, "opaque_p%d.ftsl" % pri, "%s_p%d" % (tag, pri),
                               ["-mode", mode], dev, a.spp) for pri in (1, 2)]
        pairs.append(("opaque coat in water, mode %s%s: ranked below / above the water" % (
            mode, " (device)" if dev == "gpu" else ""), lo, hi))
    # The green gel's ramps put kinks inside carrier bins, which the 24-bin carrier resolves to
    # about 8 % in blue (known-issues SPECGATHER-BIN-RESOLUTION); what the pair guards is the band
    # being seen at all (x0.27 / x5 without it), so it allows 10 %.
    for name, what, ref, tol in (
            ("partial", "coat opaque in the red only, in water",
             (["-mode", "R"], "cpu", 1024) if a.no_gpu else (["-mode", "R"], "gpu", 8192), 0.05),
            ("halfmirror", "half mirror reflecting red", (["-mode", "R"], "cpu", a.spp), 0.05),
            ("filter", "red gel filter", (["-mode", "R"], "cpu", a.spp), 0.05),
            ("greengel", "green band-pass gel", (["-mode", "R"], "cpu", a.spp), 0.10)):
        rr = render_seeds(a.exe, d, name + ".ftsl", name + "_R", ref[0], ref[1], ref[2], rgb=True)
        for dev in devs:
            mm = render_seeds(a.exe, d, name + ".ftsl", name + "_M" + ("g" if dev == "gpu" else ""),
                              modeM, dev, a.spp, rgb=True)
            pairs += [("%s, mode M%s / mode R, channel %d" % (
                what, " (device)" if dev == "gpu" else "", c), [x[..., c] for x in mm],
                [x[..., c] for x in rr], tol) for c in range(3)]
    ok = True
    for pair in pairs:
        name, x1, x0 = pair[:3]
        tol = pair[3] if len(pair) > 3 else 0.05
        ratio, z, bad = tile_verdict(tile_stats(*x1), tile_stats(*x0), tol)
        good = not bad.any()
        ok &= good
        print("%s: %s; worst tile x%.4f (%.1f sigma)" % (
            "ok" if good else "FAIL", name, ratio.flat[int(z.argmax())], z.max()), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
