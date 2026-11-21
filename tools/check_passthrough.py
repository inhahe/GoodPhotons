"""Regression check: modes A and B see through a gel and through a half mirror.

    python tools/check_passthrough.py [--out png/check_passthrough] [--exe ftrace.exe] [--no-gpu] [--spp 256]
                                      [--cases gel,mirror,...]

MODEB-PASSTHROUGH. Modes A and B connect each photon vertex to the camera along a straight leg. A `filter`
gel and a `halfmirror` pass what they pass unbent, so a leg through one is a light path carrying the gel's
T(lam) or the mirror's 1 - R(lam) (camLegThrough / dCamLegThrough). Each case, the forward mode on the CPU
and the GPU against mode R on the CPU, the frame in X, Y and Z (a tile under 2 % of a channel's brightest in
both carries no verdict):

  - check_coat_modes' red gel before a lit wall (nothing below 595 nm);
  - check_coat_modes' half mirror reflecting red above 595 nm (the blue it passes);
  - a half mirror with a soft shortpass edge at 546.5 nm (the red it passes);
  - fog between the red gel and the wall (the volume connections' legs);
  - two gels one behind the other (a leg crossing two surfaces);
  - the red gel at -max-bounce 1 in both modes: the gel is a vertex of the path, so the wall behind it,
    a second, is black in both;
  - the red gel placed as a `mesh_instance` (its asset's material), which the any-hit test that skips
    gels cannot look inside: the walk of the leg's closest hits takes it;
  - a `hide_camera` area light between the gel and the wall, facing the wall: hidden from a camera ray
    only until it crosses a surface, so through the gel its black back hides the wall in both modes;
  - mode A through a physical lens (its front-element connections);
  - mode A through a thin lens, whose film is irradiance with the lens's natural vignetting where mode R's
    is radiance (THINLENS-VIGNETTING): mode A's image is divided, pixel by pixel, by its ratio to mode R
    on the scene without the gel, fitted by a polynomial in the distance from the frame's centre;
  - mode P with the gel sliding across the shutter: its blurred split draws what is seen through the gel
    on its camera side, so its forward side must not see through it too;
  - the sky behind the gel, in place of the wall: modes A and B draw the sky a camera sees in a pass of
    their own (main.cpp prepareSky, skySeenThroughXYZ), which walks through a gel or half mirror;
  - the same with an absorbing fog between the gel and the sky, which that pass takes along the ray;
  - a glow (`glow_only`) behind the gel, which the same pass draws;
  - the sky behind the gel through a physical lens, mode A (the pass's lens samples);
  - the sky behind the gel sliding across the shutter, modes B and P (addBlurredSkyRows; mode P's split
    leaves what is seen through the gel to its camera side).

Exits 1 on any failure.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import SRGB_TO_XYZ, render_seeds, tile_stats, tile_verdict  # noqa: E402
from check_coat_modes import FILTER, HALF_MIRROR, half_mirror  # noqa: E402

TOL = 0.03


def with_camera(lines, extra):
    """`lines` with `extra` added to the camera's statement."""
    return [ln.replace("film {", extra + "  film {") if ln.startswith("camera") else ln for ln in lines]


def blurred(lines):
    """`lines` with the gel sliding across a shutter."""
    return (["scene { motion_interval -0.5 0.5 }"] +
            [ln.replace("film {", "shutter_open -0.5  shutter_close 0.5  film {") if ln.startswith("camera") else
             ("group { motion { key { translate -0.05 0 0 }  key { translate 0.05 0 0 } }  %s }" % ln
              if "material redgel" in ln and ln.startswith("quad") else ln) for ln in lines])


NO_GEL = [ln for ln in FILTER if "material redgel" not in ln]
FOG = FILTER + ['medium { sigma_t 0.3  albedo 0.9  g 0.0  bounds { min -3 0 -0.4  max 3 3 1.4 } }']
BLUR = blurred(FILTER)
SKY = [ln for ln in FILTER if not ln.startswith("quad { origin -3 0 1.5")] + ["light env { spd 1.0 }"]
SKY_FOG = SKY + ['medium { sigma_t 0.5  albedo 0.0  g 0.0  bounds { min -3 0 -0.4  max 3 3 1.4 } }']
GLOW = FILTER + ['material "glow" { type diffuse  reflect 0.3  emit 0.05  glow_only on }',
                 'quad { origin -0.3 0.3 0.5  u 0 0.6 0  v 0.6 0 0  material glow }']
PHYS = "focus 4.0  lens { preset achromat  focal 50  fstop 2.8 }"
THIN = "lens 35  fstop 4  focus 4.0"
B_ARGS = ["-mode", "B", "-n", "20000000"]
INST = ([ln for ln in FILTER if not ("material redgel" in ln and ln.startswith("quad"))] +
        ['mesh_asset "pane" { file "gel_pane.obj"  material redgel }', 'mesh_instance { of "pane" }'])
PANE_OBJ = NL.join(["v -0.6 0.05 -0.5", "v 0.6 0.05 -0.5", "v 0.6 1.25 -0.5", "v -0.6 1.25 -0.5",
                    "f 1 2 3", "f 1 3 4", ""])
HIDDEN = FILTER + ['light area { origin -0.3 0.3 0.5  u 0.6 0 0  v 0 0.6 0  normal 0 0 1  spd preset:d65  '
                   'power 20  hide_camera on }']
A_ARGS = ["-mode", "A", "-n", "20000000"]
# (name, scene, the mode's arguments, both modes' arguments, the scene each image is divided by, or None)
CASES = [
    ("gel", FILTER, B_ARGS, [], None),
    ("mirror", HALF_MIRROR, B_ARGS, [], None),
    ("mir_short", half_mirror("shortpass edge=546.5 slope=1 amp=0.9"), B_ARGS, [], None),
    ("fog", FOG, B_ARGS, [], None),
    ("two_gels", FILTER + ['quad { origin -0.6 0.05 -0.3  u 1.2 0 0  v 0 1.2 0  material redgel }'], B_ARGS, [],
     None),
    ("bounce1", FILTER, B_ARGS, ["-max-bounce", "1"], None),
    ("inst_gel", INST, B_ARGS, [], None),
    ("hidden_panel", HIDDEN, B_ARGS, [], None),
    ("lens_A", with_camera(FILTER, PHYS), A_ARGS, [], None),
    ("thin_A", with_camera(FILTER, THIN), A_ARGS, [], with_camera(NO_GEL, THIN)),
    ("blur_P", BLUR, ["-mode", "P"], [], None),
    ("sky_gel", SKY, B_ARGS, [], None),
    ("sky_fog", SKY_FOG, B_ARGS, [], None),
    ("glow_gel", GLOW, B_ARGS, [], None),
    ("sky_lens_A", with_camera(SKY, PHYS), A_ARGS, [], None),
    ("sky_blur", blurred(SKY), B_ARGS, [], None),
    ("sky_blur_P", blurred(SKY), ["-mode", "P"], [], None),
]


def falloff(a, b):
    """The smooth per-pixel ratio a / b of two images of one scene (h x w): a least-squares fit of a by
    b * (c0 + c1 r^2 + c2 r^4 + c3 r^6), r a pixel's distance from the frame's centre over the half
    diagonal -- a thin lens's natural vignetting and its scale against a camera without them."""
    h, w = a.shape
    y, x = np.mgrid[0:h, 0:w]
    r2 = (((x + 0.5 - w / 2.0) ** 2 + (y + 0.5 - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)).ravel()
    basis = np.stack([np.ones_like(r2), r2, r2 ** 2, r2 ** 3], axis=1)
    c, *_ = np.linalg.lstsq(basis * b.ravel()[:, None], a.ravel(), rcond=None)
    return (basis @ c).reshape(h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_passthrough"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    a = ap.parse_args()
    names = [c[0] for c in CASES]
    want = [w for w in a.cases.split(",") if w] or names
    unknown = [w for w in want if w not in names]
    if unknown:
        ap.error("unknown case(s) %s; the cases are %s" % (", ".join(unknown), ", ".join(names)))
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gel_pane.obj"), "w", encoding="utf-8", newline="") as f:
        f.write(PANE_OBJ)
    devs = ["cpu"] if a.no_gpu else ["cpu", "gpu"]

    def render(name, lines, args, dev):
        """Each seed's render of `lines` under `args` on `dev`, in XYZ (h x w x 3)."""
        with open(os.path.join(d, name + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)
        tag = name + "_" + args[1] + ("g" if dev == "gpu" else "")
        return [x @ SRGB_TO_XYZ.T for x in render_seeds(exe, d, name + ".ftsl", tag, args, dev, a.spp, rgb=True)]

    def tiles(xs):
        return [tile_stats(xs[0][..., c], xs[1][..., c]) for c in range(3)]

    ok = True
    for name, lines, margs, both, base in CASES:
        if name not in want:
            continue
        rr = render(name, lines, ["-mode", "R"] + both, "cpu")
        rb = render(name + "_base", base, ["-mode", "R"] + both, "cpu") if base is not None else None
        for dev in devs:
            bb = render(name, lines, margs + both, dev)
            if base is not None:   # each channel's falloff, from the scene without the gel
                ab = render(name + "_base", base, margs + both, dev)
                f = np.stack([falloff(ab[0][..., c] + ab[1][..., c], rb[0][..., c] + rb[1][..., c])
                              for c in range(3)], axis=-1)
                bb = [x / f for x in bb]
            tb, tr = tiles(bb), tiles(rr)
            good, parts = True, []
            for c, cname in enumerate("XYZ"):
                ratio, z, bad = tile_verdict(tb[c], tr[c], TOL)
                good &= not bad.any()
                parts.append("%s worst x%.3g (%.1f sigma)" % (cname, ratio.flat[int(z.argmax())], z.max()))
            ok &= good
            print("%s: %s, mode %s%s / mode R: %s" % ("ok" if good else "FAIL", name, margs[1],
                  " (device)" if dev == "gpu" else "", ", ".join(parts)), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
