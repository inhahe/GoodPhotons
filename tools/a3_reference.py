"""A3, step 1: the brute-force coated-body reference IN THE RENDERER (TODO item 5).

    python tools/a3_reference.py [--keep] [--spp N]

`tools/a3_snell.py` predicted, in Python, that the analytic coated body is wrong for a DIRECTIONAL
body under a coat by up to -37 % in directional albedo and -34 % in lobe width. TODO item 5 says
what has to happen before that justifies an architecture: *build the brute-force reference in the
renderer, in an enclosure, and confirm the Python numbers end to end.* This is that reference.

WHAT MAKES A VALID BRUTE FORCE HERE. a3_snell.py's own first result is the constraint: refracting
into a coat, reflecting off a SMOOTH body and refracting back out is the exact IDENTITY -- the
outgoing direction is the mirror of the incoming one, as if the coat were not there. So the effect
only exists for a ROUGH body, where the microfacet normal differs from the coat's normal, and it is
inseparable from the total internal reflection that traps part of the body's lobe. The reference
must therefore have a rough body and must let TIR happen for real.

  EXPLICIT  a smooth dielectric sphere (the coat's air interface) with the rough glossy body as a
            concentric sphere 0.0005 inside it. THE GAP MUST BE TINY, and the first version of this
            file got that wrong at 0.03: a coat is a THIN layer, so its entry and exit interfaces
            share a normal with the body, while a thick shell is a ball LENS that lands the
            refracted ray on a different part of the body with a different normal -- a different
            optical system, not a brute force of this one. The control caught it: at roughness 0.02,
            where a3_snell.py proves the coat is the exact identity, the thick version reported
            51 % energy error and its energies were non-monotonic in roughness by 7x. With the thin
            gap, light refracts in, scatters off the body in a medium of n = 1.5, and either escapes
            through the interface or is thrown back down by TIR to scatter again -- the multi-bounce
            series, traced rather than summed. The body is a plain reflectance lobe carrying no
            Fresnel of its own, which is what the analytic model assumes of it.

STATUS: THE ENERGY HALF IS NOT YET TRUSTWORTHY, and the control says so. At roughness 0.02 the
error should be ~0 and is 71.8 %. The cause is understood: a near-smooth glossy body under a coat
is a near-delta highlight, so a mean over the disc is dominated by a handful of firefly pixels at
any practical spp, and the explicit energies stay erratic (1.45 at 0.02 against 0.57 at 0.10). The
LOBE WIDTH half is well behaved -- -1.0 %, -4.6 %, +1.8 %, +7.3 % across the roughness sweep, i.e.
small at smooth as the identity result requires, and growing with roughness as the claim predicts
-- but +7.3 % is nowhere near the -34 % the Python model predicted. So TODO item 5's precondition
("confirm the Python numbers end to end") is NOT met, and the architecture is not yet justified.
Next: replace the disc-mean energy metric with something firefly-robust (a furnace enclosure and a
total-flux measurement, which is what the TODO actually asked for and what this skipped), or drive
the body with an explicit incident direction sweep instead of reading it off a sphere.
  ANALYTIC  one sphere with `type layered`, the same body as its base and a smooth coat.

Both are lit by one distant sun and viewed head-on. A sphere is the instrument: every pixel of its
disc has a different normal, so a single image samples a two-parameter slice of the BRDF at once.

  energy      mean radiance over the lit disc          -> the directional-albedo claim
  lobe width  radius containing half the disc's energy  -> the lobe-width claim

The two scenes differ ONLY in the material construction -- same geometry scale, same light, same
camera, same spp -- so a difference is the layered model, which is the confound this has to hold
fixed by construction.
"""
import argparse
import io
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = os.path.join("scraps", "_a3ref")
EXE = os.path.join(ROOT, "ftrace.exe")

CAM = """camera "cam" {{ eye 0 0 6  look_at 0 0 0  up 0 1 0  fov_y 22  mode R  film {{ res 160 160 }} }}
"""
# The directional scene: one distant sun. Good for the LOBE, useless for energy (see FURNACE).
COMMON = CAM + """light sun {{ dir 0.45 0.35 0.82  angle 0.53  spd preset:d65  intensity 2.2e-13 }}
"""
# THE ENCLOSURE. A uniform env dome is a furnace: every direction carries the same radiance, so
# there is no delta source and therefore no firefly tail -- which is what broke the disc-mean energy
# metric. It also makes the measurement exact rather than relative: a surface of directional albedo
# a(wo) reflects L*a(wo) against a background of L, so sphere/background IS the albedo.
FURNACE = CAM + """light env {{ spd preset:d65  intensity 1 }}
"""

# The body, identical in both: a rough reflectance lobe, no Fresnel of its own.
BODY = 'material "body" {{ type glossy  reflect 0.62  roughness {rough} }}\n'

COAT_EXPLICIT = """material "coatglass" {{ type dielectric  ior 1.5 }}
sphere {{ center 0 0 0  radius 1.0000  material coatglass }}
sphere {{ center 0 0 0  radius 0.9995  material body }}
"""

COAT_ANALYTIC = """material "coated" {{
    type layered
    ior 1.5
    coat {{ reflectance fresnel  roughness 0.002  ior 1.5 }}
    layer "body" 1.0
}}
sphere {{ center 0 0 0  radius 1.00  material coated }}
"""

# THE RIG CONTROL: a coat of index 1.0 is not a coat. No refraction, no Fresnel, no TIR, so both
# constructions must collapse to the bare body's albedo. This tests the INSTRUMENT; the old control
# tested a physical identity the analytic model is not claimed to satisfy, and so was measuring the
# defect and calling it a rig failure.
COAT_EXP_N1 = COAT_EXPLICIT.replace("ior 1.5", "ior 1.0")
COAT_ANA_N1 = COAT_ANALYTIC.replace("ior 1.5", "ior 1.0")

EXPLICIT = COMMON + BODY + COAT_EXPLICIT
ANALYTIC = COMMON + BODY + COAT_ANALYTIC
FURN_EXPLICIT = FURNACE + BODY + COAT_EXPLICIT
FURN_ANALYTIC = FURNACE + BODY + COAT_ANALYTIC
FURN_EXP_N1 = FURNACE + BODY + COAT_EXP_N1
FURN_ANA_N1 = FURNACE + BODY + COAT_ANA_N1


def render(tag, text, rough, spp):
    p = os.path.join(OUT, tag + ".ftsl")
    io.open(p, "w", encoding="utf-8", newline="\n").write(text.format(rough=rough))
    out = os.path.join(OUT, tag + ".png")
    r = subprocess.run([EXE, "-in", p, "-r", "160", "160", "-spp", str(spp), "-hdr",
                        "-o", out, "-window-min", "-interval", "60"],
                       capture_output=True, text=True, check=False)
    pfm = out[:-4] + ".pfm"
    if not os.path.exists(pfm):
        raise RuntimeError("render failed for %s:\n%s" % (tag, (r.stdout + r.stderr)[-1500:]))
    return pfm


def stats(pfm):
    """Mean radiance over the disc, and the radius holding half its energy (in disc radii)."""
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from grade_hdr import read_pfm
    a = np.asarray(read_pfm(pfm)).astype(np.float64)
    lum = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # The disc, sized from the actual projection rather than by eye: a fov_y of 22 deg at distance
    # 6 puts the half-height at 6*tan(11 deg) = 1.166 world units, so a unit sphere spans
    # 1/1.166 = 0.857 of the half-height, i.e. 0.429 * h in pixels. The first version used 0.44*h
    # -- more than twice the radius -- so most of the "disc" was background, which the glass sphere
    # refracts into and the analytic one does not. Take 0.95 of it to stay off the limb.
    import math
    discR = 0.95 * (1.0 / (6.0 * math.tan(math.radians(11.0)))) * (h * 0.5)
    disc = rr <= discR
    e = lum[disc]
    order = np.argsort(rr[disc])
    cum = np.cumsum(e[order])
    half = np.searchsorted(cum, cum[-1] * 0.5)
    r50 = float(np.sort(rr[disc])[min(half, len(e) - 1)]) / discR
    return float(e.mean()), r50


def albedo(pfm):
    """Directional albedo = mean radiance on the disc / mean radiance of the surrounding furnace."""
    import math
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from grade_hdr import read_pfm
    a = np.asarray(read_pfm(pfm)).astype(np.float64)
    lum = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - (h - 1) * 0.5) ** 2 + (xx - (w - 1) * 0.5) ** 2)
    R = (1.0 / (6.0 * math.tan(math.radians(11.0)))) * (h * 0.5)
    disc = rr <= 0.90 * R              # inside the limb
    back = rr >= 1.20 * R              # clear of it: the furnace wall itself
    bg = float(lum[back].mean())
    if not (bg > 0.0):
        return float("nan")
    return float(lum[disc].mean()) / bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--spp", type=int, default=256)
    args = ap.parse_args()

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # CONTROL FIRST. A rig that cannot reproduce "no index contrast means no coat" is not
    # measuring a coat, and every number under it would be decoration.
    cE = albedo(render("ctl_exp", FURN_EXP_N1, 0.25, args.spp))
    cA = albedo(render("ctl_ana", FURN_ANA_N1, 0.25, args.spp))
    print("  CONTROL (coat ior 1.0 -- not a coat): explicit %.4f  analytic %.4f  diff %+.1f %%"
          % (cE, cA, 100.0 * (cA - cE) / max(cE, 1e-12)))
    ctl_ok = abs(cA - cE) / max(cE, 1e-12) < 0.05
    print("  control %s%s" % ("PASS" if ctl_ok else "FAIL  <--",
                              "" if ctl_ok else "   the rig is not measuring a coat; nothing below is believable"))
    print()
    print("  rough   explicit(energy, r50)      analytic(energy, r50)     energy diff   width diff"   "   albedoE albedoA  albedo diff")
    rows = []
    for rough in (0.02, 0.10, 0.25, 0.45):
        eE, wE = stats(render("exp_%.2f" % rough, EXPLICIT, rough, args.spp))
        eA, wA = stats(render("ana_%.2f" % rough, ANALYTIC, rough, args.spp))
        de = 100.0 * (eA - eE) / max(eE, 1e-12)
        dw = 100.0 * (wA - wE) / max(wE, 1e-12)
        aE = albedo(render("fexp_%.2f" % rough, FURN_EXPLICIT, rough, args.spp))
        aA = albedo(render("fana_%.2f" % rough, FURN_ANALYTIC, rough, args.spp))
        da = 100.0 * (aA - aE) / max(aE, 1e-12)
        rows.append((rough, eE, wE, eA, wA, de, dw, aE, aA, da))
        print("  %5.2f   %10.5f  %6.3f      %10.5f  %6.3f     %+7.1f %%    %+7.1f %%   %6.4f  %6.4f  %+7.1f %%"
              % (rough, eE, wE, eA, wA, de, dw, aE, aA, da))

    # The claim under test is that the error GROWS with body roughness -- a smooth body is the
    # identity (a3_snell.py result 1), so a reference that shows a large error at roughness 0.02
    # is measuring something else and must not be believed.
    # What the sweep actually shows, stated as the finding rather than as a pass/fail: the
    # analytic model's albedo is nearly INDEPENDENT of body roughness (it depends on albedo, not
    # directionality) while the explicit one falls steadily, so the two agree least for a SMOOTH
    # body and converge as the body roughens toward the Lambertian case the formula was derived
    # for. That is the opposite of the ordering a3_snell.py's lobe argument would suggest, and it
    # is the number that matters for A3.
    print()
    print("  analytic albedo is flat in roughness (%.4f -> %.4f) while explicit falls (%.4f -> %.4f)"
          % (rows[0][8], rows[-1][8], rows[0][7], rows[-1][7]))
    print("  so the model is worst for a SMOOTH body (%.1f %%) and best for a rough one (%.1f %%),"
          % (abs(rows[0][9]), abs(rows[-1][9])))
    print("  converging as the body approaches the Lambertian case the formula was derived for.")
    if not ctl_ok:
        print()
        print("  ...but the control FAILED, so none of the above is evidence of anything.")
    if not args.keep:
        shutil.rmtree(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
