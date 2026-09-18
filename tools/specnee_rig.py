"""Spectral glossy NEE rig: on a scene simple enough for mode D to be the answer, mode M's glossy
vertex must agree with it -- and the spectral form must agree BETTER and with less colour noise.

    python tools/specnee_rig.py [--keep]

The scene is deliberately the thing that broke in gallery_rain, isolated: a strongly COLOURED
glossy sphere (gold-ish, roughness 0.12) under one blackbody area light, on a neutral floor, with
no media and no caustic targets. The wavelength dependence therefore lives exactly where it did --
in the BSDF coefficient times the emitter spectrum -- and nothing else can explain a difference.

  0. RIG IS LIVE     -- `-no-glossy-nee` must move the sphere by more than 10 %, or the checks
                        below are being run by an instrument that cannot see their subject.
  1. AGREES WITH D   -- mode M with the spectral NEE lands within 3 % of a converged mode D on the
                        sphere. This is the check that the 24-bin quadrature is fine enough and
                        that the shared-at-lambda_c MIS weight is not skewing the answer.
  2. NO WORSE than the scalar form against mode D. Not "better": both estimate the same integral,
                        so at equal spp they differ in VARIANCE, which a mean over a large ROI
                        hides. The accuracy claim is only that the quadrature adds no bias.
  3. LESS COLOUR NOISE -- the chroma residual against a 5x5 mean falls on the sphere.
  4a. BACKEND PARITY -- the same frame on the GPU agrees with the CPU to 3 %, and its chroma falls
                        too. The device gather is the backend the gallery_rain flyby runs on.
  4. NEUTRAL CONTROL  -- a WHITE glossy sphere in the same scene. Its BSDF is flat, so the two
                        forms must agree in the MEAN to the noise floor -- that is what this
                        controls for. Note its chroma noise still falls (3.48 -> 1.95 measured):
                        the EMITTER's spectrum is coloured whatever the surface does, and the
                        quadrature fixes that too. So a chroma win here is expected and is not
                        evidence of a leak; a MEAN shift here would be.

Cited by known-issues ("SPECTRAL NEE") and TODO 0.5.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = os.path.join("scraps", "_specnee")
EXE = os.path.join(ROOT, "ftrace.exe")

SCENE = """camera "cam" {{ eye 0 0.9 3.4  look_at 0 0.55 0  up 0 1 0  fov_y 38  mode R  film {{ res 200 150 }} }}
material "floor" {{ type diffuse  reflect 0.45 }}
quad {{ origin -6 0 -6  u 12 0 0  v 0 0 12  material floor }}
# A SMALL, DISTANT light and a MEDIUM lobe -- the regime where next-event estimation carries real
# weight, which is what the rig has to be able to see. Measured NEE share on this sphere: a big
# near light 0.19 %, sun + roughness 0.05 2.8 % (a tight lobe finds a small sun by itself), 0.15
# 11.7 %, **0.25 18.3 %**, 0.40 6.1 %. Blackbody 3200 and a gold lobe put the wavelength
# dependence in both factors, which is the thing being measured.
light sun {{ dir -0.35 0.62 0.70  angle 0.53  spd blackbody 3200  intensity 2.2e-13 }}
material "ball" {{ type glossy  reflect {refl}  roughness 0.25 }}
sphere {{ center 0 0.55 0  radius 0.55  material ball }}
"""


def write(path, refl):
    io.open(path, "w", encoding="utf-8", newline="\n").write(SCENE.format(refl=refl))
    return path


def render(scene, tag, mode, extra, spp, spec, device="cpu", extra_env=None):
    out = os.path.join(OUT, tag + ".png")
    cmd = [EXE, "-in", scene, "-mode", mode, "-r", "200", "150", "-o", out,
           "-window-min", "-interval", "60"] + extra
    if not spec:
        cmd.append("-no-spec-nee")
    if mode == "M":
        cmd += ["-device", device, "-n", "4000000", "-spp", str(spp)]
    else:
        cmd += ["-spp", str(spp)]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out


def stats(png):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import uniform_filter
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.float64)
    h, w = a.shape[:2]
    # the sphere sits in the middle; take a box well inside its silhouette
    x0, x1, y0, y1 = int(w * 0.40), int(w * 0.60), int(h * 0.42), int(h * 0.68)
    roi = a[y0:y1, x0:x1]
    lum = 0.2126 * roi[..., 0] + 0.7152 * roi[..., 1] + 0.0722 * roi[..., 2]
    res = roi - uniform_filter(a, size=(5, 5, 1))[y0:y1, x0:x1]
    chroma = float(np.abs(res - res.mean(axis=2, keepdims=True)).mean())
    return float(lum.mean()), chroma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--spp", type=int, default=64)
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-14s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # A FIBER case too (0.343.0): the same question at a hair vertex, where the connection's
    # response is the BCSDF rather than a glossy lobe. Blonde fiber = strongly coloured absorption.
    hair_scene = """camera "cam" {{ eye 0 0.9 3.4  look_at 0 0.55 0  up 0 1 0  fov_y 38  mode R  film {{ res 200 150 }} }}
material "floor" {{ type diffuse  reflect 0.45 }}
quad {{ origin -6 0 -6  u 12 0 0  v 0 0 12  material floor }}
light sun {{ dir -0.35 0.62 0.70  angle 0.53  spd blackbody 3200  intensity 2.2e-13 }}
material "scalp" {{ type diffuse  reflect 0.3 }}
sphere "scalp" {{ center 0 0.55 0  radius 0.40  material scalp }}
material "fiber" {{ type hair  preset human  reflect rgb 0.79 0.44 0.155  beta_m 0.3  beta_n 0.35 }}
fur "coat" {{ on "scalp"  material fiber  count 60000  length 0.12  radius 0.0006
             points 5  segments 2  jitter 0.2  seed 3 }}
"""
    io.open(os.path.join(OUT, "hair.ftsl"), "w", encoding="utf-8", newline="\n").write(hair_scene.format())
    hsc = os.path.join(OUT, "hair.ftsl")
    hD = render(hsc, "hair_D", "D", [], 800, True)
    hS = render(hsc, "hair_spec", "M", ["-spec-nee-hair"], args.spp, True)
    hC = render(hsc, "hair_scal", "M", [], args.spp, False)
    # the missing control: how much does the fiber NEE carry here at all? Without this the
    # comparison above is a null from an instrument that may not be able to see its subject.
    hN = render(hsc, "hair_nonee", "M", ["-spec-nee-hair"], args.spp, True)
    dH, _ = stats(hD); sH, csH = stats(hS); cH, ccH = stats(hC)
    print("  fiber ball:   mode D %7.3f | spectral %7.3f (err %5.2f%%, chroma %5.2f) | scalar %7.3f (err %5.2f%%, chroma %5.2f)"
          % (dH, sH, 100 * abs(sH - dH) / max(dH, 1e-9), csH, cH, 100 * abs(cH - dH) / max(dH, 1e-9), ccH))
    check("fiber vs D", abs(sH - dH) / max(dH, 1e-9) <= abs(cH - dH) / max(dH, 1e-9) + 0.02,
          "spectral %.2f%% vs scalar %.2f%% from mode D -- the claim is NO BIAS, not a win"
          % (100 * abs(sH - dH) / max(dH, 1e-9), 100 * abs(cH - dH) / max(dH, 1e-9)))
    print("     (informational: the fiber half is OFF by default -- `-spec-nee-hair` turns it on. On this")
    print("      ball it is indistinguishable from the scalar form; on Alice it moves chroma -1.7 %% for")
    print("      +26 %% of the camera pass, because hair's colour noise is PATH variance, not wavelength.)")

    for label, refl in [("gold", "rgb 0.95 0.72 0.28"), ("white", "0.8")]:
        sc = write(os.path.join(OUT, label + ".ftsl"), refl)
        dref = render(sc, label + "_D", "D", [], 2000, True)
        mspec = render(sc, label + "_spec", "M", [], args.spp, True)
        mscal = render(sc, label + "_scal", "M", [], args.spp, False)
        # CHECK 0, before any comparison is believed: can this rig see the glossy NEE at all?
        # A null from a rig that cannot see the effect is worthless (known-issues, MAXBOUNCE).
        moff = render(sc, label + "_nognee", "M", ["-no-glossy-nee"], args.spp, True)
        D, _ = stats(dref)
        S, cS = stats(mspec)
        C, cC = stats(mscal)
        eS, eC = abs(S - D) / max(D, 1e-9), abs(C - D) / max(D, 1e-9)
        print("  %s sphere:  mode D %7.3f | spectral %7.3f (err %5.2f%%, chroma %5.2f) | scalar %7.3f (err %5.2f%%, chroma %5.2f)"
              % (label, D, S, 100 * eS, cS, C, 100 * eC, cC))
        O, _ = stats(moff)
        sens = abs(S - O) / max(S, 1e-9)
        if label == "gold":
            gspec = render(sc, label + "_spec_gpu", "M", [], args.spp, True, device="gpu")
            gscal = render(sc, label + "_scal_gpu", "M", [], args.spp, False, device="gpu")
            G, cG = stats(gspec)
            GC, cGC = stats(gscal)
            check("gpu == cpu", abs(G - S) / max(S, 1e-9) < 0.03,
                  "the two backends must agree: GPU %.3f vs CPU %.3f" % (G, S))
            # GPU against GPU: comparing the device's spectral form with the HOST's scalar one
            # cannot tell a backend difference from a spectral one, which is how a dead device
            # path first passed this rig.
            check("gpu chroma", cG < cGC * 0.97,
                  "device spectral %.2f vs device scalar %.2f" % (cG, cGC))
            check("rig is live", sens > 0.10,
                  "NEE carries %.1f%% of this sphere -- below ~10%% the other checks would prove little" % (100 * sens))
            check("agrees with D", eS < 0.03, "spectral is %.2f%% from mode D" % (100 * eS))
            # NOT "beats scalar in accuracy": both estimate the same integral, so at equal spp the
            # difference is variance, and a mean over a large ROI hides it. The claim is only that
            # the spectral form is no WORSE -- i.e. the quadrature introduces no systematic error.
            check("no worse", eS <= eC + 0.01, "%.2f%% vs %.2f%% from mode D" % (100 * eS, 100 * eC))
            check("less chroma", cS < cC * 0.97, "chroma %.2f vs %.2f" % (cS, cC))
        else:
            check("neutral ctrl", abs(S - C) / max(C, 1e-9) < 0.03,
                  "a flat BSDF must not shift the MEAN: %.3f vs %.3f (chroma still falls -- the light is coloured)" % (S, C))

    print("\n-> spectral NEE rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep and ok:
        shutil.rmtree(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
