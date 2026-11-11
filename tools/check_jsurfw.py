"""Regression check: the device's point-merge MIS weights agree with the host's, chromatic bins included.

    python tools/check_jsurfw.py [--out png/check_jsurfw] [--exe ftrace.exe]

Mode J's point merges (-jsurf) weigh each photon by DSurfMergeW on the device and SurfMergeWeight
on the host. Where a medium's extinction varies with wavelength, that weight's camera half
carries two rows of JSpec's bins, read at the photon's wavelength (UPBP-W-CHROMATIC), built along
the camera subpath by DChromaRun on the device and cameraSideSumsChroma on the host. No image
check can hold them (known-issues JCHROMA-JSURF-WEAK): where they matter, beams and point merges
competing for one path, the point merges carry well under a seed's scatter of the frame. So
`ftrace -checkjsurfw` holds them directly (jsurf_tap.h): kBdptT records its first samples' camera
subpaths and the halves it built on them; the host rebuilds the same halves from the same
vertices with its own code and requires the scalar fields, the bins, which vertices are merge
sites, and the weight itself at photon states and wavelengths it draws, all to agree; and, as a
negative control where the bins are on, that the host's weight read one bin off moves
somewhere, so the run had something chromatic to hold.

Three scenes: check_jchroma's box of chromatic fog, where camera paths cross the fog to the
walls; the same box around a glass ball, whose delta vertices close the connection between a
merge site and the vertex before it; and the box in a grey fog, whose bins are off, so that the
weight reads its scalar terms (etaKm1Coef, segSumM) at the camera's wavelength. Each renders one
sample a pixel in mode J on the GPU. With no CUDA device (or a build without CUDA) it is skipped,
and says so. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_jchroma import SCENE as BOX  # noqa: E402

GREY_MAT = 'material "grey" { type diffuse reflect 0.6 }'
CHROMA_FOG = "    sigma_a rgb 0.15 0.45 0.9   sigma_s 0.5   g 0.0"
assert BOX.count(GREY_MAT) == 1 and BOX.count(CHROMA_FOG) == 1 and BOX.count("light area {") == 1, \
    "check_jchroma's scene changed: rebuild the variants here"
BALL = BOX.replace(GREY_MAT, GREY_MAT + NL + 'material "glass" { type dielectric ior 1.5 }').replace(
    "light area {", "sphere { center 0.5 0.45 0.5   radius 0.2   material glass }" + NL + "light area {")
GREY = BOX.replace(CHROMA_FOG, "    sigma_t 0.95   albedo 0.55   g 0.0")
SCENES = {"box": BOX, "ball": BALL, "grey": GREY}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jsurfw"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails, skipped = [], []
    for name, text in SCENES.items():
        ftsl = os.path.join(d, name + ".ftsl")
        with open(ftsl, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-jsurf", "-spp", "1",
               "-max-bounce", "8", "-seed", "1", "-beamfreeze", "-no-meter", "-window-min",
               "-checkjsurfw", "-o", os.path.join(d, name + ".png")]
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        with open(os.path.join(d, name + ".log"), "w", encoding="utf-8") as f:
            f.write(log)
        if "no CUDA device" in log or "built without CUDA" in log:
            print("(no GPU here: the %s scene is skipped)" % name)
            skipped.append(name)
            continue
        lines = [ln for ln in log.splitlines() if ln.startswith("[checkjsurfw]")]
        for ln in lines:
            print("%-5s %s" % (name, ln))
        ok = p.returncode == 0 and "[checkjsurfw] PASS" in lines
        if not ok:
            if not lines:
                print(log[-1500:])
            fails.append("%s (exit %d)" % (name, p.returncode))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS (skipped: no GPU, nothing was held)" if len(skipped) == len(SCENES) else "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
