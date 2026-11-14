"""Regression check: modes J and U read the floor at the base of a box standing on it as mode R does.

    python tools/check_merge_base.py [--out png/check_merge_base] [--exe ftrace.exe] [--no-gpu]

Mode J's surface merges and mode U's vertex merges divide the photons in their disc by the whole
disc's area: no footprint says how much of it is the merge's own surface (modes M and S have one,
tools/check_gather_base.py). So at the base of a box the photons of the box's top stand in for the
floor the box covers, and the estimate keeps only the radius's own bias (below). A rule that drops
another surface's photons darkens it: with the tangent-plane band of mode M's gathers
(0.442.2-0.443.0, the fix J-THIN-GAP-LEAK tried) the floor within 2 cm of the base read 0.64 of mode
R in mode J and 0.65 in mode U at radius 0.1, 0.72 / 0.71 at the default (gatherplane.h says why the
rule is M's and S's).

check_gather_base's diffuse box (30 cm square, 3 cm tall) on its lit floor, seen from straight
above: modes J and U on the CPU, J on the GPU, at a pinned radius of 0.1, against mode R on the
GPU, in rings out from the base -- 0-2 cm within 15 %, 2-5 cm within 10 %. They read 0.950 / 0.941
(J, CPU), 0.926 / 0.929 (U), 0.881 / 0.924 (J, GPU) in 0.443.1: the merges' own radius bias at a
contact, which only a footprint would take away (known-issues J-THIN-GAP-LEAK); the band read
0.639 / 0.821, 0.650 / 0.847 and 0.664 / 0.883. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_gather_base as g  # noqa: E402
from pfm import read_pfm  # noqa: E402

RINGS = ((0.0, 0.02, 0.15), (0.02, 0.05, 0.10))   # (from, to, tolerance), metres from the base
CASES = [("J_cpu", ["-mode", "J", "-device", "cpu", "-spp", "128", "-pmradius", "0.1"]),
         ("U_cpu", ["-mode", "U", "-device", "cpu", "-spp", "128", "-pmradius", "0.1"]),
         ("J_gpu", ["-mode", "J", "-device", "gpu", "-spp", "512", "-pmradius", "0.1"])]
REF = ["-mode", "R", "-device", "gpu", "-spp", "2048"]


def render(exe, scn, out, tag, args):
    png = os.path.join(out, tag + ".png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        if os.path.exists(f):
            os.remove(f)
    cmd = [exe, "-in", scn, "-hdr", "-no-meter", "-seed", "1", "-window-min", "-o", png] + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    return read_pfm(pfm, top_down=True)[..., 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_merge_base"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    lines, cam, distf, _rings, _base, _against = g.OBJECTS["diffuse"]
    scn = os.path.join(out, "box.ftsl")
    with open(scn, "w", encoding="utf-8", newline="") as f:
        f.write(g.scene(lines, cam))
    dist = distf()
    R = render(exe, scn, out, "R_gpu", REF)
    ok = True
    for tag, args in CASES:
        if a.no_gpu and tag.endswith("gpu"):
            continue
        M = render(exe, scn, out, tag, args)
        for lo, hi, tol in RINGS:
            m = (dist >= lo) & (dist < hi)
            q = float(M[m].mean() / R[m].mean())
            good = abs(q - 1.0) <= tol
            ok = ok and good
            print("%s: mode %s on the %s, the floor %d-%d cm from the box's base: %.4f of mode R"
                  % ("ok" if good else "FAIL", tag[0], "GPU" if tag.endswith("gpu") else "CPU",
                     round(lo * 100), round(hi * 100), q), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
