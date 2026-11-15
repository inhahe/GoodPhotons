"""Regression check: a merge keeps the photons of its own surface, not those a millimetre under it.

    python tools/check_gap_merge.py [--out png/check_gap_merge] [--exe ftrace.exe] [--no-gpu]

J-THIN-GAP-LEAK. Mode J's surface merges and mode U's vertex merges kept every photon within their
radius. So the light in a gap a millimetre under a plate -- photons on the floor below it, which
arrived from above -- was merged at the plate's top, where none of it can reach:
`scenes/_gap_leak.ftsl` (a plate that emits down into the gap) read mode J 167 times mode B, and
`scenes/_gap_leak2.ftsl` (a light block in the gap) 1.21 times. Since 0.445.3 they keep only their
own surface's photons and divide by that surface's share of the disc (the footprint, surfmerge.h).

Modes J, U and D, CPU and GPU, against mode B: the mean over the middle of the plate's top (lit by
a ceiling panel, and by the gap's light that escapes through its millimetre-high sides and comes
back off the room) within 3 %. Mode B is the reference because light leaving a slit a millimetre
high is what a light tracer finds and a camera path almost never does (mode D, whose light
subpaths find it as mode B does, agrees with it; mode R falls short at any practical count).

The merges run at a radius of 0.02 m, not the default (0.077 m here). On the floor just outside the
plate a merge's disc reaches under it, into the gap's trapped light, hundreds of times denser -- the
same surface, so no rule of the band's kind can tell the two apart -- and the camera paths that
bounce off that rim carry the excess everywhere: lit by the escaped light alone, the room reads 3.6
times mode B in host J at the default radius, 1.09 at 0.02, 1.03 at 0.005
(`scenes/_gap_leak_noceil.ftsl`; with the footprint 3.59 in host J). That is the merges' radius bias
at the gap's mouth (known-issues GAP-MERGE-RADIUS), not the leak this checks. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import read_pfm  # noqa: E402

SCENES = ["_gap_leak", "_gap_leak2"]
SPP = {"J": 1024, "U": 1024, "D": 256}   # CPU; the GPU takes 4x
NPH = 200_000_000                       # mode B's photons
TOL = 0.03
RADIUS = "0.02"                         # modes J and U's merge radius (-pmradius; see above)


def render(exe, scn, mode, dev, out):
    pfm = out[:-4] + ".pfm"
    for f in (out, pfm):
        if os.path.exists(f):
            os.remove(f)
    budget = ["-n", str(NPH)] if mode == "B" else ["-spp", str(SPP[mode] * (4 if dev == "gpu" else 1))]
    if mode in ("J", "U"):
        budget += ["-pmradius", RADIUS]
    cmd = [exe, "-in", scn, "-mode", mode, "-device", dev, "-max-bounce", "8",
           "-seed", "1", "-hdr", "-no-meter", "-window-min", "-o", out] + budget
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print((p.stdout + p.stderr)[-1500:])
        raise SystemExit("render failed: " + " ".join(cmd))
    img = read_pfm(pfm, top_down=True)[..., 1]
    h, w = img.shape
    return float(img[38 * h // 72:54 * h // 72, 30 * w // 96:64 * w // 96].mean())   # the plate's top


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gap_merge"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for name in SCENES:
        scn = os.path.join(ROOT, "scenes", name + ".ftsl")
        ref = render(exe, scn, "B", "gpu" if not a.no_gpu else "cpu", os.path.join(out, name + "_B.png"))
        for mode in ("J", "U", "D"):
            for dev in ["cpu"] + ([] if a.no_gpu else ["gpu"]):
                v = render(exe, scn, mode, dev, os.path.join(out, "%s_%s_%s.png" % (name, mode, dev)))
                r = v / ref
                good = abs(r - 1.0) <= TOL
                ok = ok and good
                print("%s: %s, mode %s on the %s: the plate's top %.4f of mode B's"
                      % ("ok" if good else "FAIL", name, mode, "GPU" if dev == "gpu" else "CPU", r),
                      flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
