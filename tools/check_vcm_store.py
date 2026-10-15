"""Regression check: GPU mode U renders a large frame as bright as a small one.

    D:/python314/python.exe -u tools/check_vcm_store.py [--out <dir>] [--exe ftrace.exe]

The device VCM stores every light subpath's vertices for merging and for the camera's connections
in one light-vertex store, which has a memory budget. Past it -- a large frame at a deep
-max-bounce -- the store used to keep only the first few vertices of each subpath, while every
strategy's MIS weight still counted the merges and connections at the rest, so the deepest paths
were under-weighted (known-issues VCM-SLAB-CAP): a 1600x1600 frame of a closed box read x0.989 of
the same view at 160x160. Since 0.379.3 a pass that would not fit traces fewer light subpaths
instead, each shared by several camera paths, and stores all of their vertices.

This renders check_bdpt_depth's box (diffuse walls, a ceiling-sized lamp) at -max-bounce 8 on the
GPU, with the store's budget held at 512 MB (FTRACE_VCM_BUDGET_MB), at 160x160 (one light subpath
per pixel) and at 1600x1600 (past the budget), four seeds each,
with the merge radius held at its start (-vcmalpha 1: the two run different numbers of passes, and
a shrinking radius would leave them different merge biases), and requires the frame means to
agree: it fails when they differ by more than 5 standard errors (from the seed-to-seed spread of
each mean) AND by more than 0.3 %. Exits 1 on failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm  # noqa: E402
from check_bdpt_depth import BOX  # noqa: E402

NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vcm_store"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "box.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(BOX) + NL)
    # A 512 MB store, whatever the card: 160x160 fits in it (~75 MB) and 1600x1600 (~7 GB) does
    # not, so the check exercises the pairing on any GPU.
    env = dict(os.environ)
    env["FTRACE_VCM_BUDGET_MB"] = "512"
    res = {}
    for r, spp in ((160, 256), (1600, 4)):
        means = []
        for seed in (1, 2, 3, 4):
            png = os.path.join(d, "u_r%d_s%d.png" % (r, seed))
            pfm = png[:-4] + ".pfm"
            if os.path.exists(pfm):
                os.remove(pfm)                  # never score an earlier run's image
            p = subprocess.run([exe, "-in", "box.ftsl", "-mode", "U", "-device", "gpu",
                                "-max-bounce", "8", "-vcmalpha", "1", "-r", str(r),
                                "-spp", str(spp),
                                "-seed", str(seed), "-hdr", "-no-meter", "-o", png],
                               cwd=d, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=env)
            log = p.stdout + p.stderr
            if p.returncode != 0 or not os.path.exists(pfm):
                print(log[-1500:])
                raise SystemExit("render failed at %d" % r)
            if "VCM/UPS on " not in log:
                raise SystemExit("did not run on the GPU at %d" % r)
            # The large frame must actually overflow the store and the small one fit it, or the
            # comparison tests nothing (a card with room for 1600x1600 would pass any pairing).
            shared = [ln.strip() for ln in log.splitlines()
                      if ln.startswith("[mode U]") and "light subpaths a pass" in ln]
            if (r == 1600) != bool(shared):
                raise SystemExit("at %dx%d the light-vertex store %s its budget; the check needs "
                                 "it to overflow at 1600 only" % (r, r, "overflowed" if shared
                                                                  else "fit"))
            if seed == 1 and shared:
                print("  %dx%d: %s" % (r, r, shared[0]), flush=True)
            means.append(read_pfm(pfm).sum(-1).mean())
        m = np.array(means)
        res[r] = (m.mean(), m.std(ddof=1) / np.sqrt(len(m)))
    (a1, s1), (a0, s0) = res[1600], res[160]
    ratio, z = a1 / a0, abs(a1 - a0) / max(np.hypot(s1, s0), 1e-30)
    good = not (z > 5.0 and abs(ratio - 1.0) > 0.003)
    print("%s: 1600x1600 / 160x160 frame mean x%.4f (%.1f sigma)" % ("ok" if good else "FAIL",
                                                                    ratio, z))
    print("PASS" if good else "FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
