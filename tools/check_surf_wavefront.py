"""Regression check: GPU mode J's queued point merges render what the inline gather renders.

    python tools/check_surf_wavefront.py [--out png/check_surf_wavefront] [--exe ftrace.exe]

kBdptT queues each camera vertex's surface gather (DSurfQueue) for kWfSurfGather, which gathers a
site with a whole warp and writes it to its own slot, and kWfSurfFold adds a pixel's sites to the
film in path and site order (render_cuda.cu, SURF-WAVEFRONT). The photons, their weights and the
film's terms are the inline gather's (FTRACE_NOSURFWF=1); only the order of the sums differs. So
on _fog_cornell's surface merges (`-jbeams off`, GPU, 4 spp at 160x120) it requires:

  - the queue used, every site through it (the run's point-merge line, none inline);
  - the image equal to the inline gather's to 1e-9 of each pixel;
  - the same with the queue shrunk to 4096 requests (FTRACE_SURFWFCAP), which turns sites away
    to the inline gather -- every wave overflows;
  - and two runs with the queue byte for byte the same: the fold adds a pixel's sites in a fixed
    order, so the image does not depend on which warp gathered first.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fogbox  # noqa: E402

LINE = re.compile(r"mode J point merges: (\d+) sites gathered a warp each \(([\d.]+) a path\), "
                  r"(\d+) inline")


def load(p):
    with open(p, "rb") as f:
        f.readline()
        w, h = (int(x) for x in f.readline().split())
        s = float(f.readline())
        return np.frombuffer(f.read(w * h * 12), dtype="<f4" if s < 0 else ">f4") \
                 .reshape(h, w, 3).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_surf_wavefront"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    scene = os.path.join(ROOT, "scenes", "_fog_cornell.ftsl")
    check = fogbox.Checks(62)
    logs = []

    def run(tag, env):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", scene, "-r", "160", "120", "-mode", "J", "-device", "gpu", "-spp", "4",
               "-seed", "3", "-jbeams", "off", "-beamepochs", "2", "-hdr", "-no-meter",
               "-window-min", "-o", png]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **env))
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (load(pfm) if os.path.exists(pfm) else None), pfm

    rc0, log0, inline, _ = run("inline", {"FTRACE_NOSURFWF": "1"})
    rc1, log1, queued, pfm1 = run("queued", {})
    m = LINE.search(log1)
    check("the inline gather renders", rc0 == 0 and inline is not None, "rc %d" % rc0)
    check("the queue renders, every site through it",
          rc1 == 0 and queued is not None and m is not None and int(m.group(1)) > 0
          and m.group(3) == "0",
          ("%s sites, %s a path, %s inline" % m.groups()) if m else "no point-merge line")

    def rel(x, y):
        den = np.maximum(np.abs(x), np.abs(y))
        r = np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1.0), 0.0)
        return float(r.max())

    if inline is not None and queued is not None:
        r = rel(inline, queued)
        check("...the image the inline gather's, to 1e-9 a pixel", r < 1e-9, "max %.3g" % r)
    rc2, log2, small, _ = run("small", {"FTRACE_SURFWFCAP": "4096"})
    m2 = LINE.search(log2)
    check("a 4096-request queue turns sites away inline",
          rc2 == 0 and m2 is not None and int(m2.group(3)) > 0,
          ("%s sites, %s inline" % (m2.group(1), m2.group(3))) if m2 else "rc %d" % rc2)
    if inline is not None and small is not None:
        r = rel(inline, small)
        check("...and renders the same image", r < 1e-9, "max %.3g" % r)
    rc3, _, _, pfm3 = run("queued2", {})
    same = (rc3 == 0 and os.path.exists(pfm1) and os.path.exists(pfm3)
            and open(pfm1, "rb").read() == open(pfm3, "rb").read())
    check("two runs through the queue are byte for byte the same", same)
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
