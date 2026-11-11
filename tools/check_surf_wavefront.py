"""Regression check: GPU mode J's queued point merges render what the inline gather renders.

    python tools/check_surf_wavefront.py [--out png/check_surf_wavefront] [--exe ftrace.exe]

kBdptT queues each camera vertex's surface gather (DSurfQueue) for kWfSurfGather, which gathers a
site with a whole warp and adds it to the film (render_cuda.cu, SURF-WAVEFRONT). The photons,
their weights and the film's terms are the inline gather's (FTRACE_NOSURFWF=1); only the order of
the sums differs. So it requires, against the inline gather's image of the same scene:

  - on _fog_cornell's surface merges (`-jbeams off`, 4 spp at 160x120): every site through the
    queue (the run's point-merge line, none inline) and the image the inline gather's to 1e-9 of
    each pixel; the inline run itself printing no point-merge line, i.e. bypassing the queue;
  - the same with the queue shrunk to 4096 requests (FTRACE_SURFWFCAP), which turns sites away to
    the inline gather -- every wave overflows;
  - in tools/check_jchroma.py's box of chromatic fog (beams and point merges, 8 spp at 64x64),
    where a site carries a JSPEC carrier and chromatic weights' bins in the queue's side tables:
    every site through the queue -- a site whose table the queue lacked would be counted inline --
    and the image the inline gather's to 1e-9.

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
from check_jchroma import SCENE as CHROMA_SCENE  # noqa: E402

LINE = re.compile(r"mode J point merges: (\d+) sites gathered a warp each \(([\d.]+) a path\) "
                  r"in [\d.]+ s of GPU time, (\d+) inline")


def load(p):
    with open(p, "rb") as f:
        f.readline()
        w, h = (int(x) for x in f.readline().split())
        s = float(f.readline())
        return np.frombuffer(f.read(w * h * 12), dtype="<f4" if s < 0 else ">f4") \
                 .reshape(h, w, 3).astype(np.float64)


def rel(x, y):
    den = np.maximum(np.abs(x), np.abs(y))
    r = np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1.0), 0.0)
    return float(r.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_surf_wavefront"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fog = os.path.join(ROOT, "scenes", "_fog_cornell.ftsl")
    chroma = os.path.join(d, "chroma.ftsl")
    with open(chroma, "w", encoding="utf-8", newline="") as f:
        f.write(CHROMA_SCENE)
    FOG = [fog, "-r", "160", "120", "-spp", "4", "-jbeams", "off"]
    CHROMA = [chroma, "-r", "64", "64", "-spp", "8", "-max-bounce", "8", "-jsurf"]
    check = fogbox.Checks(62)
    logs = []

    def run(tag, args, env):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in"] + args + ["-mode", "J", "-device", "gpu", "-seed", "3",
                                     "-beamepochs", "2", "-hdr", "-no-meter", "-window-min",
                                     "-o", png]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **env))
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (load(pfm) if os.path.exists(pfm) else None)

    def pair(name, args):
        """The inline gather's image and the queue's; checks both ran as asked. Returns them."""
        rc0, log0, inline = run(name + "_inline", args, {"FTRACE_NOSURFWF": "1"})
        rc1, log1, queued = run(name + "_queued", args, {})
        m = LINE.search(log1)
        check(name + ": the inline gather renders, bypassing the queue",
              rc0 == 0 and inline is not None and LINE.search(log0) is None,
              "rc %d%s" % (rc0, ", a point-merge line" if LINE.search(log0) else ""))
        check(name + ": the queue renders, every site through it",
              rc1 == 0 and queued is not None and m is not None and int(m.group(1)) > 0
              and m.group(3) == "0",
              ("%s sites, %s a path, %s inline" % m.groups()) if m else "rc %d, no point-merge line" % rc1)
        if inline is not None and queued is not None:
            r = rel(inline, queued)
            check(name + ": ...the image the inline gather's, to 1e-9 a pixel", r < 1e-9,
                  "max %.3g" % r)
        return inline

    inline = pair("fog", FOG)
    rc2, log2, small = run("fog_small", FOG, {"FTRACE_SURFWFCAP": "4096"})
    m2 = LINE.search(log2)
    check("fog: a 4096-request queue turns sites away inline",
          rc2 == 0 and m2 is not None and int(m2.group(3)) > 0,
          ("%s sites, %s inline" % (m2.group(1), m2.group(3))) if m2 else "rc %d" % rc2)
    if inline is not None and small is not None:
        r = rel(inline, small)
        check("fog: ...and renders the same image", r < 1e-9, "max %.3g" % r)
    pair("chroma", CHROMA)
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
