"""Regression check: GPU mode J's wavefront resume passes queue every beam candidate exactly once.

    python tools/check_wf_resume.py [--out png/check_wf_resume] [--exe ftrace.exe] [--cap 4096]

kWfBeamHits queues each camera segment's beam candidates for kWfBeamEval. A segment the full hit
queue turns away records how many it had queued, and a resume pass walks it again skipping
those (render_cuda.cu, DWfQueue::resume). Every candidate is evaluated once, in kWfBeamEval,
with random numbers keyed by its (segment, beam) pair, so how the candidates are split among
passes changes only the order of the film's atomic sums. So the image rendered with the hit
queue shrunk to --cap (FTRACE_WFHITCAP), which forces resume passes on every wave (~300 at
4096, most segments resumed again and again), must match the default image up to that order.
A candidate skipped or queued twice would move the merges.

The scene is a heterogeneous fog in a closed box (its transmittance estimates draw from each
candidate's keyed stream) under one light-side realization (-beamfreeze: on the device the
refresh otherwise ends on the clock, and the slower capped run would meet other maps). The
default render is made twice, as the control that the comparison can hold at all; each render
must report how many resume passes it took: none by default, some under the cap. A render that
outruns --limit fails the check: passes that re-queued what they should skip would never drain
the queue. It is asked to stop (`ftrace -stop`), which a chunk in flight does not answer, so
such a render is left running, and its pid reported.

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

SCENE = NL.join([
    "scene { units meters }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium {",
    "    sigma_a 0.3   sigma_s 1.2   g 0.3",
    "    bounds { min 0 0 0   max 1 1 1 }",
    '    density "0.4 + 0.6*saturate(1.2 - 2.0*sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2))"',
    "}",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    'camera "cam" {',
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 48 48 }",
    "}"]) + NL


def load(p):
    with open(p, "rb") as f:
        f.readline()
        w, h = (int(x) for x in f.readline().split())
        s = float(f.readline())
        return np.frombuffer(f.read(w * h * 12), dtype="<f4" if s < 0 else ">f4") \
                 .reshape(h, w, 3).astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_wf_resume"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=300, help="seconds a render may take")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "fog.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    fails = []

    def check(name, ok, detail=""):
        print("%-56s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    def render(tag, cap=None):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
            if os.path.exists(stale):
                os.remove(stale)
        env = dict(os.environ)
        env.pop("FTRACE_WFHITCAP", None)
        env.pop("FTRACE_NOWAVEFRONT", None)
        if cap:
            env["FTRACE_WFHITCAP"] = str(cap)
        cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-spp", "8", "-seed", "1",
               "-max-bounce", "8", "-beamfreeze", "-nojsurf", "-no-meter", "-hdr",
               "-window-min", "-o", png]
        logf = png[:-4] + ".log"
        with open(logf, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, cwd=d, env=env, stdout=lf, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=a.limit)
            except subprocess.TimeoutExpired:
                # Never killed: a CUDA process killed mid-kernel can wedge the driver. A chunk
                # in flight does not answer -stop (gpuSppChunks), so passes that never drain
                # leave the process running; say so rather than wait on it.
                s = subprocess.run([exe, "-stop", str(p.pid)], capture_output=True)
                if s.returncode == 0:
                    p.wait()
                raise SystemExit("FAILED: %s ran past %d s (passes that never drain the queue?)%s"
                                 % (tag, a.limit, "" if s.returncode == 0 else
                                    "; pid %d ignores -stop, mid-chunk" % p.pid))
        with open(logf, encoding="utf-8", errors="replace") as lf:
            log = lf.read()
        if rc != 0:
            print(log[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        m = re.search(r"(\d+) resume passes for (\d+) segments", log)
        if not m:
            print(log[-1500:])
            raise SystemExit("no wavefront report in the log: is mode J gathering on the GPU?")
        return load(png[:-4] + ".pfm"), int(m.group(1)), int(m.group(2))

    a1, r1, _ = render("default1")
    a2, _, _ = render("default2")
    b, rb, sb = render("capped", a.cap)
    check("the default render takes no resume pass", r1 == 0, "%d" % r1)
    check("the capped one takes some", rb > 0, "%d passes, %d segments" % (rb, sb))
    check("the image is not empty", a1.mean() > 0.0, "mean %.4g" % a1.mean())

    def diff(x, y):
        rel = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-12)
        return rel.max(), int((x != y).sum())

    c_rel, c_n = diff(a1, a2)
    check("two default renders agree (the control)", c_rel < 1e-5,
          "largest change %.2g of the pixel, %d values differ" % (c_rel, c_n))
    rel, n = diff(a1, b)
    check("the capped render agrees with the default", rel < 1e-5,
          "largest change %.2g of the pixel, %d values differ; mean ratio %.6f"
          % (rel, n, b.mean() / a1.mean()))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
