"""Regression check: modes J (CPU and GPU) and M (CPU) at a fixed -spp render the same image every
time.

    python tools/check_jdeterminism.py [--out png/check_jdeterminism] [--exe ftrace.exe]

`-device cpu` is the reference and validation device, so it must repeat itself, and so must the
GPU, whose renders the checks hold to tolerances a run-to-run scatter would eat. Mode J splits a
render into light-side epochs, each drawing a new beam map; on the CPU at a fixed -spp an epoch
ends on work counts (workcount.h: vertices traced, connections, beams and photons weighed, the
beam map's build), not on the clock, which made which samples met which map depend on the
machine's speed (known-issues J-CPU-NONDET). Mode M's loop does the same (MODEM-CPU-NONDET), and
its camera pass stratifies its wavelengths by a sample's absolute index rather than within each
call, whose size also followed the clock. The scene is scraps/jspec2/hetmed's -- a heterogeneous
fog, whose transmittance estimates draw from the camera's stream, so a sample that meets another
map takes other draws -- rendered at 32x32 in mode J (256 spp, beam merges only), under the knee rule and under the
default fixed map, and in mode M (-beams, 64 spp, -beamrefresh 0.5), each:

  - twice, byte for byte the same;
  - and at one thread, the same again: the counts are sums over the frame, which no division of
    it among threads changes.

On the GPU (GPU-J-CLOCK) mode J's camera pass counts its own work (cudaLastBdptWork), and under
-jsurf each device surface realization covers a block of samples the counts set, not a chunk the
clock sizes. Mode J on the GPU, under the knee rule with and without -jsurf, must repeat itself
twice and at one thread too; and the beams' render with every segment gathered inline
(FTRACE_NOWAVEFRONT=1) must be the queued one, since a segment the full queue turns away (which
ones, a race) draws each hit from the stream the queue would have given it. All to
pfm.ORDER_TOL rather than byte for byte, the film's atomic sums landing in any order; an epoch or
a realization laid by the clock moves pixels by per cent. With no CUDA device (or a build without
CUDA) the GPU cases are skipped, and say so.

Each needs more than one epoch to mean anything, and says how many it got; and renders each in
one call (FTRACE_JWORK=1 prints a line a call), epoch 0 included: its first sample prices a
sample and the call runs on (SppProgress::span), where a one-sample call and a second one had set
the device's maps up twice. And mode M resumed
from a checkpoint draws its light side under salts of its own, not the first session's, at the
first session's radii, and the same way each time.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import ORDER_TOL, image_diff, read_pfm  # noqa: E402

SCENE = NL.join([
    "scene { units meters  spectral 360 830 1 }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium {",
    "    sigma_a rgb 0.15 0.45 0.9   sigma_s 0.5   g 0.0",
    "    bounds { min 0 0 0   max 1 1 1 }",
    '    density "0.4 + 0.6*saturate(1.2 - 2.0*sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2))"',
    "}",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    'camera "cam" {',
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 32 32 }",
    "}"]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jdeterminism"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "jdet.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    fails = []

    # J: the knee rule (-beamcount and -beamk select it); Jfixed: the default fixed map (-jbeams).
    # Jgpu: the knee rule on the GPU; Jgpu-jsurf: with point merges, whose surface photons the
    # device redraws every block of samples.
    KNEE = ["-mode", "J", "-spp", "256", "-beamcount", "100000", "-beamk", "2"]
    CASES = {"J": ["-device", "cpu"] + KNEE + ["-nojsurf"],
             "Jfixed": ["-device", "cpu", "-mode", "J", "-spp", "256", "-nojsurf"],
             "M": ["-device", "cpu", "-mode", "M", "-beams", "-n", "20000", "-spp", "64",
                   "-beamrefresh", "0.5"],
             "Jgpu": ["-device", "gpu", "-window-min"] + KNEE + ["-nojsurf"],
             "Jgpu-jsurf": ["-device", "gpu", "-window-min"] + KNEE + ["-jsurf"]}

    def render(mode, tag, extra=(), env=None):
        """The render's PFM bytes, its light-side epoch count and its render calls (a line each
        under FTRACE_JWORK); (None, 0, 0) for a GPU case where there is no GPU to run it."""
        png = os.path.join(d, "%s_%s.png" % (mode, tag))
        if os.path.exists(png[:-4] + ".pfm"):
            os.remove(png[:-4] + ".pfm")
        cmd = [exe, "-in", ftsl, "-max-bounce", "8", "-no-meter", "-hdr",
               "-seed", "1", "-o", png] + CASES[mode] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **(env or {})))
        log = p.stdout + p.stderr
        on_gpu = "gpu" in CASES[mode]
        if on_gpu and ("no CUDA device" in log or "built without CUDA" in log):
            return None, 0, 0
        if p.returncode != 0 or "[stop] interrupted" in log or not os.path.exists(png[:-4] + ".pfm"):
            print(log[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        if on_gpu and "on GPU" not in log:
            print(log[-1500:])
            raise SystemExit("mode %s did not run on the GPU" % mode)
        m = re.search(r"\[jstats\] (\d+) epochs|averaged (\d+) independent light-side", log)
        calls = len(re.findall(r"^\[jwork\] (?:mode M )?epoch \d+:", log, re.M))
        with open(png[:-4] + ".pfm", "rb") as f:
            return f.read(), int(m.group(1) or m.group(2)) if m else 0, calls

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    def same(mode, tag, a, b):
        """Byte for byte on the CPU; on the GPU to ORDER_TOL (atomic sums). Returns the
        check's name and verdict and its detail."""
        if "gpu" not in CASES[mode]:
            return "mode %s: %s byte for byte the first" % (mode, tag), a == b, ""
        pa, pb = (os.path.join(d, "%s_%s.pfm" % (mode, t)) for t in ("run1", tag))
        r, n = image_diff(read_pfm(pa, top_down=False), read_pfm(pb, top_down=False))
        return ("mode %s: %s the first, to %g of a pixel" % (mode, tag, ORDER_TOL),
                r < ORDER_TOL, "largest change %.2g, %d values differ" % (r, n))

    for mode in CASES:
        first, epochs, calls = render(mode, "run1", (), {"FTRACE_JWORK": "1"})
        if first is None:
            print("(no GPU here: mode %s is skipped)" % mode)
            continue
        check("mode %s: more than one light-side epoch" % mode, epochs > 1, "%d epochs" % epochs)
        check("mode %s: ...each in one render call" % mode, calls == epochs,
              "%d calls for %d epochs" % (calls, epochs))
        second, _, _ = render(mode, "run2")
        check(*same(mode, "run2", first, second))
        single, _, _ = render(mode, "t1", ("-t", "1"))
        check(*same(mode, "t1", first, single))
        if mode == "Jgpu":
            # Every segment gathered inline (FTRACE_NOWAVEFRONT=1): each hit drawn from the stream
            # the queue gives it, so a segment the full queue turns away -- which ones is a race --
            # renders as the queue would have.
            inline, _, _ = render(mode, "inline", (), {"FTRACE_NOWAVEFRONT": "1"})
            check(*same(mode, "inline", first, inline))

    # A resumed mode M render must not gather from a light side the checkpoint already averaged,
    # so it draws its own under salts past the loaded samples'; but at the radii they were
    # gathered at, or the average converges to neither estimator. And it repeats itself.
    png = os.path.join(d, "M_resume.png")
    def resumed(spp, extra):
        cmd = [exe, "-in", ftsl, "-device", "cpu", "-max-bounce", "8", "-no-meter", "-hdr",
               "-seed", "1", "-o", png, "-mode", "M", "-beams", "-n", "20000", "-spp", str(spp),
               "-beamrefresh", "0.5", "-checkpoint"] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        with open(png[:-4] + ".pfm", "rb") as f:
            return p.stdout, f.read()
    def radii(out):
        return re.findall(r"adaptive gather radius: \S+ -> (\S+)", out)
    out1, _ = resumed(16, ())
    ckpt = [f for f in os.listdir(d) if f.startswith("M_resume") and f.endswith(".ftbuf")]
    if len(ckpt) != 1:
        raise SystemExit("expected one M_resume checkpoint, found %r" % ckpt)
    ckpt = os.path.join(d, ckpt[0])
    with open(ckpt, "rb") as f:
        saved = f.read()
    out2, img2 = resumed(32, ("-resume",))
    check("mode M: a resume loads the checkpoint", "[resume] loaded" in out2)
    check("mode M: ...and draws its light side afresh", "the light side is drawn afresh" in out2)
    check("mode M: ...at the first session's radii", radii(out1) != [] and radii(out2) == radii(out1),
          "%s, then %s" % (radii(out1), radii(out2)))
    with open(ckpt, "wb") as f:
        f.write(saved)
    _, img3 = resumed(32, ("-resume",))
    check("mode M: ...and repeats itself", img3 == img2)
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
