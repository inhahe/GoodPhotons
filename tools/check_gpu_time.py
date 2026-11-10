"""Regression check: a `-time` budget ends a GPU mode J chunk in flight, not after it.

    python tools/check_gpu_time.py [--out png/check_gpu_time] [--exe ftrace.exe]

A GPU mode D/J chunk is at least one sample of the whole frame, and the budget's own test runs
between chunks; gpuSppChunks therefore cuts the chunk in flight when SppProgress::deadline
passes, through the device stop flag a -stop raises, and drops it -- once the film already holds
a sample, so the first sample of a render is never cut. The scene is check_gpu_stop's: the
density-field haze at 960x720 with a fixed `-n 30000` map, ~10-20 s a sample on an RTX 4090.

  first   -time far inside the first sample: the sample runs on (an image needs one), the
          render says so, and ends with 1 spp at elapsed E1, the sample's own S1 of it
          (FTRACE_CHUNK_DEBUG) and the setup before it the rest.
  frozen  -beamfreeze, -time E1 + S1/2: the deadline lands in the epoch's second chunk (this
          call's own sample is the film's), which is cut -- the render ends within a fifth of
          a sample of the budget with 1 spp, where a build without the cut ends a sample after
          E1 with 2.
  epochs  -beamrefresh 100, the same budget: an epoch ends at its first report, so every sample
          is its own light-side epoch (-beamepochs splits only a fixed -spp) and the deadline
          lands in the second epoch's FIRST chunk; the earlier epoch's sample is the film's
          (SppProgress::sampleBase), and the cut is answered with that image.
  resume  -resume from first's checkpoint, -time E1 - S1/2 (its setup and half a sample): the
          deadline lands in this session's first chunk, the loaded sample is the film's, and the
          render ends with the 1 spp it loaded.

Each case requires exit 0, the image written, the final status line's spp, and its elapsed time.
Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fogbox  # noqa: E402
from check_haze_domain import FIELD, scene  # noqa: E402

def seconds(text):
    """humanDur's '31.2s' / '1:02' / '1:00:02' (duration.h) back to seconds."""
    text = text.strip()
    if text.endswith("s"):
        return float(text[:-1])
    total = 0
    for part in text.split(":"):
        total = total * 60 + int(part)
    return float(total)


def render(exe, d, ftsl, name, budget, extra):
    png = os.path.join(d, name + ".png")
    for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-time", "%.2f" % budget,
           "-seed", "1", "-n", "30000", "-interval", "3600", "-hdr", "-no-meter",
           "-window-min", "-o", png] + extra
    logf = os.path.join(d, name + ".log")
    env = dict(os.environ, FTRACE_CHUNK_DEBUG="1")   # each chunk's GPU time
    with open(logf, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=d, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    with open(logf, encoding="utf-8", errors="replace") as f:
        out = f.read()
    last = re.findall(r"\[time\] (.+?) / (.+?), (\d+) spp", out)
    if not last:
        return rc, out, None, None, png
    el, _, spp = last[-1]
    return rc, out, seconds(el), int(spp), png


def chunk_secs(out):
    """The GPU time of each chunk that landed (gpuSppChunks' FTRACE_CHUNK_DEBUG lines)."""
    return [float(s) for s in re.findall(r"\[chunk\] gpu \d+ spp in ([\d.]+) s", out)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gpu_time"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    ftsl = fogbox.write(d, "field.ftsl", scene(True, medium=FIELD, res=(960, 720)))
    check = fogbox.Checks(60)
    tails = []

    rc, out, e1, spp, png = render(exe, d, ftsl, "first", 2.0, ["-beamfreeze"])
    tails.append(out[-800:])
    chunks = chunk_secs(out)
    s1 = chunks[0] if chunks else 0.0     # the sample's own time; e1 adds the setup before it
    check("first: exits 0", rc == 0, "rc %d" % rc)
    check("first: one whole sample", spp == 1, "spp %s" % spp)
    check("first: says the first sample runs on",
          "budget ran out inside the render's first sample" in out)
    check("first: the image was written", os.path.exists(png[:-4] + ".pfm"))
    ok = e1 is not None and s1 >= 6.0
    check("first: a sample long enough to cut", ok, "sample %.1f s, E1 %s" % (s1, e1))
    if not ok:
        return check.verdict(NL.join(tails))

    # Half a sample into the second: past the epochs case's light-side rebuild (a fraction of a
    # second), with half of it left to cut. The resume's: half a sample into its first.
    later = e1 + 0.5 * s1
    cases = (("frozen", "frozen", later, ["-beamfreeze"]),
             ("epochs", "epochs", later, ["-beamrefresh", "100"]),
             ("resume", "first", e1 - 0.5 * s1, ["-beamfreeze", "-resume"]))
    for name, out_name, budget, extra in cases:
        t0 = time.monotonic()
        rc, out, el, spp, png = render(exe, d, ftsl, out_name, budget, extra)
        wall = time.monotonic() - t0
        tails.append(out[-800:])
        check("%s: exits 0" % name, rc == 0, "rc %d" % rc)
        if name == "epochs":
            check("epochs: the second sample is a refresh's", "light-side refresh" in out)
        if name == "resume":
            check("resume: the checkpoint's sample was loaded", "1 spp accumulated so far" in out)
        check("%s: the cut sample is dropped" % name, spp == 1, "spp %s" % spp)
        check("%s: ends within a fifth of a sample of -time" % name,
              el is not None and budget <= el < budget + 0.2 * s1,
              "%s s against %.1f s (a sample %.1f s; wall %.1f s)"
              % ("?" if el is None else "%.1f" % el, budget, s1, wall))
        check("%s: the image was written" % name, os.path.exists(png[:-4] + ".pfm"))
    return check.verdict(NL.join(tails))


if __name__ == "__main__":
    sys.exit(main())
