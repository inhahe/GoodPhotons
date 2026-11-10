"""Regression check: `ftrace -stop` cuts a GPU mode J chunk short, and the frame keeps whole samples.

    python tools/check_gpu_stop.py [--out png/check_gpu_stop] [--exe ftrace.exe]

renderBdptCuda's chunks are abortable: inside a DeviceStopScope the kernels read the device
stop flag, and a chunk a stop cut short is dropped, gpuSppChunks flushing the whole chunks
before it as final. A build that runs a chunk whole before it stops fails here, as does one that
keeps a cut chunk. The scene is check_haze_domain's density field at 48x36 under -beamepochs spp,
which pins every GPU chunk to one sample, tens of seconds each (J-FIELD-SLOW) -- and makes each
sample its own light-side epoch, so the stop cuts an epoch's first and only chunk. The check
waits for the first sample to land (the refresh that opens the second epoch), times it, asks the
render to stop a quarter of a sample later -- inside the second sample's GPU chunk, its light
side rebuilt in about a second -- and requires:

  - the process to be gone in under a third of a sample's time, which only a chunk cut short
    allows;
  - the final status line to count the whole samples, 1, and the image to be written. With
    -interval longer than the render no periodic write happens, so the image is the stop's own
    final flush: an epoch loop that left without one writes none.

Never kills the render: one that ignores the stop is reported, and left running.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gpu_stop"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    ftsl = fogbox.write(d, "field.ftsl", scene(True, medium=FIELD, res=(48, 36)))
    png = os.path.join(d, "field.png")
    for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
        if os.path.exists(stale):
            os.remove(stale)
    logf = os.path.join(d, "field.log")
    cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-spp", "8", "-seed", "1",
           "-beamepochs", "spp", "-interval", "3600", "-hdr", "-no-meter", "-window-min",
           "-o", png]
    check = fogbox.Checks(56)
    with open(logf, "w", encoding="utf-8") as lf:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, cwd=d, stdout=lf, stderr=subprocess.STDOUT)

        def log():
            with open(logf, encoding="utf-8", errors="replace") as f:
                return f.read()

        # The first sample: the refresh that opens the second epoch, timed from the start.
        t1 = None
        while p.poll() is None and time.monotonic() - t0 < 900:
            if "light-side refresh" in log():
                t1 = time.monotonic()
                break
            time.sleep(0.25)
        if t1 is None:
            print(log()[-1500:])
            raise SystemExit("FAILED: no first sample (the render %s)" %
                             ("exited" if p.poll() is not None else "is still running, pid %d"
                              % p.pid))
        sample = t1 - t0
        # A quarter of a sample into the second: past its light-side rebuild (~1 s against the
        # sample's ~25), so inside its GPU chunk, with three quarters of it left to cut.
        time.sleep(sample * 0.25)
        ts = time.monotonic()
        s = subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if s.returncode != 0:
            raise SystemExit("FAILED: pid %d ignored -stop (%s); left running"
                             % (p.pid, (s.stdout + s.stderr).strip()[-200:]))
        rc = p.wait()
        stopSec = time.monotonic() - ts
    out = log()
    check("the first sample took long enough to cut", sample > 10.0, "%.1f s" % sample)
    check("the render stopped within a third of a sample", stopSec < sample / 3.0,
          "%.1f s against %.1f s" % (stopSec, sample))
    check("the render exits 0", rc == 0, "rc %d" % rc)
    finals = re.findall(r"\[spp\] (\d+) / 8", out)
    check("the frame counts whole samples only", finals != [] and finals[-1] == "1",
          "last status %s / 8" % (finals[-1] if finals else "?"))
    check("the image was written", os.path.exists(png[:-4] + ".pfm"))
    return check.verdict(out[-1500:])


if __name__ == "__main__":
    sys.exit(main())
