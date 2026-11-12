"""Regression check: a -radcache render in mode R or P ends when it is told to (RADCACHE-BATCHES).

    python tools/check_radcache_stop.py [--out png/check_radcache_stop] [--exe ftrace.exe]

Under -radcache the CPU's modes R and P rendered a fixed -spp in a schedule of its samples (1, 9,
then 62 even batches), not the clock's, because the cache advanced between batches. Two ways that
schedule kept a render from ending, both on scenes/cornell.ftsl:

  - Under a -time budget there is no total to split, and mode R's third batch was about 16
    million samples: `-radcache -time 6` never ended (0.434.26 was still running at 50 s, and its
    -stop failed). It must end, cleanly and with its image, within a few seconds of its budget.
  - At -spp 1000000 a batch is some 16 000 spp, minutes here. A -stop sent inside one must land
    within seconds, in mode R and in mode P. 0.434.26 waited for the batch: 28 s for one of 3200
    spp, past -stop's wait for these. Since 0.439.3 the cache's batches are its own
    (RadCacheSchedule) and the chunks follow the clock again, cut at the batches' boundaries
    (tools/check_radcache_det.py holds the image to them): mode R's image then holds the
    chunks the stop came after, past the first two batches.

Exits 1 on any failure. A render that will not stop is reported by pid and left running (this
project never force-kills ftrace).
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "cornell.ftsl")
NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_radcache_stop"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails, left = [], []

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    def start(tag, extra):
        for f in os.listdir(d):
            if f.startswith(tag + "."):
                os.remove(os.path.join(d, f))
        log = open(os.path.join(d, tag + ".log"), "w", encoding="utf-8")
        cmd = [exe, "-in", SCENE, "-device", "cpu", "-radcache", "-hdr", "-no-meter",
               "-window-min", "-o", os.path.join(d, tag + ".png")] + extra
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log

    def stop(p):
        """-stop the render; seconds it took and -stop's exit code."""
        t0 = time.time()
        r = subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return time.time() - t0, r.returncode

    def finish(p, log, tag):
        """Wait for the render to exit; its exit code, or None (left running)."""
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            left.append("%s (pid %d)" % (tag, p.pid))
        log.close()
        return p.returncode

    def logtext(tag):
        with open(os.path.join(d, tag + ".log"), encoding="utf-8", errors="replace") as f:
            return f.read()

    # A budget: no total to split, so the clock sets the chunks.
    t0 = time.time()
    p, log = start("budget", ["-mode", "R", "-time", "6"])
    try:
        p.wait(timeout=60)
        took = time.time() - t0
        log.close()
        check("mode R -radcache -time 6 ends on its budget", took < 20 and p.returncode == 0
              and os.path.exists(os.path.join(d, "budget.png")),
              "%.1f s, exit %d" % (took, p.returncode))
    except subprocess.TimeoutExpired:
        sec, rc = stop(p)
        finish(p, log, "budget")
        check("mode R -radcache -time 6 ends on its budget", False,
              "still running at 60 s; -stop took %.0f s, exit %d" % (sec, rc))

    # A stop inside a long batch of a fixed -spp, in mode R and in mode P.
    for mode, extra in (("R", []), ("P", ["-n", "100000"])):
        tag = "long_" + mode
        p, log = start(tag, ["-mode", mode, "-spp", "1000000"] + extra)
        time.sleep(8)   # past the 1 and 9 spp batches, inside the third
        sec, rc = stop(p)
        code = finish(p, log, tag)
        check("mode %s: -stop inside a long batch lands within seconds" % mode,
              rc == 0 and sec < 15 and code == 0, "%.1f s, -stop exit %d, render exit %s"
              % (sec, rc, code))
        if mode == "R":
            m = re.search(r"interrupted at (\d+) spp", logtext(tag))
            check("mode R: ...keeping the clock's chunks before it",
                  m is not None and int(m.group(1)) > 10, "%s spp" % (m.group(1) if m else "?"))

    if left:
        print("LEFT RUNNING (stop them with ftrace -stop <pid>): " + ", ".join(left))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
