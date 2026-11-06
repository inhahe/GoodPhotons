"""Regression check: a -radcache render in mode R ends when it is told to (RADCACHE-BATCHES).

    python tools/check_radcache_stop.py [--out png/check_radcache_stop] [--exe ftrace.exe]

Under -radcache the CPU's mode R renders a fixed -spp in a schedule of its samples (1, 9, then
62 even batches), not the clock's, because the cache advances between batches. Two ways that
schedule kept a render from ending, both on scenes/cornell.ftsl:

  - Under a -time budget there is no total to split, and the schedule's third batch was about
    16 million samples: `-radcache -time 6` never ended (0.434.26 was still running at 50 s, and
    its -stop failed). It must end within a few seconds of its budget.
  - A fixed -spp 200000 has batches of about 3200 spp, some 40 s each here. A -stop sent inside
    one must land within seconds, the batch dropped and the image holding the 10 spp before it.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "cornell.ftsl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_radcache_stop"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails = []

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    def start(tag, extra):
        log = open(os.path.join(d, tag + ".log"), "w", encoding="utf-8")
        cmd = [exe, "-in", SCENE, "-mode", "R", "-device", "cpu", "-radcache", "-hdr", "-no-meter",
               "-window-min", "-o", os.path.join(d, tag + ".png")] + extra
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log

    def stop(p):
        t0 = time.time()
        r = subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return time.time() - t0, r.returncode

    # A budget: no total to split, so the clock sets the chunks.
    t0 = time.time()
    p, log = start("budget", ["-time", "6"])
    try:
        p.wait(timeout=60)
        took = time.time() - t0
        check("-radcache -time 6 ends on its budget", took < 20, "%.1f s" % took)
    except subprocess.TimeoutExpired:
        sec, rc = stop(p)
        check("-radcache -time 6 ends on its budget", False,
              "still running at 60 s; -stop took %.0f s, exit %d" % (sec, rc))
    log.close()

    # A stop inside a long batch of a fixed -spp.
    p, log = start("long", ["-spp", "200000"])
    time.sleep(8)   # past the 1 and 9 spp batches, inside the third
    sec, rc = stop(p)
    try:
        p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        pass
    log.close()
    check("-stop inside a long batch lands within seconds", rc == 0 and sec < 20,
          "%.1f s, exit %d" % (sec, rc))
    with open(os.path.join(d, "long.log"), encoding="utf-8", errors="replace") as f:
        m = re.search(r"interrupted at (\d+) spp", f.read())
    check("...dropping the batch, the image holding the 10 spp before it",
          m is not None and m.group(1) == "10", "%s spp" % (m.group(1) if m else "?"))

    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
