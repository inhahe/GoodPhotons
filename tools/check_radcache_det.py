"""Regression check: a -radcache render is the same however it is cut, threaded or stopped (RADCACHE-BATCHES).

    python tools/check_radcache_det.py [--out png/check_radcache_det] [--exe ftrace.exe]

The CPU's radiance cache advances at its schedule's boundaries -- 1, 10, then an even split of
the rest into 62 -- over the render's absolute samples (RadCacheSchedule), and nowhere else; its
merge is canonical (radCachePend: a key's cell takes the least of its batch's marks, the
verification records are summed sorted); and a checkpoint carries the batch under way. So at a
fixed -spp the image is a function of the samples alone. Until 0.439.3 the cache advanced at the
end of each chunk, which made the chunks a fixed schedule of their own -- long batches the
-interval writes, the window and a stop waited on -- and its merge followed the threads' order.

On scenes/cornell.ftsl at 64x64, 4096 spp, cells of 0.12 (129 of them, a fifth ready by the end,
answering some 5 % of the consults) and 1024 update rays a cell -- so a round serves about 34
cells, part of its work list, and a resume must go on from the same place in its rotation, each
cell in the slot it had -- mode R must render byte for byte the same image:
  - twice;
  - at one thread (`-t 1`);
  - in chunks of 7 (FTRACE_CHUNK_SPP);
  - and, at two threads, stopped 3 s in (`ftrace -stop`) and resumed to the same total.
And mode P (-n 200000) twice. Exits 1 on any failure.
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
SPP = 4096


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_radcache_det"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    fails = []

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    def cmd(tag, extra):
        return [exe, "-in", SCENE, "-device", "cpu", "-radcache", "-radcache-cell", "0.12",
                "-radcache-rays", "1024", "-r", "64", "64", "-seed", "1", "-hdr", "-no-meter",
                "-window-min",
                "-o", os.path.join(d, tag + ".png")] + extra

    def clear(tag):
        for f in os.listdir(d):
            if f.startswith(tag + ".") or f.startswith(tag + "_"):
                os.remove(os.path.join(d, f))

    def render(tag, extra, env=None, keep=False):
        """The render's PFM bytes and its log; `keep` leaves the tag's files (a resume's)."""
        if not keep:
            clear(tag)
        p = subprocess.run(cmd(tag, extra), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **(env or {})))
        log = p.stdout + p.stderr
        pfm = os.path.join(d, tag + ".pfm")
        if p.returncode != 0 or not os.path.exists(pfm):
            print(log[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd(tag, extra)))
        with open(pfm, "rb") as f:
            return f.read(), log

    R = ["-mode", "R", "-spp", str(SPP)]
    first, log = render("R", R)
    m = re.search(r"(\d+)/(\d+) consults terminated", log)
    check("mode R: the cache answered (the check holds a live cache)",
          m is not None and int(m.group(1)) > 0,
          "%s of %s consults" % (m.group(1), m.group(2)) if m else "no [radcache] line")
    check("mode R: twice, byte for byte", render("R2", R)[0] == first)
    check("mode R: at one thread, byte for byte", render("Rt1", R + ["-t", "1"])[0] == first)
    check("mode R: in chunks of 7, byte for byte",
          render("Rc7", R, {"FTRACE_CHUNK_SPP": "7"})[0] == first)

    # Stopped part-way, then resumed to the same total: the first image. At two threads, so the
    # stop lands inside it (the image is the same at any count); at the same total, so its batches
    # are the others' (a longer one would serve the whole work list each round).
    RS = R + ["-t", "2"]
    clear("Rs")
    log_path = os.path.join(d, "Rs.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd("Rs", RS + ["-checkpoint"]), stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(3)
        subprocess.run([exe, "-stop", str(p.pid)], capture_output=True)
        p.wait(timeout=180)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        slog = f.read()
    m = re.search(r"interrupted at (\d+) spp", slog)
    have = int(m.group(1)) if m else SPP
    if have >= SPP:
        check("mode R: the stop landed inside the render", False, "it finished first")
    else:
        resumed, rlog = render("Rs", ["-mode", "R", "-spp", str(SPP - have), "-t", "2",
                                      "-checkpoint", "-resume"], keep=True)
        check("mode R: stopped at %d spp and resumed, byte for byte" % have, resumed == first,
              "the resume loaded the checkpoint" if "[resume] loaded" in rlog
              else "no [resume] line")

    P = ["-mode", "P", "-spp", "64", "-n", "200000"]
    pfirst, _ = render("P", P)
    check("mode P: twice, byte for byte", render("P2", P)[0] == pfirst)

    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
