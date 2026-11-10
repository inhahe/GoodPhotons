"""Regression check: `-jbeams off | auto | <N>` picks mode J's beam map as documented (0.435.0).

    python tools/check_jbeams.py [--out png/check_jbeams] [--exe ftrace.exe]

Renders check_haze_domain's still haze (an unbounded sunlit medium round a ball) in CPU mode J at
32x24 and 1 spp under each way of choosing the map, and reads what the log says it built:

  - `-jbeams 64` builds a fixed map of exactly 64 raw beams, from FEWER than 256 subpaths: a
    fixed map has no kPathsMin floor (BeamBudgetReq::exact), which the knee rule keeps and which
    on this scene alone is ~450 raw beams;
  - no flag builds the default fixed map, 1000 beams;
  - `-jbeams auto`, `-beamk 32` and `-beamcount 200000` each select the knee rule (the budget
    line), and `-n 3000` too, its budget off: exactly 3000 subpaths, no pilot;
  - an explicit `-jbeams` wins over `-nobeams` and `-beamk`, whichever comes first;
  - `-jbeams off` renders the same image as `-nobeams`, bit for bit, and says the beams are off;
  - `-jbeams 0` and `-jbeams many` are refused with exit code 2.

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
import fogbox  # noqa: E402
from check_haze_domain import scene  # noqa: E402

FIXED = re.compile(r"mode J: beam map: pilot of \d+ subpaths measured [\d.]+ raw beams/subpath "
                   r"-> (\d+) beams from (\d+) subpaths \(fixed, -jbeams\)")
KNEE = re.compile(r"mode J: beam budget: pilot of \d+ subpaths")
TRACED = re.compile(r"mode J: \d+ beams from (\d+) light subpaths")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jbeams"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    ftsl = fogbox.write(d, "still.ftsl", scene(True, res=(32, 24)))
    check = fogbox.Checks(60)
    logs = []

    def run(tag, extra):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "cpu", "-spp", "1", "-seed", "1",
               "-hdr", "-no-meter", "-o", png] + extra
        # From the repo root: the scene's presets resolve from there.
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (open(pfm, "rb").read() if os.path.exists(pfm) else None)

    rc, log, _ = run("fixed64", ["-jbeams", "64"])
    m = FIXED.search(log)
    check("-jbeams 64: a fixed map of exactly 64 beams", rc == 0 and m is not None and m.group(1) == "64",
          m.group(0)[-50:] if m else "no fixed-map line (rc %d)" % rc)
    check("...from fewer than 256 subpaths (no kPathsMin floor)",
          m is not None and int(m.group(2)) < 256, "%s subpaths" % (m.group(2) if m else "?"))

    rc, log, _ = run("default", [])
    m = FIXED.search(log)
    check("no flag: the default fixed map of 1000 beams", rc == 0 and m is not None and m.group(1) == "1000",
          m.group(0)[-50:] if m else "no fixed-map line (rc %d)" % rc)

    for tag, extra in (("auto", ["-jbeams", "auto"]), ("beamk", ["-beamk", "32"]),
                       ("beamcount", ["-beamcount", "200000"])):
        rc, log, _ = run(tag, extra)
        check("%s: the knee rule" % " ".join(extra),
              rc == 0 and KNEE.search(log) is not None and FIXED.search(log) is None,
              "rc %d" % rc)

    rc, log, _ = run("n", ["-n", "3000"])
    m = TRACED.search(log)
    check("-n 3000: the knee rule's budget off, exactly 3000 subpaths",
          rc == 0 and KNEE.search(log) is None and FIXED.search(log) is None and m is not None
          and m.group(1) == "3000", "%s subpaths" % (m.group(1) if m else "?"))

    for tag, extra in (("over_nobeams", ["-nobeams", "-jbeams", "64"]),
                       ("over_beamk", ["-jbeams", "64", "-beamk", "32"])):
        rc, log, _ = run(tag, extra)
        m = FIXED.search(log)
        check("%s: -jbeams wins" % " ".join(extra), rc == 0 and m is not None and m.group(1) == "64",
              "rc %d" % rc)

    rc1, log1, img1 = run("off", ["-jbeams", "off"])
    rc2, _, img2 = run("nobeams", ["-nobeams"])
    check("-jbeams off: the beams are off, and it says so",
          rc1 == 0 and "the beams are off (-jbeams off)" in log1, "rc %d" % rc1)
    check("-jbeams off renders -nobeams' image bit for bit",
          rc2 == 0 and img1 is not None and img1 == img2,
          "same" if img1 is not None and img1 == img2 else "differ")

    for bad in ("0", "many"):
        rc, log, _ = run("bad_" + bad, ["-jbeams", bad])
        check("-jbeams %s is refused (exit 2)" % bad,
              rc == 2 and "-jbeams wants off, auto or a beam count" in log, "rc %d" % rc)
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
