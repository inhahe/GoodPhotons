"""Regression check: `-jbeams off | auto | <N>` picks mode J's beam map as documented (0.435.0-0.436.0).

    python tools/check_jbeams.py [--out png/check_jbeams] [--exe ftrace.exe] [--no-gpu]

Renders check_haze_domain's still haze (an unbounded sunlit medium round a ball) in mode J at
32x24 and 1 spp (CPU unless said) under each way of choosing the map, and reads what the log says
was BUILT:

  - `-jbeams 64` deposits about 64 raw beams (the chords the build counts, within 2x: the pilot's
    rate is an average), from FEWER than 256 subpaths -- a fixed map has no kPathsMin floor
    (BeamBudgetReq::exact), which the knee rule keeps and which on this scene alone is ~450
    beams -- builds them with no -beamk floor, and leaves the -jsurf surface map its own count,
    one subpath per pixel (768), not the beams' few dozen (0.435.1);
  - no flag builds the default fixed map, about 1000 beams;
  - `-jbeams auto`, `-beamk 32` and `-beamcount 200000` each select the knee rule (the budget
    line), and `-n 3000` too, its budget off: exactly 3000 subpaths, no pilot;
  - an explicit `-jbeams` wins over `-nobeams`, `-beamk` and `-n`, which then only caps the pass;
    and `-n` below 256 (here 100) is a cap, not a crossed clamp;
  - `-jbeams off` renders `-nobeams`' image bit for bit, deposits no beams, sizes its pass for the
    surface map, and each names the flag that turned the beams off; at 128x96, where its pass is
    big enough for the pilot to run, it still lays no beams and prints no beam budget (0.436.0);
  - a scene's `render { jbeams auto }` selects the knee rule, and a command-line -jbeams beats it;
    a scene's `jbeams off` stays off under `-n` (0.436.0: those pick `auto` only where no rule is
    named);
  - a checkpoint written under the fixed map refuses to resume under `auto`, and resumes under
    the same map;
  - a two-frame flight that keeps only epoch 0 (FTRACE_JEPOCHKEEP=1) re-traces its second frame's
    other epochs at the subpath count epoch 0 settled on, not the inert default (0.436.0: the
    counts live with the batch, where a frame that takes epoch 0 from it finds them);
  - on the GPU, the device draws the surface map from one subpath per pixel and the host traces
    only the beams' subpaths (BeamBudgetReq::hostSurf, 0.436.0);
  - `-jbeams 0`, `many`, `inf` and `1e30` are refused with exit code 2.

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
                   r"-> ~\d+ beams from (\d+) subpaths of the pass's (\d+) \(fixed, -jbeams (\d+)\)")
KNEE = re.compile(r"mode J: beam budget: pilot of \d+ subpaths")
CHORDS = re.compile(r"mode J: photon beams: medium \d+: (\d+) chords")
FLOOR0 = re.compile(r"-beamk floor 0\)")
TRACED = re.compile(r"mode J: \d+ beams from (\d+) light subpaths")
SURF = re.compile(r"mode J: \d+ surface photons from (\d+) subpaths")
DEVSURF = re.compile(r"mode J: the device draws the surface photon map, from (\d+) subpaths")
REDRAW = re.compile(r"light-side refresh .{0,4}redrawing (\d+) subpaths")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jbeams"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    ftsl = fogbox.write(d, "still.ftsl", scene(True, res=(32, 24)))
    pixels = 32 * 24
    text = open(ftsl, encoding="utf-8").read()
    ftsl_auto = fogbox.write(d, "still_auto.ftsl", text + NL + "render { jbeams auto }" + NL)
    ftsl_off = fogbox.write(d, "still_off.ftsl", text + NL + "render { jbeams off }" + NL)
    flight = ('camera_path "fly" { look_at 3 1 0  up 0 1 0  fov_y 50  frames 2  film { res 32 24 }'
              + NL + "key 0 3 2 9" + NL + "key 1 4 2 9" + NL + "}")
    ftsl_fly = fogbox.write(d, "still_fly.ftsl", text + NL + flight + NL)
    check = fogbox.Checks(66)
    logs = []

    def run(tag, extra, scene_path=ftsl, out=None, clean=True, dev="cpu", env=None):
        png = out or os.path.join(d, tag + ".png")
        if clean:
            for stale in (png, png[:-4] + ".pfm", png + ".ftbuf"):
                if os.path.exists(stale):
                    os.remove(stale)
        cmd = [exe, "-in", scene_path, "-mode", "J", "-device", dev, "-spp", "1", "-seed", "1",
               "-hdr", "-no-meter", "-window-min", "-o", png] + extra
        # From the repo root: the scene's presets resolve from there.
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **(env or {})))
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-700:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (open(pfm, "rb").read() if os.path.exists(pfm) else None)

    def chords(log):
        return sum(int(c) for c in CHORDS.findall(log))

    rc, log, _ = run("fixed64", ["-jbeams", "64"])
    m, s = FIXED.search(log), SURF.search(log)
    c = chords(log)
    check("-jbeams 64: a fixed map of about 64 beams", rc == 0 and m is not None and 32 <= c <= 128,
          "%d chords (rc %d)" % (c, rc))
    check("...from fewer than 256 subpaths (no kPathsMin floor)",
          m is not None and int(m.group(1)) < 256, "%s subpaths" % (m.group(1) if m else "?"))
    check("...built with no -beamk floor", FLOOR0.search(log) is not None)
    check("...while the surface map keeps one subpath per pixel",
          m is not None and s is not None and int(s.group(1)) == pixels == int(m.group(2)),
          "surface %s, pass %s" % (s.group(1) if s else "?", m.group(2) if m else "?"))

    rc, log, _ = run("default", [])
    c = chords(log)
    m = FIXED.search(log)
    check("no flag: the default fixed map, about 1000 beams",
          rc == 0 and m is not None and m.group(3) == "1000" and 500 <= c <= 2000, "%d chords" % c)

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
                       ("over_beamk", ["-jbeams", "64", "-beamk", "32"]),
                       ("over_n", ["-jbeams", "64", "-n", "500"]),
                       ("over_n100", ["-jbeams", "64", "-n", "100"])):
        rc, log, _ = run(tag, extra)
        m = FIXED.search(log)
        check("%s: -jbeams wins" % " ".join(extra), rc == 0 and m is not None, "rc %d" % rc)
        if "-n" in extra:
            cap = int(extra[-1])
            check("...-n only capping the pass", m is not None and int(m.group(2)) <= cap,
                  "pass %s" % (m.group(2) if m else "?"))

    rc1, log1, img1 = run("off", ["-jbeams", "off"])
    rc2, log2, img2 = run("nobeams", ["-nobeams"])
    check("-jbeams off: says so, by that flag", rc1 == 0 and "the beams are off (-jbeams off)" in log1)
    check("-nobeams: says so, by that flag", rc2 == 0 and "the beams are off (-nobeams)" in log2)
    check("-jbeams off: no beams, a pass sized for the surface map",
          chords(log1) == 0 and KNEE.search(log1) is None
          and "the beams are off, so the light pass is sized for the SURFACE map" in log1)
    check("-jbeams off renders -nobeams' image bit for bit",
          img1 is not None and img1 == img2, "same" if img1 is not None and img1 == img2 else "differ")
    rc, log, _ = run("off_big", ["-jbeams", "off", "-r", "128", "96"])
    check("-jbeams off at 128x96 (the pilot runs): no beams, no beam budget",
          rc == 0 and chords(log) == 0 and KNEE.search(log) is None and "beam budget" not in log,
          "rc %d" % rc)

    rc, log, _ = run("scene_auto", [], scene_path=ftsl_auto)
    check("render { jbeams auto }: the knee rule, and it says so",
          rc == 0 and KNEE.search(log) is not None and "[scene] mode J beam map: -jbeams auto" in log,
          "rc %d" % rc)
    rc, log, _ = run("scene_auto_cli", ["-jbeams", "64"], scene_path=ftsl_auto)
    check("...a command-line -jbeams beats it", rc == 0 and FIXED.search(log) is not None)
    rc, log, _ = run("scene_off_n", ["-n", "5000"], scene_path=ftsl_off)
    check("render { jbeams off } stays off under -n",
          rc == 0 and chords(log) == 0 and "the scene's render { jbeams off }" in log, "rc %d" % rc)

    ck = os.path.join(d, "ck.png")
    run("ck_write", ["-jbeams", "64", "-checkpoint"], out=ck)
    rc, log, _ = run("ck_auto", ["-jbeams", "auto", "-checkpoint", "-resume", "-spp", "2"],
                     out=ck, clean=False)
    check("a fixed map's checkpoint refuses to resume under auto",
          "does not match this render" in log, "rc %d" % rc)
    run("ck_write2", ["-jbeams", "64", "-checkpoint"], out=ck)
    rc, log, _ = run("ck_same", ["-jbeams", "64", "-checkpoint", "-resume", "-spp", "2"],
                     out=ck, clean=False)
    check("...and resumes under the same map",
          "does not match" not in log and "[resume] loaded" in log, "rc %d" % rc)

    rc, log, _ = run("flight", ["-camera", "fly", "-frames", "0", "1", "-spp", "4",
                                "-beamepochs", "4", "-jbeams", "64"],
                     scene_path=ftsl_fly, out=os.path.join(d, "fly", "fly.png"),
                     env={"FTRACE_JEPOCHKEEP": "1"})
    redraws = [int(x) for x in REDRAW.findall(log)]
    check("a flight keeping only epoch 0: both frames re-trace epoch 0's count",
          rc == 0 and len(redraws) == 2 and redraws[0] == redraws[1] == pixels,
          "redraws %s (rc %d)" % (redraws, rc))

    if not a.no_gpu:
        rc, log, _ = run("gpu64", ["-jbeams", "64"], dev="gpu")
        m, g = FIXED.search(log), DEVSURF.search(log)
        check("GPU -jbeams 64: the device draws the surface map from one subpath per pixel",
              rc == 0 and g is not None and int(g.group(1)) == pixels, "rc %d" % rc)
        check("...the host tracing only the beams' subpaths",
              m is not None and int(m.group(2)) == int(m.group(1)) < 256,
              "pass %s, beams' %s" % (m.group(2) if m else "?", m.group(1) if m else "?"))
        rc, log, _ = run("gpuoff", ["-jbeams", "off", "-r", "128", "96"], dev="gpu")
        g = DEVSURF.search(log)
        check("GPU -jbeams off: the device draws the surface map, no beam budget",
              rc == 0 and g is not None and int(g.group(1)) == 128 * 96 and "beam budget" not in log,
              "rc %d" % rc)

    for bad in ("0", "many", "inf", "1e30"):
        rc, log, _ = run("bad_" + bad, ["-jbeams", bad])
        check("-jbeams %s is refused (exit 2)" % bad,
              rc == 2 and "-jbeams wants off, auto or a beam count" in log, "rc %d" % rc)
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
