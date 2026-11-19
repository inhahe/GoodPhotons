"""Regression check: every GPU mode asks the VRAM preflight before it takes the card, and falls back
to the CPU when the card cannot afford its kernels.

    python -u tools/check_vram_preflight.py [--out png/check_vram_preflight] [--exe ftrace.exe]

The preflight (main.cpp, vramPreflight) weighs a mode's kernels' local memory
(cudaMegakernelLocalBytes) and its films against the card's free memory, and under `-device auto`
sends a render that would not fit to the CPU, where a GPU launch would spill to host memory and
look hung. Each mode decides its device in its own place -- runRender's shared decision (R, B, D,
J), mode S's and mode U's dispatches, run()'s mode-M routes for one camera and for a flight of
them, the shared forward flight of mode B's cameras, the exposure meter's frames -- and each must
ask. FTRACE_VRAMFREE=<GB> makes the card read as having no more than that
free. For each mode, on a small room:

  - with the card as it is: the run's `[vram]` line names the mode's kernel, and the mode renders
    on the GPU;
  - with FTRACE_VRAMFREE=0 under `-device auto`: the refusal names the mode's kernel, the image is
    still written, and the modes that announce their device after asking (M, S, U) do not announce
    the GPU;
  - with FTRACE_VRAMFREE=0 under `-device gpu`: the refusal, and the GPU taken anyway, as asked.

And of a locked two-frame path metered in mode D: its [vram] line ahead of the meter's, and with the
card read as full a refusal (or a forced warning) for the meter as well as each frame.

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

ROOM = """scene { units meters }
material "w" { type diffuse reflect 0.7 }
quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }
sphere { center 0 0.3 0  radius 0.3  material w }
light area { origin -0.2 1.5 -0.2  u 0.4 0 0  v 0 0 0.4  normal 0 -1 0  power 20  spd preset:d65 }
camera "a" { eye 0 0.8 2  look_at 0 0.3 0  up 0 1 0  fov_y 45  film { res 64 48 } }
camera "b" { eye 0.5 0.8 1.8  look_at 0 0.3 0  up 0 1 0  fov_y 45  film { res 64 48 } }
camera_path "fly" { look_at 0 0.3 0  up 0 1 0  fov_y 45  frames 2  film { res 64 48 }
                    key 0 0 0.8 2  key 1 0.5 0.8 1.8  exposure_lock }
"""
# (tag, mode, extra arguments, whether the mode announces its device only after asking)
CASES = [
    ("R", "R", ["-camera", "a", "-spp", "4"], False),
    ("B", "B", ["-camera", "a", "-n", "200000"], False),
    ("D", "D", ["-camera", "a", "-spp", "4"], False),
    ("J", "J", ["-camera", "a", "-spp", "4"], False),
    ("S", "S", ["-camera", "a", "-spp", "4", "-n", "200000"], True),
    ("U", "U", ["-camera", "a", "-spp", "4", "-n", "200000"], True),
    ("M", "M", ["-camera", "a", "-spp", "4", "-n", "200000"], True),
    ("M flight", "M", ["-spp", "4", "-n", "200000"], True),      # both cameras: the shared path
    ("B flight", "B", ["-n", "200000"], True),                      # both cameras: the shared flight
]
REFUSED = "NOT ENOUGH FREE VRAM for the GPU path"
FORCED = "-device gpu given explicitly; proceeding"
# The device a render took: the dispatch's announcement, or a shared forward flight's start line,
# which names its backend (the card, or "N CPU threads").
ON_GPU = re.compile(r"\[device\] (auto|gpu) -> GPU|shared model-\w pass: .* on (?!\d+ CPU threads)")
# Where a mode announcing its device before it asks (R, B, D, J) really rendered: its start line's,
# mode J's camera pass's.
RAN = re.compile(r"^mode (\w): [^\n]*?\bon (GPU|\d+ CPU threads)", re.M)
WANTS = re.compile(r"kernel wants ([\d.]+) GB local")


def ran_on(log, mode):
    """'GPU' or 'CPU' as the mode's start line says, or None."""
    for m in RAN.finditer(log):
        if m.group(1) == mode:
            return "GPU" if m.group(2) == "GPU" else "CPU"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vram_preflight"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    room = os.path.join(d, "room.ftsl")
    with open(room, "w", encoding="utf-8", newline="") as f:
        f.write(ROOM)
    check = fogbox.Checks(66)
    logs = []

    def run(tag, mode, args, device, free, scene=room, meter=False):
        png = os.path.join(d, "%s_%s_%s.png" % (tag.replace(" ", "_"), device, "full" if free is None else "none"))
        stem = png[:-4]
        for name in os.listdir(d):
            if name.startswith(os.path.basename(stem)) and name.endswith(".png"):
                os.remove(os.path.join(d, name))
        env = dict(os.environ)
        env.pop("FTRACE_VRAMFREE", None)
        if free is not None:
            env["FTRACE_VRAMFREE"] = str(free)
        cmd = ([exe, "-in", scene] + args + ["-mode", mode, "-device", device, "-seed", "1"] +
               ([] if meter else ["-no-meter"]) + ["-window-min", "-o", png])
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
        log = p.stdout + p.stderr
        logs.append("--- %s %s %s" % (tag, device, free) + NL + log[-500:])
        wrote = any(n.startswith(os.path.basename(stem)) and n.endswith(".png") for n in os.listdir(d))
        return p.returncode, log, wrote

    for tag, mode, args, announcesAfter in CASES:
        kernel = "mode-%s kernel wants" % mode
        rc, log, wrote = run(tag, mode, args, "auto", None)
        w = WANTS.search(log)
        weighed = w is not None and float(w.group(1)) > 0.0   # the mode's kernels weighed, not taken as 0
        where = ran_on(log, mode)
        check("%s: asks the preflight, and renders on the GPU" % tag,
              rc == 0 and wrote and kernel in log and weighed and REFUSED not in log
              and ON_GPU.search(log) is not None and where in (None, "GPU"),
              "rc %d, %s, %s, %s" % (rc, ("asked, %s GB" % w.group(1)) if kernel in log and w else "never asked",
                                     "refused" if REFUSED in log else "afforded",
                                     "GPU" if ON_GPU.search(log) and where in (None, "GPU") else "not on the GPU"))
        rc, log, wrote = run(tag, mode, args, "auto", 0)
        refused = ("the mode-%s megakernel" % mode) in log and REFUSED in log
        where = ran_on(log, mode)
        left = ON_GPU.search(log) is None if announcesAfter else where == "CPU"
        check("%s: a full card sends -device auto to the CPU" % tag,
              rc == 0 and wrote and refused and FORCED not in log and left,
              "rc %d, %s, %s, %s" % (rc, "refused" if refused else "not refused", "written" if wrote else "no image",
                                     "on the CPU" if left else "stayed on the GPU"))
        rc, log, wrote = run(tag, mode, args, "gpu", 0)
        where = ran_on(log, mode)
        check("%s: ...and -device gpu takes the GPU anyway" % tag,
              rc == 0 and wrote and REFUSED in log and FORCED in log and ON_GPU.search(log) is not None
              and where in (None, "GPU"),
              "rc %d, %s, %s" % (rc, "forced" if FORCED in log else "not forced",
                                 "GPU" if ON_GPU.search(log) and where in (None, "GPU") else "not on the GPU"))

    # The exposure meter's frames: a locked path of two frames, its exposure automatic (no light's
    # `power`), metered in mode D before its first frame renders. The meter must ask before it meters:
    # its [vram] line ahead of the meter's own (both on stdout, whose order the log keeps); and with the
    # card read as full, a refusal (or under -device gpu a forced warning) for the meter and each frame.
    metered = os.path.join(d, "metered.ftsl")
    with open(metered, "w", encoding="utf-8", newline="") as f:
        f.write(ROOM.replace("  power 20  spd preset:d65", "  spd preset:d65"))
    for device, free, what, word in (("auto", None, "asks the preflight before it meters", None),
                                     ("auto", 0, "a full card meters on the CPU", REFUSED),
                                     ("gpu", 0, "...and -device gpu meters on the GPU anyway", FORCED)):
        rc, log, _ = run("metered", "D", ["-camera", "fly", "-spp", "4"], device, free, metered, meter=True)
        at, mt = log.find("mode-D kernel wants"), log.find("[meter]")
        n = log.count(word) if word else 0
        good = rc == 0 and mt >= 0 and 0 <= at < mt and (word is None or n == 3)
        check("D metered: " + what, good,
              "rc %d, %s%s" % (rc, "no meter line" if mt < 0 else "asked first" if 0 <= at < mt
                               else "never asked" if at < 0 else "asked after metering",
                               (", %d of 3 %s" % (n, "refusals" if word == REFUSED else "forced")) if word else ""))
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
