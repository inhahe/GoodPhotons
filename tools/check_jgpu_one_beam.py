"""Regression check: GPU mode J gathers a light-side map of a single sub-beam.

    python tools/check_jgpu_one_beam.py [--out png/check_jgpu_one_beam] [--exe ftrace.exe]

GPU mode J builds its beam tree on the device (an LBVH; the host skips its own). An LBVH of one
primitive has no internal node: its one leaf must be the root, or the map reaches the device with
no tree, its merges keeping their weight and gathering nothing -- the realization silently dark
by their share. renderBdptCuda ends the process on a map with beams and no tree, so such a
regression fails this check loudly. The scene is fogbox.py's box filled by a homogeneous fog, so
its light lies inside the medium: with one light subpath a realization (-n 1), no bounce after
the first (-max-bounce 1) and no splitting (-beamsplit 1000), every realization is exactly one
chord, whatever the random numbers. Sixteen of them (-beamepochs spp). The render must exit 0,
build the one-leaf LBVH, and write its image.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jgpu_one_beam"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    medium = "medium { sigma_a 0.2   sigma_s 2.0   bounds { min 0 0 0   max 1 1 1 } }"
    ftsl = fogbox.write(d, "one.ftsl", fogbox.scene(medium))
    png = os.path.join(d, "one.png")
    if os.path.exists(png):   # a render that writes nothing must not pass
        os.remove(png)
    cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-spp", "16", "-seed", "1",
           "-n", "1", "-max-bounce", "1", "-beamsplit", "1000", "-beamepochs", "spp",
           "-no-meter", "-window-min", "-o", png]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log = p.stdout + p.stderr
    with open(os.path.join(d, "one.log"), "w", encoding="utf-8") as f:
        f.write(log)
    check = fogbox.Checks(50)
    m = re.search(r"mode J: (\d+) beams from (\d+) light subpaths", log)
    check("epoch 0's map is one chord", m is not None and m.group(1) == "1",
          "%s beams" % (m.group(1) if m else "?"))
    check("the render exits 0", p.returncode == 0, "rc %d" % p.returncode)
    check("no map reached the device without a tree", "without a tree" not in log)
    check("the LBVH over one sub-beam is its leaf",
          re.search(r"\[jlbvh\] device LBVH over 1 sub-beams .*LBVH has 1\)", log) is not None)
    check("the image was written", os.path.exists(png))
    return check.verdict(log[-1500:])


if __name__ == "__main__":
    sys.exit(main())
