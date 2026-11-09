"""Regression check: GPU mode J gathers a light-side map of a single sub-beam.

    python tools/check_jgpu_one_beam.py [--out png/check_jgpu_one_beam] [--exe ftrace.exe]

GPU mode J builds its beam tree on the device (an LBVH) and the host skips its own, but an LBVH
of one primitive has no internal node, so `buildBeamLbvhDevice` refuses it; BeamMap::build then
keeps the host tree, and uploadBeamMapCuda must upload it. Until 0.434.36 the upload converted
no host nodes whenever the LBVH was wanted, so a one-sub-beam map reached the device with no
tree: its merges kept their weight and gathered nothing, the realization silently dark by their
share. renderBdptCuda now refuses a map with beams and no tree (std::logic_error), so the old
upload fails this check loudly. The scene is a closed box lit from above with a 5 cm medium in
a corner, traced with 64 light subpaths a realization and no beam splitting (-beamsplit 1000):
at seed 1 epoch 0's map is one chord, and of the 16 realizations (-beamepochs spp) many hold one
or none. The render must exit 0 and say it kept the host tree for a one-sub-beam map.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENE = NL.join([
    "scene { units meters }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium { sigma_a 0.2   sigma_s 2.0   bounds { min 0.1 0.1 0.1   max 0.15 0.15 0.15 } }",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    'camera "cam" {',
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 48 48 }",
    "}"]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jgpu_one_beam"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "tiny.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    png = os.path.join(d, "tiny.png")
    if os.path.exists(png):   # a render that writes nothing must not pass
        os.remove(png)
    cmd = [exe, "-in", ftsl, "-mode", "J", "-device", "gpu", "-spp", "16", "-seed", "1",
           "-n", "64", "-beamsplit", "1000", "-beamepochs", "spp", "-no-meter", "-window-min",
           "-o", png]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log = p.stdout + p.stderr
    with open(os.path.join(d, "tiny.log"), "w", encoding="utf-8") as f:
        f.write(log)
    fails = []

    def check(name, ok, detail=""):
        print("%-50s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    m = re.search(r"mode J: (\d+) beams from (\d+) light subpaths", log)
    check("epoch 0's map is one chord", m is not None and m.group(1) == "1",
          "%s beams" % (m.group(1) if m else "?"))
    check("the render exits 0", p.returncode == 0, "rc %d" % p.returncode)
    check("no map reached the device without a tree", "without a tree" not in log)
    check("a one-sub-beam map kept the host tree",
          re.search(r"\[jlbvh\] a map of 1 sub-beam\(s\): no LBVH; keeping the host tree "
                    r"\(1 node\(s\)\)", log) is not None)
    check("the image was written", os.path.exists(png))
    if fails:
        print(log[-1500:])
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
