"""Regression check: a scene whose only light is a fire that emits nothing counts as unlit.

    D:/python314/python.exe -u tools/check_cold_fire.py [--out <dir>] [--exe ftrace.exe]

A `temperature` medium that emits nothing in the band (here `emission_scale 0`) is demoted at
build to the smoke it is made of (Scene::finalizeEmissiveVolumes). The loader's "has the scene a
light?" test runs before the build, where the fire still looks like one, so it asks again after
(ftsl.h, the post-build check): a render must be refused with the usual "scene has no light"
error, and a preview must be lit by the photo studio, as for any scene without a light. Uses
check_fire_r's fire with its emission_scale set to 0, and the same fire at its own scale as the
control, which must render and must not be given the studio. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "loom"))
from check_fire_r import write_fire  # noqa: E402

STUDIO = "has no light, so the preview lights it"
COLD = "has a temperature grid but emits nothing"
UNLIT = "scene has no light"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_cold_fire"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    write_fire(d)
    with open(os.path.join(d, "fire.ftsl"), encoding="utf-8") as f:
        warm = f.read()
    if warm.count("emission_scale 4") != 1:
        raise SystemExit("check_fire_r's scene changed: no single `emission_scale 4` to zero")
    with open(os.path.join(d, "cold.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(warm.replace("emission_scale 4", "emission_scale 0"))

    def run(scene, args, out):
        path = os.path.join(d, out)
        if os.path.exists(path):
            os.remove(path)                       # never judge an earlier run's file
        p = subprocess.run([exe, "-in", scene] + args + ["-o", out], cwd=d, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout + p.stderr, os.path.exists(path)

    ok = True

    def verdict(good, what, log):
        nonlocal ok
        ok &= good
        print("%s: %s" % ("ok" if good else "FAIL", what), flush=True)
        if not good:
            print("    " + log.strip().replace(chr(10), chr(10) + "    ")[-1500:])

    render = ["-mode", "B", "-n", "100000", "-no-meter"]
    rc, log, wrote = run("cold.ftsl", render, "cold_b.png")
    verdict(rc != 0 and UNLIT in log and COLD in log and not wrote,
            "a render of the cold fire is refused as unlit (exit %d)" % rc, log)
    rc, log, wrote = run("cold.ftsl", ["-raster"], "cold_raster.png")
    verdict(rc == 0 and STUDIO in log and wrote,
            "a preview of the cold fire is lit by the studio (exit %d)" % rc, log)
    rc, log, wrote = run("fire.ftsl", render, "warm_b.png")
    verdict(rc == 0 and wrote and COLD not in log, "the warm fire renders (exit %d)" % rc, log)
    rc, log, wrote = run("fire.ftsl", ["-raster"], "warm_raster.png")
    verdict(rc == 0 and wrote and STUDIO not in log,
            "a preview of the warm fire keeps its own light (exit %d)" % rc, log)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
