"""Regression check: a GPU run that renders a mode M camera and then a mode J camera completes.

    python tools/check_gpu_mixed_modes.py [--out png/check_gpu_mixed_modes] [--exe ftrace.exe]

Mode M's GPU gather publishes a mapped host flag in the device global `g_dGatherStop`, which the
gather's kernels poll for `-stop`, and frees the flag when it returns. Kernels of other modes
read the same global (dGatherStopped). Until 0.434.35 the global was left pointing at the freed
mapping, so the next mode J camera of the same run faulted in its first wavefront launch ("wf-hits
kernel failed: an illegal memory access was encountered") -- GATHER-STOP-DANGLING. The scene is
a small fog box with two cameras, `m` in mode M (with -beams) and then `j` in mode J; the run
must exit 0 and write both images, and mode J must report its wavefront gather.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

CAMERA = NL.join([
    'camera "%s" {',
    "    mode %s",
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 32 32 }",
    "}"])
SCENE = NL.join([
    "scene { units meters }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium { sigma_a 0.3   sigma_s 1.2   g 0.3   bounds { min 0 0 0   max 1 1 1 } }",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    CAMERA % ("m", "M"),
    CAMERA % ("j", "J")]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gpu_mixed_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "two.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)
    png = os.path.join(d, "two.png")
    outs = [os.path.join(d, "two_%s.png" % c) for c in ("m", "j")]
    for stale in outs:   # a render that writes nothing must not pass
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [exe, "-in", ftsl, "-device", "gpu", "-spp", "2", "-n", "100000", "-beams",
           "-beamcount", "20000", "-seed", "1", "-no-meter", "-window-min", "-o", png]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log = p.stdout + p.stderr
    with open(os.path.join(d, "two.log"), "w", encoding="utf-8") as f:
        f.write(log)
    fails = []

    def check(name, ok, detail=""):
        print("%-52s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    check("the run exits 0", p.returncode == 0, "rc %d" % p.returncode)
    check("no CUDA fault", "illegal memory access" not in log and "kernel failed" not in log)
    check("mode M's camera was gathered on the GPU", "shared photon map (mode M) on" in log)
    check("mode J's wavefront gather ran", "mode J wavefront gather" in log)
    for o in outs:
        check("wrote " + os.path.basename(o), os.path.exists(o))
    if fails:
        print(log[-1200:])
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
