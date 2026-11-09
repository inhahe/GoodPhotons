"""Regression check: a GPU run that renders a mode M camera and then a mode J camera completes.

    python tools/check_gpu_mixed_modes.py [--out png/check_gpu_mixed_modes] [--exe ftrace.exe]

Mode M's GPU gather publishes device globals its kernels read -- the stop flag
(`g_dGatherStop`, DeviceStopScope) and its progress counter (`g_dGatherDone`) -- and kernels of
other modes read the same globals. One left pointing at memory freed when the gather returned
makes the next mode J camera of the run fault in its first wavefront launch ("wf-hits kernel
failed: an illegal memory access was encountered"; GATHER-STOP-DANGLING). The scene is
fogbox.py's box under a homogeneous fog with two cameras, `m` in mode M (with -beams) and then
`j` in mode J; the run must exit 0 and write both images, and mode J must report its wavefront
gather.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fogbox  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gpu_mixed_modes"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    medium = "medium { sigma_a 0.3   sigma_s 1.2   g 0.3   bounds { min 0 0 0   max 1 1 1 } }"
    ftsl = fogbox.write(d, "two.ftsl", fogbox.scene(medium, [fogbox.camera("m", "M", 32),
                                                             fogbox.camera("j", "J", 32)]))
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
    check = fogbox.Checks(52)
    check("the run exits 0", p.returncode == 0, "rc %d" % p.returncode)
    check("no CUDA fault", "illegal memory access" not in log and "kernel failed" not in log)
    check("mode M's camera was gathered on the GPU", "shared photon map (mode M) on" in log)
    check("mode J's wavefront gather ran", "mode J wavefront gather" in log)
    for o in outs:
        check("wrote " + os.path.basename(o), os.path.exists(o))
    return check.verdict(log[-1200:])


if __name__ == "__main__":
    sys.exit(main())
