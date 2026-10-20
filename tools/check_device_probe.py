"""Regression check: the GPU half of ftrace agrees with the host half.

    python tools/check_device_probe.py [--exe ftrace.exe]

nvcc compiles render_cuda.cu twice -- the host half with the host compiler, the device half with
its own front end -- and both halves read the HOST compiler's headers. A host-toolchain change can
therefore break the device alone with no CUDA error and a plausible-looking image. It happened:
with MSVC 14.5x (VS 2026) the float overload of copysign compiled, on the device, to a function
that returned 0 for every input; onb() NaN-ed every sampled direction, mode B rendered black and
modes D and R read 0.58x and 0.72x of the truth, while the deterministic mode W looked perfect.

`FTRACE_DEVICE_PROBE=1` makes ftrace print, before any render, the sizeof() of every struct the
host uploads as each half sees it, and the value of every math function the device code calls
(float and double, edge inputs included) as each half computes it. This runs it on a tiny mode-W
render and fails on any disagreement. Run it after every change to the host compiler or the CUDA
toolkit -- it is a few seconds, and it is the only thing that catches this class of bug.
Exits 0 when the two halves agree, 1 otherwise (or when no GPU is available).
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    a = ap.parse_args()
    env = dict(os.environ, FTRACE_DEVICE_PROBE='1')
    out = os.path.join(ROOT, 'scraps', '_device_probe.png')
    r = subprocess.run([os.path.abspath(a.exe), '-in', os.path.join(ROOT, 'scenes', 'cornell.ftsl'),
                        '-mode', 'W', '-device', 'gpu', '-spp', '1', '-r', '16', '12', '-no-meter',
                        '-o', out], capture_output=True, text=True, env=env)
    text = r.stdout + r.stderr
    lines = [l for l in text.splitlines() if l.startswith('[probe]')]
    for l in lines:
        print(l)
    m = re.search(r'\[probe\] (\d+) disagreement', text)
    if not m:
        print('FAIL: the probe did not run (no GPU, or a binary older than 0.385.0).')
        return 1
    bad = int(m.group(1))
    print('PASS' if bad == 0 else f'FAIL: {bad} host/device disagreement(s)')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
