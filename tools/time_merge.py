"""Wall time of the merge modes (U, J, M, S), two builds paired and interleaved, on the CPU and the GPU.

    python -u tools/time_merge.py --old <exe> [--new ftrace.exe] [--rounds 6] [--only <substring>]

The cases: each merge at its own radius on a dense scene (scenes/_merge_glossy_bowl.ftsl for mode U,
scenes/_caustic_box.ftsl for J, M and S; mode M at 4 M photons on the CPU and 20 M on the GPU, where
building the 129 M-photon map is most of the time), and modes U and J at a fine radius over a sparse
10 m box (png/check_vcm_grid_cap/box.ftsl, which tools/check_vcm_grid_cap.py writes; those cases are
skipped without it). Mode J takes -beamfreeze, so its light side does not pace by the work it counts.

A case runs its rounds before the next case begins, each round once with each build, the order
alternating (old first in the even rounds, new first in the odd; an odd --rounds is taken up to the
next even one), so whatever the second of two runs gains or loses (the first's teardown still in
flight, clocks ramping) falls on both builds alike; prints each case's median per build and the
ratio new / old, with every run's time. Run it on an idle machine: a check sweep's load moves the
times by more than most changes it is meant to measure (known-issues MERGE-HASH-TIMING).
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOWL = os.path.join(ROOT, "scenes", "_merge_glossy_bowl.ftsl")
CBOX = os.path.join(ROOT, "scenes", "_caustic_box.ftsl")
BOX = os.path.join(ROOT, "png", "check_vcm_grid_cap", "box.ftsl")
CASES = [
    ("bowl_U_cpu", BOWL, ["-mode", "U", "-device", "cpu", "-spp", "128", "-max-bounce", "24"]),
    ("bowl_U_cpu1", BOWL, ["-mode", "U", "-device", "cpu", "-spp", "16", "-max-bounce", "24", "-t", "1"]),
    ("bowl_U_gpu", BOWL, ["-mode", "U", "-device", "gpu", "-spp", "512", "-max-bounce", "24"]),
    ("cbox_J_cpu", CBOX, ["-mode", "J", "-device", "cpu", "-spp", "128", "-beamfreeze"]),
    ("cbox_J_gpu", CBOX, ["-mode", "J", "-device", "gpu", "-spp", "256", "-beamfreeze"]),
    ("cbox_M_cpu", CBOX, ["-mode", "M", "-device", "cpu", "-n", "4000000", "-spp", "8", "-beamfreeze"]),
    ("cbox_M_gpu", CBOX, ["-mode", "M", "-device", "gpu", "-n", "20000000", "-spp", "32", "-beamfreeze"]),
    ("cbox_S_gpu", CBOX, ["-mode", "S", "-device", "gpu", "-spp", "64"]),
    ("box_U_cpu_fine", BOX, ["-mode", "U", "-device", "cpu", "-spp", "16", "-pmradius", "1e-4"]),
    ("box_U_gpu_fine", BOX, ["-mode", "U", "-device", "gpu", "-spp", "64", "-pmradius", "1e-4"]),
    ("box_J_cpu_fine", BOX, ["-mode", "J", "-device", "cpu", "-spp", "16", "-jsurf-radius", "1e-4", "-beamfreeze"]),
    ("box_J_gpu_fine", BOX, ["-mode", "J", "-device", "gpu", "-spp", "64", "-jsurf-radius", "1e-4", "-beamfreeze"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True, help="the build to time against")
    ap.add_argument("--new", default=os.path.join(ROOT, "ftrace.exe"), help="the build to time")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--only", default="", help="run only the cases whose name holds this")
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "time_merge"))
    a = ap.parse_args()
    exe = {"old": os.path.abspath(a.old), "new": os.path.abspath(a.new)}
    for lab, path in exe.items():
        if not os.path.exists(path):
            print("no %s build at %s" % (lab, path))
            return 2
    os.makedirs(a.out, exist_ok=True)
    for tag, scene, args in CASES:
        if a.only not in tag:
            continue
        if not os.path.exists(scene):
            print("%-16s skipped: no %s%s" % (tag, os.path.relpath(scene, ROOT),
                                              " (tools/check_vcm_grid_cap.py writes it)" if scene == BOX else ""))
            continue
        t = {"old": [], "new": []}
        for rnd in range(a.rounds + a.rounds % 2):   # as many old-first rounds as new-first
            for lab in (("old", "new") if rnd % 2 == 0 else ("new", "old")):
                png = os.path.join(a.out, "%s_%s.png" % (tag, lab))
                cmd = [exe[lab], "-in", scene, "-no-meter", "-seed", "7", "-window-min", "-o", png] + args
                t0 = time.perf_counter()
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
                t[lab].append(time.perf_counter() - t0)
                if r.returncode != 0:
                    print("%s %s failed: %s" % (tag, lab, (r.stdout + r.stderr)[-400:]), flush=True)
                    return 1
        mo, mn = statistics.median(t["old"]), statistics.median(t["new"])
        print("%-16s old %.2f s  new %.2f s  new/old %.3f  (old %s; new %s)"
              % (tag, mo, mn, mn / mo, " ".join("%.2f" % x for x in t["old"]),
                 " ".join("%.2f" % x for x in t["new"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
