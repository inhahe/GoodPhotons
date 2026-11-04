"""Regression check: -denoise-shot's temporal denoisers (ROADMAP 11 stage D).

    D:/python314/python.exe -u tools/check_temporal.py [--out <dir>] [--exe ftrace.exe]

  * On temporal_compare's shot, small (10 frames, 96x72, 4 spp, reference 1024 spp): `svgf`, and
    `optix` when ftrace was built with it, come out with less error AND less flicker than the
    noisy shot; `oidn` (when its DLLs are found) with less error.
  * `svgf` keeps no ghost of the light a still floor had before the ball's contact shadow
    reached it: its error about the ball is under half what it is with its history clamp off
    (`-clamp 0`).
  * `svgf`'s history converges: on the same shot held still, its last frame's error is under half
    its first's, while the noisy frames' stays where it was.
  * -denoise-shot refuses a pattern with no frame number, an option it does not know, an output
    that would overwrite the render it denoises (leaving the render as it was), a frame range it
    cannot read, and options before it; makes the output's directory; and names -write-aovs when
    a frame's guides are missing.
Exits 1 on any failure.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_compare as tc  # noqa: E402
from check_motion import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_temporal"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-64s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""))

    n, W, H = 10, 96, 72
    moving = os.path.join(out, "moving")
    res = tc.compare(exe, moving, n, 4, 1024, W, H, ["none", "oidn", "svgf", "optix"], fresh=True, video=False)
    base = res["none"]
    verdict("the noisy shot scores", base is not None)
    for m, need_flicker in (("svgf", True), ("optix", True), ("oidn", False)):
        r = res.get(m)
        if r is None:
            print("(%s not available here: skipped)" % m)
            continue
        good = r[0] < base[0] and (r[1] < base[1] or not need_flicker)
        verdict("%s: less error%s than the noisy shot" % (m, " and flicker" if need_flicker else ""), good,
                "error %.4f vs %.4f, flicker %.4f vs %.4f" % (r[0], base[0], r[1], base[1]))

    # The ball's surroundings, its contact shadow among them: where the light changes as it rolls.
    if res.get("svgf") is not None:
        os.makedirs(os.path.join(moving, "svgf_noclamp"), exist_ok=True)
        p = subprocess.run([exe, "-denoise-shot", "svgf", os.path.join(moving, "noisy", "f%04d.pfm"),
                            os.path.join(moving, "svgf_noclamp", "f%04d.png"), "0", str(n - 1), "-clamp", "0"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        ref = [read_pfm(os.path.join(moving, "ref", "f%04d.pfm" % k)) for k in range(n)]
        eps2 = (0.1 * np.mean([r.mean() for r in ref])) ** 2

        def about_ball(method):
            e = []
            for k in range(n):
                pos = read_pfm(os.path.join(moving, "noisy", "f%04d.position.pfm" % k))
                near = np.linalg.norm(pos - np.array([tc.ball_x(k, n), 0.35, 0.2]), axis=2) < 0.45
                d = read_pfm(os.path.join(moving, method, "f%04d.pfm" % k))
                e.append((((d - ref[k]) ** 2) / (ref[k] ** 2 + eps2)).mean(axis=2)[near].mean())
            return float(np.mean(e))
        if p.returncode == 0:
            ec, eu = about_ball("svgf"), about_ball("svgf_noclamp")
            verdict("svgf leaves no ghost in the ball's moving shadow (vs -clamp 0)", ec < 0.5 * eu,
                    "error about the ball %.3f, unclamped %.3f" % (ec, eu))
        else:
            verdict("svgf -clamp 0 runs", False, p.stderr.strip()[-200:])

    still = os.path.join(out, "still")
    rs = tc.compare(exe, still, n, 4, 1024, W, H, ["none", "svgf"], fresh=True, video=False, still=True)
    verdict("the still shot denoises", rs.get("svgf") is not None)
    es, en = tc.frame_errors(still, "svgf", n), tc.frame_errors(still, "none", n)
    verdict("svgf's history converges on a still shot", es[-1] < 0.5 * es[0] and en[-1] > 0.5 * en[0],
            "svgf %.4f -> %.4f, noisy %.4f -> %.4f" % (es[0], es[-1], en[0], en[-1]))

    p = subprocess.run([exe, "-denoise-shot", "svgf", os.path.join(moving, "noisy", "f.pfm"),
                        os.path.join(out, "x%04d.png"), "0", "1"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    verdict("refuses a pattern with no frame number", p.returncode != 0 and "frame number" in p.stderr,
            p.stderr.strip()[-200:] if p.returncode == 0 else "")
    p = subprocess.run([exe, "-denoise-shot", "svgf", os.path.join(moving, "noisy", "f%04d.pfm"),
                        os.path.join(out, "x%04d.png"), "0", "1", "-pases", "2"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    verdict("refuses an option it does not know", p.returncode != 0 and "-pases" in p.stderr,
            p.stderr.strip()[-200:])
    noisy = os.path.join(moving, "noisy", "f%04d.pfm")
    for name, cmd, phrase in (
            ("refuses to overwrite the render it denoises",
             ["-denoise-shot", "svgf", noisy, os.path.join(moving, "noisy", "f%04d.png"), "0", "1"], "overwrite"),
            ("refuses a frame range it cannot read", ["-denoise-shot", "svgf", noisy, os.path.join(out, "x%04d.png"),
                                                      "0", "4o"], "frame numbers"),
            ("refuses options before it", ["-exposure", "2", "-denoise-shot", "svgf", noisy,
                                            os.path.join(out, "x%04d.png"), "0", "1"], "comes first")):
        before = open(os.path.join(moving, "noisy", "f0000.pfm"), "rb").read()
        p = subprocess.run([exe] + cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        kept = open(os.path.join(moving, "noisy", "f0000.pfm"), "rb").read() == before
        verdict(name, p.returncode != 0 and phrase in p.stderr and kept, p.stderr.strip()[-160:])
    fresh = os.path.join(out, "made", "here")
    shutil.rmtree(os.path.join(out, "made"), ignore_errors=True)
    p = subprocess.run([exe, "-denoise-shot", "none", noisy, os.path.join(fresh, "f%04d.png"), "0", "0"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    verdict("makes the output's directory", p.returncode == 0 and os.path.exists(os.path.join(fresh, "f0000.png")),
            (p.stdout + p.stderr).strip()[-160:])
    bare = os.path.join(out, "bare")
    os.makedirs(bare, exist_ok=True)
    shutil.copy(os.path.join(moving, "noisy", "f0000.pfm"), os.path.join(bare, "f0000.pfm"))
    p = subprocess.run([exe, "-denoise-shot", "svgf", os.path.join(bare, "f%04d.pfm"),
                        os.path.join(bare, "o%04d.png"), "0", "0"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    verdict("names -write-aovs when a frame's guides are missing",
            p.returncode != 0 and "-write-aovs" in p.stderr, p.stderr.strip()[-200:])

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
