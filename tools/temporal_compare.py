"""The temporal denoisers compared on one shot (ROADMAP 11 stage D).

    D:/python314/python.exe -u tools/temporal_compare.py [--out png/temporal_compare] [--frames 24]
        [--spp 8] [--ref-spp 2048] [--res 160 120] [--methods none,oidn,svgf,optix] [--exe ftrace.exe]

Renders a short shot -- a striped ball rolling across the checkered floor of a closed room, lit
mostly indirectly by a small panel, while the camera dollies past -- twice: noisy (--spp, with -hdr -write-aovs) and as a reference (--ref-spp). Each frame's
scene carries motion keys about its own instant (the ball's and the camera's), so the guide pass
knows how everything moves, and no shutter, so the frames are sharp. It then denoises the noisy
shot with each method (`ftrace -denoise-shot`) and scores the result against the reference:

  error    relMSE: (d - r)^2 / (r^2 + (0.1 m)^2) per pixel and channel, m the reference's mean,
           averaged over the shot
  flicker  for each pair of neighbouring frames, the mean |(d_t - d_t-1) - (r_t - r_t-1)| over the
           reference's mean: how far the result's frame-to-frame change strays from the true one

and writes report.txt, and compare.mp4: the noisy shot, then each method, then the reference,
side by side (in --methods order), all developed at the reference's first-frame exposure.
Renders that exist are reused (--fresh renders again). A method the binary lacks (optix without
the OptiX headers) is reported and skipped.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_motion import read_pfm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def ball_x(f, n):
    return -1.2 + 2.4 * f / (n - 1)


def eye(f, n):
    return (-0.6 + 1.2 * f / (n - 1), 1.1, 4.0)


def v3(v):
    return " ".join(repr(float(x)) for x in v)


def frame_scene(k, n, W, H, still=False):
    """Frame k of n, with motion keys at k - 0.5, k and k + 0.5 (`still`: everything held at frame
    0's place, every frame)."""
    t = (lambda d: 0.0) if still else (lambda d: k + d)
    keys = "  ".join("key { translate %r 0.35 0.2  rotate 0 0 %r }" % (ball_x(t(d), n), -60.0 * t(d))
                     for d in (-0.5, 0.0, 0.5))
    cams = "  ".join("key { eye %s  look_at 0 0.5 0 }" % v3(eye(t(d), n)) for d in (-0.5, 0.0, 0.5))
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'pattern "chk" { expr "0.2 + 0.6 * smoothstep(-0.05, 0.05, sin(12 * x) * sin(12 * z))" }',
        'pattern "str" { expr "0.25 + 0.6 * smoothstep(-0.1, 0.1, sin(20 * y))" }',
        'material "floor" { type diffuse reflect pattern:chk }',
        'material "wall"  { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "green" { type diffuse reflect rgb 0.15 0.6 0.2 }',
        'material "ball"  { type diffuse reflect pattern:str }',
        # a closed room, lit by a small panel: mostly indirect light, the noise a denoiser is for
        "quad { origin -2 0 -2  u 0 0 7  v 4 0 0  material floor }",
        "quad { origin -2 3 -2  u 4 0 0  v 0 0 7  material wall }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material wall }",
        "quad { origin -2 0 5  u 0 3 0  v 4 0 0  material wall }",
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 7  material red }",
        "quad { origin 2 0 -2  u 0 0 7  v 0 3 0  material green }",
        "light area { origin -0.3 2.98 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  spd preset:bb6500  power 60 }",
        "group { motion { %s }  sphere { center 0 0 0  radius 0.35  material ball } }" % keys,
        'camera "c" { up 0 1 0  fov_y 45  motion { %s }  film { res %d %d } }' % (cams, W, H),
    ]) + NL


def run(cmd, what):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-2000:])
        raise SystemExit(what + " failed: " + " ".join(cmd))
    return p.stdout + p.stderr


def compare(exe, out, n, spp, ref_spp, W, H, methods, fresh=False, video=True, still=False):
    """Render the shot (noisy and reference), denoise it with each of `methods`, score each, write
    report.txt (and compare.mp4 when `video`). Returns {method: (error, flicker) or None}."""
    for d in ("scenes", "noisy", "ref"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # --- the two renders of the shot ---
    for k in range(n):
        scene = os.path.join(out, "scenes", "f%04d.ftsl" % k)
        with open(scene, "w", encoding="utf-8", newline="") as f:
            f.write(frame_scene(k, n, W, H, still))
        for kind, kspp, extra in (("noisy", spp, ["-write-aovs"]), ("ref", ref_spp, [])):
            png = os.path.join(out, kind, "f%04d.png" % k)
            if os.path.exists(png[:-4] + ".pfm") and not fresh:
                continue
            run([exe, "-in", scene, "-mode", "R", "-spp", str(kspp), "-seed", str(k + 1), "-hdr", "-no-meter",
                 "-o", png] + extra, "render")
        print("rendered frame %d of %d" % (k + 1, n), flush=True)
    return score(exe, out, n, spp, ref_spp, W, H, methods, video)


def frame_errors(out, method, n):
    """`method`'s relMSE against the reference, frame by frame."""
    ref = [read_pfm(os.path.join(out, "ref", "f%04d.pfm" % k)) for k in range(n)]
    eps2 = (0.1 * np.mean([r.mean() for r in ref])) ** 2
    return [float(np.mean((read_pfm(os.path.join(out, method, "f%04d.pfm" % k)) - ref[k]) ** 2 /
                          (ref[k] ** 2 + eps2))) for k in range(n)]


def score(exe, out, n, spp, ref_spp, W, H, methods, video):
    """The second half of compare(): the shot is rendered."""
    ref = [read_pfm(os.path.join(out, "ref", "f%04d.pfm" % k)) for k in range(n)]
    lum0 = ref[0] @ np.array([0.2126, 0.7152, 0.0722])
    expo = float(0.9 / max(np.percentile(lum0, 99), 1e-30))   # a float: repr of numpy's is "np.float64(...)"

    # --- each method ---
    def denoise(method, src, name):
        """`ftrace -denoise-shot method` over shot `src`, into <out>/<name>; (ok, last log line)."""
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        p = subprocess.run([exe, "-denoise-shot", method, os.path.join(out, src, "f%04d.pfm"),
                            os.path.join(d, "f%04d.png"), "0", str(n - 1), "-exposure", repr(expo)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode == 0, ((p.stdout + p.stderr).strip().splitlines() or [""])[-1]

    rows, videos = [], [os.path.join(out, "noisy_dev")]
    denoise("none", "noisy", "noisy_dev")            # the noisy shot, developed alike
    mean_ref = np.mean([r.mean() for r in ref])
    eps2 = (0.1 * mean_ref) ** 2        # relMSE's floor, in the images' own scale
    for method in methods:
        ok, msg = denoise(method, "noisy", method)
        if not ok:
            rows.append((method, None, None, msg))
            continue
        res = [read_pfm(os.path.join(out, method, "f%04d.pfm" % k)) for k in range(n)]
        err = np.mean([np.mean((d - r) ** 2 / (r ** 2 + eps2)) for d, r in zip(res, ref)])
        flick = np.mean([np.mean(np.abs((res[k] - res[k - 1]) - (ref[k] - ref[k - 1])))
                         for k in range(1, n)]) / mean_ref
        rows.append((method, err, flick, ""))
        videos.append(os.path.join(out, method))
    denoise("none", "ref", "ref_dev")
    videos.append(os.path.join(out, "ref_dev"))

    lines = ["temporal denoisers on %d frames at %dx%d, %d spp (reference %d spp)" % (n, W, H, spp, ref_spp),
             "%-8s %12s %12s" % ("method", "error", "flicker")]
    for method, err, flick, msg in rows:
        lines.append("%-8s %12s %12s  %s" % (method, "%.5g" % err if err is not None else "-",
                                             "%.5g" % flick if flick is not None else "-", msg))
    lines.append("video: " + " | ".join(os.path.basename(v) for v in videos))
    report = NL.join(lines) + NL
    print(report)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(report)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for v in videos:
        cmd += ["-framerate", "12", "-i", os.path.join(v, "f%04d.png")]
    cmd += ["-filter_complex", "".join("[%d:v]" % i for i in range(len(videos))) +
            "hstack=inputs=%d,scale=trunc(iw/2)*2:trunc(ih/2)*2" % len(videos),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", os.path.join(out, "compare.mp4")]
    if video and len(videos) > 1:
        subprocess.run(cmd)
    return {m: ((e, f) if e is not None else None) for m, e, f, _ in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "temporal_compare"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--spp", type=int, default=8)
    ap.add_argument("--ref-spp", type=int, default=2048)
    ap.add_argument("--res", type=int, nargs=2, default=(160, 120))
    ap.add_argument("--methods", default="none,oidn,svgf,optix")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    compare(os.path.abspath(a.exe), os.path.abspath(a.out), a.frames, a.spp, a.ref_spp, a.res[0], a.res[1],
            [m for m in a.methods.split(",") if m], a.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
