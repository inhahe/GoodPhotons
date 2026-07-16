#!/usr/bin/env python3
r"""Render the showcase camera-curve flyby and assemble it into a video/GIF.

Runs the `camera_curve "fly"` flyby in scenes/gallery_settled.ftsl through
ftrace (rasterized preview by default, or any transport mode you pick), then
converts the per-frame PNGs into an animated GIF/MP4 with ffmpeg at a chosen
playback speed.

Every run prints its full CLI options, the input scene file, and every
resolved parameter value before doing any work.

Examples
--------
    # Defaults: rasterized, 640x400, 30 fps, -> .\showcase.gif
    python tools/showcase_flyby.py

    # 1280x720 BDPT flyby at 24 fps into an mp4; 8 s budget per frame
    python tools/showcase_flyby.py --mode d --res 1280 720 --fps 24 \
        --time 8 --out showcase.mp4
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root = parent of this tools/ dir, so the script works from any cwd.
ROOT = Path(__file__).resolve().parent.parent
FTRACE = ROOT / "build" / "bin" / "ftrace.exe"
DEFAULT_SCENE = "scenes/gallery_settled.ftsl"
DEFAULT_CAMERA = "fly"          # the camera_curve path base name in the scene
FRAME_DIR = ROOT / "png" / "showcase_fly"   # flyby series gets its own subdir
FRAME_STEM = "showcase"          # -> png/showcase_fly/showcase_fly000.png ...


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="showcase_flyby.py",
        description="Render the showcase flyby and convert it to a GIF/MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--res", nargs=2, type=int, metavar=("W", "H"),
                   default=[640, 400],
                   help="render resolution in pixels")
    p.add_argument("--fps", type=float, default=30.0,
                   help="playback speed of the output video (frames per second)")
    p.add_argument("--out", default="showcase.gif",
                   help="output video filename; extension picks the format "
                        "(.gif, .mp4, ...) - converted with ffmpeg")
    p.add_argument("--mode", default="raster",
                   help="render mode: 'raster' (fast solid-shaded preview) or a "
                        "transport mode letter (a, b, c, d, u, m, r, ...)")
    p.add_argument("--time", type=float, default=None, metavar="SEC",
                   help="per-frame time budget in seconds for non-raster modes. "
                        "If omitted, the window is held open (run until you close "
                        "it) after the flyby renders.")
    p.add_argument("--scene", default=DEFAULT_SCENE,
                   help="input scene (.ftsl) file")
    p.add_argument("--camera", default=DEFAULT_CAMERA,
                   help="camera_curve/path base name to render (selects all its "
                        "frames)")
    p.add_argument("--spp", type=int, default=None,
                   help="samples per pixel per frame (non-raster modes only)")
    p.add_argument("--keep-frames", action="store_true",
                   help="keep the per-frame PNGs after building the video "
                        "(default: leave them in png/showcase_fly/ anyway)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the ftrace/ffmpeg commands but do not run them")
    return p


def print_run_banner(parser: argparse.ArgumentParser, args: argparse.Namespace,
                     raster: bool) -> None:
    print("=" * 72)
    print("showcase_flyby.py - showcase camera-curve flyby renderer")
    print("=" * 72)
    # Full CLI options every run, as requested.
    print(parser.format_help())
    print("-" * 72)
    print(f"input scene file : {args.scene}")
    print("resolved parameters:")
    print(f"  mode           : {args.mode} ({'rasterized preview' if raster else 'transport mode'})")
    print(f"  resolution     : {args.res[0]} x {args.res[1]}")
    print(f"  fps (playback) : {args.fps}")
    print(f"  output         : {args.out}")
    print(f"  camera path    : {args.camera}")
    if raster:
        time_desc = "(n/a for raster - animates all frames then exits)"
    elif args.time is not None:
        time_desc = f"{args.time} s/frame"
    else:
        time_desc = "(none -> hold window open until you close it)"
    print(f"  per-frame time : {time_desc}")
    print(f"  spp            : {args.spp if args.spp is not None else '(scene default)'}")
    print(f"  frame PNG dir  : {FRAME_DIR}")
    print(f"  frame stem     : {FRAME_STEM}")
    print(f"  ftrace exe     : {FTRACE}")
    print(f"  keep frames    : {args.keep_frames}")
    print(f"  dry run        : {args.dry_run}")
    print("-" * 72)


def build_ftrace_cmd(args: argparse.Namespace, raster: bool) -> list[str]:
    frame_out = FRAME_DIR / f"{FRAME_STEM}.png"
    cmd = [str(FTRACE),
           "-in", args.scene,
           "-camera", args.camera,
           "-r", str(args.res[0]), str(args.res[1]),
           "-window",
           "-o", str(frame_out)]
    if raster:
        # Raster flyby animates every frame in the window then exits, writing one
        # PNG per frame - exactly what we want before handing off to ffmpeg.
        cmd.insert(cmd.index("-camera"), "-raster")
    else:
        cmd[cmd.index("-camera"):cmd.index("-camera")] = ["-mode", args.mode]
        if args.time is not None:
            # Per-frame budget; window still shown, then the flyby advances.
            cmd += ["-time", str(args.time)]
        else:
            # No budget: hold the window open (run until the user closes it).
            cmd += ["-keepwindow"]
        if args.spp is not None:
            cmd += ["-spp", str(args.spp)]
    return cmd


def find_frames(camera: str) -> list[Path]:
    return sorted(FRAME_DIR.glob(f"{FRAME_STEM}_{camera}*.png"))


def detect_pattern(frames: list[Path]) -> tuple[str, int]:
    """Return (ffmpeg %0Nd pattern, start_number) for a numbered frame set."""
    m = re.search(r"(\d+)\.png$", frames[0].name)
    if not m:
        raise SystemExit(f"[error] cannot parse frame number from {frames[0].name}")
    pad = len(m.group(1))
    start = int(m.group(1))
    prefix = frames[0].name[: m.start(1)]
    pattern = str(FRAME_DIR / f"{prefix}%0{pad}d.png")
    return pattern, start


def build_ffmpeg_cmd(pattern: str, start: int, args: argparse.Namespace) -> list[str]:
    out = args.out
    ext = os.path.splitext(out)[1].lower()
    base = ["ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-start_number", str(start),
            "-i", pattern]
    if ext == ".gif":
        # Two filters in one graph: build an optimal 256-colour palette, then map.
        vf = ("split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
              "[s1][p]paletteuse=dither=bayer:bayer_scale=3")
        return base + ["-vf", vf, "-loop", "0", out]
    # mp4/webm/etc: yuv420p for broad compatibility, even dims via scale pad.
    return base + ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                   "-pix_fmt", "yuv420p", out]


def run(cmd: list[str], sandbox_note: str = "") -> int:
    print(f"[run] {' '.join(cmd)}")
    if sandbox_note:
        print(f"      ({sandbox_note})")
    return subprocess.call(cmd)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    raster = args.mode.lower() in ("raster", "r-raster", "preview")

    print_run_banner(parser, args, raster)

    if not FTRACE.exists():
        print(f"[error] ftrace.exe not found at {FTRACE} - build it first.")
        return 2
    if shutil.which("ffmpeg") is None:
        print("[error] ffmpeg not found on PATH - install it to build the video.")
        return 2

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale frames from a previous run so ffmpeg only sees this set.
    for old in FRAME_DIR.glob(f"{FRAME_STEM}_{args.camera}*.png"):
        if not args.dry_run:
            old.unlink()

    ftrace_cmd = build_ftrace_cmd(args, raster)
    print(f"[plan] ftrace : {' '.join(ftrace_cmd)}")
    print(f"[plan] ffmpeg : will assemble frames from {FRAME_DIR} at {args.fps} fps -> {args.out}")

    if args.dry_run:
        print("[dry-run] not executing.")
        return 0

    rc = run(ftrace_cmd)
    if rc != 0:
        print(f"[error] ftrace exited with code {rc}")
        return rc

    frames = find_frames(args.camera)
    if not frames:
        print(f"[error] no frames matched {FRAME_DIR}/{FRAME_STEM}_{args.camera}*.png")
        return 3
    print(f"[info] rendered {len(frames)} frames: {frames[0].name} .. {frames[-1].name}")

    pattern, start = detect_pattern(frames)
    ffmpeg_cmd = build_ffmpeg_cmd(pattern, start, args)
    rc = run(ffmpeg_cmd)
    if rc != 0:
        print(f"[error] ffmpeg exited with code {rc}")
        return rc

    out_path = Path(args.out).resolve()
    print("-" * 72)
    print(f"[done] wrote {out_path} ({len(frames)} frames @ {args.fps} fps)")
    if not args.keep_frames:
        print(f"[info] per-frame PNGs left in {FRAME_DIR} (use --keep-frames "
              "to silence this note; they are not deleted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
