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
    # Defaults: rasterized, 640x480, 30 fps, -> .\showcase.gif
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

_NUM = r"[0-9]*\.?[0-9]+"        # a bare integer or decimal (no sign/exponent needed here)


def _strip_ftsl_comments(text: str) -> str:
    """Drop FTSL line comments (# ... to end of line) so a commented-out `fps`
    can't be mistaken for a real one."""
    return re.sub(r"#[^\n]*", "", text)


def _block_body(text: str, brace_idx: int) -> str:
    """Return the body between the `{` at brace_idx and its matching `}`."""
    depth = 0
    for i in range(brace_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx + 1:i]
    return text[brace_idx + 1:]


def read_scene_fps(scene: str, camera: str) -> float | None:
    """Read the playback fps the scene authors for this flyby, mirroring ftrace's
    resolution order: the flyby camera's own `fps`, else the scene-level `fps`
    default. Returns None if neither is present (caller falls back to 30)."""
    p = Path(scene)
    if not p.is_absolute() and not p.exists():
        p = ROOT / scene                         # tolerate being run from another cwd
    try:
        text = _strip_ftsl_comments(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    # 1. The flyby camera's own `fps` (camera_curve/path/orbit "camera" { ... fps N }).
    m = re.search(r'camera_(?:curve|path|orbit)\s+"' + re.escape(camera) + r'"\s*\{', text)
    if m:
        fm = re.search(r"\bfps\s+(" + _NUM + r")", _block_body(text, m.end() - 1))
        if fm:
            return float(fm.group(1))
    # 2. The scene-level default (scene { ... fps N }).
    sm = re.search(r"\bscene\s*\{", text)
    if sm:
        fm = re.search(r"\bfps\s+(" + _NUM + r")", _block_body(text, sm.end() - 1))
        if fm:
            return float(fm.group(1))
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="showcase_flyby.py",
        description="Render the showcase flyby and convert it to a GIF/MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--res", nargs=2, type=int, metavar=("W", "H"),
                   default=[640, 480],
                   help="render resolution in pixels")
    p.add_argument("--fps", type=float, default=None,
                   help="playback speed of the output video (frames per second). "
                        "If omitted, read from the scene: the flyby camera's `fps`, "
                        "else the scene's top-level `fps` default, else 30")
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
    p.add_argument("--preview", dest="preview", action="store_true",
                   help="also show the live ANSI terminal thumbnail (-preview); "
                        "off by default, the OS live window (-window) is always on")
    p.add_argument("--explore", action="store_true",
                   help="skip rendering the flyby: open the interactive fly viewer at "
                        "the first camera frame and let you explore it yourself "
                        "(raster; Space/Shift to fly, mouse to look, wheel to dolly, "
                        "Ctrl+wheel = step size, C = wall collision, P prints a camera "
                        "block, close the window to finish). Implies --no-meter.")
    p.add_argument("--noclip", action="store_true",
                   help="start the interactive fly viewer with wall collision OFF "
                        "(-noclip), so you can fly through geometry to place a camera "
                        "outside the room or inside glass. Collision is on by default; "
                        "press C in the viewer to cycle slide/stop/off live")
    p.add_argument("--no-meter", action="store_true",
                   help="skip the exposure-lock metering pre-pass (-no-meter); frames "
                        "auto-expose per frame instead of metering the whole flyby. "
                        "Faster startup; --explore turns this on automatically")
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
    if args.explore:
        print("  fps (playback) : (n/a in --explore; the interactive viewer renders "
              "as fast as it can, no video is assembled)")
    else:
        print(f"  fps (playback) : {args.fps:g}  [{getattr(args, 'fps_source', '--fps')}]")
    print(f"  output         : {args.out}")
    print(f"  camera path    : {args.camera}")
    print(f"  explore        : {'on (interactive fly viewer, no render)' if args.explore else 'off'}")
    if args.explore:
        print(f"  collision      : {'off (-noclip; fly through walls)' if args.noclip else 'on (slide; C cycles slide/stop/off)'}")
    print(f"  metering       : {'off (-no-meter; auto-expose per frame)' if (args.no_meter or args.explore) else 'on (exposure-lock pre-pass)'}")
    if raster:
        time_desc = "(n/a for raster - animates all frames then exits)"
    elif args.time is not None:
        time_desc = f"{args.time} s/frame"
    else:
        time_desc = "(none -> hold window open until you close it)"
    print(f"  per-frame time : {time_desc}")
    print(f"  spp            : {args.spp if args.spp is not None else '(scene default)'}")
    print(f"  preview        : {'on (-window + -preview ANSI thumbnail)' if args.preview else 'window only (-window)'}")
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
    if args.preview:
        cmd.append("-preview")   # live ANSI thumbnail in the terminal too
    if args.explore:
        # Interactive fly-through: ftrace seeds the raster viewer at the first frame
        # of the selected camera path and hands control to the user - no full render.
        # -explore already implies -no-meter inside ftrace.
        cmd.insert(cmd.index("-camera"), "-explore")
        if args.noclip:
            cmd.insert(cmd.index("-camera"), "-noclip")   # start with collision off
        return cmd
    if args.no_meter:
        cmd.insert(cmd.index("-camera"), "-no-meter")
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

    # Resolve the playback fps: --fps wins; else the scene authors it (flyby
    # camera's `fps`, then the scene-level `fps` default); else fall back to 30.
    # Skipped entirely under --explore, which assembles no video (fps is unused).
    if args.explore:
        args.fps_source = "n/a"
    elif args.fps is not None:
        args.fps_source = "--fps"
    else:
        scene_fps = read_scene_fps(args.scene, args.camera)
        if scene_fps is not None:
            args.fps = scene_fps
            args.fps_source = f"scene ({args.scene})"
        else:
            args.fps = 30.0
            args.fps_source = "default"

    print_run_banner(parser, args, raster)

    if not FTRACE.exists():
        print(f"[error] ftrace.exe not found at {FTRACE} - build it first.")
        return 2
    # ffmpeg is only needed to assemble the video - skip that check in --explore.
    if not args.explore and shutil.which("ffmpeg") is None:
        print("[error] ffmpeg not found on PATH - install it to build the video.")
        return 2

    # Interactive fly-through: hand off to ftrace's viewer, no frames, no ffmpeg.
    if args.explore:
        ftrace_cmd = build_ftrace_cmd(args, raster)
        print(f"[plan] ftrace : {' '.join(ftrace_cmd)}")
        print("[plan] explore mode: interactive fly viewer, no frames rendered, "
              "no video assembled.")
        if args.dry_run:
            print("[dry-run] not executing.")
            return 0
        rc = run(ftrace_cmd)
        if rc != 0:
            print(f"[error] ftrace exited with code {rc}")
            return rc
        print("-" * 72)
        print("[done] explore session ended.")
        return 0

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
