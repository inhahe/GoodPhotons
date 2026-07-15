"""OBJ-sequence -> MP4 driver: render animated geometry as a video.

Renders a *sequence of OBJ files* (one mesh per animation frame -- e.g. a cloth
sim, a growing crystal, an exported Blender/Houdini point-cache baked to OBJ per
frame) with ftrace, one frame at a time, then encodes the frames into an MP4 with
ffmpeg. This is the geometry-animation counterpart to the built-in camera
animation (camera_path / camera_orbit / camera_curve), which moves the camera over
ONE static scene; here the camera can be fixed while the *mesh* changes each frame.

Self-contained: no new C++ / renderer dependencies -- it just drives the existing
`ftrace` binary per frame and shells out to `ffmpeg` for the encode.

--------------------------------------------------------------------------------
How it works
--------------------------------------------------------------------------------
You supply a *template* FTSL scene that sets up the camera, lights, materials and
render mode, with a placeholder token `{obj}` wherever the animated mesh goes:

    # anim.ftsl  (template)
    camera "cam" { preset portable  look_at 0 1 0  from 0 1.5 4 }
    light  "key" { type area  ... }
    material "body" { type diffuse reflect 0.8 0.7 0.6 }
    mesh { file "{obj}"  material "body"  scale 1 }

For each OBJ in the sequence the driver substitutes the placeholder, writes a
per-frame scene, renders it to `frames/frame_NNNNN.png`, then ffmpeg concatenates
the PNGs into the output MP4. Placeholders available in the template:
    {obj}        absolute path to this frame's OBJ (forward slashes)
    {frame}      zero-based frame index, zero-padded (e.g. 00007)
    {frame1}     one-based frame index, zero-padded
    {obj_stem}   the OBJ filename without directory or extension

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python tools/obj_sequence_to_video.py FRAMES --template anim.ftsl -o out.mp4 [opts]

    FRAMES is either a glob ("anim/frame_*.obj") or a directory (all *.obj inside,
    naturally sorted). Quote globs so the shell doesn't expand them inconsistently.

Common options:
    -o, --out PATH        output video (default png/obj_seq.mp4)
    --template PATH       template scene with a {obj} placeholder (required unless
                          --encode-only or --write-template)
    --fps N               frames per second (default 24)
    --mode M              render mode letter passed as -mode (e.g. R, B, M)
    -r, --res W [H]       resolution; one value = square, two = W H
    --time S | --spp N | --noise PCT   per-frame budget (progressive stop)
    --device cpu|gpu|auto passed through as -device
    --window / --preview  live view per frame (window = GDI, preview = ANSI)
    --start / --end / --step   render a sub-range of the sequence
    --ftrace-arg ARG      extra raw arg forwarded to ftrace (repeatable)
    --frames-dir DIR      where PNGs are written (default <out>_frames)
    --resume              skip frames whose PNG already exists
    --keep-frames         don't delete the PNG frames after a successful encode
    --encode-only         skip rendering; just encode existing PNGs in --frames-dir
    --no-encode           render frames only; don't run ffmpeg
    --crf N               x264 quality (default 18; lower = better/larger)
    --codec / --pix-fmt   ffmpeg video codec (default libx264) / pixel format
                          (default yuv420p -- required for broad player support)
    --ftrace PATH         ftrace binary (auto-detected under build*/bin otherwise)
    --dry-run             print the commands instead of running them
    --write-template PATH write a starter template to PATH and exit

Examples:
    # Render an OBJ cache to a 24fps 1080-ish clip, ~4s/frame, on the GPU
    python tools/obj_sequence_to_video.py "cache/*.obj" --template anim.ftsl \
        -o png/growth.mp4 --mode B --device gpu -r 960 --time 4 --fps 24

    # Re-encode already-rendered frames at 30fps without re-rendering
    python tools/obj_sequence_to_video.py --encode-only --frames-dir png/growth_frames \
        -o png/growth30.mp4 --fps 30

    # Emit a starter template you can edit
    python tools/obj_sequence_to_video.py --write-template anim.ftsl
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STARTER_TEMPLATE = """\
# Starter template for tools/obj_sequence_to_video.py
# The {obj} token is replaced with each frame's OBJ path. Set up the camera,
# lights and materials once here; only the mesh changes per frame. This is a
# plain Cornell-box stage -- edit the walls/light/camera to taste.

scene { units meters  spectral 360 830 1 }

material "white" { type diffuse reflect whitewall 0.75 }
material "red"   { type diffuse reflect redwall }
material "green" { type diffuse reflect greenwall }
material "body"  { type diffuse reflect 0.65 }

quad { origin 0 0 0  u 1 0 0  v 0 0 1  material white }   # floor
quad { origin 0 1 0  u 0 0 1  v 1 0 0  material white }   # ceiling
quad { origin 0 0 0  u 1 0 0  v 0 1 0  material white }   # back
quad { origin 0 0 0  u 0 0 1  v 0 1 0  material red   }   # left wall
quad { origin 1 0 0  u 0 1 0  v 0 0 1  material green }   # right wall

# The animated mesh. {obj} is substituted per frame (absolute path). Adjust
# scale/translate so your model sits inside the box (or replace the box).
mesh {
    file      "{obj}"
    material  body
    scale     0.3
    translate 0.5 0.3 0.5
}

light area {
    origin 0.35 0.999 0.35  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0
    spd preset:bb6500
}

camera "cam" {
    eye 0.5 0.5 2.4  look_at 0.5 0.4 0.5  up 0 1 0  fov_y 40
    mode B
    film { res 512 512 }
}
"""


def natural_key(s):
    """Sort key that orders frame_2 before frame_10 (numeric-aware)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]


def find_ftrace(explicit):
    if explicit:
        if os.path.exists(explicit):
            return explicit
        sys.exit(f"ftrace not found at --ftrace path: {explicit}")
    # Prefer a built binary in the repo; fall back to PATH.
    cands = []
    for sub in ("build_cuda2", "build_cuda", "build"):
        for exe in ("ftrace.exe", "ftrace"):
            cands.append(os.path.join(ROOT, sub, "bin", exe))
    for c in cands:
        if os.path.exists(c):
            return c
    onpath = shutil.which("ftrace")
    if onpath:
        return onpath
    sys.exit("ftrace binary not found (looked under build*/bin and PATH). "
             "Pass --ftrace <path>.")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found on PATH. Install ffmpeg (e.g. winget install "
                 "Gyan.FFmpeg) or add it to PATH.")
    return exe


def collect_frames(spec):
    """Resolve a directory or glob into a naturally-sorted list of .obj paths."""
    if os.path.isdir(spec):
        files = glob.glob(os.path.join(spec, "*.obj"))
    else:
        files = glob.glob(spec)
        # Keep only OBJs when a broad glob was given.
        objs = [f for f in files if f.lower().endswith(".obj")]
        files = objs if objs else files
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files


def substitute(template_text, obj_path, idx, pad):
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    absobj = os.path.abspath(obj_path).replace("\\", "/")
    return (template_text
            .replace("{obj}", absobj)
            .replace("{obj_stem}", stem)
            .replace("{frame1}", str(idx + 1).zfill(pad))
            .replace("{frame}", str(idx).zfill(pad)))


def run(cmd, dry):
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if dry:
        print("  [dry-run]", printable)
        return 0
    return subprocess.call(cmd)


def build_ftrace_cmd(args, ftrace, scene_path, png_path):
    cmd = [ftrace, "-in", scene_path, "-o", png_path]
    if args.mode:
        cmd += ["-mode", args.mode]
    if args.res:
        cmd += ["-r"] + [str(v) for v in args.res]
    if args.time is not None:
        cmd += ["-time", str(args.time)]
    if args.spp is not None:
        cmd += ["-spp", str(args.spp)]
    if args.noise is not None:
        cmd += ["-noise", str(args.noise)]
    if args.device:
        cmd += ["-device", args.device]
    if args.window:
        cmd += ["-window"]
    if args.preview:
        cmd += ["-preview"]
    for extra in (args.ftrace_arg or []):
        cmd += extra.split() if " " in extra else [extra]
    return cmd


def encode(args, ffmpeg, frames_dir, pad, out_path, n_expected):
    pattern = os.path.join(frames_dir, f"frame_%0{pad}d.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [ffmpeg, "-y",
           "-framerate", str(args.fps),
           "-start_number", "0",
           "-i", pattern,
           "-c:v", args.codec,
           "-pix_fmt", args.pix_fmt]
    if args.codec in ("libx264", "libx265"):
        cmd += ["-crf", str(args.crf)]
    # -vf pad to even dims keeps yuv420p happy for odd-sized renders.
    cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd += [out_path]
    print(f"[encode] {n_expected} frames @ {args.fps}fps -> {out_path}")
    rc = run(cmd, args.dry_run)
    if rc != 0 and not args.dry_run:
        sys.exit(f"ffmpeg failed (exit {rc}).")
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="Render an OBJ sequence to an MP4 via ftrace + ffmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="?",
                    help="OBJ sequence: a directory of *.obj or a glob "
                         '("anim/frame_*.obj"). Quote globs.')
    ap.add_argument("--template", help="template FTSL scene with a {obj} placeholder")
    ap.add_argument("-o", "--out", default=os.path.join("png", "obj_seq.mp4"),
                    help="output video path (default png/obj_seq.mp4)")
    ap.add_argument("--fps", type=float, default=24, help="frames per second (default 24)")
    ap.add_argument("--mode", help="render mode letter (-mode)")
    ap.add_argument("-r", "--res", nargs="+", type=int, metavar="N",
                    help="resolution: one value (square) or W H")
    ap.add_argument("--time", type=float, help="per-frame time budget in seconds")
    ap.add_argument("--spp", type=int, help="per-frame samples-per-pixel budget")
    ap.add_argument("--noise", type=float, help="per-frame noise%% stop target")
    ap.add_argument("--device", help="cpu | gpu | auto (-device)")
    ap.add_argument("--window", action="store_true", help="live GDI window per frame")
    ap.add_argument("--preview", action="store_true", help="live ANSI preview per frame")
    ap.add_argument("--start", type=int, default=0, help="first frame index (inclusive)")
    ap.add_argument("--end", type=int, help="last frame index (inclusive)")
    ap.add_argument("--step", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--ftrace-arg", action="append",
                    help="extra raw arg forwarded to ftrace (repeatable)")
    ap.add_argument("--frames-dir", help="PNG output dir (default <out>_frames)")
    ap.add_argument("--scene-dir", help="per-frame scene dir (default <frames-dir>/scenes)")
    ap.add_argument("--resume", action="store_true", help="skip frames whose PNG exists")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep PNG frames after a successful encode")
    ap.add_argument("--encode-only", action="store_true",
                    help="don't render; encode existing PNGs in --frames-dir")
    ap.add_argument("--no-encode", action="store_true", help="render frames only")
    ap.add_argument("--crf", type=int, default=18, help="x264/x265 quality (default 18)")
    ap.add_argument("--codec", default="libx264", help="ffmpeg video codec")
    ap.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pixel format")
    ap.add_argument("--ftrace", help="path to the ftrace binary")
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run")
    ap.add_argument("--write-template", metavar="PATH",
                    help="write a starter template to PATH and exit")
    args = ap.parse_args()

    if args.write_template:
        with open(args.write_template, "w", encoding="utf-8") as f:
            f.write(STARTER_TEMPLATE)
        print(f"wrote starter template -> {args.write_template}")
        print("Edit it (camera/lights/materials), keep the {obj} token, then run:")
        print(f'  python tools/obj_sequence_to_video.py FRAMES --template '
              f'{args.write_template} -o out.mp4')
        return

    # Frames dir + pad width.
    out_abs = os.path.abspath(args.out)
    frames_dir = args.frames_dir or (os.path.splitext(out_abs)[0] + "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    ffmpeg = find_ffmpeg()

    if args.encode_only:
        existing = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")),
                          key=lambda p: natural_key(os.path.basename(p)))
        if not existing:
            sys.exit(f"--encode-only: no frame_*.png in {frames_dir}")
        pad = len(re.search(r"frame_(\d+)\.png", os.path.basename(existing[0])).group(1))
        encode(args, ffmpeg, frames_dir, pad, args.out, len(existing))
        print(f"[done] {args.out}")
        return

    # Render path: need a sequence + template.
    if not args.frames:
        sys.exit("missing FRAMES (a directory or glob of .obj files). "
                 "See --help, or --write-template to start a scene.")
    if not args.template:
        sys.exit("--template is required to render (a scene with a {obj} token). "
                 "Use --write-template PATH to generate one.")
    if not os.path.exists(args.template):
        sys.exit(f"template not found: {args.template}")

    seq = collect_frames(args.frames)
    if not seq:
        sys.exit(f"no .obj files matched: {args.frames}")

    # Apply start/end/step range over the resolved sequence.
    end = args.end if args.end is not None else len(seq) - 1
    picked = [(i, seq[i]) for i in range(args.start, min(end, len(seq) - 1) + 1, args.step)]
    if not picked:
        sys.exit("frame range selected zero frames (check --start/--end/--step).")

    with open(args.template, "r", encoding="utf-8") as f:
        template_text = f.read()
    if "{obj}" not in template_text:
        sys.exit("template has no {obj} placeholder -- nothing to animate. "
                 "Add `file \"{obj}\"` to the mesh block.")

    ftrace = find_ftrace(args.ftrace)
    scene_dir = args.scene_dir or os.path.join(frames_dir, "scenes")
    os.makedirs(scene_dir, exist_ok=True)

    pad = max(5, len(str(len(picked) - 1)))
    print(f"[obj-seq] {len(picked)} frame(s) from {args.frames}")
    print(f"[obj-seq] ftrace: {ftrace}")
    print(f"[obj-seq] frames -> {frames_dir}")

    rendered = 0
    for out_idx, (src_idx, obj_path) in enumerate(picked):
        png_path = os.path.join(frames_dir, f"frame_{str(out_idx).zfill(pad)}.png")
        if args.resume and os.path.exists(png_path) and not args.dry_run:
            print(f"[{out_idx+1}/{len(picked)}] skip (exists): "
                  f"{os.path.basename(png_path)}")
            rendered += 1
            continue

        scene_text = substitute(template_text, obj_path, out_idx, pad)
        scene_path = os.path.join(scene_dir, f"frame_{str(out_idx).zfill(pad)}.ftsl")
        if not args.dry_run:
            with open(scene_path, "w", encoding="utf-8") as f:
                f.write(scene_text)

        print(f"[{out_idx+1}/{len(picked)}] {os.path.basename(obj_path)} "
              f"-> {os.path.basename(png_path)}")
        cmd = build_ftrace_cmd(args, ftrace, scene_path, png_path)
        rc = run(cmd, args.dry_run)
        if rc != 0 and not args.dry_run:
            sys.exit(f"ftrace failed on frame {out_idx} (exit {rc}): {obj_path}")
        if not args.dry_run and not os.path.exists(png_path):
            sys.exit(f"ftrace produced no PNG for frame {out_idx}: {png_path}")
        rendered += 1

    print(f"[obj-seq] rendered/collected {rendered} frame(s).")

    if args.no_encode:
        print("[obj-seq] --no-encode: stopping before ffmpeg.")
        return

    encode(args, ffmpeg, frames_dir, pad, args.out, rendered)

    if not args.keep_frames and not args.dry_run:
        # Remove PNGs + per-frame scenes, keep the dir if it holds anything else.
        for p in glob.glob(os.path.join(frames_dir, "frame_*.png")):
            os.remove(p)
        shutil.rmtree(scene_dir, ignore_errors=True)
        print("[obj-seq] cleaned intermediate frames (use --keep-frames to retain).")

    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
