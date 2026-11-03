"""
``python -m loom.shot <scene.py>`` — render a loom file as a shot, every frame through ftrace.

``ftrace -shot <scene.py> …`` runs this (with ``--ftrace`` naming itself), so either entry
point renders the same shot. The scene file follows the viewer's ``build`` contract (a
module-level ``build(clock=None, **params) -> Scene``; ``--func`` names another, such as an
example's ``build_scene``), and may declare its timeline as module constants, which the flags
override:

    FRAMES = 96          # frames in the timeline            (--frames)
    FPS = 24.0           # playback rate                     (--fps)
    LOOP = False         # an open, one-shot timeline         (--open / --loop)
    SHUTTER = (-0.25, 0.25)   # the camera's shutter, frames  (--shutter O C)
    MOTION = Motion((-0.5, 0.5), 5)                           (--motion A B KEYS)

A shutter brings motion keys: without ``MOTION`` / ``--motion`` the keys span the shutter and
the instant (``(-0.5, 0.5)`` for a shutter inside it) with enough of them to land one on the
instant. Frames go to ``png/<name>/`` (``--outdir``), each frame's ``.ftsl`` beside its image,
and a frame whose checkpoint already holds the requested ``-spp`` is skipped, so rerunning the
same command resumes the shot. ``--mp4`` assembles ``<name>.mp4`` from the frames when the
range covers the whole timeline (ffmpeg). Everything after ``--`` goes to ftrace for every
frame; by default each frame also gets ``-window-min`` (``--no-window`` drops it).

A stop file (``--stop-file``, which ``ftrace -shot`` passes) ends the shot cleanly: the frame
in progress is stopped with ``ftrace -stop`` and keeps what it rendered, and no further frame
starts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .ftsl_emit import Motion


def _default_motion(shutter) -> Motion:
    """Keys spanning the shutter and the instant, with one on the instant."""
    a, b = min(-0.5, float(shutter[0])), max(0.5, float(shutter[1]))
    for keys in range(5, 65):
        s = -a / (b - a) * (keys - 1)
        if abs(s - round(s)) < 1e-9:
            return Motion((a, b), keys)
    raise ValueError(f"no key count up to 64 lands a key on the instant for the interval "
                     f"({a}, {b}); pass --motion A B KEYS")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(prog="python -m loom.shot",
                                 description="Render a loom scene file as a shot through ftrace.")
    ap.add_argument("scene", help="the loom scene .py")
    ap.add_argument("--func", default="build", help="its scene-building callable (default build)")
    ap.add_argument("--frames", type=int, help="frames in the timeline (else the file's FRAMES)")
    ap.add_argument("--range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="render only these frames, inclusive")
    ap.add_argument("--fps", type=float, help="playback rate (else the file's FPS, else 30)")
    tl = ap.add_mutually_exclusive_group()
    tl.add_argument("--open", action="store_true", help="an open, one-shot timeline")
    tl.add_argument("--loop", action="store_true", help="a closed loop (the default)")
    ap.add_argument("--shutter", nargs=2, type=float, metavar=("OPEN", "CLOSE"),
                    help="the camera's shutter, in frames around each frame's instant")
    ap.add_argument("--motion", nargs=3, metavar=("A", "B", "KEYS"),
                    help="motion keys: KEYS spread over [A, B] frames")
    ap.add_argument("--no-motion", action="store_true", help="write no motion keys")
    ap.add_argument("--name", help="the shot's name (default: the scene file's)")
    ap.add_argument("--outdir", help="where frames go (default png/<name>)")
    ap.add_argument("--ftrace", help="the ftrace binary (default: the repo's)")
    ap.add_argument("--mp4", action="store_true", help="assemble <name>.mp4 at the end")
    ap.add_argument("--no-window", action="store_true", help="no live preview per frame")
    ap.add_argument("--stop-file", help="stop the shot cleanly when this file appears")
    a = ap.parse_args(argv)

    from .viewer import load_build, build_scene
    build = load_build(a.scene, func=a.func)
    module = sys.modules[build.__module__]   # load_build registers the scene file there

    frames = a.frames if a.frames is not None else getattr(module, "FRAMES", None)
    if frames is None:
        ap.error("the timeline's length is unknown: give --frames, or FRAMES in the scene file")
    fps = a.fps if a.fps is not None else float(getattr(module, "FPS", 30.0))
    loop = (False if a.open else True if a.loop else bool(getattr(module, "LOOP", True)))
    shutter = tuple(a.shutter) if a.shutter else getattr(module, "SHUTTER", None)

    scene = build_scene(build)
    if shutter is not None:
        if not hasattr(scene.camera, "shutter"):
            ap.error(f"a shutter needs a loom Camera; this scene's camera is a "
                     f"{type(scene.camera).__name__}")
        if shutter[1] < shutter[0]:
            ap.error(f"the shutter closes before it opens: {shutter}")
        scene.camera.shutter = (float(shutter[0]), float(shutter[1]))

    motion: Optional[Motion] = None
    if a.no_motion:
        motion = None
    elif a.motion:
        motion = Motion((float(a.motion[0]), float(a.motion[1])), int(a.motion[2]))
    elif getattr(module, "MOTION", None) is not None:
        motion = module.MOTION
    elif shutter is not None:
        motion = _default_motion(shutter)

    name = a.name or Path(a.scene).stem
    from .drive import render_range, assemble_mp4, default_outdir
    outdir = Path(a.outdir) if a.outdir else default_outdir(name)
    if not a.no_window and not any(f in extra for f in ("-window", "-window-min", "-minimized")):
        extra = ["-window-min"] + extra
    fr = tuple(a.range) if a.range else None
    print(f"[shot] {name}: {frames} frames at {fps:g} fps ({'loop' if loop else 'open'}), "
          f"{'frames %d..%d' % fr if fr else 'all frames'}"
          f"{', shutter %g..%g' % tuple(shutter) if shutter else ''}"
          f"{', %d motion keys over %g..%g' % (motion.keys, *motion.interval) if motion else ''}"
          f" -> {outdir}", flush=True)
    pngs = render_range(scene, frames, name=name, outdir=outdir, fps=fps, loop=loop,
                        window=False, skip_existing=True, motion=motion, frame_range=fr,
                        ftrace=a.ftrace, stop_file=a.stop_file, extra_args=extra)
    stopped = a.stop_file is not None and os.path.exists(a.stop_file)
    whole = fr is None or (fr[0] == 0 and fr[1] == frames - 1)
    if a.mp4 and not stopped:
        if whole:
            assemble_mp4(pngs, outdir / f"{name}.mp4", fps=fps)
            print(f"[shot] wrote {outdir / (name + '.mp4')}", flush=True)
        else:
            print("[shot] --mp4 skipped: the range is not the whole timeline", flush=True)
    print(f"[shot] {'stopped' if stopped else 'done'}: {len(pngs)} frame(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
