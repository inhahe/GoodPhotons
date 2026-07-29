"""
Loom drivers / IO — turn an animated :class:`~loom.scene.Scene` into a frame
range of ``.ftsl`` files, render each with ftrace, and assemble a seamless loop.

Design notes:
- Loom animates *geometry*, which ftrace cannot do internally (its own animation
  is a moving *camera* via ``camera_curve``), so we emit one scene per frame and
  invoke ftrace once per frame.
- Per the project render rules every ftrace call passes ``-window`` (a live,
  watchable preview) plus a crash-safe ``-interval`` and ``-checkpoint``.  We do
  NOT pass ``-keepwindow`` on a batch (it blocks until the user closes the
  window, which would stall a multi-frame loop); the per-frame window flashing by
  *is* the animation preview.  A single still (``render_still``) does hold.
- When you launch these from a shell, run ftrace in the interactive Console
  session (Claude: ``dangerouslyDisableSandbox: true``) or the GDI window is
  created invisibly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .signals.core import Clock, Cache
from .scene import Scene


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_ftrace() -> Path:
    """Locate the newest built ftrace.exe under the repo's build dirs."""
    root = repo_root()
    cands = [p for p in [
        root / "build" / "bin" / "ftrace.exe",
        root / "build_cuda" / "bin" / "ftrace.exe",
        root / "build_cuda2" / "bin" / "ftrace.exe",
    ] if p.exists()]
    if not cands:
        cands = list(root.glob("**/ftrace.exe"))
    if not cands:
        raise FileNotFoundError("ftrace.exe not found; build the renderer first")
    return max(cands, key=lambda p: p.stat().st_mtime)


def default_outdir(name: str) -> Path:
    return repo_root() / "png" / name


def _budget_args(noise: Optional[float], time_s: Optional[float],
                 n: Optional[int]) -> List[str]:
    if noise is not None:
        return ["-noise", f"{noise:g}"]
    if time_s is not None:
        return ["-time", f"{time_s:g}"]
    if n is not None:
        return ["-n", str(int(n))]
    return ["-noise", "3"]  # sensible default: stop at 3% graininess


def emit_frames(scene: Scene, frames: int, outdir: os.PathLike, name: str,
                *, fps: float = 30.0, loop: bool = True) -> List[Path]:
    """Emit ``frames`` ``.ftsl`` files; return their paths.

    ``loop=True`` (default) maps frames onto a closed loop (``t=(k % frames)/
    frames``) so frame ``frames`` would equal frame 0 — a seamless cycle.
    ``loop=False`` maps them onto an **open** timeline (``t=k/(frames-1)``,
    endpoints distinct) — a one-shot animation (DESIGN.md §11.6).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scene.check_cycles()
    width = max(3, len(str(frames - 1)))
    paths: List[Path] = []
    for k in range(frames):
        clock = Clock.at_frame(k, frames, fps, loop=loop)
        tag = f"{k:0{width}d}"
        text = scene.emit(clock, Cache(), assets_dir=outdir, tag=tag)
        p = outdir / f"{name}{k:0{width}d}.ftsl"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def render_range(scene: Scene, frames: int, *, name: str = "loom",
                 outdir: Optional[os.PathLike] = None, fps: float = 30.0,
                 window: bool = True, interval: float = 5.0,
                 noise: Optional[float] = None, time_s: Optional[float] = None,
                 n: Optional[int] = None, loop: bool = True,
                 skip_existing: bool = False,
                 extra_args: Sequence[str] = ()) -> List[Path]:
    """Emit and render a frame range; return the rendered PNG paths.

    ``loop=True`` (default) renders a **seamless closed loop**; ``loop=False``
    renders an **open** one-shot timeline with distinct endpoints (§11.6).
    ``noise``/``time_s``/``n`` pick the per-frame stop budget (default: 3% noise).

    ``skip_existing=True`` leaves already-rendered PNGs alone, so a long sequence
    interrupted by a crash (or a Ctrl-C) resumes instead of starting over.  It is
    opt-in precisely because it cannot tell a stale frame from a fresh one — clear
    the directory when the scene changes.
    """
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    ftrace = find_ftrace()
    ftsl_paths = emit_frames(scene, frames, outdir, name, fps=fps, loop=loop)
    budget = _budget_args(noise, time_s, n)
    pngs: List[Path] = []
    for i, fp in enumerate(ftsl_paths):
        png = fp.with_suffix(".png")
        if skip_existing and png.is_file() and png.stat().st_size > 0:
            print(f"[loom] frame {i + 1}/{len(ftsl_paths)}: {png.name} exists, skipping",
                  flush=True)
            pngs.append(png)
            continue
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png),
               "-interval", f"{interval:g}", "-checkpoint", *budget]
        if window:
            cmd.append("-window")
        cmd.extend(extra_args)
        print(f"[loom] frame {i + 1}/{len(ftsl_paths)}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(repo_root()))
        if r.returncode != 0:
            raise RuntimeError(f"ftrace failed on {fp} (exit {r.returncode})")
        pngs.append(png)
    return pngs


def render_still(scene: Scene, *, t: float = 0.0, name: str = "loom_still",
                 outdir: Optional[os.PathLike] = None,
                 interval: float = 5.0, noise: Optional[float] = None,
                 time_s: Optional[float] = None, n: Optional[int] = None,
                 hold: bool = True, extra_args: Sequence[str] = ()) -> Path:
    """Render a single frame at phase ``t`` with a held live window (per rules)."""
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    outdir.mkdir(parents=True, exist_ok=True)
    scene.check_cycles()
    ftrace = find_ftrace()
    clock = Clock(t=t, frame=0, frames=1, fps=30.0)
    fp = outdir / f"{name}.ftsl"
    fp.write_text(scene.emit(clock, Cache(), assets_dir=outdir, tag=""),
                  encoding="utf-8")
    png = fp.with_suffix(".png")
    cmd = [str(ftrace), "-in", str(fp), "-o", str(png),
           "-interval", f"{interval:g}", "-checkpoint",
           *_budget_args(noise, time_s, n)]
    cmd.append("-keepwindow" if hold else "-window")
    cmd.extend(extra_args)
    print(f"[loom] still: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(repo_root()))
    if r.returncode != 0:
        raise RuntimeError(f"ftrace failed (exit {r.returncode})")
    return png


def _find_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found on PATH (needed for video assembly)")
    return ff


_FRAME_PATTERN = "f%06d.png"


def _stage_frames(pngs: Sequence[os.PathLike], stage: Path) -> str:
    """Link an arbitrary frame list into ``stage`` as ``f000000.png`` … and return the
    ``-i`` pattern for it.

    ffmpeg's ``image2`` demuxer only reads a contiguously numbered ``%0Nd`` *pattern*,
    which would force every caller to have already named its frames that way in one
    directory; staging decouples the two.  It is **not** done with the ``concat``
    demuxer, which looks like the general answer and is a trap here: concat needs the
    last entry repeated for its ``duration`` to apply, and that repeat is a real extra
    frame — on a *closed loop* it duplicates the frame before the seam and shows up as
    a visible hitch every cycle (caught by ``tests/test_drive.py``).

    Staging costs no copy in the normal case: frames are hard-linked, falling back to a
    copy only when the link fails (a different volume, or a filesystem without them).
    """
    stage.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pngs):
        src, dst = Path(p).resolve(), stage / (_FRAME_PATTERN % i)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return str(stage / _FRAME_PATTERN)


def assemble_gif_ffmpeg(pngs: Sequence[os.PathLike], out_gif: os.PathLike,
                        *, fps: float = 30.0, loop: int = 0,
                        max_colors: int = 256,
                        dither: str = "sierra2_4a") -> Path:
    """Assemble a looping GIF through ffmpeg's ``palettegen``/``paletteuse``.

    Prefer this over :func:`assemble_gif` for anything with real shading: Pillow
    quantizes each frame independently against a fixed web-ish palette, which bands
    gradients and makes the banding *crawl* between frames.  ``palettegen`` with
    ``stats_mode=diff`` instead derives one 256-colour palette weighted toward the
    pixels that actually *change* across the sequence, so a loop's moving parts get the
    colour budget and the static background doesn't shimmer.

    ``loop=0`` repeats forever (the GIF convention), ``loop=-1`` plays once.

    Note the frame *delay* a GIF stores is an integer number of centiseconds, so an
    ``fps`` that doesn't divide 100 gets rounded and plays back at the wrong rate —
    prefer 25 / 20 / 10 (and 50 if the viewer honours 2 cs).
    """
    ff = _find_ffmpeg()
    out_gif = Path(out_gif)
    pngs = [Path(p) for p in pngs]
    if not pngs:
        raise ValueError("no frames to assemble")
    vf = (f"split[a][b];"
          f"[a]palettegen=max_colors={int(max_colors)}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle")
    with tempfile.TemporaryDirectory(prefix="loom_gif_") as tmp:
        pat = _stage_frames(pngs, Path(tmp) / "frames")
        cmd = [ff, "-y", "-framerate", f"{fps:g}", "-i", pat,
               "-filter_complex", vf, "-loop", str(int(loop)), str(out_gif)]
        print(f"[loom] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr.decode("utf-8", "replace")[-2000:])
    print(f"[loom] wrote {out_gif} ({len(pngs)} frames @ {fps} fps)", flush=True)
    return out_gif


def assemble_mp4(pngs: Sequence[os.PathLike], out_mp4: os.PathLike,
                 *, fps: float = 30.0, crf: int = 17, preset: str = "slow",
                 codec: str = "libx264") -> Path:
    """Assemble an MP4 from rendered frames (needs ffmpeg).

    The companion to :func:`assemble_gif_ffmpeg` for anything a GIF handles badly:
    full colour instead of 256 entries, and roughly an order of magnitude smaller for
    a long or highly detailed sequence.  A *loop* is a player-side property here (an
    MP4 has no loop flag), so the file is simply the closed cycle's frames — play it
    with ``loop`` set and the seam is still exactly the one loom built.

    ``crf`` is x264's quality knob (lower = better; 17 is near-visually-lossless and
    the right default for render output, which has none of the camera noise that
    normally hides compression).  The frames are padded to even dimensions because
    ``yuv420p`` — the profile every player accepts — cannot represent an odd width or
    height, and the pad is *after* the render rather than a resolution constraint on it.
    """
    ff = _find_ffmpeg()
    out_mp4 = Path(out_mp4)
    pngs = [Path(p) for p in pngs]
    if not pngs:
        raise ValueError("no frames to assemble")
    with tempfile.TemporaryDirectory(prefix="loom_mp4_") as tmp:
        pat = _stage_frames(pngs, Path(tmp) / "frames")
        cmd = [ff, "-y", "-framerate", f"{fps:g}", "-i", pat,
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", codec, "-crf", str(int(crf)), "-preset", preset,
               "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
               "-movflags", "+faststart", str(out_mp4)]
        print(f"[loom] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr.decode("utf-8", "replace")[-2000:])
    print(f"[loom] wrote {out_mp4} ({len(pngs)} frames @ {fps} fps)", flush=True)
    return out_mp4


def assemble_gif(pngs: Sequence[os.PathLike], out_gif: os.PathLike,
                 *, fps: float = 30.0, loop: int = 0) -> Path:
    """Assemble a seamless looping GIF from rendered frames (needs Pillow).

    Dependency-light but quality-limited — see :func:`assemble_gif_ffmpeg`, which
    derives a change-weighted shared palette and is the better default when ffmpeg
    is available.
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("assemble_gif needs Pillow (pip install pillow)") from e
    out_gif = Path(out_gif)
    imgs = [Image.open(str(p)).convert("RGB") for p in pngs]
    if not imgs:
        raise ValueError("no frames to assemble")
    duration_ms = max(1, int(round(1000.0 / fps)))
    imgs[0].save(str(out_gif), save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=loop, optimize=True)
    print(f"[loom] wrote {out_gif} ({len(imgs)} frames @ {fps} fps)", flush=True)
    return out_gif
