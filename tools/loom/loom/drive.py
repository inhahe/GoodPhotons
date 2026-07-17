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
import subprocess
import sys
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
                 extra_args: Sequence[str] = ()) -> List[Path]:
    """Emit and render a frame range; return the rendered PNG paths.

    ``loop=True`` (default) renders a **seamless closed loop**; ``loop=False``
    renders an **open** one-shot timeline with distinct endpoints (§11.6).
    ``noise``/``time_s``/``n`` pick the per-frame stop budget (default: 3% noise).
    """
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    ftrace = find_ftrace()
    ftsl_paths = emit_frames(scene, frames, outdir, name, fps=fps, loop=loop)
    budget = _budget_args(noise, time_s, n)
    pngs: List[Path] = []
    for i, fp in enumerate(ftsl_paths):
        png = fp.with_suffix(".png")
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


def assemble_gif(pngs: Sequence[os.PathLike], out_gif: os.PathLike,
                 *, fps: float = 30.0, loop: int = 0) -> Path:
    """Assemble a seamless looping GIF from rendered frames (needs Pillow)."""
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
