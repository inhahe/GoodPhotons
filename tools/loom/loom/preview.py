"""
Loom driver — the **resident preview server** (M12).

``render_range`` (see :mod:`loom.drive`) spawns a *fresh* ftrace process per frame.
For a quick, interactive preview loop that is wasteful: every frame re-pays the cost
of process start-up, live-window creation, CUDA context init, and building the
spectral / spectral-upsampling tables — fixed overhead that dwarfs a cheap preview
render.

:class:`PreviewServer` keeps **one** ftrace process resident (``ftrace -serve``) and
streams it one scene path per frame over stdin, so all of that global state is paid
for *once*.  The renderer holds its live window open across frames, giving a smooth
in-place preview instead of a window flashing per frame.

Honest scope (mirrors the C++ ``-serve`` note): this delivers only the
*resident-process* win.  Each frame is still a full, independent render — there is no
incremental delta rendering, no static-geometry / BVH caching between frames, and no
reduced preview level-of-detail yet.  The live window keeps the first frame's
resolution for the session, so keep ``-r`` constant across a preview run.

Protocol (line-oriented; matches ``runServe`` in ``src/main.cpp``):
    <- "[serve] ready"            once, before the first frame
    -> "<path/to/frame.ftsl>\\n"   request a render
    <- "[serve] done <path>"      after each frame
    -> "quit" / EOF               end the loop
    <- "[serve] shutdown"

Because ftrace's GDI live window needs the interactive Console session, launch your
driver script from a real console (Claude: ``dangerouslyDisableSandbox: true``).

Example::

    from loom.preview import PreviewServer

    with PreviewServer(res=480, noise=4.0) as srv:
        for k in range(frames):
            srv.render_frame(scene, k, frames)     # streams to the resident renderer
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .signals.core import Clock, Cache
from .scene import Scene
from .drive import find_ftrace, repo_root, default_outdir, _budget_args


class PreviewServer:
    """A resident ``ftrace -serve`` process fed one scene path per frame.

    Parameters
    ----------
    outdir      where per-frame ``.ftsl`` files are written (default ``png/<name>``).
    name        base name for the emitted scene files.
    res         render resolution (``-r``); fixed for the session (see module note).
    noise/time_s/n   per-frame stop budget (default: 4% graininess for a snappy preview).
    interval    crash-safe write / window-refresh cadence in seconds (``-interval``).
    window      open the live preview window (default True; the whole point of preview).
    extra_args  any extra ftrace flags appended verbatim.
    """

    def __init__(self, *, outdir: Optional[os.PathLike] = None, name: str = "preview",
                 res: Optional[int] = None, noise: Optional[float] = None,
                 time_s: Optional[float] = None, n: Optional[int] = None,
                 interval: float = 2.0, window: bool = True,
                 extra_args: Sequence[str] = ()):
        self.outdir = Path(outdir) if outdir is not None else default_outdir(name)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.res = res
        # Snappy default: stop at 4% graininess — but any explicit budget wins.
        if noise is None and time_s is None and n is None:
            noise = 4.0
        self._budget = _budget_args(noise, time_s, n)
        self.interval = interval
        self.window = window
        self.extra_args = list(extra_args)
        self.proc: Optional[subprocess.Popen] = None
        self._frame_no = 0

    # -- lifecycle ---------------------------------------------------------
    def _build_cmd(self, initial_ftsl: os.PathLike) -> List[str]:
        """Assemble the ``ftrace -serve`` command line (also unit-testable)."""
        ftrace = find_ftrace()
        cmd = [str(ftrace), "-serve", "-in", str(initial_ftsl),
               "-interval", f"{self.interval:g}", *self._budget]
        if self.res is not None:
            cmd += ["-r", str(int(self.res))]
        if self.window:
            cmd.append("-window")
        cmd.extend(self.extra_args)
        return cmd

    def start(self, initial_ftsl: os.PathLike) -> "PreviewServer":
        """Launch ``ftrace -serve`` with ``initial_ftsl`` as the first frame."""
        if self.proc is not None:
            raise RuntimeError("PreviewServer already started")
        cmd = self._build_cmd(initial_ftsl)
        print(f"[preview] launching resident server: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, cwd=str(repo_root()),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self._wait_for("[serve] ready")
        # The initial -in frame renders immediately; consume its done marker.
        self._wait_for("[serve] done")
        return self

    def _wait_for(self, marker: str) -> str:
        """Read ftrace stdout until a line starts with ``marker``; echo the rest."""
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if line == "":  # EOF: the process died
                rc = self.proc.poll()
                raise RuntimeError(f"ftrace -serve exited unexpectedly (rc={rc})")
            line = line.rstrip("\n")
            if line.startswith(marker):
                return line
            # Pass through the renderer's own progress chatter.
            print(f"    {line}", flush=True)

    # -- rendering ---------------------------------------------------------
    def render_path(self, ftsl_path: os.PathLike) -> None:
        """Ask the resident renderer to render an already-emitted ``.ftsl`` file."""
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("PreviewServer not started")
        self.proc.stdin.write(f"{ftsl_path}\n")
        self.proc.stdin.flush()
        self._wait_for("[serve] done")

    def render_frame(self, scene: Scene, k: int, frames: int, *,
                     fps: float = 30.0, loop: bool = True) -> Path:
        """Emit frame ``k`` of ``scene`` and stream it to the resident renderer."""
        if self.proc is None:
            # First frame: emit and boot the server on it.
            fp = self._emit(scene, k, frames, fps=fps, loop=loop)
            self.start(fp)
            return fp
        fp = self._emit(scene, k, frames, fps=fps, loop=loop)
        self.render_path(fp)
        return fp

    def _emit(self, scene: Scene, k: int, frames: int, *,
              fps: float, loop: bool) -> Path:
        scene.check_cycles()
        width = max(3, len(str(max(0, frames - 1))))
        clock = Clock.at_frame(k, frames, fps, loop=loop)
        tag = f"{k:0{width}d}"
        text = scene.emit(clock, Cache(), assets_dir=self.outdir, tag=tag)
        fp = self.outdir / f"{self.name}{k:0{width}d}.ftsl"
        fp.write_text(text, encoding="utf-8")
        return fp

    # -- shutdown ----------------------------------------------------------
    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def __enter__(self) -> "PreviewServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def preview_range(scene: Scene, frames: int, *, name: str = "preview",
                  outdir: Optional[os.PathLike] = None, fps: float = 30.0,
                  loop: bool = True, res: Optional[int] = None,
                  noise: Optional[float] = None, time_s: Optional[float] = None,
                  n: Optional[int] = None, interval: float = 2.0,
                  extra_args: Sequence[str] = ()) -> List[Path]:
    """Animate ``scene`` over ``frames`` through one resident ftrace process.

    Returns the emitted ``.ftsl`` paths.  The single live window updates in place
    as each frame renders — a smooth preview with no per-frame process churn.
    """
    paths: List[Path] = []
    with PreviewServer(outdir=outdir, name=name, res=res, noise=noise,
                       time_s=time_s, n=n, interval=interval,
                       extra_args=extra_args) as srv:
        for k in range(frames):
            paths.append(srv.render_frame(scene, k, frames, fps=fps, loop=loop))
            print(f"[preview] frame {k + 1}/{frames} done", flush=True)
    return paths
