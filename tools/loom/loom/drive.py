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

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .signals.core import Clock, Cache
from .scene import Scene
from .ftsl_emit import Motion


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_ftrace() -> Path:
    """The repo-root ftrace.exe, which build.bat installs after every build; else the newest
    build*/bin/ftrace.exe. A fixed list of build directories goes stale whenever build.bat moves
    to a new one, and then silently renders with an old binary."""
    root = repo_root()
    if (root / "ftrace.exe").is_file():
        return root / "ftrace.exe"
    cands = [p for p in root.glob("build*/bin/ftrace.exe") if p.is_file()]
    if not cands:
        raise FileNotFoundError("ftrace.exe not found; build the renderer first (build.bat)")
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


# The ftrace flags that set a render's budget themselves: given one, a caller that chose no
# budget of its own gets none added (render_range's `-noise 3` default would ride beside it).
_BUDGET_FLAGS = {"-spp", "-noise", "-time", "-n", "-forever"}


# ---------------------------------------------------------------------------
# Telling a finished frame from an interrupted one
# ---------------------------------------------------------------------------
# ftrace writes the output PNG at EVERY `-interval` tick, not just at the end — that is
# the whole point of the crash-safety rule.  So "the PNG exists" says nothing about
# whether the frame converged, and a `skip_existing` resume that trusts it will silently
# adopt a frame that was interrupted at 1 of 8 spp.  In a loop that is a visible noise
# pop, and it is invisible in the logs because the resume cheerfully prints "skipping".
# (Observed for real: a render killed by memory pressure at frame 244 left a 3-spp frame
# that the next run skipped.)
#
# The `-checkpoint` sidecar is the honest record.  Its header is packed little-endian:
#
#     magic "FTBUF01\n" (8)  resX (i32)  resY (i32)  mode (i32)  N (i64)  ...
#
# and for a deterministic `-spp` render `N` is the accumulated sample count
# (`src/main.cpp` ~4099 prints it as "holds %lld spp").  So a frame is finished iff its
# sidecar reports at least the requested spp.
_FTBUF_MAGIC = b"FTBUF01\n"


def checkpoint_spp(png: os.PathLike) -> Optional[int]:
    """Samples-per-pixel recorded in ``png``'s ``.ftbuf`` sidecar, or ``None``.

    ``None`` means "cannot tell" — no sidecar, or one that is truncated or not a
    checkpoint at all — and callers must treat that as *not known to be finished*
    rather than assuming either answer.
    """
    fb = Path(str(png) + ".ftbuf")
    try:
        with open(fb, "rb") as f:
            head = f.read(28)
    except OSError:
        return None
    if len(head) < 28 or head[:8] != _FTBUF_MAGIC:
        return None
    return int.from_bytes(head[20:28], "little", signed=True)


def _target_spp(extra_args: Sequence[str]) -> Optional[int]:
    """The spp a finished frame must reach, read off the caller's own ``-spp``.

    ``None`` when the budget is not spp-based (``-noise`` / ``-time`` have no fixed
    target), in which case completeness cannot be checked and we say so out loud.
    """
    a = list(extra_args)
    if "-spp" not in a:
        return None
    try:
        return int(a[a.index("-spp") + 1])
    except (IndexError, ValueError):
        return None


# The render flags that shape how a finished film is DEVELOPED into its PNG, and that a
# re-develop (``-topng``) must therefore repeat: the denoiser's and the exposure
# compensation. Each takes one value, except ``-denoise``, whose amount is optional.
_DEVELOP_FLAGS = {"-denoise-chroma", "-denoise-luma", "-denoise-levels", "-fireflies",
                  "-ev", "-exposure"}


def _develop_args(extra_args: Sequence[str]) -> List[str]:
    """The develop flags (see ``_DEVELOP_FLAGS``) out of a render's own arguments, in order.

    ``stabilize_exposure`` re-develops every frame from its checkpoint; without these a
    sequence rendered with ``-denoise`` would be overwritten by undenoised frames.
    """
    a = list(extra_args)
    out: List[str] = []
    i = 0
    while i < len(a):
        t = a[i][1:] if a[i].startswith("--") and len(a[i]) > 2 else a[i]   # ftrace takes both
        if t == "-denoise":
            out.append(t)
            if i + 1 < len(a) and not a[i + 1].startswith("-"):
                try:
                    float(a[i + 1])
                    out.append(a[i + 1])
                    i += 1
                except ValueError:
                    pass
        elif t in _DEVELOP_FLAGS and i + 1 < len(a):
            out += [t, a[i + 1]]
            i += 1
        i += 1
    return out


def emit_frames(scene: Scene, frames: int, outdir: os.PathLike, name: str,
                *, fps: float = 30.0, loop: bool = True,
                motion: Optional[Motion] = None,
                frame_range: Optional[Tuple[int, int]] = None) -> List[Path]:
    """Emit ``frames`` ``.ftsl`` files; return their paths.

    ``loop=True`` (default) maps frames onto a closed loop (``t=(k % frames)/
    frames``) so frame ``frames`` would equal frame 0 — a seamless cycle.
    ``loop=False`` maps them onto an **open** timeline (``t=k/(frames-1)``,
    endpoints distinct) — a one-shot animation (DESIGN.md §11.6).

    ``motion`` gives each frame motion keys (:meth:`Scene.emit`), for a camera ``shutter``
    to blur. ``frame_range=(first, last)`` (inclusive) emits only those frames of the
    ``frames``-long timeline.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scene.check_cycles()
    width = max(3, len(str(frames - 1)))
    paths: List[Path] = []
    first, last = frame_range if frame_range is not None else (0, frames - 1)
    if not (0 <= first <= last < frames):
        raise ValueError(f"frame_range {frame_range} is outside the {frames}-frame timeline")
    for k in range(first, last + 1):
        clock = Clock.at_frame(k, frames, fps, loop=loop)
        tag = f"{k:0{width}d}"
        text = scene.emit(clock, Cache(), assets_dir=outdir, tag=tag, motion=motion)
        p = outdir / f"{name}{k:0{width}d}.ftsl"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def render_range(scene: Scene, frames: int, *, name: str = "loom",
                 outdir: Optional[os.PathLike] = None, fps: float = 30.0,
                 window: bool = True, interval: float = 5.0,
                 noise: Optional[float] = None, time_s: Optional[float] = None,
                 n: Optional[int] = None, loop: bool = True,
                 skip_existing: bool = False, retries: int = 2,
                 stabilize: bool = True, motion: Optional[Motion] = None,
                 frame_range: Optional[Tuple[int, int]] = None,
                 ftrace: Optional[os.PathLike] = None,
                 stop_file: Optional[os.PathLike] = None,
                 extra_args: Sequence[str] = ()) -> List[Path]:
    """Emit and render a frame range; return the rendered PNG paths.

    ``motion`` (a :class:`~loom.ftsl_emit.Motion`) writes each frame with motion keys, so a
    camera given a ``shutter`` renders it motion-blurred — in mode R, the mode that blurs.
    ``frame_range=(first, last)`` renders only those frames (inclusive) of the timeline;
    ``ftrace`` names the binary (default :func:`find_ftrace`). When no budget is given
    (``noise``/``time_s``/``n``) and ``extra_args`` sets one (``-spp``, ``-noise``, …), none
    is added beside it.

    ``stop_file``: when that file appears, the frame being rendered is stopped cleanly
    (``ftrace -stop``: it writes its image and checkpoint) and no further frame starts; the
    frames finished so far are returned. ``python -m loom.shot`` uses it to let
    ``ftrace -stop`` reach a whole shot.

    ``loop=True`` (default) renders a **seamless closed loop**; ``loop=False``
    renders an **open** one-shot timeline with distinct endpoints (§11.6).
    ``noise``/``time_s``/``n`` pick the per-frame stop budget (default: 3% noise).

    ``stabilize=True`` (default) runs :func:`stabilize_exposure` at the end, which
    re-develops the finished frames through one shared auto-exposure anchor.  Without
    it every frame auto-exposes independently and a scene with a moving specular
    highlight flickers — see that function for the measurement.  It costs no
    re-rendering (it works off the ``-checkpoint`` sidecars), so it is on by default.

    ``skip_existing=True`` resumes a long sequence that was interrupted by a crash
    or a Ctrl-C instead of starting over.  A frame counts as done only when its
    ``-checkpoint`` sidecar reports the full requested spp — **not** merely because
    a PNG is there, since ftrace writes the PNG at every interval tick and an
    interrupted frame would otherwise be adopted at whatever spp it reached.  A
    partial frame is *continued* with ``-resume``, so the samples it already has are
    kept.  When the budget is not spp-based there is no completeness test available,
    so existence is all we have and the fact is logged rather than hidden.

    ``skip_existing`` still cannot tell a *stale* frame from a fresh one — clear the
    directory when the scene changes.

    ``retries`` re-attempts a frame that exits non-zero (default 2 extra tries).
    A long unattended sequence is exactly where a transient failure — the machine
    briefly running out of commit, say — should cost one frame's time rather than
    the whole run; each attempt resumes from the checkpoint, so nothing is redone.
    A frame that fails every attempt still raises.
    """
    # Absolute: the scenes are written from this process's cwd, and each frame's ftrace runs in
    # the repo root, so a relative directory would name two different places.
    outdir = (Path(outdir) if outdir is not None else default_outdir(name)).resolve()
    ftrace = Path(ftrace) if ftrace is not None else find_ftrace()
    ftsl_paths = emit_frames(scene, frames, outdir, name, fps=fps, loop=loop, motion=motion,
                             frame_range=frame_range)
    own = noise is None and time_s is None and n is None
    budget = ([] if own and any(a in _BUDGET_FLAGS for a in extra_args)
              else _budget_args(noise, time_s, n))
    stop = Path(stop_file) if stop_file is not None else None
    want = _target_spp(extra_args)
    if skip_existing and want is None:
        print("[loom] note: budget is not -spp based, so a resume cannot verify that an "
              "existing frame finished; falling back to existence alone", flush=True)
    pngs: List[Path] = []
    for i, fp in enumerate(ftsl_paths):
        png = fp.with_suffix(".png")
        k = (frame_range[0] if frame_range is not None else 0) + i   # its place on the timeline
        tag = f"[loom] frame {k + 1}/{frames}"
        if stop is not None and stop.exists():
            print(f"{tag}: stop requested -- not starting it", flush=True)
            return pngs
        have = checkpoint_spp(png) if want is not None else None
        stopped_mark = Path(str(png) + ".stopped")   # left by a stop: the frame is not done
        if skip_existing and png.is_file() and png.stat().st_size > 0 and not stopped_mark.exists():
            if want is None or (have is not None and have >= want):
                print(f"{tag}: {png.name} done, skipping", flush=True)
                pngs.append(png)
                continue
            print(f"{tag}: {png.name} is INCOMPLETE "
                  f"({'no checkpoint' if have is None else f'{have}/{want} spp'}), "
                  f"re-rendering", flush=True)
        # -frame-label: each frame is its own ftrace process and scene file, so only this loop
        # knows the frame's place in the sequence; the live window's title shows it.
        cmd = [str(ftrace), "-in", str(fp), "-o", str(png),
               "-interval", f"{interval:g}", "-checkpoint", *budget,
               "-frame-label", f"frame {k + 1}/{frames}"]
        if window:
            cmd.append("-window")
        cmd.extend(extra_args)
        for attempt in range(retries + 1):
            # Continue from whatever the sidecar holds rather than discarding it.  A
            # mismatched (stale-scene) checkpoint is rejected by ftrace's own identity
            # guard, which restarts the frame and says so, so this is safe to pass
            # whenever a usable sidecar exists.
            #
            # Note `-spp` is ADDITIONAL samples under `-resume`, not a total: resuming a
            # 3-spp frame with `-spp 8` renders it to 11.  So ask for exactly the
            # shortfall, which lands every frame on the same spp.  Because the sample
            # lattice is indexed by ABSOLUTE sample index, `3 + 5` is bit-identical to a
            # fresh `8`, so a resumed frame is not merely close to an un-interrupted one --
            # it is the same image.  That is measured, not assumed: scraps/resume_check.py
            # checks it by filecmp on both CPU and GPU, and its first run FAILED, which is
            # how the GPU `-resume` bug (gpuSppChunks never applied prog.sampleBase) was
            # found.  Requires ftrace >= 0.117.1; see known-issues.md.
            have_now = checkpoint_spp(png) or 0
            run = list(cmd)
            if have_now > 0 and want is not None and have_now < want:
                run[run.index("-spp") + 1] = str(want - have_now)
                run.append("-resume")
            elif have_now > 0 and want is None:
                run.append("-resume")
            print(f"{tag}"
                  f"{f' (attempt {attempt + 1}/{retries + 1})' if attempt else ''}: "
                  f"{' '.join(run)}", flush=True)
            rc, stopped = _run_frame(run, ftrace, stop)
            if stopped:
                stopped_mark.write_text("stopped before it finished; a rerun resumes it\n",
                                        encoding="utf-8")
                print(f"{tag}: stop requested -- {png.name} keeps what it rendered; "
                      f"no further frames", flush=True)
                return pngs
            if rc == 0:
                if stopped_mark.exists():
                    stopped_mark.unlink()
                break
            if attempt == retries:
                raise RuntimeError(f"ftrace failed on {fp} (exit {rc}) after "
                                   f"{retries + 1} attempts")
            print(f"{tag}: exit {rc}, retrying", flush=True)
        pngs.append(png)
    if stabilize:
        stabilize_exposure(pngs, anchor_file=outdir / f"{name}_exposure_anchor.txt",
                           develop_args=_develop_args(extra_args), ftrace=ftrace)
    return pngs


def _run_frame(cmd: Sequence[str], ftrace: Path,
               stop: Optional[Path]) -> Tuple[int, bool]:
    """Run one frame's ftrace and wait for it; returns (exit code, whether it was stopped).

    It is never killed. When ``stop`` appears it is asked to stop cleanly (``ftrace -stop``:
    it writes its image and checkpoint, then exits). On Ctrl-C -- which the console hands
    ftrace too, and which it answers the same way -- this waits for it to finish rather than
    letting :func:`subprocess.run` terminate it a quarter second later, mid-way through its
    CUDA teardown and checkpoint, then reports the frame as stopped and passes the Ctrl-C on.
    """
    p = subprocess.Popen(list(cmd), cwd=str(repo_root()))
    try:
        while True:
            try:
                return p.wait(timeout=0.25), False
            except subprocess.TimeoutExpired:
                pass
            if stop is not None and stop.exists():
                subprocess.run([str(ftrace), "-stop", str(p.pid)], cwd=str(repo_root()))
                return p.wait(), True
    except KeyboardInterrupt:
        print("[loom] Ctrl-C: waiting for ftrace to write its image and checkpoint", flush=True)
        while True:
            try:
                p.wait()
                break
            except KeyboardInterrupt:
                continue
        raise


# ftrace prints exactly one of these per image write; the number is the auto-exposure
# gain it chose for that film.
_ANCHOR_RE = re.compile(r"auto-exposure=([0-9.eE+-]+)")


def _frame_anchor(ftrace: Path, ftbuf: Path, png: Path,
                  develop_args: Sequence[str] = ()) -> Optional[float]:
    """Develop one checkpoint with its OWN auto-exposure and report the gain chosen."""
    r = subprocess.run([str(ftrace), "-topng", str(ftbuf), str(png), *develop_args],
                       capture_output=True, text=True, cwd=str(repo_root()))
    if r.returncode != 0:
        return None
    m = _ANCHOR_RE.search(r.stdout)
    return float(m.group(1)) if m else None


def stabilize_exposure(pngs: Sequence[os.PathLike], *,
                       anchor_file: Optional[os.PathLike] = None,
                       quiet: bool = False,
                       develop_args: Sequence[str] = (),
                       ftrace: Optional[os.PathLike] = None) -> Optional[float]:
    """Re-develop a rendered sequence through ONE shared auto-exposure anchor.

    A loom sequence is rendered one ftrace process per frame, so every frame picks its
    own auto-exposure — and that anchor is a single order statistic (the 99th luminance
    percentile).  When a scene has a bright, compact specular or emitter population the
    luminance histogram is *bimodal*: ordinary shading in a low mode, the highlight in a
    far brighter one, with almost no mass between them.  As the highlight's **area**
    sweeps across 1% of the frame the p99 rank falls off the cliff from one mode to the
    other and the anchor jumps discontinuously.  Measured on ``png/pastel_jack_ring``:
    up to **42% between adjacent frames**, while every measure of the actual picture
    (the static background, p95, the median) moved less than 2.3% over the same step.
    Each frame is individually defensible; the assembled movie visibly flickers.

    The fix is one anchor for the whole sequence, and it costs no re-rendering: the
    ``-checkpoint`` sidecars still hold the raw linear film, so only the tone map has to
    be redone.  Two cheap passes over the checkpoints — measure every frame's own
    anchor, then re-develop them all through the **median** of those.

    The median specifically, not the first frame's: "first frame wins" is what ftrace's
    ``-exposure-lock`` does *within* one process, but frame 0 of a loop is an arbitrary
    phase of it.  On pastel_jack_ring frame 0 lands at the 93rd percentile of the
    sequence's anchors, so anchoring there would have left the whole movie a full stop
    dark.  The median leaves the typical frame's exposure exactly where it already was
    and pulls only the outliers into line.

    ``develop_args`` are the render's develop flags (``_develop_args`` of its own
    arguments: ``-denoise`` and its knobs, ``-ev``), passed to every re-develop so a
    frame is developed the way it was rendered.

    ``ftrace`` is the binary that rendered them (default :func:`find_ftrace`), so they are
    developed by the build that wrote them.

    Returns the chosen anchor, or ``None`` if no checkpoint was usable (a sequence
    rendered without ``-checkpoint`` cannot be re-developed, and is left alone).
    """
    ftrace = Path(ftrace) if ftrace is not None else find_ftrace()
    frames = [Path(p) for p in pngs]
    pairs = [(p, Path(str(p) + ".ftbuf")) for p in frames]
    usable = [(p, b) for p, b in pairs if b.is_file()]
    if not usable:
        if not quiet:
            print("[loom] exposure: no .ftbuf checkpoints found, leaving frames as "
                  "rendered (render with -checkpoint to enable this)", flush=True)
        return None
    if len(usable) < len(pairs) and not quiet:
        print(f"[loom] exposure: {len(pairs) - len(usable)} of {len(pairs)} frames have "
              f"no checkpoint and will keep their own exposure", flush=True)

    if not quiet:
        print(f"[loom] exposure: metering {len(usable)} frames…", flush=True)
    anchors = []
    for png, ftbuf in usable:
        a = _frame_anchor(ftrace, ftbuf, png, develop_args)
        if a is not None and a > 0.0:
            anchors.append((png, ftbuf, a))
    if not anchors:
        if not quiet:
            print("[loom] exposure: could not read any frame anchor, leaving frames as "
                  "rendered", flush=True)
        return None

    vals = sorted(a for _, _, a in anchors)
    shared = vals[len(vals) // 2]
    if not quiet:
        spread = vals[-1] / vals[0] if vals[0] > 0 else float("inf")
        print(f"[loom] exposure: per-frame anchors span {spread:.2f}x "
              f"({math.log2(spread):.2f} stops); sharing the median {shared:.6g}",
              flush=True)
    if anchor_file is not None:
        Path(anchor_file).write_text(f"{shared!r}\n", encoding="utf-8")

    n_re = 0
    for png, ftbuf, own in anchors:
        if own == shared:
            continue          # the meter pass already developed it at exactly this gain
        r = subprocess.run([str(ftrace), "-topng", str(ftbuf), str(png),
                            "-exposure-anchor", repr(shared), *develop_args],
                           capture_output=True, text=True, cwd=str(repo_root()))
        if r.returncode != 0:
            print(f"[loom] exposure: WARNING re-develop failed for {png.name} "
                  f"(exit {r.returncode}); it keeps its own exposure", flush=True)
            continue
        n_re += 1
    if not quiet:
        print(f"[loom] exposure: re-developed {n_re} frames through the shared anchor",
              flush=True)
    return shared


def render_still(scene: Scene, *, t: float = 0.0, name: str = "loom_still",
                 outdir: Optional[os.PathLike] = None,
                 interval: float = 5.0, noise: Optional[float] = None,
                 time_s: Optional[float] = None, n: Optional[int] = None,
                 hold: bool = True, motion: Optional[Motion] = None,
                 frames: Optional[int] = None, loop: bool = True,
                 extra_args: Sequence[str] = ()) -> Path:
    """Render a single frame at phase ``t`` with a held live window (per rules).

    ``motion`` gives it motion keys (see :func:`render_range`); a key's distance from the
    instant is in frames, so it needs the animation's ``frames`` (and ``loop``) to know how far
    one frame moves the phase."""
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    outdir.mkdir(parents=True, exist_ok=True)
    scene.check_cycles()
    ftrace = find_ftrace()
    if motion is not None and frames is None:
        raise ValueError("render_still(motion=...) needs frames=: a motion key's offset is in "
                         "frames, and one frame's step in phase depends on the frame count")
    clock = (Clock(t=t, frame=0, frames=1, fps=30.0) if frames is None else
             Clock(t=t, frame=int(round(t * (frames if loop else frames - 1))), frames=frames,
                   fps=30.0, loop=loop))
    fp = outdir / f"{name}.ftsl"
    fp.write_text(scene.emit(clock, Cache(), assets_dir=outdir, tag="", motion=motion),
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


def find_gifski() -> Optional[str]:
    """Locate the ``gifski`` encoder, or ``None`` if it isn't installed.

    Checked in order: ``PATH``; then the npm global package, whose shipped binary is
    NOT put on ``PATH`` by ``npm install -g gifski`` on Windows (the package has no
    ``bin`` entry — it exposes a JS API and drops the platform executables under
    ``node_modules/gifski/bin/<platform>/``), which is exactly the trap that makes a
    freshly installed gifski look missing.

    Install with ``npm install -g gifski`` (or ``cargo install gifski``).
    """
    exe = shutil.which("gifski")
    if exe:
        return exe
    plat = {"win32": ("windows", "gifski.exe"),
            "darwin": ("macos", "gifski"),
            "linux": ("debian", "gifski")}.get(sys.platform)
    if plat is None:
        return None
    roots = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "npm" / "node_modules")
    npm_prefix = os.environ.get("NPM_CONFIG_PREFIX")
    if npm_prefix:
        roots.append(Path(npm_prefix) / "node_modules")
    roots += [Path.home() / ".npm-global" / "lib" / "node_modules",
              Path("/usr/local/lib/node_modules"),
              Path("/usr/lib/node_modules")]
    for root in roots:
        cand = root / "gifski" / "bin" / plat[0] / plat[1]
        if cand.is_file():
            return str(cand)
    return None


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


def assemble_gif_gifski(pngs: Sequence[os.PathLike], out_gif: os.PathLike,
                        *, fps: float = 20.0, width: Optional[int] = None,
                        quality: int = 65, loop: int = 0,
                        fallback: bool = True) -> Path:
    """Assemble a looping GIF with **gifski**, falling back to ffmpeg if it's absent.

    Use this for anything that has to be *small* as well as clean — a README embed,
    where GitHub is the only viewer that matters.  gifski quantizes across the whole
    sequence with per-frame dithering tuned to the shared palette; ffmpeg's
    ``palettegen``/``paletteuse`` (:func:`assemble_gif_ffmpeg`) is a fine general
    encoder but is measurably worse per byte on smooth shading.  Measured on the
    ``pastel_jack_ring`` loop (144 frames, 320², 20 fps): ffmpeg bottomed out at
    **5.3 MB** for acceptable quality, gifski 1.7.1 at ``--quality 65`` produced
    **2.7 MB** with visibly less banding on the gyroid glass.

    ``width`` downscales (gifski's own Lanczos), ``loop=0`` repeats forever.

    Frames are passed in the order given — gifski does *not* sort or renumber them —
    so a caller applying a stride just slices its own list.

    Raises if gifski is missing and ``fallback`` is False; otherwise degrades to
    :func:`assemble_gif_ffmpeg` with a printed notice (which ignores ``width``, so a
    fallback GIF comes out at the source resolution and will be bigger).
    """
    out_gif = Path(out_gif)
    pngs = [Path(p) for p in pngs]
    if not pngs:
        raise ValueError("no frames to assemble")
    exe = find_gifski()
    if exe is None:
        if not fallback:
            raise RuntimeError("gifski not found (npm install -g gifski / cargo install gifski)")
        print("[loom] gifski not found — falling back to ffmpeg palettegen "
              "(bigger file; `npm install -g gifski` for the small one)", flush=True)
        return assemble_gif_ffmpeg(pngs, out_gif, fps=fps, loop=loop)
    cmd = [exe, "-o", str(out_gif), "--fps", f"{fps:g}", "--quality", str(int(quality))]
    if width:
        cmd += ["--width", str(int(width))]
    if loop != 0:
        cmd += ["--repeat", str(int(loop) if loop > 0 else -1)]
    cmd += [str(p) for p in pngs]
    print(f"[loom] {exe} -o {out_gif} --fps {fps:g} "
          f"{'--width ' + str(width) + ' ' if width else ''}--quality {quality} "
          f"({len(pngs)} frames)", flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("gifski failed:\n" + r.stderr.decode("utf-8", "replace")[-2000:])
    size = out_gif.stat().st_size
    print(f"[loom] wrote {out_gif} ({len(pngs)} frames @ {fps} fps, "
          f"{size / 1e6:.2f} MB)", flush=True)
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
