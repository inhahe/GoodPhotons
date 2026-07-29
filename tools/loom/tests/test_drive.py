"""Driver / IO layer: frame emission and video assembly.

Everything here avoids invoking ftrace (the render calls need a built binary and a
GPU); what's covered is the pure-Python half — how a frame range maps onto the loop,
and the ffmpeg/Pillow assembly of those frames into a GIF / MP4.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Scene, Camera, Sphere, phase_drift, emit_frames, assemble_gif,
    assemble_gif_ffmpeg, assemble_mp4,
)
from loom.drive import _stage_frames  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False


def _scene() -> Scene:
    # a monotone ramp, not a Sine: a sine is symmetric about its peak, so several
    # frames of a short loop would emit the identical radius and the "no frame
    # repeats" assertions below would be about the *signal*, not about emit_frames
    s = Scene(camera=Camera(eye=(0, 0, 3), look_at=(0, 0, 0)))
    s.add(Sphere(center=(0, 0, 0), radius=0.5 + 0.25 * phase_drift(1.0), material="m"))
    return s


def _frames(tmp: Path, n: int = 4, size: int = 8) -> list:
    """n tiny PNGs whose colour walks, so a palette/codec has something to encode."""
    out = []
    for k in range(n):
        img = Image.new("RGB", (size, size))
        img.putdata([((k * 37 + i) % 256, (i * 11) % 256, (k * 91) % 256)
                     for i in range(size * size)])
        p = tmp / f"f{k:03d}.png"
        img.save(p)
        out.append(p)
    return out


# --- emit_frames ------------------------------------------------------------

def test_emit_frames_closed_loop_wraps(tmp_path):
    paths = emit_frames(_scene(), 6, tmp_path, "loop", loop=True)
    assert [p.name for p in paths] == [f"loop{k:03d}.ftsl" for k in range(6)]
    # a closed loop maps t = k/frames, so frame 0 is the seam and no frame repeats it
    texts = [p.read_text(encoding="utf-8") for p in paths]
    assert len(set(texts)) == 6
    # emitting one extra frame lands exactly back on frame 0
    wrap = emit_frames(_scene(), 6, tmp_path / "w", "loop", loop=True)
    assert wrap[0].read_text(encoding="utf-8") == texts[0]


def test_emit_frames_open_timeline_has_distinct_endpoints(tmp_path):
    a = [p.read_text(encoding="utf-8")
         for p in emit_frames(_scene(), 5, tmp_path / "a", "shot", loop=False)]
    b = [p.read_text(encoding="utf-8")
         for p in emit_frames(_scene(), 5, tmp_path / "b", "shot", loop=True)]
    # open spans [0,1] inclusive: 5 distinct frames, first != last
    assert len(set(a)) == 5 and a[0] != a[-1]
    # ... and it must NOT collapse onto the closed mapping.  The last open frame sits
    # at t=1, which a periodic wrap would fold to t=0 -- i.e. back onto frame 0 (the
    # TimeFn-wraps-an-open-clock bug).  Only frame 0 may agree between the two.
    assert a[0] == b[0]
    assert all(x != y for x, y in zip(a[1:], b[1:]))


def test_emit_frames_pads_index_width_to_the_frame_count(tmp_path):
    assert emit_frames(_scene(), 1200, tmp_path, "big")[0].name == "big0000.ftsl"


# --- the concat script ------------------------------------------------------

def test_stage_frames_renumbers_without_adding_or_dropping_any(tmp_path):
    src = tmp_path / "dir with space"       # ffmpeg-hostile name, deliberately
    src.mkdir()
    files = []
    for i, name in enumerate(("z.png", "a.png", "m.png")):
        f = src / name
        f.write_bytes(bytes([i]))
        files.append(f)
    stage = tmp_path / "stage"
    pat = _stage_frames(files, stage)
    staged = sorted(stage.iterdir())
    assert [p.name for p in staged] == ["f000000.png", "f000001.png", "f000002.png"]
    # exactly one staged frame per input, in the *given* order (not sorted by name)
    assert [p.read_bytes() for p in staged] == [f.read_bytes() for f in files]
    assert pat.endswith("f%06d.png")


def test_assemblers_reject_an_empty_frame_list(tmp_path):
    for fn, name in ((assemble_gif_ffmpeg, "x.gif"), (assemble_mp4, "x.mp4")):
        with pytest.raises(ValueError):
            fn([], tmp_path / name)


# --- real assembly ----------------------------------------------------------

@pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_PIL), reason="needs ffmpeg + Pillow")
def test_assemble_gif_ffmpeg_writes_a_looping_gif(tmp_path):
    frames = _frames(tmp_path)
    out = assemble_gif_ffmpeg(frames, tmp_path / "o.gif", fps=25.0)
    assert out.is_file() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert im.n_frames == len(frames)
        assert im.info.get("loop") == 0            # loop=0 == forever
        # 25 fps must land on an exact 4-centisecond delay
        assert im.info["duration"] == 40


@pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_PIL), reason="needs ffmpeg + Pillow")
def test_assemble_gif_ffmpeg_play_once(tmp_path):
    out = assemble_gif_ffmpeg(_frames(tmp_path), tmp_path / "once.gif", fps=25.0, loop=-1)
    with Image.open(out) as im:
        # GIF has no "play once" *value*: it's the absence of the NETSCAPE application
        # block, which ffmpeg omits for loop=-1.  Pillow only sets info["loop"] when
        # that block exists, so "no key" is exactly what play-once looks like.
        assert "loop" not in im.info


@pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_PIL), reason="needs ffmpeg + Pillow")
def test_assemble_mp4_pads_odd_dimensions_and_keeps_every_frame(tmp_path):
    # 7x7 is odd on both axes, which yuv420p cannot represent unless padded
    frames = _frames(tmp_path, n=5, size=7)
    out = assemble_mp4(frames, tmp_path / "o.mp4", fps=25.0, preset="ultrafast")
    assert out.is_file() and out.stat().st_size > 0
    probe = shutil.which("ffprobe")
    if probe is None:  # pragma: no cover - ffmpeg without ffprobe is unusual
        return
    r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries",
                        "stream=width,height,pix_fmt,nb_read_frames",
                        "-of", "default=nw=1:nk=1", str(out)],
                       capture_output=True, text=True)
    w, h, pix, n = r.stdout.split()
    assert (int(w), int(h)) == (8, 8)      # padded up to even
    assert pix == "yuv420p"
    assert int(n) == len(frames)           # no dropped or duplicated frame


@pytest.mark.skipif(not HAVE_PIL, reason="needs Pillow")
def test_assemble_gif_pillow_still_works(tmp_path):
    out = assemble_gif(_frames(tmp_path), tmp_path / "p.gif", fps=25.0)
    with Image.open(out) as im:
        assert im.n_frames == 4 and im.info.get("loop") == 0
