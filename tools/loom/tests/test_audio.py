"""Tests for loom.audio.SampleBuffer (E3 — offline procedural audio)."""
import math
import os
import struct
import tempfile
import wave

import pytest

from loom import SampleBuffer
from loom.signals import Sine


# -- construction ----------------------------------------------------------
def test_construct_by_duration_and_frames():
    b = SampleBuffer(rate=48000, channels=2, duration=0.5)
    assert b.rate == 48000
    assert b.channels == 2
    assert len(b) == 24000
    assert b.frames == 24000
    assert abs(b.duration - 0.5) < 1e-12

    c = SampleBuffer(rate=8000, frames=100)
    assert len(c) == 100
    assert abs(c.duration - 100 / 8000) < 1e-12


def test_construct_validation():
    with pytest.raises(ValueError):
        SampleBuffer(rate=0, frames=10)
    with pytest.raises(ValueError):
        SampleBuffer(channels=0, frames=10)
    with pytest.raises(ValueError):
        SampleBuffer(frames=10, duration=1.0)   # both given
    with pytest.raises(ValueError):
        SampleBuffer(rate=48000)                # neither given
    with pytest.raises(ValueError):
        SampleBuffer(duration=-1.0)


def test_starts_silent():
    b = SampleBuffer(frames=32)
    assert b.peak() == 0.0
    assert b.rms() == 0.0
    assert all(v == 0.0 for v in b.channel(0))


# -- single-sample ops -----------------------------------------------------
def test_add_set_mul_get():
    b = SampleBuffer(frames=8)
    b.add(2, 0.5)
    b.add(2, 0.25)                 # accumulates
    assert b.get(2) == 0.75
    b.set(2, -0.3)
    assert b.get(2) == -0.3
    b.mul(2, 2.0)
    assert abs(b.get(2) - -0.6) < 1e-12


def test_out_of_range_ignored():
    b = SampleBuffer(frames=4)
    b.add(-1, 1.0)                 # ignored, no raise
    b.add(100, 1.0)               # ignored
    b.set(-5, 1.0)
    b.mul(9, 2.0)
    assert b.peak() == 0.0
    assert b.get(-1) == 0.0
    assert b.get(99) == 0.0


def test_bad_channel_raises():
    b = SampleBuffer(frames=4, channels=1)
    with pytest.raises(IndexError):
        b.add(0, 1.0, ch=1)
    with pytest.raises(IndexError):
        b.get(0, ch=-1)


# -- range ops -------------------------------------------------------------
def test_add_range_overlap_add_and_clipping():
    b = SampleBuffer(frames=5)
    b.add_range(1, [0.1, 0.2, 0.3])
    assert [round(b.get(i), 6) for i in range(5)] == [0.0, 0.1, 0.2, 0.3, 0.0]
    b.add_range(2, [1.0, 1.0, 1.0, 1.0, 1.0])  # runs off the end → clipped away
    assert [round(b.get(i), 6) for i in range(5)] == [0.0, 0.1, 1.2, 1.3, 1.0]
    # negative start: leading part clipped
    b2 = SampleBuffer(frames=3)
    b2.add_range(-1, [5.0, 0.5, 0.5])
    assert [b2.get(i) for i in range(3)] == [0.5, 0.5, 0.0]


def test_mul_range():
    b = SampleBuffer(frames=6)
    for i in range(6):
        b.set(i, 1.0)
    b.mul_range(2, 4, 0.5)
    assert [b.get(i) for i in range(6)] == [1.0, 1.0, 0.5, 0.5, 1.0, 1.0]


def test_fade_linear_ramp():
    b = SampleBuffer(frames=5)
    for i in range(5):
        b.set(i, 1.0)
    b.fade(0, 5, 0.0, 1.0)         # ramp 0 -> 1 over indices 0..4
    vals = [b.get(i) for i in range(5)]
    assert abs(vals[0] - 0.0) < 1e-12
    assert abs(vals[4] - 1.0) < 1e-12
    assert abs(vals[2] - 0.5) < 1e-12
    # monotone increasing
    assert all(vals[i] <= vals[i + 1] for i in range(4))


# -- cursor front-end ------------------------------------------------------
def test_emit_next_equivalent_to_add():
    a = SampleBuffer(frames=6)
    b = SampleBuffer(frames=6)
    seq = [0.1, -0.2, 0.3, 0.4]
    for k, v in enumerate(seq):
        a.add(k, v)
    for v in seq:
        b.emit_next(v)
    assert [a.get(i) for i in range(6)] == [b.get(i) for i in range(6)]
    assert b.tell() == len(seq)


def test_seek_and_emit_accumulates():
    b = SampleBuffer(frames=6)
    b.seek(3)
    b.emit_next(0.5)
    b.emit_next(0.25)
    assert b.tell() == 5
    assert b.get(3) == 0.5
    assert b.get(4) == 0.25
    b.seek(3)
    b.emit_next(0.1)              # accumulates onto existing 0.5
    assert abs(b.get(3) - 0.6) < 1e-12


def test_emit_next_off_end_ignored():
    b = SampleBuffer(frames=2)
    b.seek(1)
    b.emit_next(0.5)
    b.emit_next(0.9)             # index 2 — off the end, ignored
    assert b.tell() == 3
    assert b.get(1) == 0.5


# -- producers -------------------------------------------------------------
def test_render_fn_modes():
    b = SampleBuffer(rate=4, frames=4)
    b.render_fn(lambda i, t: float(i), mode="set")
    assert [b.get(i) for i in range(4)] == [0.0, 1.0, 2.0, 3.0]
    b.render_fn(lambda i, t: 1.0, mode="add")
    assert [b.get(i) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
    b.render_fn(lambda i, t: 2.0, mode="mul")
    assert [b.get(i) for i in range(4)] == [2.0, 4.0, 6.0, 8.0]


def test_render_fn_time_is_seconds():
    b = SampleBuffer(rate=10, frames=10)
    seen = {}
    b.render_fn(lambda i, t: seen.setdefault(i, t) or 0.0)
    assert abs(seen[0] - 0.0) < 1e-12
    assert abs(seen[5] - 0.5) < 1e-12
    assert abs(seen[9] - 0.9) < 1e-12


def test_render_fn_start_count_and_gain():
    b = SampleBuffer(rate=8, frames=8)
    b.render_fn(lambda i, t: 1.0, start=2, count=3, gain=0.5, mode="set")
    assert [b.get(i) for i in range(8)] == [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]


def test_render_fn_bad_mode():
    b = SampleBuffer(frames=4)
    with pytest.raises(ValueError):
        b.render_fn(lambda i, t: 0.0, mode="bogus")


def test_render_signal_seamless_loop():
    # Sine(cycles=1) over the region, closed loop: starts at 0, never revisits t=1.
    b = SampleBuffer(rate=64, frames=64)
    b.render_signal(Sine(cycles=1.0), loop=True, mode="set")
    assert abs(b.get(0) - 0.0) < 1e-9
    # quarter cycle -> peak ~1
    assert abs(b.get(16) - 1.0) < 1e-6
    # last sample must not duplicate frame-0 (t < 1)
    assert abs(b.get(63)) > 1e-6


def test_render_signal_gain_and_add():
    b = SampleBuffer(rate=16, frames=16)
    b.set(0, 0.5)
    b.render_signal(Sine(cycles=1.0), gain=0.25, mode="add")
    # at frame 0 sine=0, so add leaves the pre-existing 0.5
    assert abs(b.get(0) - 0.5) < 1e-9


# -- mix -------------------------------------------------------------------
def test_mix_pairwise_channels():
    dst = SampleBuffer(frames=4, channels=2)
    src = SampleBuffer(frames=4, channels=2)
    for i in range(4):
        src.set(i, 0.2, ch=0)
        src.set(i, 0.4, ch=1)
    dst.mix(src, gain=0.5)
    assert [round(dst.get(i, ch=0), 6) for i in range(4)] == [0.1] * 4
    assert [round(dst.get(i, ch=1), 6) for i in range(4)] == [0.2] * 4


def test_mix_route_to_channel_and_offset():
    dst = SampleBuffer(frames=6, channels=2)
    src = SampleBuffer(frames=2, channels=1)
    src.set(0, 1.0)
    src.set(1, 1.0)
    dst.mix(src, at=3, ch=1)
    assert [dst.get(i, ch=0) for i in range(6)] == [0.0] * 6
    assert [dst.get(i, ch=1) for i in range(6)] == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_mix_rate_mismatch_raises():
    a = SampleBuffer(rate=48000, frames=4)
    b = SampleBuffer(rate=44100, frames=4)
    with pytest.raises(ValueError):
        a.mix(b)


# -- analysis --------------------------------------------------------------
def test_peak_and_rms():
    b = SampleBuffer(frames=4)
    b.set(0, 0.3)
    b.set(1, -0.9)
    b.set(2, 0.6)
    assert abs(b.peak() - 0.9) < 1e-12
    expect = math.sqrt((0.3**2 + 0.9**2 + 0.6**2 + 0.0) / 4)
    assert abs(b.rms() - expect) < 1e-12


# -- finalize / WAV round-trip --------------------------------------------
def _read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        return {
            "channels": w.getnchannels(),
            "width": w.getsampwidth(),
            "rate": w.getframerate(),
            "frames": n,
            "raw": raw,
        }


def _unpack16(raw):
    return list(struct.unpack("<%dh" % (len(raw) // 2), raw))


def _unpack24(raw):
    out = []
    for j in range(0, len(raw), 3):
        b0, b1, b2 = raw[j], raw[j + 1], raw[j + 2]
        val = b0 | (b1 << 8) | (b2 << 16)
        if val & 0x800000:
            val -= 1 << 24
        out.append(val)
    return out


def test_finalize_16bit_roundtrip():
    b = SampleBuffer(rate=8000, channels=1, frames=4)
    b.set(0, 0.0)
    b.set(1, 1.0)
    b.set(2, -1.0)
    b.set(3, 0.5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=16, dither=False)
        meta = _read_wav(p)
        assert meta["channels"] == 1
        assert meta["width"] == 2
        assert meta["rate"] == 8000
        assert meta["frames"] == 4
        s = _unpack16(meta["raw"])
        assert s[0] == 0
        assert s[1] == 32767
        assert s[2] == -32767      # -1.0 * 32767 rounded
        assert abs(s[3] - round(0.5 * 32767)) <= 1


def test_finalize_24bit_roundtrip():
    b = SampleBuffer(rate=8000, channels=1, frames=3)
    b.set(0, 0.0)
    b.set(1, 1.0)
    b.set(2, -0.5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=24, dither=False)
        meta = _read_wav(p)
        assert meta["width"] == 3
        s = _unpack24(meta["raw"])
        maxv = (1 << 23) - 1
        assert s[0] == 0
        assert s[1] == maxv
        assert abs(s[2] - round(-0.5 * maxv)) <= 1


def test_finalize_clip():
    b = SampleBuffer(rate=8000, frames=2)
    b.set(0, 2.5)                  # over-range
    b.set(1, -3.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=16, dither=False, clip=True)
        s = _unpack16(_read_wav(p)["raw"])
        assert s[0] == 32767
        assert s[1] == -32767 or s[1] == -32768


def test_finalize_normalize():
    b = SampleBuffer(rate=8000, frames=4)
    b.set(0, 0.25)
    b.set(1, -0.5)                 # peak = 0.5
    b.set(2, 0.1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=16, dither=False, normalize=1.0)
        s = _unpack16(_read_wav(p)["raw"])
        # 0.5 scaled to 1.0 → full scale
        assert abs(abs(s[1]) - 32767) <= 1
        assert abs(s[0] - round(0.5 * 32767)) <= 1


def test_finalize_normalize_silent_buffer():
    b = SampleBuffer(rate=8000, frames=4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=16, dither=False, normalize=1.0)
        s = _unpack16(_read_wav(p)["raw"])
        assert all(v == 0 for v in s)


def test_finalize_dither_deterministic():
    def make():
        bb = SampleBuffer(rate=8000, frames=64)
        bb.render_fn(lambda i, t: 0.3 * math.sin(2 * math.pi * 440 * t))
        return bb
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.wav")
        p2 = os.path.join(d, "b.wav")
        make().finalize(p1, bit_depth=16, dither=True, seed=42)
        make().finalize(p2, bit_depth=16, dither=True, seed=42)
        assert _read_wav(p1)["raw"] == _read_wav(p2)["raw"]
        p3 = os.path.join(d, "c.wav")
        make().finalize(p3, bit_depth=16, dither=True, seed=7)
        assert _read_wav(p3)["raw"] != _read_wav(p1)["raw"]


def test_finalize_bad_bit_depth():
    b = SampleBuffer(frames=4)
    with pytest.raises(ValueError):
        b.finalize("x.wav", bit_depth=8)


def test_finalize_stereo_interleaved():
    b = SampleBuffer(rate=8000, channels=2, frames=2)
    b.set(0, 0.5, ch=0)
    b.set(0, -0.5, ch=1)
    b.set(1, 0.25, ch=0)
    b.set(1, -0.25, ch=1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        b.finalize(p, bit_depth=16, dither=False)
        meta = _read_wav(p)
        assert meta["channels"] == 2
        s = _unpack16(meta["raw"])
        # interleaved L,R,L,R
        assert abs(s[0] - round(0.5 * 32767)) <= 1
        assert abs(s[1] - round(-0.5 * 32767)) <= 1
        assert abs(s[2] - round(0.25 * 32767)) <= 1
        assert abs(s[3] - round(-0.25 * 32767)) <= 1
