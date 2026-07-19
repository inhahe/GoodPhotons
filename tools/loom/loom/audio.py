"""
Loom procedural audio — one buffer back-end, per-tick as a thin front-end.

loom is an **offline** toolkit: it renders finite, static products, never real-time
streams (Python is far too slow to synthesize audio live, and the whole point of the
buffer model — editing arbitrary past/future samples to mix voices, run reverb tails
past a note's end, fade a range, normalize the whole file — is fundamentally
incompatible with commit-as-you-go streaming).  So there is exactly **one** output
model here, per the design decision in ``TODO.md`` §E3:

* :class:`SampleBuffer` is the **single source of truth** — a per-channel float array
  you may read / write / **accumulate** at any index.  Additive and subtractive
  synthesis *are* this model (``buf[t] += v`` / ``-= v`` / ``*= g``).
* "one sample per tick" is **not** a second pipeline — it is a thin cursor wrapper on
  top: :meth:`SampleBuffer.emit_next` is literally ``buf[cursor] += v; cursor += 1``.
* a single :meth:`SampleBuffer.finalize` does gain → normalize → dither → clip →
  PCM-encode → write once at the end.

So "per-tick" and "whole-file" are two front-ends over one back-end, not two ways to
output audio — no divergent dither / clip / normalize / format-writer code paths.

Example::

    buf = SampleBuffer(rate=48000, channels=1, duration=2.0)
    # a 220 Hz sine by arbitrary function of time (full flexibility)
    buf.render_fn(lambda i, t: 0.4 * math.sin(2 * math.pi * 220 * t))
    # overlap-add a decaying blip one second in
    buf.render_fn(lambda i, t: 0.3 * math.sin(2*math.pi*660*t) * math.exp(-4*t),
                  start=buf.rate)                 # start = sample index 1.0 s
    buf.finalize("out.wav", normalize=0.98)
"""

from __future__ import annotations

import array
import math
import random
import struct
import wave
from typing import Callable, List, Optional, Sequence

from .signals.core import Signal, Clock, Cache


__all__ = ["SampleBuffer"]


def _clip_unit(x: float) -> float:
    """Hard-clip a normalized sample to ``[-1, 1]``."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


class SampleBuffer:
    """A mutable multi-channel float sample buffer — loom's whole audio back-end.

    Samples are normalized floats (nominally ``[-1, 1]``; they may exceed that
    while you build up a mix — clipping happens only at :meth:`finalize`).  Every
    channel is a fixed-length :class:`array.array` of ``float64``; you may
    ``add`` / ``set`` / ``mul`` / read at any index, scatter-write ranges, or mix
    another buffer in.  The per-tick :meth:`emit_next` cursor is a convenience over
    the same store, not a separate system.

    Parameters
    ----------
    rate        sample rate in Hz (default 48000).
    channels    channel count (default 1 = mono; 2 = stereo; N supported).
    duration    length in **seconds** (mutually exclusive with ``frames``).
    frames      length in **samples per channel** (mutually exclusive with ``duration``).
    """

    def __init__(self, *, rate: int = 48000, channels: int = 1,
                 duration: Optional[float] = None,
                 frames: Optional[int] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if (duration is None) == (frames is None):
            raise ValueError("give exactly one of duration= or frames=")
        if frames is None:
            if duration < 0:                                   # type: ignore[operator]
                raise ValueError("duration must be >= 0")
            frames = int(round(duration * rate))               # type: ignore[operator]
        if frames < 0:
            raise ValueError("frames must be >= 0")
        self.rate = int(rate)
        self.channels = int(channels)
        self._n = int(frames)
        self._chan: List[array.array] = [
            array.array("d", bytes(8 * self._n)) for _ in range(self.channels)
        ]
        self._cursor: List[int] = [0] * self.channels

    # -- geometry ----------------------------------------------------------
    def __len__(self) -> int:
        """Number of samples per channel."""
        return self._n

    @property
    def frames(self) -> int:
        return self._n

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self._n / self.rate if self.rate else 0.0

    def _ck_ch(self, ch: int) -> None:
        if not (0 <= ch < self.channels):
            raise IndexError(f"channel {ch} out of range [0, {self.channels})")

    # -- random-access single-sample ops -----------------------------------
    def add(self, i: int, v: float, *, ch: int = 0) -> "SampleBuffer":
        """``buf[i] += v`` on channel ``ch`` (the core additive op).  Out-of-range
        indices are silently ignored so a tail that runs off the end just stops —
        the buffer is a bounded canvas, not an error if you paint past its edge."""
        self._ck_ch(ch)
        if 0 <= i < self._n:
            self._chan[ch][i] += v
        return self

    def set(self, i: int, v: float, *, ch: int = 0) -> "SampleBuffer":
        """``buf[i] = v`` (overwrite)."""
        self._ck_ch(ch)
        if 0 <= i < self._n:
            self._chan[ch][i] = v
        return self

    def mul(self, i: int, v: float, *, ch: int = 0) -> "SampleBuffer":
        """``buf[i] *= v`` (e.g. a per-sample envelope)."""
        self._ck_ch(ch)
        if 0 <= i < self._n:
            self._chan[ch][i] *= v
        return self

    def get(self, i: int, *, ch: int = 0) -> float:
        """Read one sample (0.0 outside the buffer)."""
        self._ck_ch(ch)
        return self._chan[ch][i] if 0 <= i < self._n else 0.0

    # -- range / scatter ops ----------------------------------------------
    def add_range(self, start: int, values: Sequence[float], *, ch: int = 0) -> "SampleBuffer":
        """Overlap-**add** ``values`` into the buffer beginning at sample ``start``
        (the standard overlap-add mixing primitive).  Parts that fall outside the
        buffer are clipped away."""
        self._ck_ch(ch)
        buf = self._chan[ch]
        i = start
        for v in values:
            if 0 <= i < self._n:
                buf[i] += v
            i += 1
        return self

    def mul_range(self, start: int, stop: int, factor: float, *, ch: int = 0) -> "SampleBuffer":
        """Multiply every sample in ``[start, stop)`` by ``factor`` (a flat gain /
        mute of a region).  For a *ramp* fade use :meth:`fade`."""
        self._ck_ch(ch)
        buf = self._chan[ch]
        a = max(0, start)
        b = min(self._n, stop)
        for i in range(a, b):
            buf[i] *= factor
        return self

    def fade(self, start: int, stop: int, g0: float, g1: float, *, ch: int = 0) -> "SampleBuffer":
        """Linear gain ramp from ``g0`` at ``start`` to ``g1`` at ``stop-1`` applied
        multiplicatively over ``[start, stop)`` — a fade-in/out over a range."""
        self._ck_ch(ch)
        buf = self._chan[ch]
        a = max(0, start)
        b = min(self._n, stop)
        span = max(1, stop - start - 1)
        for i in range(a, b):
            g = g0 + (g1 - g0) * ((i - start) / span)
            buf[i] *= g
        return self

    def mix(self, other: "SampleBuffer", *, at: int = 0, gain: float = 1.0,
            ch: Optional[int] = None) -> "SampleBuffer":
        """Overlap-add another buffer's samples into this one starting at sample
        ``at``, scaled by ``gain``.  Rates must match.  With ``ch=None`` (default)
        channels are added pairwise (a mono source folds into channel 0); pass
        ``ch=k`` to route ``other``'s channel 0 into this buffer's channel ``k``."""
        if other.rate != self.rate:
            raise ValueError(f"rate mismatch: {other.rate} vs {self.rate}")
        if ch is None:
            pairs = [(c, c) for c in range(min(self.channels, other.channels))]
        else:
            self._ck_ch(ch)
            pairs = [(ch, 0)]
        for dst, src in pairs:
            self.add_range(at, (gain * s for s in other._chan[src]), ch=dst)
        return self

    # -- per-tick cursor front-end (thin wrapper over the same store) -------
    def seek(self, i: int, *, ch: int = 0) -> "SampleBuffer":
        """Move channel ``ch``'s write cursor to sample ``i``."""
        self._ck_ch(ch)
        self._cursor[ch] = i
        return self

    def tell(self, *, ch: int = 0) -> int:
        """Current write-cursor position for channel ``ch``."""
        self._ck_ch(ch)
        return self._cursor[ch]

    def emit_next(self, v: float, *, ch: int = 0) -> "SampleBuffer":
        """Append one sample at the write cursor: ``buf[cursor] += v; cursor += 1``.

        This *is* the "one sample per tick" streaming model — a monotone write
        cursor with no look-back/ahead — expressed as a cursor over the buffer, so
        a sequential generator can just call ``emit_next`` in a loop while every
        random-access capability stays available on the same data."""
        self._ck_ch(ch)
        c = self._cursor[ch]
        if 0 <= c < self._n:
            self._chan[ch][c] += v
        self._cursor[ch] = c + 1
        return self

    # -- producers ---------------------------------------------------------
    def render_fn(self, fn: Callable[[int, float], float], *, ch: int = 0,
                  start: int = 0, count: Optional[int] = None,
                  gain: float = 1.0, mode: str = "add") -> "SampleBuffer":
        """Fill from an arbitrary ``fn(i, t) -> sample`` where ``i`` is the sample
        index (from ``start``) and ``t`` is time in **seconds** (``i / rate``).

        ``mode`` selects the write op — ``"add"`` (overlap-add, the default and the
        basis of additive/subtractive synthesis), ``"set"`` (overwrite), or
        ``"mul"`` (ring-mod / envelope).  ``count`` defaults to the rest of the
        buffer.  This is the general offline-synthesis entry point (write
        ``math.sin(2*math.pi*freq*t)`` etc.)."""
        self._ck_ch(ch)
        if count is None:
            count = self._n - start
        buf = self._chan[ch]
        inv = 1.0 / self.rate
        for k in range(count):
            i = start + k
            if not (0 <= i < self._n):
                continue
            v = gain * fn(i, i * inv)
            if mode == "add":
                buf[i] += v
            elif mode == "set":
                buf[i] = v
            elif mode == "mul":
                buf[i] *= v
            else:
                raise ValueError(f'mode must be "add"/"set"/"mul", got {mode!r}')
        return self

    def render_signal(self, signal: Signal, *, ch: int = 0, start: int = 0,
                      count: Optional[int] = None, gain: float = 1.0,
                      mode: str = "add", loop: bool = True) -> "SampleBuffer":
        """Sample a loom :class:`~loom.signals.core.Signal` into the buffer at audio
        rate, one evaluation per sample.  The signal is clocked with
        :meth:`Clock.at_frame` using ``fps = rate`` and ``frames = count`` (the
        region length), so a periodic leaf authored over the region — e.g.
        ``Sine(cycles=N)`` — completes ``N`` cycles across it and (with ``loop=True``)
        closes seamlessly at the region seam.  A fresh per-sample :class:`Cache`
        keeps shared sub-graphs evaluated once each."""
        self._ck_ch(ch)
        if count is None:
            count = self._n - start
        buf = self._chan[ch]
        cache = Cache()
        for k in range(count):
            i = start + k
            if not (0 <= i < self._n):
                continue
            clock = Clock.at_frame(k, count, fps=float(self.rate), loop=loop)
            v = gain * signal.at(clock, cache)
            if mode == "add":
                buf[i] += v
            elif mode == "set":
                buf[i] = v
            elif mode == "mul":
                buf[i] *= v
            else:
                raise ValueError(f'mode must be "add"/"set"/"mul", got {mode!r}')
        return self

    # -- analysis ----------------------------------------------------------
    def peak(self) -> float:
        """Largest absolute sample across all channels (0.0 for an empty buffer)."""
        m = 0.0
        for ch in self._chan:
            for v in ch:
                a = -v if v < 0.0 else v
                if a > m:
                    m = a
        return m

    def rms(self) -> float:
        """Root-mean-square level across all channels."""
        total = self.channels * self._n
        if total == 0:
            return 0.0
        s = 0.0
        for ch in self._chan:
            for v in ch:
                s += v * v
        return math.sqrt(s / total)

    def channel(self, ch: int) -> array.array:
        """Direct access to a channel's underlying ``array('d')`` (for tests /
        advanced producers).  Mutating it mutates the buffer."""
        self._ck_ch(ch)
        return self._chan[ch]

    # -- finalize ----------------------------------------------------------
    def finalize(self, path: str, *, bit_depth: int = 16, gain: float = 1.0,
                 normalize: Optional[float] = None, dither: Optional[bool] = None,
                 clip: bool = True, seed: int = 0) -> str:
        """The single output stage: gain → normalize → dither → clip → PCM-encode →
        write a ``.wav``.  Returns ``path``.

        Parameters
        ----------
        bit_depth   16 or 24 (signed little-endian PCM; ``wave`` stdlib).
        gain        scalar applied to every sample first.
        normalize   if set, scale so the post-gain peak equals this target
                    (e.g. ``0.98`` ≈ -0.2 dBFS); overrides ``gain``'s effect on level
                    but not its sign.  ``None`` = no normalize.
        dither      add TPDF dither at the LSB before quantizing (decorrelates
                    quantization error).  Default: on for 16-bit, off for 24-bit.
        clip        hard-clip to [-1, 1] before quantizing (default on).  Off lets
                    the encoder wrap — almost never what you want; kept for parity.
        seed        RNG seed for reproducible dither.
        """
        if bit_depth not in (16, 24):
            raise ValueError("bit_depth must be 16 or 24")
        if dither is None:
            dither = (bit_depth == 16)

        g = gain
        if normalize is not None:
            # Scale so the post-normalize peak hits |normalize|, keeping gain's sign.
            raw_pk = self.peak()
            if raw_pk > 0.0:
                g = (abs(normalize) / raw_pk) * (1.0 if gain >= 0.0 else -1.0)
            else:
                g = 0.0

        width = bit_depth // 8
        maxv = (1 << (bit_depth - 1)) - 1          # 32767 / 8388607
        rng = random.Random(seed)
        lsb = 1.0 / maxv

        # Interleave channels sample-by-sample into a bytes buffer.
        out = bytearray()
        n = self._n
        chans = self._chan
        C = self.channels
        for i in range(n):
            for ch in range(C):
                x = chans[ch][i] * g
                if dither:
                    x += (rng.random() - rng.random()) * lsb   # TPDF, ±1 LSB
                if clip:
                    x = _clip_unit(x)
                q = int(round(x * maxv))
                if q > maxv:
                    q = maxv
                elif q < -maxv - 1:
                    q = -maxv - 1
                if width == 2:
                    out += struct.pack("<h", q)
                else:                                          # 24-bit little-endian
                    out += struct.pack("<i", q)[0:3]

        with wave.open(path, "wb") as w:
            w.setnchannels(C)
            w.setsampwidth(width)
            w.setframerate(self.rate)
            w.writeframes(bytes(out))
        return path
