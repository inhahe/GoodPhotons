"""M11 tests: the two-pass spacetime transform-video tool (:mod:`loom.xvideo`).

Cover clip materialization (array / Canvas2D), the general **open** rotate
(static feature -> synthesized motion; open boundary handling), the constrained
**seamless-loop** shear on the 2-torus (bit-exact seam; integer-winding
requirement), shape/axis handling, and validation.  Runnable directly or under
pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import Canvas2D, X, Y, sexpr  # noqa: E402
from loom.xvideo import Clip, spacetime_rotate, spacetime_shear  # noqa: E402


def _bar_clip(T=16, H=32, W=32):
    blk = np.zeros((T, H, W))
    blk[:, :, 14:18] = 1.0  # static vertical bar, all frames
    return Clip.from_array(blk)


def _centroid_x(frame):
    col = frame.mean(axis=(0, 2))  # (H, W, C) -> per-column weight
    xs = np.arange(len(col))
    return float((xs * col).sum() / (col.sum() + 1e-9))


def _grating_clip(T=16, H=24, W=32):
    xx = np.linspace(0, 2 * np.pi, W, endpoint=False)
    g = (0.5 + 0.5 * np.sin(xx))[None, None, :] * np.ones((T, H, 1))
    return Clip.from_array(g)


def test_clip_shapes_and_channel_expand():
    c = Clip.from_array(np.zeros((5, 8, 9)))  # greyscale -> RGB
    assert (c.frames, c.height, c.width, c.channels) == (5, 8, 9, 3)


def test_from_canvas_materializes():
    canv = Canvas2D(24, 24, view=(-1, -1, 1, 1))
    canv.field((0.5 * X + 0.5, 0.5 * Y + 0.5, sexpr(0.2)))
    clip = Clip.from_canvas(canv, frames=6, loop=True)
    assert clip.block.shape == (6, 24, 24, 3)
    assert clip.block.min() >= 0.0 and clip.block.max() <= 1.0


def test_rotate_synthesizes_motion():
    # a static bar becomes a moving bar under an x-t rotation
    out = spacetime_rotate(_bar_clip(), math.radians(35), axis="x", mode="clamp")
    c0 = _centroid_x(out.block[0])
    cl = _centroid_x(out.block[-1])
    assert abs(cl - c0) > 5.0  # genuine motion introduced by the time rotation


def test_rotate_zero_angle_is_identity():
    clip = _bar_clip()
    out = spacetime_rotate(clip, 0.0, axis="x", mode="clamp")
    assert np.allclose(out.block, clip.block)


def test_rotate_blank_vs_clamp_differ_at_edges():
    clip = _bar_clip()
    a = spacetime_rotate(clip, math.radians(40), axis="x", mode="clamp")
    b = spacetime_rotate(clip, math.radians(40), axis="x", mode="blank")
    assert not np.allclose(a.block, b.block)  # boundaries handled differently


def test_shear_is_bit_seamless():
    # A grating that tiles in x, winding=1, with W a multiple of n so the per-frame
    # spatial step (winding*S/n = 2 px) is an exact integer -> no interpolation
    # error.  Then frame k is frame 0 rolled by +step*k, and the frame after the
    # last (a whole-period scroll) wraps *exactly* back to frame 0.
    clip = _grating_clip(T=16, W=32)          # step = 1*32/16 = 2 px, integer
    sh = spacetime_shear(clip, axis="x", winding=1, out_frames=16)
    n, S, step = sh.frames, 32, 2
    row0 = sh.block[0, 0, :, 0]
    # each frame is the previous rolled by +step, exactly
    for k in range(1, n):
        expected = np.roll(row0, step * k)
        assert np.allclose(sh.block[k, 0, :, 0], expected, atol=1e-9)
    # the seam: rolling the last frame by one more step lands bit-exactly on frame 0
    last = sh.block[-1, 0, :, 0]
    assert np.allclose(np.roll(last, step), row0, atol=1e-9)


def test_shear_winding_zero_is_static_scroll():
    # winding=0 over a time-constant clip => every output frame identical (no
    # spatial scroll, and time sampling of an identical block is a no-op).
    clip = _grating_clip(T=10, W=20)
    sh = spacetime_shear(clip, axis="x", winding=0, out_frames=10)
    for k in range(1, sh.frames):
        assert np.allclose(sh.block[k], sh.block[0], atol=1e-9)


def test_shear_rejects_noninteger_winding():
    clip = _grating_clip()
    for bad in (1.5, 0.3):
        try:
            spacetime_shear(clip, winding=bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for non-integer winding")


def test_axis_validation():
    clip = _bar_clip()
    for fn in (lambda: spacetime_rotate(clip, 0.1, axis="z"),
               lambda: spacetime_shear(clip, axis="q")):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a bad axis")


def test_out_frames_controls_length():
    out = spacetime_rotate(_bar_clip(T=16), math.radians(20), out_frames=40)
    assert out.frames == 40


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
