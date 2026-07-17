"""
Loom example: **transform video** — the two-pass spacetime transform (M11).

A spacetime transform mixes a spatial axis with *time*, which the streaming
emitter can't do (each of its frames is a pure function of one loop phase).  So
this offline tool **materializes** a clip into a 4-D block, **transforms** it, and
**re-slices** it back to frames (DESIGN.md §11.8).

Two honest cases are shown:

- ``--rotate``  the general/default **open** case: a static grating is rotated in
  the (x, t) plane, so it *appears to move* — motion synthesized purely by tilting
  space into time.  It does not loop (open boundaries held at the edges).
- ``--shear``   the constrained **seamless-loop** case: an integer-winding shear on
  the 2-torus.  A tiling grating scrolls exactly ``winding`` whole periods over one
  time loop, so the output is bit-seamless.

Run:
  python examples/transform_video.py            # print what each pass does
  python examples/transform_video.py --rotate   # open spacetime rotation (+GIF)
  python examples/transform_video.py --shear     # seamless torus shear (+GIF)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import Clip, spacetime_rotate, spacetime_shear  # noqa: E402


def _grating(T=48, H=288, W=288, cycles=6):  # W a multiple of T => integer scroll
    """A clip that tiles in x (a vertical sine grating), constant over time.
    Tiling is required for the seamless torus shear."""
    xx = np.linspace(0, 2 * np.pi * cycles, W, endpoint=False)
    row = 0.5 + 0.5 * np.sin(xx)
    band = np.stack([row * 0.35, row * 0.65, 0.25 + row * 0.70], axis=-1)  # (W,3)
    frame = np.broadcast_to(band[None, :, :], (H, W, 3))
    return Clip.from_array(np.broadcast_to(frame[None], (T, H, W, 3)).copy())


def _stripe(T=48, H=288, W=288):
    """A clip with a single *static* bright vertical band (non-tiling), constant
    over time — so the open rotate's synthesized motion is visibly a *sweep*."""
    xs = np.arange(W)
    band = np.exp(-((xs - W * 0.5) / (W * 0.06)) ** 2)         # gaussian stripe
    rgb = np.stack([0.30 + band * 0.70, 0.40 + band * 0.55, 0.95 * band + 0.10],
                   axis=-1)                                    # (W, 3)
    frame = np.broadcast_to(rgb[None, :, :], (H, W, 3))
    return Clip.from_array(np.broadcast_to(frame[None], (T, H, W, 3)).copy())


def _outdir(name):
    from loom.drive import default_outdir
    return default_outdir(name)


def main() -> int:
    if "--rotate" in sys.argv:
        clip = _stripe()
        out = spacetime_rotate(clip, math.radians(35), axis="x",
                               coupling=1.5, mode="clamp")
        paths = out.save(_outdir("xvid_rotate"), name="xvid_rotate", fps=24, gif=True)
        print(f"[xvideo] wrote {len(paths)} rotated frames + GIF to {paths[0].parent}")
        return 0
    if "--shear" in sys.argv:
        clip = _grating()
        out = spacetime_shear(clip, axis="x", winding=1)  # 1 whole scroll / loop
        paths = out.save(_outdir("xvid_shear"), name="xvid_shear", fps=24, gif=True)
        print(f"[xvideo] wrote {len(paths)} seamless shear frames + GIF to {paths[0].parent}")
        return 0

    clip = _grating(T=48)
    print(f"# materialized clip: block {clip.block.shape} (T, H, W, C)")
    r = spacetime_rotate(clip, math.radians(30), axis="x", coupling=1.5)
    s = spacetime_shear(clip, axis="x", winding=1)
    # seam residual for the shear (bit-exact when W is a multiple of T)
    step = int(round(1 * clip.width / clip.frames))
    seam = float(np.abs(np.roll(s.block[-1], step, axis=1) - s.block[0]).max())
    print(f"# open rotate -> {r.frames} frames (static grating now appears to move)")
    print(f"# torus shear -> {s.frames} frames, seam residual {seam:.3g} "
          f"({'bit-seamless' if seam < 1e-6 else 'approx'})")
    print("# run with --rotate / --shear to write the frames + a GIF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
