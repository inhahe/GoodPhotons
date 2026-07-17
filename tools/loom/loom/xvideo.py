"""
Loom transform-video — a **two-pass spacetime transform** over a whole clip (M11).

The streaming emitter's core invariant is *each frame is a pure function of its
loop phase* ``t`` (DESIGN.md §11.8).  A **spacetime transform** — one that mixes a
spatial axis with the *time* axis — violates that: a transformed frame draws on a
*range* of input times, not one.  So this lives **outside** the emitter as a
separate offline tool with an explicit two-pass shape:

1. **materialize** the clip into a 4-D block ``(T, H, W, C)`` — from image files,
   a numpy array, or a :class:`loom.Canvas2D` rendered over the loop;
2. **transform** — resample the block under a spacetime map in a (spatial, time)
   plane;
3. **re-slice** the rotated block back to output frames (PNG / GIF).

**Two honest cases (DESIGN.md §11.8).**

- :func:`spacetime_rotate` — the **general / default, open** case: a metric
  rotation by an arbitrary angle in the (axis, t) plane.  Rotating space into time
  turns a static feature into a moving one (and tilts a moving trajectory).  The
  result is **not** claimed to loop: loop-time is a circle (S¹) and rotating a
  periodic axis into a non-periodic one is no longer periodic.  Out-of-range
  samples are held (``mode="clamp"``) or blanked (``mode="blank"``).

- :func:`spacetime_shear` — the **constrained, seamless-loop** case: an
  integer-**winding** shear on the 2-torus.  Over one output loop, time advances
  exactly one period while the coupled spatial axis scrolls exactly ``winding``
  whole periods, so the lattice maps to itself and the output is **bit-seamless**
  (both axes wrapped).  This is a rotation's torus-preserving cousin — the only
  spacetime map that is genuinely seamless for arbitrary tiling content.  It
  requires content that already tiles in the coupled axis.

Interpolation uses SciPy's ``ndimage.map_coordinates`` (installed).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

# axis name -> spatial block axis index (block is (T, H, W, C))
_SPATIAL_AXIS = {"x": 2, "y": 1}


class Clip:
    """A materialized clip: a ``(T, H, W, C)`` float array in ``[0, 1]``.

    Build one with :meth:`from_array`, :meth:`from_frames`, or
    :meth:`from_canvas`, transform it with :func:`spacetime_rotate` /
    :func:`spacetime_shear`, and write it out with :meth:`save`.
    """

    def __init__(self, block) -> None:
        import numpy as np
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 3:  # (T, H, W) greyscale -> add a channel
            block = np.repeat(block[..., None], 3, axis=-1)
        if block.ndim != 4:
            raise ValueError("clip block must be (T, H, W) or (T, H, W, C)")
        self.block = block

    # ---- shape --------------------------------------------------------------
    @property
    def frames(self) -> int:
        return self.block.shape[0]

    @property
    def height(self) -> int:
        return self.block.shape[1]

    @property
    def width(self) -> int:
        return self.block.shape[2]

    @property
    def channels(self) -> int:
        return self.block.shape[3]

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_array(cls, arr) -> "Clip":
        """Wrap a numpy ``(T, H, W[, C])`` array (values assumed in ``[0, 1]``)."""
        return cls(arr)

    @classmethod
    def from_frames(cls, paths: Sequence[os.PathLike]) -> "Clip":
        """Load a list of image files into a block (all must share H×W)."""
        import numpy as np
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Clip.from_frames needs Pillow") from e
        imgs = []
        for p in paths:
            im = Image.open(str(p)).convert("RGB")
            imgs.append(np.asarray(im, dtype=np.float64) / 255.0)
        if not imgs:
            raise ValueError("no frames given")
        shapes = {a.shape for a in imgs}
        if len(shapes) != 1:
            raise ValueError(f"frames differ in size: {shapes}")
        return cls(np.stack(imgs, axis=0))

    @classmethod
    def from_canvas(cls, canvas, frames: int, *, loop: bool = True) -> "Clip":
        """Render a :class:`loom.Canvas2D` over ``frames`` into a block.

        ``loop`` picks the clock sampling (closed loop vs. open timeline), exactly
        like :func:`loom.render_canvas`.
        """
        import numpy as np
        from .signals.core import Clock, Cache
        arrs = []
        for k in range(frames):
            clock = Clock.at_frame(k, frames, loop=loop)
            img = canvas.rasterize(clock, Cache())
            arrs.append(np.asarray(img, dtype=np.float64) / 255.0)
        return cls(np.stack(arrs, axis=0))

    # ---- output -------------------------------------------------------------
    def to_images(self):
        import numpy as np
        from PIL import Image
        buf = np.clip(self.block, 0.0, 1.0)
        buf = (buf * 255.0 + 0.5).astype(np.uint8)
        mode = "RGB" if self.channels == 3 else "RGBA"
        return [Image.fromarray(buf[k], mode=mode) for k in range(self.frames)]

    def save(self, outdir: os.PathLike, name: str = "xvideo", *,
             fps: float = 30.0, gif: bool = True) -> List[Path]:
        """Write per-frame PNGs (and optionally an assembled GIF)."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        imgs = self.to_images()
        width = max(3, len(str(self.frames - 1)))
        paths: List[Path] = []
        for k, im in enumerate(imgs):
            p = outdir / f"{name}{k:0{width}d}.png"
            im.save(str(p))
            paths.append(p)
        if gif and imgs:
            dur = max(1, int(round(1000.0 / fps)))
            imgs[0].save(str(outdir / f"{name}.gif"), save_all=True,
                         append_images=imgs[1:], duration=dur, loop=0, optimize=True)
        return paths


def _resample_plane(np, plane, ti, si, order, nd_mode, cval):
    """Resample a (T, S) source plane at fractional (time, space) coords."""
    from scipy.ndimage import map_coordinates
    coords = np.vstack([ti.ravel(), si.ravel()])
    out = map_coordinates(plane, coords, order=order, mode=nd_mode, cval=cval)
    return out.reshape(ti.shape)


def _apply_st_map(clip, axis, ti, si, order, nd_mode, cval):
    """Apply a precomputed (n_out, S) source-index map (ti time, si coupled-space)
    to every pass-through slice of the block, returning a new (n_out, H, W, C)."""
    import numpy as np
    sp_ax = _SPATIAL_AXIS[axis]
    blk = clip.block
    T, H, W, C = blk.shape
    n_out, S = ti.shape
    out = np.empty((n_out, H, W, C), dtype=np.float64)
    other_ax = 1 if sp_ax == 2 else 2       # the spatial axis NOT coupled to time
    O = blk.shape[other_ax]
    for o in range(O):
        for c in range(C):
            plane = blk[:, o, :, c] if sp_ax == 2 else blk[:, :, o, c]  # (T, S)
            sampled = _resample_plane(np, plane, ti, si, order, nd_mode, cval)
            if sp_ax == 2:
                out[:, o, :, c] = sampled
            else:
                out[:, :, o, c] = sampled
    return Clip(out)


def spacetime_rotate(
    clip: Clip,
    angle: float,
    *,
    axis: str = "x",
    out_frames: Optional[int] = None,
    coupling: float = 1.0,
    order: int = 1,
    mode: str = "clamp",
) -> Clip:
    """Rotate the (``axis``, *time*) plane of ``clip`` by ``angle`` radians (open).

    The general/default spacetime transform.  For every output sample we map back
    through the **inverse** rotation into the source block and resample.  Rotating
    space into time synthesizes motion from the time axis.  This does **not** loop
    (see module docstring); use :func:`spacetime_shear` for a seamless loop.

    ``axis``      spatial axis coupled to time: ``"x"`` or ``"y"``.
    ``angle``     rotation angle in radians in the (axis, t) plane.
    ``out_frames`` output frame count (default: same as input ``T``).
    ``coupling``  spatial-vs-time scale.  Both axes are normalized to ``[-1, 1]``
                  across their extent; the spatial axis is scaled by ``coupling``
                  before rotating (larger = more spatial travel per unit time).
    ``mode``      open-boundary handling: ``"clamp"`` holds the edge value,
                  ``"blank"`` fills out-of-range samples with 0.
    """
    import numpy as np
    if axis not in _SPATIAL_AXIS:
        raise ValueError("axis must be 'x' or 'y'")
    sp_ax = _SPATIAL_AXIS[axis]
    blk = clip.block
    T = blk.shape[0]
    S = blk.shape[sp_ax]
    n_out = int(out_frames) if out_frames is not None else T
    ca, sa = math.cos(angle), math.sin(angle)

    def _norm(i, n):
        return (i - (n - 1) / 2.0) / (max(n - 1, 1) / 2.0)

    def _denorm(v, n):
        return v * (max(n - 1, 1) / 2.0) + (n - 1) / 2.0

    tn = _norm(np.arange(n_out), n_out)
    sn = _norm(np.arange(S), S) * coupling
    TT, SS = np.meshgrid(tn, sn, indexing="ij")          # (n_out, S)
    src_s = (ca * SS + sa * TT) / coupling               # inverse rotation
    src_t = (-sa * SS + ca * TT)
    ti = _denorm(src_t, T)
    si = _denorm(src_s, S)

    if mode == "blank":
        nd_mode, cval = "constant", 0.0
    elif mode == "clamp":
        nd_mode, cval = "nearest", 0.0
    else:
        raise ValueError("mode must be 'clamp' or 'blank'")
    return _apply_st_map(clip, axis, ti, si, order, nd_mode, cval)


def spacetime_shear(
    clip: Clip,
    *,
    axis: str = "x",
    winding: int = 1,
    out_frames: Optional[int] = None,
    order: int = 1,
) -> Clip:
    """Seamless-loop spacetime shear on the 2-torus (the constrained case).

    Over one output loop, source **time advances one whole period** while the
    coupled spatial axis **scrolls ``winding`` whole periods**.  Both axes are
    wrapped, so for content that already tiles in ``axis`` the output is
    **bit-seamless** (frame ``n`` continues into frame ``0``).  ``winding`` must be
    an integer; a non-integer would break the torus lattice and the seam.

    ``axis``      spatial axis coupled to time: ``"x"`` or ``"y"``.
    ``winding``   integer spatial periods scrolled per time loop (may be negative;
                  ``0`` is a pure re-time / no-op shear).
    ``out_frames`` output frame count (default: same as input ``T``).
    """
    import numpy as np
    if axis not in _SPATIAL_AXIS:
        raise ValueError("axis must be 'x' or 'y'")
    if int(winding) != winding:
        raise ValueError("winding must be an integer (else the loop seam breaks)")
    winding = int(winding)
    sp_ax = _SPATIAL_AXIS[axis]
    blk = clip.block
    T = blk.shape[0]
    S = blk.shape[sp_ax]
    n_out = int(out_frames) if out_frames is not None else T

    phase = np.arange(n_out) / n_out               # closed loop in [0, 1)
    ti_line = phase * T                            # source time advances one period
    s_line = np.arange(S)
    # shear: source space = s - winding * phase * S   (wrapped)
    TI, SI = np.meshgrid(ti_line, s_line, indexing="ij")     # (n_out, S)
    shift = (winding * S) * phase[:, None]
    si = SI - shift
    ti = TI
    return _apply_st_map(clip, axis, ti, si, order, "grid-wrap", 0.0)
