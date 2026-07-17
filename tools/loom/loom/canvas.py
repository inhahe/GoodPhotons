"""
Loom 2D backend (M10) — a parallel output driver for **seamless-looping motion
graphics**, over the *same* dimension-agnostic modulation DAG that drives the 3-D
ftrace scenes.  Nothing here forks the core: a 2-D animation is just Signals
sampled at the current :class:`~loom.signals.core.Clock`, exactly like a 3-D one.

The primitive is the user's mental model — *plot an RGB value at an (x, y)
location at the current time*:

    canvas = Canvas2D(512, 512, view=(-1, -1, 1, 1))
    # a dot orbiting the origin; Sine/Cosine are 1-periodic -> the loop closes
    canvas.plot(x=Cosine(cycles=1) * 0.7, y=Sine(cycles=1) * 0.7,
                color=vec(0.9, 0.4, 0.1), radius=8)

Because ``x``/``y``/``color``/``radius`` are Signals, a single ``plot`` traces a
*moving, colour-cycling* marker over the loop; compose it from periodic leaves and
the motion returns bit-for-bit at the wrap (seamless), or from :class:`loom.Ramp`
/ :class:`loom.Ease` for a one-shot ``loop=False`` timeline.

Two output formats (the user asked for **both**), honest about what each does well:

- **SVG** — resolution-independent *vector* primitives (markers + strokes).  A
  per-pixel :meth:`Canvas2D.field` cannot be expressed as vectors, so SVG omits it.
- **raster PNG** — a real pixel buffer (Pillow), so it renders markers, strokes
  **and** a full-canvas :meth:`Canvas2D.field` (per-pixel RGB), and the PNG
  sequence assembles into a seamless GIF via :func:`loom.assemble_gif`.

Coordinates: ``view=(xmin, ymin, xmax, ymax)`` is world space, **y-up** (world +y
is screen-up); ``radius``/``width`` are in **pixels**.  Colours are RGB in [0, 1].
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .signals.core import (
    Signal, Clock, Cache, as_signal, Number, detect_signal_cycle,
)
from .signals.vector import VecSignal, Vecish, vec
from .spatial import SpatialExpr

Point = Tuple[Union[Signal, Number], Union[Signal, Number]]
# a per-pixel field: fn(X, Y, clock, cache) -> (R, G, B).  With ``vectorized``
# X/Y/R/G/B are numpy arrays over the whole canvas; otherwise plain floats.
FieldFn = Callable[..., Tuple]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(x: float) -> str:
    """Compact SVG number (trim trailing zeros)."""
    return f"{x:.4g}"


def _clamp8(v: float) -> int:
    return 0 if v <= 0.0 else 255 if v >= 1.0 else int(v * 255.0 + 0.5)


def _hex(rgb: Sequence[float]) -> str:
    r, g, b = (_clamp8(rgb[0]), _clamp8(rgb[1]), _clamp8(rgb[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _as_rgb(color: Vecish) -> VecSignal:
    c = VecSignal.of(color)
    if c.dim != 3:
        raise ValueError(f"color must be RGB (3 components), got {c.dim}")
    return c


# ---------------------------------------------------------------------------
# primitives (each is a DAG-integrated, per-frame-sampled drawable)
# ---------------------------------------------------------------------------

class Marker:
    """A time-driven point: position, colour and radius are all Signals."""

    def __init__(self, x: Union[Signal, Number], y: Union[Signal, Number],
                 color: Vecish, *, radius: Union[Signal, Number] = 4.0,
                 opacity: Union[Signal, Number] = 1.0,
                 shape: str = "circle") -> None:
        if shape not in ("circle", "square"):
            raise ValueError('shape must be "circle" or "square"')
        self.x = as_signal(x)
        self.y = as_signal(y)
        self.color = _as_rgb(color)
        self.radius = as_signal(radius)
        self.opacity = as_signal(opacity)
        self.shape = shape

    def roots(self) -> List:
        return [self.x, self.y, self.color, self.radius, self.opacity]

    def sample(self, clock: Clock, cache: Cache):
        return (self.x.at(clock, cache), self.y.at(clock, cache),
                self.color.at(clock, cache), self.radius.at(clock, cache),
                self.opacity.at(clock, cache))


class Stroke:
    """A polyline through a list of ``(x, y)`` control points (each a Signal).

    A whole animated path in one drawable — feed it a
    :class:`~loom.interp.LoopCurve` sampled at ``n`` params (see
    :func:`curve_points`) for a seamless looping scribble, or any explicit list.
    """

    def __init__(self, points: Sequence[Point], color: Vecish, *,
                 width: Union[Signal, Number] = 2.0,
                 opacity: Union[Signal, Number] = 1.0,
                 closed: bool = False) -> None:
        self.points: List[Tuple[Signal, Signal]] = [
            (as_signal(px), as_signal(py)) for px, py in points]
        if len(self.points) < 2:
            raise ValueError("a stroke needs at least 2 points")
        self.color = _as_rgb(color)
        self.width = as_signal(width)
        self.opacity = as_signal(opacity)
        self.closed = closed

    def roots(self) -> List:
        out: List = [self.color, self.width, self.opacity]
        for px, py in self.points:
            out.extend((px, py))
        return out

    def sample(self, clock: Clock, cache: Cache):
        pts = [(px.at(clock, cache), py.at(clock, cache)) for px, py in self.points]
        return (pts, self.color.at(clock, cache),
                self.width.at(clock, cache), self.opacity.at(clock, cache))


def curve_points(curve, n: int, *, axes: Tuple[int, int] = (0, 1),
                 scale: float = 1.0) -> List[Point]:
    """Sample a :class:`~loom.interp.LoopCurve` (or any VecSignal-of-``u``) at ``n``
    params, projecting components ``axes`` to 2-D — ready for :class:`Stroke`.

    The curve is re-parameterised by cloning it at ``n`` fixed ``u`` values, so
    each returned point is itself a live Signal pair that animates per frame.
    """
    from .interp import LoopCurve
    if not isinstance(curve, LoopCurve):
        raise TypeError("curve_points expects a LoopCurve")
    if n < 2:
        raise ValueError("curve_points needs n >= 2")
    i, j = axes
    pts: List[Point] = []
    for k in range(n):
        u = k / n if curve.closed else k / (n - 1)
        # a fresh LoopCurve pinned to param ``u`` over the *same* control path:
        # it stays at u but still animates as the control points move per frame.
        c = LoopCurve(curve.path, u, closed=curve.closed)
        pts.append((c[i] * scale, c[j] * scale))
    return pts


# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------

class Canvas2D:
    """A 2-D drawing surface: accumulate time-driven primitives, emit per frame.

    ``view`` is the world box ``(xmin, ymin, xmax, ymax)`` mapped onto the
    ``width`` x ``height`` pixel canvas, **y-up**.  ``background`` is an RGB tuple.
    """

    def __init__(self, width: int = 512, height: int = 512, *,
                 view: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0),
                 background: Vecish = (0.0, 0.0, 0.0)) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width < 1 or self.height < 1:
            raise ValueError("canvas dimensions must be >= 1")
        self.view = tuple(float(v) for v in view)
        if self.view[0] == self.view[2] or self.view[1] == self.view[3]:
            raise ValueError("degenerate view box")
        self.background = _as_rgb(background)
        self.markers: List[Marker] = []
        self.strokes: List[Stroke] = []
        # (kind, payload, vectorized, static): kind in {"call", "expr"}
        self._field: Optional[Tuple] = None
        self._static_raster = None  # cached HxWx3 buffer for a time-independent field

    # ---- authoring --------------------------------------------------------
    def plot(self, x: Union[Signal, Number], y: Union[Signal, Number],
             color: Vecish, *, radius: Union[Signal, Number] = 4.0,
             opacity: Union[Signal, Number] = 1.0,
             shape: str = "circle") -> "Canvas2D":
        """Plot an RGB ``color`` at ``(x, y)`` — each argument a Signal or number."""
        self.markers.append(Marker(x, y, color, radius=radius,
                                    opacity=opacity, shape=shape))
        return self

    def stroke(self, points: Sequence[Point], color: Vecish, *,
               width: Union[Signal, Number] = 2.0,
               opacity: Union[Signal, Number] = 1.0,
               closed: bool = False) -> "Canvas2D":
        self.strokes.append(Stroke(points, color, width=width,
                                   opacity=opacity, closed=closed))
        return self

    def field(self, fn, *, vectorized: bool = True,
              static: Optional[bool] = None) -> "Canvas2D":
        """Set a full-canvas per-pixel field.  Two authoring styles, one entry point:

        - a :class:`~loom.spatial.SpatialExpr` (or a 3-tuple of them for R, G, B) —
          the **shared** spatial layer, evaluated numerically here and emit-able as
          an ftsl string on the 3-D side.  A scalar expr paints greyscale.  loom can
          *introspect* it, so a time-independent field (no ``T`` / temporal
          coefficient) is auto-detected and its raster is **baked once**.
        - a plain callable ``fn(X, Y, clock, cache) -> (R, G, B)`` — an opaque numpy
          (``vectorized=True``, whole-canvas arrays) or per-pixel (``vectorized=
          False``, floats) function.  loom can't introspect it, so pass
          ``static=True`` yourself to bake a time-independent one once.

        Raster (PNG) only — SVG has no per-pixel surface, so :meth:`emit_svg`
        ignores the field.
        """
        self._static_raster = None
        if isinstance(fn, SpatialExpr) or (
                isinstance(fn, (tuple, list))
                and all(isinstance(e, SpatialExpr) for e in fn)):
            exprs = (fn,) if isinstance(fn, SpatialExpr) else tuple(fn)
            if len(exprs) not in (1, 3):
                raise ValueError("an expr field must be 1 (greyscale) or 3 (RGB) exprs")
            auto = not any(e.uses_time() for e in exprs)
            self._field = ("expr", exprs, True, auto if static is None else static)
        elif callable(fn):
            self._field = ("call", fn, bool(vectorized),
                           False if static is None else bool(static))
        else:
            raise TypeError("field expects a SpatialExpr, a 3-tuple of them, "
                            "or a callable")
        return self

    # ---- graph safety -----------------------------------------------------
    def _drawables(self):
        return [*self.strokes, *self.markers]

    def check_cycles(self) -> None:
        detect_signal_cycle(self.background)
        for d in self._drawables():
            for r in d.roots():
                detect_signal_cycle(r)
        if self._field is not None and self._field[0] == "expr":
            for e in self._field[1]:
                for s in e.time_signals():
                    detect_signal_cycle(s)

    # ---- coordinate mapping ----------------------------------------------
    def _to_px(self, x: float, y: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.view
        px = (x - x0) / (x1 - x0) * self.width
        py = (1.0 - (y - y0) / (y1 - y0)) * self.height  # y-up
        return px, py

    # ---- SVG output (vector: markers + strokes) ---------------------------
    def emit_svg(self, clock: Clock, cache: Optional[Cache] = None) -> str:
        cache = cache if cache is not None else Cache()
        W, H = self.width, self.height
        out: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">',
            f'<rect width="{W}" height="{H}" fill="'
            f'{_hex(self.background.at(clock, cache))}"/>',
        ]
        for s in self.strokes:
            pts, col, wid, op = s.sample(clock, cache)
            xy = " ".join(f"{_f(px)},{_f(py)}"
                          for px, py in (self._to_px(x, y) for x, y in pts))
            tag = "polygon" if s.closed else "polyline"
            out.append(
                f'<{tag} points="{xy}" fill="none" stroke="{_hex(col)}" '
                f'stroke-width="{_f(wid)}" stroke-opacity="{_f(op)}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')
        for m in self.markers:
            x, y, col, rad, op = m.sample(clock, cache)
            px, py = self._to_px(x, y)
            if m.shape == "circle":
                out.append(
                    f'<circle cx="{_f(px)}" cy="{_f(py)}" r="{_f(rad)}" '
                    f'fill="{_hex(col)}" fill-opacity="{_f(op)}"/>')
            else:
                out.append(
                    f'<rect x="{_f(px - rad)}" y="{_f(py - rad)}" '
                    f'width="{_f(2 * rad)}" height="{_f(2 * rad)}" '
                    f'fill="{_hex(col)}" fill-opacity="{_f(op)}"/>')
        out.append("</svg>")
        return "\n".join(out)

    # ---- raster output (pixels: field + markers + strokes) ----------------
    def _field_grid(self, np):
        W, H = self.width, self.height
        x0, y0, x1, y1 = self.view
        # pixel-centre world coords, y-up
        xs = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)
        ys = y1 - (np.arange(H) + 0.5) / H * (y1 - y0)
        return np.meshgrid(xs, ys), xs, ys

    def _field_array(self, clock: Clock, cache: Cache):
        import numpy as np
        kind, payload, vectorized, static = self._field  # type: ignore[misc]
        if static and self._static_raster is not None:
            return self._static_raster                     # baked once, reuse
        W, H = self.width, self.height
        (X, Y), xs, ys = self._field_grid(np)
        if kind == "expr":
            Z = np.zeros_like(X)
            chans = [e.eval_np((X, Y, Z), clock, cache) for e in payload]
            if len(chans) == 1:
                chans = chans * 3
            arr = np.stack([np.broadcast_to(c, X.shape).astype(np.float64)
                            for c in chans], axis=-1)
        elif vectorized:
            r, g, b = payload(X, Y, clock, cache)
            arr = np.stack(np.broadcast_arrays(r, g, b), axis=-1)
        else:
            arr = np.empty((H, W, 3), dtype=np.float64)
            for iy in range(H):
                for ix in range(W):
                    arr[iy, ix] = payload(float(xs[ix]), float(ys[iy]), clock, cache)
        arr = np.clip(arr, 0.0, 1.0)
        if static:
            self._static_raster = arr                       # cache the baked field
        return arr

    def rasterize(self, clock: Clock, cache: Optional[Cache] = None):
        """Render one frame to a Pillow ``Image`` (RGB)."""
        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("raster output needs numpy + Pillow "
                               "(pip install numpy pillow)") from e
        cache = cache if cache is not None else Cache()
        W, H = self.width, self.height
        if self._field is not None:
            arr = self._field_array(clock, cache)
            buf = (arr * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(buf, mode="RGB")
        else:
            bg = tuple(_clamp8(c) for c in self.background.at(clock, cache))
            img = Image.new("RGB", (W, H), bg)
        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        for s in self.strokes:
            pts, col, wid, op = s.sample(clock, cache)
            xy = [self._to_px(x, y) for x, y in pts]
            if s.closed:
                xy = xy + [xy[0]]
            rgba = (*(_clamp8(c) for c in col), _clamp8(op))
            draw.line(xy, fill=rgba, width=max(1, int(round(wid))), joint="curve")
        for m in self.markers:
            x, y, col, rad, op = m.sample(clock, cache)
            px, py = self._to_px(x, y)
            rgba = (*(_clamp8(c) for c in col), _clamp8(op))
            box = (px - rad, py - rad, px + rad, py + rad)
            if m.shape == "circle":
                draw.ellipse(box, fill=rgba)
            else:
                draw.rectangle(box, fill=rgba)
        return img.convert("RGB")


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def render_canvas(canvas: Canvas2D, frames: int, *, name: str = "loom2d",
                  outdir: Optional[os.PathLike] = None, fps: float = 30.0,
                  fmt: str = "png", loop: bool = True,
                  gif: bool = True) -> List[Path]:
    """Render ``frames`` frames of ``canvas`` to ``fmt`` in ("png", "svg", "both").

    ``loop=True`` maps frames onto a closed cycle (seamless when composed with
    periodic leaves); ``loop=False`` spans an open timeline.  For a PNG sequence
    with ``gif=True`` a seamless looping ``<name>.gif`` is assembled alongside.
    """
    if fmt not in ("png", "svg", "both"):
        raise ValueError('fmt must be "png", "svg" or "both"')
    from .drive import default_outdir
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    outdir.mkdir(parents=True, exist_ok=True)
    canvas.check_cycles()
    width = max(3, len(str(frames - 1)))
    want_png = fmt in ("png", "both")
    want_svg = fmt in ("svg", "both")
    pngs: List[Path] = []
    outputs: List[Path] = []
    for k in range(frames):
        clock = Clock.at_frame(k, frames, fps, loop=loop)
        cache = Cache()
        tag = f"{k:0{width}d}"
        if want_svg:
            p = outdir / f"{name}{tag}.svg"
            p.write_text(canvas.emit_svg(clock, cache), encoding="utf-8")
            outputs.append(p)
        if want_png:
            p = outdir / f"{name}{tag}.png"
            canvas.rasterize(clock, cache).save(str(p))
            pngs.append(p)
            outputs.append(p)
    print(f"[loom2d] wrote {len(outputs)} file(s) to {outdir}", flush=True)
    if want_png and gif and pngs:
        from .drive import assemble_gif
        assemble_gif(pngs, outdir / f"{name}.gif", fps=fps)
    return outputs


def render_canvas_still(canvas: Canvas2D, *, t: float = 0.0,
                        name: str = "loom2d_still",
                        outdir: Optional[os.PathLike] = None,
                        fmt: str = "png") -> Path:
    """Render a single frame at phase ``t`` (png or svg)."""
    from .drive import default_outdir
    outdir = Path(outdir) if outdir is not None else default_outdir(name)
    outdir.mkdir(parents=True, exist_ok=True)
    canvas.check_cycles()
    clock = Clock(t=t, frame=0, frames=1, fps=30.0)
    if fmt == "svg":
        p = outdir / f"{name}.svg"
        p.write_text(canvas.emit_svg(clock, Cache()), encoding="utf-8")
    elif fmt == "png":
        p = outdir / f"{name}.png"
        canvas.rasterize(clock, Cache()).save(str(p))
    else:
        raise ValueError('fmt must be "png" or "svg"')
    print(f"[loom2d] wrote {p}", flush=True)
    return p
