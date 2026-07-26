"""
Loom spatial-expression tree (M10.5) — one pattern definition, evaluated **two
ways**: numerically over pixels (the 2-D backend) *and* emitted as an ftsl string
(the 3-D isosurface / material backend).

DESIGN.md §11.10: loom's temporal DAG is a function of *time* (the clock), cached
per frame.  A *field* is a function of *space* (x, y, z) — a different axis — so it
does **not** belong in the time-DAG (its per-frame cache would be wrong: one value
per pixel, not per frame).  Coordinates live here, in a small **spatial** algebra
whose leaves are the coordinate variables :data:`X`, :data:`Y`, :data:`Z` and the
loop phase :data:`T`, and whose *coefficients* may be temporal :class:`Signal`\\s
(baked per frame — exactly how the 3-D side already animates a static formula).

The leaf family (:class:`Surface`) also carries the *surface* inputs :data:`U`,
:data:`V` (the ftrace pattern vars ``u``/``v`` — emit-only) and the material
*albedo* placeholder :data:`A` (no ftrace variable — a pure binding slot).  A
:class:`SpatialExpr` reports its named inputs with :meth:`SpatialExpr.free_inputs`
and binds them by rewrite with :meth:`SpatialExpr.substitute` — this is the
substrate for materials-as-bundles (``gold(u=v, a=1)`` / ``(a=x*.5)`` authoring):
loom resolves every binding to a concrete field in real ftrace variables at emit,
so it never writes literal bundle syntax and stays renderable at every step.

A :class:`SpatialExpr` evaluates two ways:

- :meth:`SpatialExpr.eval_np` — numerically over numpy coordinate arrays (2-D
  raster fields), and
- :meth:`SpatialExpr.emit` — as an ftsl expression string in ``x``/``y``/``z``
  with the temporal coefficients baked to numbers (3-D isosurfaces / patterns).

Because it also exposes ``build(cx, cy, cz, ctx)`` and ``param_signals()`` it drops
straight into :class:`loom.Isosurface` and :class:`loom.FuncPattern` through their
existing duck-typed template protocol — no changes there.  Every function name
emitted (:func:`sin`, :func:`sign`, :func:`clamp`, …) is a real ftsl pattern
builtin (``src/pattern.h``), so the emitted string always parses; the numpy path
computes the *same* mathematics (``noise`` is intentionally absent — ftrace's value
noise has no bit-identical numpy twin, so it would break the "one definition, two
backends" honesty).

One leaf is not a coordinate but a *photograph*: :class:`Image` samples a decoded
image file as a scalar **term inside a formula**, so a picture can be multiplied,
thresholded, warped or blended like any other subexpression rather than only bound
wholesale to a material slot.  It honours the two-backend rule the same way as the
rest of the algebra: it emits ftrace's ``tex:<name>(u, v)`` pattern op (``PatOp::Tex``
in ``src/pattern.h``, with the GPU twin in ``render_cuda.cu``) and its
:meth:`Image.eval_np` is a faithful port of ``Texture::sampleRgb``/``scalarAt`` —
same ``-0.5`` texel offset, same ``v``-flip, same repeat/clamp/mirror wrapping, same
mean-of-RGB reduction.  Its ``u``/``v`` arguments are ordinary sub-expressions, so
the coordinates can themselves be warped or rebound.
"""

from __future__ import annotations

import operator
from typing import Callable, List, Sequence, Tuple, Union

from .signals.core import Signal, Number
from .ftsl_emit import fmt

try:  # numpy is only needed for the 2-D numeric path
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------

class SpatialExpr:
    """A scalar function of space (and optionally the loop phase ``T``).

    Build with the leaves :data:`X`/:data:`Y`/:data:`Z`/:data:`T`, Python numbers,
    temporal :class:`~loom.signals.core.Signal`\\s (used as animated coefficients),
    the arithmetic operators, and the module math functions.
    """

    # ---- operators (coerce the other operand into the spatial algebra) ----
    def __add__(self, o): return _Bin("+", self, _coerce(o))
    def __radd__(self, o): return _Bin("+", _coerce(o), self)
    def __sub__(self, o): return _Bin("-", self, _coerce(o))
    def __rsub__(self, o): return _Bin("-", _coerce(o), self)
    def __mul__(self, o): return _Bin("*", self, _coerce(o))
    def __rmul__(self, o): return _Bin("*", _coerce(o), self)
    def __truediv__(self, o): return _Bin("/", self, _coerce(o))
    def __rtruediv__(self, o): return _Bin("/", _coerce(o), self)
    def __neg__(self): return _Neg(self)
    def __pow__(self, o): return spow(self, o)
    def __abs__(self): return sabs(self)

    # ---- tree walk (leaves override the hooks) ----------------------------
    def children(self) -> Tuple["SpatialExpr", ...]:
        return ()

    def _time_signal(self):
        return None  # a _Sig leaf returns its wrapped Signal

    def _is_time(self) -> bool:
        return False  # T and _Sig leaves are time-dependent

    def _walk(self):
        stack: List[SpatialExpr] = [self]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children())

    # ---- named-input inspection / binding (J3b materials-as-bundles) -------
    def _input_name(self):
        return None  # a Surface leaf returns its binding name

    def _image_texture(self):
        return None  # an Image leaf returns the loom Texture it needs declared

    def image_textures(self) -> List:
        """The :class:`loom.scene.Texture` declarations required by the
        :class:`Image` leaves in this tree, deduped by name and in encounter order.

        :meth:`loom.Scene.add` collects these automatically, so an ``Image`` term
        drops into a field without the author having to declare the texture
        separately — but the list is public so a hand-rolled emit path can too."""
        out, seen = [], set()
        for n in self._walk():
            t = n._image_texture()
            if t is not None and t.name not in seen:
                seen.add(t.name)
                out.append(t)
        return out

    def free_inputs(self, include_coords: bool = False) -> "frozenset[str]":
        """The set of named input leaves present in the tree.

        By default this is the *bindable* free-input set — the surface params
        (:data:`U`, :data:`V`) and the albedo (:data:`A`) — which is exactly what
        a material exposes for binding (``gold(u=v, a=1)``).  Pass
        ``include_coords=True`` to also include the system-provided spatial
        coordinates :data:`X`/:data:`Y`/:data:`Z`."""
        out = set()
        for n in self._walk():
            nm = n._input_name()
            if nm is None:
                continue
            if include_coords or not getattr(n, "is_coord", False):
                out.add(nm)
        return frozenset(out)

    def substitute(self, mapping) -> "SpatialExpr":
        """Return a copy of the tree with every named input leaf whose name is a
        key of ``mapping`` replaced by ``mapping[name]`` (coerced into the spatial
        algebra).  This is how a material binds its free inputs at emit —
        ``gold(u=v)`` rewrites the :data:`U` leaf to the consumer's expression,
        so loom always emits a concrete field in real ftrace variables and never
        literal bundle syntax."""
        kids = self.children()
        if not kids:
            return self
        return self._rebuild([k.substitute(mapping) for k in kids])

    def _rebuild(self, new_children) -> "SpatialExpr":
        return self  # leaves have no children; interior nodes override

    def time_signals(self) -> List[Signal]:
        """The temporal Signals embedded as coefficients (deduped) — the DAG roots
        an :class:`Isosurface`/:class:`FuncPattern` must expose for cycle/cache."""
        out: List[Signal] = []
        seen = set()
        for n in self._walk():
            s = n._time_signal()
            if s is not None and id(s) not in seen:
                seen.add(id(s))
                out.append(s)
        return out

    def uses_time(self) -> bool:
        """True if any ``T`` leaf or temporal-Signal coefficient is present — i.e.
        the field varies over the loop (else a 2-D raster can be baked once)."""
        return any(n._is_time() for n in self._walk())

    # ---- evaluation (subclasses implement) --------------------------------
    def emit(self, coords: Tuple[str, str, str], ctx) -> str:
        raise NotImplementedError

    def eval_np(self, coords, clock, cache):
        raise NotImplementedError

    # ---- duck-typed template protocol (Isosurface / FuncPattern) ----------
    def build(self, cx: str, cy: str, cz: str, ctx) -> str:
        return self.emit((cx, cy, cz), ctx)

    def param_signals(self) -> List[Signal]:
        return self.time_signals()


def _coerce(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    if isinstance(v, SpatialExpr):
        return v
    if isinstance(v, Signal):
        return _Sig(v)
    if isinstance(v, (int, float)):
        return _Const(float(v))
    raise TypeError(f"cannot use {type(v).__name__} in a spatial expression")


def sexpr(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    """Coerce a number / temporal Signal / SpatialExpr into the spatial algebra."""
    return _coerce(v)


# ---------------------------------------------------------------------------
# leaves
# ---------------------------------------------------------------------------

class _Const(SpatialExpr):
    def __init__(self, v: float) -> None:
        self.v = float(v)

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.v)})"

    def eval_np(self, coords, clock, cache):
        return self.v


class Surface(SpatialExpr):
    """A named input leaf of the field / material grammar (J3b).

    Six singletons live on this class:

    - :data:`X` / :data:`Y` / :data:`Z` — the spatial coordinates (axes 0/1/2).
      Evaluated *both* ways: ``eval_np`` indexes the coordinate arrays and
      ``emit`` writes the coordinate token.
    - :data:`U` / :data:`V` — the surface parameters (ftrace pattern variables
      ``u`` / ``v``).  **Emit-only**: they are real ftsl tokens so they render,
      but they have no numpy twin, so ``eval_np`` raises (the 2-D raster backend
      has no surface UV).
    - :data:`A` — the material *albedo* input.  ftrace's pattern VM has **no**
      ``a`` variable, so :data:`A` is a pure binding placeholder: it must be
      substituted away (``substitute({'a': ...})``) or defaulted by the material
      before ``emit``; emitting a bare :data:`A` raises.

    A material's free-input set (:meth:`SpatialExpr.free_inputs`) is the union of
    its properties' input leaves; binding rewrites those leaves by name with
    :meth:`SpatialExpr.substitute` — ``gold(u=v, a=1)``."""

    def __init__(self, name: str, *, axis: int = None, emit_ok: bool = True) -> None:
        self.name = name
        self.axis = axis
        self.is_coord = axis is not None
        self._emit_ok = emit_ok

    def emit(self, coords, ctx) -> str:
        if self.axis is not None:
            return f"({coords[self.axis]})"
        if not self._emit_ok:
            raise ValueError(
                f"the material input '{self.name}' has no ftrace pattern "
                f"variable; bind it (e.g. {self.name}=<expr>) or give the "
                f"material an albedo default before emitting")
        return f"({self.name})"

    def eval_np(self, coords, clock, cache):
        if self.axis is not None:
            return coords[self.axis]
        raise ValueError(
            f"the surface input '{self.name}' is emit-only (no numpy twin); it "
            f"exists on the 3-D / material backend, not the 2-D raster path")

    def _input_name(self):
        return self.name

    def substitute(self, mapping) -> "SpatialExpr":
        if self.name in mapping:
            return _coerce(mapping[self.name])
        return self


class Image(SpatialExpr):
    """An **image sampled as a term inside a formula** (J3b item 3b).

    ``Image("bark.png")`` is a scalar leaf whose value at a surface point is the
    image's grayscale level there, so a photo can be *multiplied into* a procedural
    field rather than pasted over it::

        grime = Image("grime.png")
        rough = 0.05 + 0.9 * grime * (0.5 + 0.5 * sin(30 * X))

    This is the complement of :func:`loom.scene.skin` — that binds a whole image to
    a material slot; this makes the image one operand of an expression.

    The coordinates default to the surface params :data:`U`/:data:`V` but are
    ordinary :class:`SpatialExpr`\\s, so the image can be warped
    (``Image("p.png", u=U * 2 + 0.1 * sin(10 * V))``) and stays rebindable — a
    :meth:`SpatialExpr.substitute` reaches inside them, which is how a material
    bundle's ``u=``/``v=`` binding flows into an image term.

    **Both backends.**  :meth:`emit` writes ftrace's ``tex:<name>(u, v)`` pattern-VM
    call (``PatOp::Tex``) and the required ``texture`` declaration is collected by
    :meth:`SpatialExpr.image_textures` — :meth:`loom.Scene.add` picks it up, so the
    author never declares it.  :meth:`eval_np` is an exact port of ftrace's
    ``Texture::sampleRgb`` / ``scalarAt`` (same v-flip, same wrap, same bilerp, same
    channel mean), so the 2-D raster path agrees with the render wherever the
    coordinates are computable there — i.e. whenever ``u``/``v`` are expressed in
    :data:`X`/:data:`Y`/:data:`Z`; bare :data:`U`/:data:`V` stay emit-only, as they
    are everywhere else.

    ``encoding`` defaults to ``"linear"`` (not ``"srgb"``, the default for a
    :class:`~loom.scene.Texture` *skin*) because a value used as a NUMBER wants the
    stored levels, not a de-gamma'd colour — the same advice ftrace gives for its
    scalar maps.
    """

    _cache: dict = {}          # path -> decoded float32 HxWx3 in [0,1] (LINEAR)

    def __init__(self, path, *, u=None, v=None, name: str = None,
                 encoding: str = "linear", filter: str = "bilinear",
                 wrap: str = "repeat") -> None:
        self.path = str(path).replace("\\", "/")
        self.u = _coerce(U if u is None else u)
        self.v = _coerce(V if v is None else v)
        if encoding not in ("srgb", "linear"):
            raise ValueError('image encoding must be "srgb" or "linear"')
        if filter not in ("bilinear", "nearest"):
            raise ValueError('image filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('image wrap must be "repeat", "clamp" or "mirror"')
        self.encoding = encoding
        self.filter = filter
        self.wrap = wrap
        self.name = name if name is not None else self._auto_name()

    # ---- texture declaration ---------------------------------------------
    def _auto_name(self) -> str:
        """A deterministic ftsl identifier for this image+sampler settings.

        Two ``Image`` leaves over the same file with the same sampler settings get
        the SAME name, so they share one ``texture`` block; differing settings get
        different names, so they don't silently collide."""
        import hashlib, re, os
        stem = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(self.path))[0])
        key = f"{self.path}|{self.encoding}|{self.filter}|{self.wrap}"
        return f"img_{stem}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"

    def _image_texture(self):
        from .scene import Texture   # lazy: scene.py is the higher layer
        return Texture(self.name, self.path, encoding=self.encoding,
                       filter=self.filter, wrap=self.wrap)

    # ---- tree ------------------------------------------------------------
    def children(self):
        return (self.u, self.v)

    def _rebuild(self, new_children):
        return Image(self.path, u=new_children[0], v=new_children[1],
                     name=self.name, encoding=self.encoding,
                     filter=self.filter, wrap=self.wrap)

    # ---- emit ------------------------------------------------------------
    def emit(self, coords, ctx) -> str:
        return f"tex:{self.name}({self.u.emit(coords, ctx)},{self.v.emit(coords, ctx)})"

    # ---- numpy twin (exact port of Texture::sampleRgb + scalarAt) --------
    @classmethod
    def _load(cls, path: str, encoding: str):
        key = (path, encoding)
        hit = cls._cache.get(key)
        if hit is not None:
            return hit
        from PIL import Image as _PILImage
        with _PILImage.open(path) as im:
            a = _np.asarray(im.convert("RGB"), dtype=_np.float32) / 255.0
        if encoding == "srgb":       # sRGB EOTF, matching ftrace's texture decode
            a = _np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        cls._cache[key] = a
        return a

    def _wrap_index(self, i, n):
        if self.wrap == "clamp":
            return _np.clip(i, 0, n - 1)
        if self.wrap == "mirror":
            period = 2 * n
            m = _np.mod(i, period)
            return _np.where(m < n, m, period - 1 - m)
        return _np.mod(i, n)         # repeat

    def eval_np(self, coords, clock, cache):
        if _np is None:              # pragma: no cover
            raise ImportError("Image.eval_np needs numpy")
        u = _np.asarray(self.u.eval_np(coords, clock, cache), dtype=_np.float64)
        v = _np.asarray(self.v.eval_np(coords, clock, cache), dtype=_np.float64)
        img = self._load(self.path, self.encoding)
        h, w = img.shape[0], img.shape[1]
        if self.filter == "nearest":
            x = self._wrap_index(_np.floor(u * w).astype(_np.int64), w)
            y = self._wrap_index(_np.floor((1.0 - v) * h).astype(_np.int64), h)
            c = img[y, x]
        else:
            tu = u * w - 0.5
            tv = (1.0 - v) * h - 0.5      # ftrace flips v (OBJ convention)
            flx = _np.floor(tu)
            fly = _np.floor(tv)
            fx = (tu - flx)[..., None]
            fy = (tv - fly)[..., None]
            x0 = self._wrap_index(flx.astype(_np.int64), w)
            x1 = self._wrap_index(flx.astype(_np.int64) + 1, w)
            y0 = self._wrap_index(fly.astype(_np.int64), h)
            y1 = self._wrap_index(fly.astype(_np.int64) + 1, h)
            a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
            b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
            c = a * (1 - fy) + b * fy
        return c.mean(axis=-1)            # Texture::scalarAt = mean of linear rgb


class _Time(SpatialExpr):
    """The loop phase ``t`` in [0, 1) at the current frame."""

    def emit(self, coords, ctx) -> str:
        return f"({fmt(ctx.clock.t)})"

    def eval_np(self, coords, clock, cache):
        return clock.t

    def _is_time(self) -> bool:
        return True


class _Sig(SpatialExpr):
    """A temporal Signal used as an animated coefficient (baked per frame)."""

    def __init__(self, sig: Signal) -> None:
        self.sig = sig

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.sig.at(ctx.clock, ctx.cache))})"

    def eval_np(self, coords, clock, cache):
        return self.sig.at(clock, cache)

    def _time_signal(self):
        return self.sig

    def _is_time(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

_BINOPS = {"+": operator.add, "-": operator.sub,
           "*": operator.mul, "/": operator.truediv}


class _Bin(SpatialExpr):
    def __init__(self, op: str, a: SpatialExpr, b: SpatialExpr) -> None:
        self.op = op
        self.a = a
        self.b = b

    def children(self):
        return (self.a, self.b)

    def _rebuild(self, new_children):
        return _Bin(self.op, new_children[0], new_children[1])

    def emit(self, coords, ctx) -> str:
        return f"({self.a.emit(coords, ctx)}{self.op}{self.b.emit(coords, ctx)})"

    def eval_np(self, coords, clock, cache):
        return _BINOPS[self.op](self.a.eval_np(coords, clock, cache),
                                self.b.eval_np(coords, clock, cache))


class _Neg(SpatialExpr):
    def __init__(self, a: SpatialExpr) -> None:
        self.a = a

    def children(self):
        return (self.a,)

    def _rebuild(self, new_children):
        return _Neg(new_children[0])

    def emit(self, coords, ctx) -> str:
        return f"(-({self.a.emit(coords, ctx)}))"

    def eval_np(self, coords, clock, cache):
        return -self.a.eval_np(coords, clock, cache)


class _Fn(SpatialExpr):
    """An ftsl builtin call — ``name`` must exist in ``src/pattern.h``; ``npfn`` is
    its numpy twin (same argument order/semantics)."""

    def __init__(self, name: str, args: Sequence[SpatialExpr], npfn: Callable) -> None:
        self.name = name
        self.args = list(args)
        self.npfn = npfn

    def children(self):
        return tuple(self.args)

    def _rebuild(self, new_children):
        return _Fn(self.name, new_children, self.npfn)

    def emit(self, coords, ctx) -> str:
        inner = ",".join(a.emit(coords, ctx) for a in self.args)
        return f"{self.name}({inner})"

    def eval_np(self, coords, clock, cache):
        return self.npfn(*[a.eval_np(coords, clock, cache) for a in self.args])


# ---------------------------------------------------------------------------
# leaf singletons + math functions (each emits a real ftsl pattern builtin)
# ---------------------------------------------------------------------------

X = Surface("x", axis=0)
Y = Surface("y", axis=1)
Z = Surface("z", axis=2)
U = Surface("u")                    # surface param (emit-only)
V = Surface("v")                    # surface param (emit-only)
A = Surface("a", emit_ok=False)     # albedo binding placeholder (no ftrace var)
T = _Time()


def _mk(name: str, npfn: Callable) -> Callable[..., SpatialExpr]:
    def f(*args):
        return _Fn(name, [_coerce(a) for a in args], npfn)
    f.__name__ = name
    return f


def _np_step(edge, x):
    return _np.where(_np.asarray(x) >= edge, 1.0, 0.0)


def _np_clamp(x, lo, hi):
    return _np.minimum(hi, _np.maximum(lo, x))


def _np_mix(a, b, t):
    return a + (b - a) * t


def _np_smoothstep(e0, e1, x):
    t = _np.clip((_np.asarray(x) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _np_fract(x):
    return x - _np.floor(x)


# unary
sin = _mk("sin", lambda x: _np.sin(x))
cos = _mk("cos", lambda x: _np.cos(x))
tan = _mk("tan", lambda x: _np.tan(x))
sqrt = _mk("sqrt", lambda x: _np.sqrt(x))
exp = _mk("exp", lambda x: _np.exp(x))
log = _mk("log", lambda x: _np.log(x))
floor = _mk("floor", lambda x: _np.floor(x))
fract = _mk("fract", _np_fract)
sign = _mk("sign", lambda x: _np.sign(x))
saturate = _mk("saturate", lambda x: _np.clip(x, 0.0, 1.0))
sabs = _mk("abs", lambda x: _np.abs(x))       # ``abs`` shadows the builtin -> sabs
# binary
smin = _mk("min", lambda a, b: _np.minimum(a, b))
smax = _mk("max", lambda a, b: _np.maximum(a, b))
spow = _mk("pow", lambda a, b: _np.power(a, b))
atan2 = _mk("atan2", lambda a, b: _np.arctan2(a, b))
step = _mk("step", _np_step)
# ternary
clamp = _mk("clamp", _np_clamp)
mix = _mk("mix", _np_mix)
smoothstep = _mk("smoothstep", _np_smoothstep)


# ---------------------------------------------------------------------------
# preset patterns (shared 2-D-numeric / 3-D-emitted) — compose your own too
# ---------------------------------------------------------------------------

def waves(freq: Union[SpatialExpr, Signal, Number] = 1.0, axis: int = 0) -> SpatialExpr:
    """1-D sinusoid along one axis, remapped to [0, 1]."""
    c = (X, Y, Z)[axis]
    return 0.5 + 0.5 * sin(freq * c)


Coord3 = Tuple[Union[SpatialExpr, Signal, Number],
               Union[SpatialExpr, Signal, Number],
               Union[SpatialExpr, Signal, Number]]


def _offset(coord: SpatialExpr, c: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    # ``coord - c``, but a literal 0 offset drops out — so a default-centered
    # source emits the plain ``x``/``y``/``z`` (byte-identical to the old rings)
    # and costs no per-pixel subtraction.
    if isinstance(c, (int, float)) and c == 0.0:
        return coord
    return coord - c


def _radial(freq: Union[SpatialExpr, Signal, Number],
            center: Coord3 = (0.0, 0.0, 0.0)) -> SpatialExpr:
    """Signed radial wave ``sin(freq * |p - center|)`` from a point source at
    ``center``.  ``freq`` and each ``center`` component may be a number or an
    animated :class:`~loom.signals.core.Signal` (baked per frame)."""
    cx, cy, cz = center
    dx, dy, dz = _offset(X, cx), _offset(Y, cy), _offset(Z, cz)
    return sin(freq * sqrt(dx * dx + dy * dy + dz * dz))


def rings(freq: Union[SpatialExpr, Signal, Number] = 1.0,
          center: Coord3 = (0.0, 0.0, 0.0)) -> SpatialExpr:
    """Concentric shells ``0.5 + 0.5 sin(freq |p - center|)`` in [0, 1] from a
    point source at ``center`` (default origin)."""
    return 0.5 + 0.5 * _radial(freq, center)


def interference(freq: Union[SpatialExpr, Signal, Number] = 1.0,
                 source_a: Coord3 = (-0.5, 0.0, 0.0),
                 source_b: Coord3 = (0.5, 0.0, 0.0)) -> SpatialExpr:
    """Two-source interference in [0, 1]: the superposition (sum) of two radial
    waves from ``source_a`` and ``source_b``.  Where the two path lengths differ
    by a constant the crests reinforce, tracing the classic hyperbolic two-slit
    fringes; feed a :class:`~loom.signals.core.Signal` into a source coordinate
    to move an emitter and the fringes sweep (loop-safe if it returns by whole
    cycles per loop).  This is the *spatial* counterpart of the temporal beat you
    get for free from ``Sine(cycles=a) + Sine(cycles=b)``."""
    return 0.5 + 0.25 * (_radial(freq, source_a) + _radial(freq, source_b))


def moire(freq: Union[SpatialExpr, Signal, Number] = 1.0,
          angle: Union[Signal, Number] = 0.2,
          freq2: Union[SpatialExpr, Signal, Number, None] = None) -> SpatialExpr:
    """Moiré in [0, 1]: the superposition of two line gratings, the second
    rotated by ``angle`` radians (and optionally ruled at its own ``freq2``).
    The slow beat between the two nearly-aligned rulings is the moiré envelope;
    an animated ``angle`` :class:`~loom.signals.core.Signal` rotates one grating
    and the fringes crawl."""
    f2 = freq if freq2 is None else freq2
    xr = cos(angle) * X - sin(angle) * Y      # X of the grating rotated by `angle`
    return 0.5 + 0.25 * (sin(freq * X) + sin(f2 * xr))


def checker(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """3-D checkerboard in [0, 1] (sign of the product of three sines)."""
    return 0.5 + 0.5 * sign(sin(freq * X) * sin(freq * Y) * sin(freq * Z))


def gyroid(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """Schoen gyroid field ``sin x cos y + sin y cos z + sin z cos x`` (an
    isosurface field at ``=0``, or a signed pattern)."""
    fx, fy, fz = freq * X, freq * Y, freq * Z
    return sin(fx) * cos(fy) + sin(fy) * cos(fz) + sin(fz) * cos(fx)


SPATIAL_PATTERNS = {
    "waves": waves, "rings": rings, "checker": checker, "gyroid": gyroid,
    "interference": interference, "moire": moire,
}
